/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_FILTER_MODE_CUH__
#define __CUDA_FILTER_MODE_CUH__


#include "cudaFilterMode.h"
#include "cudaVector.h"
#include "cudaMath.h"


//////////////////////////////////////////////////////////////////////////////////////////
/// @name CUDA device functions for reading a single pixel, either in HWC or CHW layout.
/// @ingroup cudaFilter
//////////////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * CUDA device function for reading a pixel from an image, either in HWC or CHW layout.
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample
 * @param y desired y-coordinate to sample
 * @param width width of the input image
 * @param height height of the input image
 *
 * @returns the raw pixel data from the input image
 */
template<cudaDataFormat layout, typename T>
__device__ inline T cudaReadPixel( T* input, int x, int y, int width, int height )
{
	return input[y * width + x];
}

template<> __device__ inline 
float2 cudaReadPixel<FORMAT_CHW>( float2* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	return make_float2(ptr[offset], ptr[width * height + offset]);
}

template<> __device__ inline 
float3 cudaReadPixel<FORMAT_CHW>( float3* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	const int pixels = width * height;
	return make_float3(ptr[offset], ptr[pixels + offset], ptr[pixels * 2 + offset]);
}

template<> __device__ inline 
float4 cudaReadPixel<FORMAT_CHW>( float4* input, int x, int y, int width, int height )
{
	float* ptr = (float*)input;
	const int offset = y * width + x;
	const int pixels = width * height;
	return make_float4(ptr[offset], ptr[pixels + offset], ptr[pixels * 2 + offset], ptr[pixels * 3 + offset]);
}

///@}



// linear.
template<cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel_linear( T* input, float x, float y, int width, int height, float max_value=255.0f )
{
	const int x1 = __float2int_rd(x);
	const int y1 = __float2int_rd(y);
	const int x2 = x1 + 1;
	const int y2 = y1 + 1;
	const int x2_read = ::min(x2, width - 1);
	const int y2_read = ::min(y2, height - 1);

	const float4 pix_in[4] = {
		make_float4(cudaReadPixel<format>(input, x1, y1, width, height)),
		make_float4(cudaReadPixel<format>(input, x2_read, y1, width, height)),
		make_float4(cudaReadPixel<format>(input, x1, y2_read, width, height)),
		make_float4(cudaReadPixel<format>(input, x2_read, y2_read, width, height)),
	};
	const float weight[4] = {
		(x2 - x) * (y2 - y),
		(x - x1) * (y2 - y),
		(x2 - x) * (y - y1),
		(x - x1) * (y - y1),
	};

	const float4 pix_tmp =
		pix_in[0] * weight[0]
		+ pix_in[1] * weight[1]
		+ pix_in[2] * weight[2]
		+ pix_in[3] * weight[3];

	const T out = cast_vec<T>(clamp(pix_tmp, 0.0f, max_value));

	return out;
}



// cubic.
template <typename T>
static __device__ inline T calc_cubic_coef(T d, T a)
{
	T d_abs = abs(d);

	// T w1 = T(1) - (a + T(3)) * d_abs * d_abs + (a + T(2)) * d_abs * d_abs * d_abs;
	// T w2 = T(-4) * a + T(8) * a * d_abs + T(-5) * a * d_abs * d_abs + a * d_abs * d_abs * d_abs;
	T w1 = T(1) + ((-a + T(-3)) + (a + T(2)) * d_abs) * d_abs * d_abs;
	T w2 = (T(-4) + (T(8) + (T(-5) + d_abs) * d_abs) * d_abs) * a;

	T w = (d_abs < T(1)) ? w1 : (d_abs < T(2)) ? w2 : T(0);

	return w;
}
static __device__ inline float calc_cubic_coef_f(float d, float a)
{
	float d_abs = abs(d);

	float w1 = __fmaf_rn(__fmaf_rn(a + 2.0f, d_abs, -a - 3.0f), d_abs * d_abs, 1.0f);
	float w2 = __fmaf_rn(__fmaf_rn(__fmaf_rn(d_abs, 1.0f, -5.0f), d_abs, 8.0f), d_abs, -4.0f) * a;

	float w = (d_abs < 1.0f) ? w1 : (d_abs < 2.0f) ? w2 : 0.0f;

	return w;
}
template<cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel_cubic( T* input, float x, float y, int width, int height, float max_value = 255.0f )
{
	const int xc = __float2int_rd(x);
	const int yc = __float2int_rd(y);
	const int x_btm = xc - 1;
	const int y_btm = yc - 1;
	// const int x_top = xc + 2;
	const int y_top = yc + 2;
	const float xd = x - xc;
	const float yd = y - yc;

	constexpr float a = -0.75f;	// CPU ver. = -0.75, GPU ver. = -0.5
	const float wx[4] = {
		calc_cubic_coef_f(-1.0f - xd, a),
		calc_cubic_coef_f(0.0f - xd, a),
		calc_cubic_coef_f(1.0f - xd, a),
		calc_cubic_coef_f(2.0f - xd, a),
	};
	const float wx_sum = wx[0] + wx[1] + wx[2] + wx[3];

	float4 pix_sum = {};
	float w_sum = 0.0f;
	for (int i = y_btm; i <= y_top; i++) {
		const float wy = calc_cubic_coef_f(i - yd - yc, a);

		const int pos_x[4] = {
			::max(x_btm, 0),
			x_btm + 1,
			::min(x_btm + 2, width - 1),
			::min(x_btm + 3, width - 1),
		};
		const int pos_y = ::clamp(i, 0, height - 1);

		const float4 pix[4] = {
			make_float4(cudaReadPixel<format>(input, pos_x[0], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[1], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[2], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[3], pos_y, width, height)),
		};

		pix_sum += (pix[0] * wx[0] + pix[1] * wx[1] + pix[2] * wx[2] + pix[3] * wx[3]) * wy;
		w_sum += wx_sum * wy;
	}

	// const T out = (w_sum == 0.0f) ? T{} : cast_vec<T>(clamp((pix_sum / w_sum), 0.0f, max_value));
	float4 pix_out = {
		__fdividef(pix_sum.x, w_sum),
		__fdividef(pix_sum.y, w_sum),
		__fdividef(pix_sum.z, w_sum),
		__fdividef(pix_sum.w, w_sum),
	};
	const T out = (w_sum == 0.0f) ? T{} : cast_vec<T>(clamp(pix_out, 0.0f, max_value));

	return out;
}



// area.
template<cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel_area( T* input, float x, float y, int width, int height, float2 scale, float max_value = 255.0f )
{
	float fsx1 = x;
	float fsx2 = ::min(fsx1 + scale.x, (float)width);

	int sx1 = __float2int_ru(fsx1);
	int sx2 = __float2int_rd(fsx2);

	float fsy1 = y;
	float fsy2 = ::min(fsy1 + scale.y, (float)height);

	int sy1 = __float2int_ru(fsy1);
	int sy2 = __float2int_rd(fsy2);

	float sx = ::min(float(scale.x), width - fsx1);
	float sy = ::min(float(scale.y), height - fsy1);
	float scale_xy = 1.f / (sx * sy);

	float4 out = {};

	for (int dy = sy1; dy < sy2; ++dy)
	{
		// inner rectangle.
		for (int dx = sx1; dx < sx2; ++dx)
			out += make_float4(cudaReadPixel<format>(input, dx, dy, width, height));

		// v-edge line(left).
		if (sx1 > fsx1)
			out += make_float4(cudaReadPixel<format>(input, (sx1 -1), dy, width, height) * (sx1 - fsx1));

		// v-edge line(right).
		if (sx2 < fsx2)
			out += make_float4(cudaReadPixel<format>(input, sx2, dy, width, height) * (fsx2 - sx2));
	}

	// h-edge line(top).
	if (sy1 > fsy1)
		for (int dx = sx1; dx < sx2; ++dx)
			out += make_float4(cudaReadPixel<format>(input, dx, (sy1 - 1), width, height) * (sy1 - fsy1));

	// h-edge line(bottom).
	if (sy2 < fsy2)
		for (int dx = sx1; dx < sx2; ++dx)
			out += make_float4(cudaReadPixel<format>(input, dx, sy2, width, height) * (fsy2 - sy2));

	// corner(top, left).
	if ((sy1 > fsy1) &&  (sx1 > fsx1))
		out += make_float4(cudaReadPixel<format>(input, (sx1 - 1), (sy1 - 1), width, height) * (sy1 - fsy1) * (sx1 - fsx1));

	// corner(top, right).
	if ((sy1 > fsy1) &&  (sx2 < fsx2))
		out += make_float4(cudaReadPixel<format>(input, sx2, (sy1 - 1), width, height) * (sy1 - fsy1) * (fsx2 - sx2));

	// corner(bottom, left).
	if ((sy2 < fsy2) &&  (sx2 < fsx2))
		out += make_float4(cudaReadPixel<format>(input, sx2, sy2, width, height) * (fsy2 - sy2) * (fsx2 - sx2));

	// corner(bottom, right).
	if ((sy2 < fsy2) &&  (sx1 > fsx1))
		out += make_float4(cudaReadPixel<format>(input, (sx1 - 1), sy2, width, height) * (fsy2 - sy2) * (sx1 - fsx1));

	out *= scale_xy;

	return cast_vec<T>(clamp(out, 0.0f, max_value));
}



// lanczos4.
//// /* # SLOW. */
// template <typename T>
// static __device__  inline T calc_sinc(T x)
// {
// 	return (x == T(0)) ? T(1) : sin(x * M_PI) / (x * M_PI);
// }
// template <typename T>
// static __device__ inline T calc_lanczos_coef(T d, T n)
// {
// 	T d_abs = abs(d);
// 	T w = (d_abs > n) ? T(0) : calc_sinc(d_abs) * calc_sinc(d_abs / n);

// 	return w;
// }
//// /* # FAST. */
template <typename T>
static __device__ inline T calc_lanczos_coef(T d, T n)
{
	T d_abs = abs(d);

	T pi_d = M_PI * d_abs;
	T cos_k1 = cos(pi_d * 0.25f);
	T cos_k2 = sqrt(1.0f - cos_k1 * cos_k1);

	constexpr auto zero = 1e-3f;
	T w = (d_abs >= n) ? T(0) :
		(d_abs < T(zero)) ? T(1) :
		(T(16) * cos_k1 * cos_k2 * (cos_k1 - cos_k2) * (cos_k1 + cos_k2) * cos_k2) / (pi_d * pi_d);

	return w;
}
static __device__ inline float calc_lanczos_coef_f(float d, float n)
{
	float d_abs = abs(d);

	float pi_d = (float)M_PI * d_abs;
	float cos_k1 = __cosf(pi_d * 0.25f);
	float cos_k2 = __fsqrt_rn(1.0f - cos_k1 * cos_k1);

	constexpr auto zero = 1e-3f;
	float w = (d_abs >= n) ? 0.0f :
		(d_abs < zero) ? 1.0f :
		__fdividef(16.0f * cos_k1 * cos_k2 * (cos_k1 - cos_k2) * (cos_k1 + cos_k2) * cos_k2, (pi_d * pi_d));

	return w;
}
template<cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel_lanczos4( T* input, float x, float y, int width, int height, float max_value = 255.0f )
{
	constexpr float tap = 4.0f;

	const int xc = __float2int_rd(x);
	const int yc = __float2int_rd(y);
	const int x_btm = xc - 3;
	const int y_btm = yc - 3;
	// const int x_top = xc + 4;
	const int y_top = yc + 4;
	const float xd = x - xc;
	const float yd = y - yc;

	const float wx[8] = {
		calc_lanczos_coef_f(-3.0f - xd, tap),
		calc_lanczos_coef_f(-2.0f - xd, tap),
		calc_lanczos_coef_f(-1.0f - xd, tap),
		calc_lanczos_coef_f(0.0f - xd, tap),
		calc_lanczos_coef_f(1.0f - xd, tap),
		calc_lanczos_coef_f(2.0f - xd, tap),
		calc_lanczos_coef_f(3.0f - xd, tap),
		calc_lanczos_coef_f(4.0f - xd, tap),
	};
	const float wx_sum = wx[0] + wx[1] + wx[2] + wx[3] + wx[4] + wx[5] + wx[6] + wx[7];

	float4 pix_sum = {};
	float w_sum = 0.0f;
	for (int i = y_btm; i <= y_top; i++) {
		const float wy = calc_lanczos_coef_f(i - yd - yc, tap);

		const int pos_x[8] = {
			::max(x_btm, 0),
			::max(x_btm + 1, 0),
			::max(x_btm + 2, 0),
			x_btm + 3,
			::min(x_btm + 4, width - 1),
			::min(x_btm + 5, width - 1),
			::min(x_btm + 6, width - 1),
			::min(x_btm + 7, width - 1),
		};
		const int pos_y = ::clamp(i, 0, height - 1);

		const float4 pix[8] = {
			make_float4(cudaReadPixel<format>(input, pos_x[0], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[1], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[2], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[3], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[4], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[5], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[6], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[7], pos_y, width, height)),
		};

		const float4 pix_0 = pix[0] * wx[0] + pix[1] * wx[1] + pix[2] * wx[2] + pix[3] * wx[3];
		const float4 pix_1 = pix[4] * wx[4] + pix[5] * wx[5] + pix[6] * wx[6] + pix[7] * wx[7];
		pix_sum += (pix_0 + pix_1) * wy;
		w_sum += wx_sum * wy;
	}

	// const T out = (w_sum == 0.0f) ? T{} : cast_vec<T>(clamp(pix_sum / w_sum, 0.0f, max_value));
	float4 pix_out = {
		__fdividef(pix_sum.x, w_sum),
		__fdividef(pix_sum.y, w_sum),
		__fdividef(pix_sum.z, w_sum),
		__fdividef(pix_sum.w, w_sum),
	};
	const T out = (w_sum == 0.0f) ? T{} : cast_vec<T>(clamp(pix_out, 0.0f, max_value));

	return out;
}



// spline36.
template <typename T>
static __device__ inline T calc_spline36_coef(T d)
{
	T d_abs = abs(d);

	T w = (d_abs > T(3)) ? T(0)
		: (d_abs > T(2)) ? (((T(19) * d_abs + T(-159)) * d_abs + T(434)) * d_abs + T(-384)) / T(209)
		: (d_abs > T(1)) ? (((T(-114) * d_abs + T(612)) * d_abs + T(-1038)) * d_abs + T(540)) / T(209)
		: (((T(247) * d_abs + T(-453)) * d_abs + T(-3)) * d_abs + T(209)) / T(209);

	return w;
}
static __device__ inline float calc_spline36_coef_f(float d)
{
	float d_abs = abs(d);

	float w = (d_abs > 3.0f) ? 0.0f
		: (d_abs > 2.0f) ? __fdividef(__fmaf_rn(__fmaf_rn(__fmaf_rn(19.0f, d_abs, -159.0f), d_abs, 434.0f), d_abs, -384.0f), 209.0f)
		: (d_abs > 1.0f) ? __fdividef(__fmaf_rn(__fmaf_rn(__fmaf_rn(-114.0f, d_abs, 612.0f), d_abs, -1038.0f), d_abs, 540.0f), 209.0f)
		: __fdividef(__fmaf_rn(__fmaf_rn(__fmaf_rn(247.0f, d_abs, -453.0f), d_abs, -3.0f), d_abs, 209.0f), 209.0f);

	return w;
}
template<cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel_spline36( T* input, float x, float y, int width, int height, float max_value = 255.0f )
{
	const int xc = __float2int_rd(x);
	const int yc = __float2int_rd(y);
	const int x_btm = xc - 2;
	const int y_btm = yc - 2;
	// const int x_top = xc + 3;
	const int y_top = yc + 3;
	const float xd = x - xc;
	const float yd = y - yc;

	const float wx[6] = {
		calc_spline36_coef_f(-2.0f - xd),
		calc_spline36_coef_f(-1.0f - xd),
		calc_spline36_coef_f(0.0f - xd),
		calc_spline36_coef_f(1.0f - xd),
		calc_spline36_coef_f(2.0f - xd),
		calc_spline36_coef_f(3.0f - xd),
	};
	const float wx_sum = wx[0] + wx[1] + wx[2] + wx[3] + wx[4] + wx[5];

	float4 pix_sum = {};
	float w_sum = 0.0f;
	for (int i = y_btm; i <= y_top; i++) {
		const float wy = calc_spline36_coef_f(i - yd - yc);

		const int pos_x[6] = {
			::max(x_btm, 0),
			::max(x_btm + 1, 0),
			x_btm + 2,
			::min(x_btm + 3, width - 1),
			::min(x_btm + 4, width - 1),
			::min(x_btm + 5, width - 1),
		};
		const int pos_y = ::clamp(i, 0, height - 1);

		const float4 pix[6] = {
			make_float4(cudaReadPixel<format>(input, pos_x[0], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[1], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[2], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[3], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[4], pos_y, width, height)),
			make_float4(cudaReadPixel<format>(input, pos_x[5], pos_y, width, height)),
		};

		const float4 pix6 = pix[0] * wx[0] + pix[1] * wx[1] + pix[2] * wx[2]
							+ pix[3] * wx[3] + pix[4] * wx[4] + pix[5] * wx[5];
		pix_sum += pix6 * wy;
		w_sum += wx_sum * wy;
	}

	// const T out = (w_sum == 0.0f) ? T{} : cast_vec<T>(clamp(pix_sum / w_sum, 0.0f, max_value));
	float4 pix_out = {
		__fdividef(pix_sum.x, w_sum),
		__fdividef(pix_sum.y, w_sum),
		__fdividef(pix_sum.z, w_sum),
		__fdividef(pix_sum.w, w_sum),
	};
	const T out = (w_sum == 0.0f) ? T{} : cast_vec<T>(clamp(pix_out, 0.0f, max_value));

	return out;
}

/**
 * CUDA device function for sampling a pixel with bilinear or point filtering.
 * cudaFilterPixel() is for use inside of other CUDA kernels, and accepts a
 * cudaFilterMode template parameter which sets the filtering mode, in addition
 * to a cudaDataFormat template parameter which sets the format (HWC or CHW).
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample
 * @param y desired y-coordinate to sample
 * @param width width of the input image
 * @param height height of the input image
 *
 * @returns the filtered pixel from the input image
 * @ingroup cudaFilter
 */ 
template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel( T* input, float x, float y, int width, int height, float max_value = 255.0f )
{
	if( filter == FILTER_POINT )
	{
		const int x1 = int(x);
		const int y1 = int(y);

		return cudaReadPixel<format>(input, x1, y1, width, height); //input[y1 * width + x1];
	}
	else if ( filter == FILTER_CUBIC )
	{
		return cudaFilterPixel_cubic<format>(input, x, y, width, height, max_value);
	}
	else if ( filter == FILTER_LANCZOS4 )
	{
		return cudaFilterPixel_lanczos4<format>(input, x, y, width, height, max_value);
	}
	else if ( filter == FILTER_SPLINE36 )
	{
		return cudaFilterPixel_spline36<format>(input, x, y, width, height, max_value);
	}
	else // FILTER_LINEAR
	{
		return cudaFilterPixel_linear<format>(input, x, y, width, height, max_value);
	}
}

/**
 * CUDA device function for sampling a pixel with bilinear or point filtering.
 * cudaFilterPixel() is for use inside of other CUDA kernels, and samples a
 * pixel from an input image from the scaled coordinates of an output image.
 *
 * @param input pointer to image in CUDA device memory
 * @param x desired x-coordinate to sample (in coordinate space of output image)
 * @param y desired y-coordinate to sample (in coordinate space of output image)
 * @param input_width width of the input image
 * @param input_height height of the input image
 * @param output_width width of the output image
 * @param output_height height of the output image
 *
 * @returns the filtered pixel from the input image
 * @ingroup cudaFilter
 */ 
 template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
 __device__ inline T cudaFilterPixel( T* input, float x, float y,
								int input_width, int input_height,
								int output_width, int output_height,
								float2 scale, float max_value = 255.0f )
 {
	 const float px =
	 	(filter == FILTER_POINT || filter == FILTER_AREA)
		 ? (x * scale.x)
		 : ((x + 0.5f) * scale.x - 0.5f);
	 const float py =
	 	(filter == FILTER_POINT || filter == FILTER_AREA)
		 ? (y * scale.y)
		 : ((y + 0.5f) * scale.y - 0.5f);

	 if ( filter == FILTER_AREA ) {
		 if (scale.x > 1.0f && scale.y > 1.0f) {
			 return cudaFilterPixel_area<format>(input, px, py, input_width, input_height, scale, max_value);
		 } else {
			 return cudaFilterPixel_linear<format>(input, px, py, input_width, input_height, max_value);
		 }
	 } else {
		 return cudaFilterPixel<filter, format>(input, px, py, input_width, input_height, max_value);
	 }
 }
 template<cudaFilterMode filter, cudaDataFormat format=FORMAT_HWC, typename T>
__device__ inline T cudaFilterPixel( T* input, float x, float y,
						       int input_width, int input_height,
						       int output_width, int output_height,
							   float max_value = 255.0f )
{
	const float2 scale = {
		__fdividef(float(input_width), float(output_width)),
		__fdividef(float(input_height), float(output_height)),
	};

	return cudaFilterPixel<filter, format>(input, x, y, input_width, input_height, output_width, output_height, scale, max_value);
}


#endif
