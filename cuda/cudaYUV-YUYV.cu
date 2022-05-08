/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "cudaYUV.h"
#include "imageFormat.h"
#include "cudaYUV_internal.h"


//-----------------------------------------------------------------------------------
// YUV to RGB colorspace conversion
//-----------------------------------------------------------------------------------
// static inline __device__ float clamp( float x )
// {
// 	return fminf(fmaxf(x, 0.0f), 255.0f);
// }

// static inline __device__ float3 YUV2RGB(float Y, float U, float V)
// {
// 	U -= 128.0f;
// 	V -= 128.0f;

// #if 1
// 	return make_float3(clamp(Y + 1.4065f * V),
//     				    clamp(Y - 0.3455f * U - 0.7169f * V),
// 				    clamp(Y + 1.7790f * U));
// #else
// 	return make_float3(clamp(Y + 1.402f * V),
//     				    clamp(Y - 0.344f * U - 0.714f * V),
// 				    clamp(Y + 1.772f * U));
// #endif
// }

static inline __device__ __host__ float3 YUV2RGB(const float3 &yuv, int width, int height, float in_max, float out_max, bool is_limited = true)
{
	float3 rgb;

	if (width > 1920 || height > 1080) {
		rgb = is_limited ? YUV2RGB_2020_limited(yuv, in_max, out_max) : YUV2RGB_2020_full(yuv, in_max, out_max);
	} else if (width > 720 || height > 576) {
		rgb = is_limited ? YUV2RGB_709_limited(yuv, in_max, out_max) : YUV2RGB_709_full(yuv, in_max, out_max);
	} else {
		rgb = is_limited ? YUV2RGB_601_limited(yuv, in_max, out_max) : YUV2RGB_601_full(yuv, in_max, out_max);
	}

	return rgb;
}

static inline __device__ __host__ float3 RGB2YUV(const float3 &rgb, int width, int height, float in_max, float out_max, bool is_limited = true)
{
	float3 yuv;

	if (width > 1920 || height > 1080) {
		yuv = is_limited ? RGB2YUV_2020_limited(rgb, in_max, out_max) : RGB2YUV_2020_full(rgb, in_max, out_max);
	} else if (width > 720 || height > 576) {
		yuv = is_limited ? RGB2YUV_709_limited(rgb, in_max, out_max) : RGB2YUV_709_full(rgb, in_max, out_max);
	} else {
		yuv = is_limited ? RGB2YUV_601_limited(rgb, in_max, out_max) : RGB2YUV_601_full(rgb, in_max, out_max);
	}

	return yuv;
}


//-----------------------------------------------------------------------------------
// YUYV/UYVY are macropixel formats, and two RGB pixels are output at once.
// Define vectors with 6 and 8 elements so they can be written at one time.
// These are similar to those from cudaVector.h, except for 6/8 elements.
//-----------------------------------------------------------------------------------
struct /*__align__(6)*/ uchar6
{
   uint8_t x0, y0, z0, x1, y1, z1;
};

struct __align__(8) uchar8
{
   uint8_t x0, y0, z0, w0, x1, y1, z1, w1;
};

struct /*__align__(24)*/ float6
{
   float x0, y0, z0, x1, y1, z1;
};

struct __align__(32) float8
{
   float x0, y0, z0, w0, x1, y1, z1, w1;
};

template<class T> struct vecTypeInfo;

template<> struct vecTypeInfo<uchar2> { typedef uint8_t Base; };
template<> struct vecTypeInfo<uchar6> { typedef uint8_t Base; };
template<> struct vecTypeInfo<uchar8> { typedef uint8_t Base; };

template<> struct vecTypeInfo<float2> { typedef float Base; };
template<> struct vecTypeInfo<float6> { typedef float Base; };
template<> struct vecTypeInfo<float8> { typedef float Base; };

template<typename T> struct vec_assert_false : std::false_type { };

#define BaseType typename vecTypeInfo<T>::Base

template<typename T> inline __host__ __device__ T make_vec(BaseType x0, BaseType y0, BaseType z0, BaseType w0, BaseType x1, BaseType y1, BaseType z1, BaseType w1) { static_assert(vec_assert_false<T>::value, "invalid vector type - supported types are uchar6, uchar8, float6, float8");  }

template<> inline __host__ __device__ uchar6 make_vec( uint8_t x0, uint8_t y0, uint8_t z0, uint8_t w0, uint8_t x1, uint8_t y1, uint8_t z1, uint8_t w1 )	{ return {x0, y0, z0, x1, y1, z1}; }
template<> inline __host__ __device__ uchar8 make_vec( uint8_t x0, uint8_t y0, uint8_t z0, uint8_t w0, uint8_t x1, uint8_t y1, uint8_t z1, uint8_t w1 )	{ return {x0, y0, z0, w1, x1, y1, z1, w1}; }

template<> inline __host__ __device__ float6 make_vec( float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )				{ return {x0, y0, z0, x1, y1, z1}; }
template<> inline __host__ __device__ float8 make_vec( float x0, float y0, float z0, float w0, float x1, float y1, float z1, float w1 )				{ return {x0, y0, z0, w1, x1, y1, z1, w1}; }

// make_float8.
template<typename T> inline __host__ __device__ float8 make_float8(T &v, float a) { static_assert(vec_assert_false<T>::value, "invalid vector type - supported types are uchar2, float2, uchar6, uchar8, float6, float8");  }
template<> inline __host__ __device__ float8 make_float8(uchar2 &v, float a) { return make_vec<float8>(v.x, v.x, v.x, a, v.y, v.y, v.y, a); }
template<> inline __host__ __device__ float8 make_float8(uchar6 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, a, v.x1, v.y1, v.z1, a); }
template<> inline __host__ __device__ float8 make_float8(uchar8 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, v.w0, v.x1, v.y1, v.z1, v.w1); }
template<> inline __host__ __device__ float8 make_float8(float2 &v, float a) { return make_vec<float8>(v.x, v.x, v.x, a, v.y, v.y, v.y, a); }
template<> inline __host__ __device__ float8 make_float8(float6 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, a, v.x1, v.y1, v.z1, a); }
template<> inline __host__ __device__ float8 make_float8(float8 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, v.w0, v.x1, v.y1, v.z1, v.w1); }
template<> inline __host__ __device__ float8 make_float8(const uchar2 &v, float a) { return make_vec<float8>(v.x, v.x, v.x, a, v.y, v.y, v.y, a); }
template<> inline __host__ __device__ float8 make_float8(const uchar6 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, a, v.x1, v.y1, v.z1, a); }
template<> inline __host__ __device__ float8 make_float8(const uchar8 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, v.w0, v.x1, v.y1, v.z1, v.w1); }
template<> inline __host__ __device__ float8 make_float8(const float2 &v, float a) { return make_vec<float8>(v.x, v.x, v.x, a, v.y, v.y, v.y, a); }
template<> inline __host__ __device__ float8 make_float8(const float6 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, a, v.x1, v.y1, v.z1, a); }
template<> inline __host__ __device__ float8 make_float8(const float8 &v, float a) { return make_vec<float8>(v.x0, v.y0, v.z0, v.w0, v.x1, v.y1, v.z1, v.w1); }

//-----------------------------------------------------------------------------------
// YUYV/UYVY to RGBA
//-----------------------------------------------------------------------------------
template <typename T, imageFormat format>
__global__ void YUYVToRGBA( uchar4* src, T* dst, int halfWidth, int height, float in_max = 255.0f, float out_max = 255.0f )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= halfWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * halfWidth + x];

	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U and V is the color of both pixels.
	float y0, y1, u, v;

	if( format == IMAGE_YUYV )
	{
		// YUYV [ Y0 | U0 | Y1 | V0 ]
		y0 = macroPx.x;
		y1 = macroPx.z;
		u  = macroPx.y;
		v  = macroPx.w;
	}
	else if( format == IMAGE_YVYU )
	{
		// YVYU [ Y0 | V0 | Y1 | U0 ]
		y0 = macroPx.x;
		y1 = macroPx.z;
		u  = macroPx.w;
		v  = macroPx.y;
	}
	else // if( format == IMAGE_UYVY )
	{
		// UYVY [ U0 | Y0 | V0 | Y1 ]
		y0 = macroPx.y;
		y1 = macroPx.w;
		u  = macroPx.x;
		v  = macroPx.z;
	}

	float a0, a1;
	if (format == IMAGE_UYVA) {
		const uchar2 *src_alpha = (uchar2 *)(&src[height * halfWidth]);
		const uchar2 alphaPx = src_alpha[y * halfWidth + x];
		a0 = alphaPx.x;
		a1 = alphaPx.y;
	} else {
		a0 = out_max;
		a1 = out_max;
	}

	// this function outputs two pixels from one YUYV macropixel
	float3 yuv0 = make_float3(y0, u, v);
	float3 yuv1 = make_float3(y1, u, v);
	constexpr bool is_limited = true;
	const float3 px0 = YUV2RGB(yuv0, halfWidth * 2, height, in_max, out_max, is_limited);
	const float3 px1 = YUV2RGB(yuv1, halfWidth * 2, height, in_max, out_max, is_limited);

	dst[y * halfWidth + x] = make_vec<T>(px0.x, px0.y, px0.z, a0,
								  px1.x, px1.y, px1.z, a1);
}

// GRAY/RGB/RGBA to UYVA.
// src: [0, in_max].
// dst: [0, out_max].
template<typename T, bool is_BGR>	// T: uchar2, float2, uchar6, float6, uchar8, float8.
__global__ void ConvertToUYVA( T* src, uint8_t* dst, int halfWidth, int height, float in_max, float out_max )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= halfWidth || y >= height )
		return;

	const T macroPx = src[y * halfWidth + x];

	const float8 rgb01 = make_float8(macroPx, in_max);
	const float3 rgb0 = is_BGR ? float3{ rgb01.z0, rgb01.y0, rgb01.x0 } : float3{ rgb01.x0, rgb01.y0, rgb01.z0 };
	const float3 rgb1 = is_BGR ? float3{ rgb01.z1, rgb01.y1, rgb01.x1 } : float3{ rgb01.x1, rgb01.y1, rgb01.z1 };
	const float a0 = rgb01.w0;
	const float a1 = rgb01.w1;

	constexpr bool is_limited = true;
	float3 yuv0 = RGB2YUV(rgb0, halfWidth * 2, height, in_max, out_max, is_limited);
	float3 yuv1 = RGB2YUV(rgb1, halfWidth * 2, height, in_max, out_max, is_limited);

	float y0 = yuv0.x, y1 = yuv1.x;
	float u = yuv0.y, v = yuv0.z;
	const float scale = out_max / in_max;
	float alpha0 = a0 * scale;
	float alpha1 = a1 * scale;

	dst[(y * halfWidth + x) * 4 + 0] = static_cast<uint8_t>(u);
	dst[(y * halfWidth + x) * 4 + 1] = static_cast<uint8_t>(y0);
	dst[(y * halfWidth + x) * 4 + 2] = static_cast<uint8_t>(v);
	dst[(y * halfWidth + x) * 4 + 3] = static_cast<uint8_t>(y1);
	dst[height * halfWidth * 4 + (y * halfWidth + x) * 2 + 0] = static_cast<uint8_t>(alpha0);
	dst[height * halfWidth * 4 + (y * halfWidth + x) * 2 + 1] = static_cast<uint8_t>(alpha1);
}

// GRAY/RGB/RGBA to PA16.
// src: [0, in_max].
// dst: [0, out_max].
template<typename T, bool is_BGR>	// T: uchar2, float2, uchar6, float6, uchar8, float8.
__global__ void ConvertToPA16( T* src, uint16_t* dst, int halfWidth, int height, float in_max, float out_max )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= halfWidth || y >= height )
		return;

	const T macroPx = src[y * halfWidth + x];

	const float8 rgb01 = make_float8(macroPx, in_max);
	const float3 rgb0 = is_BGR ? float3{ rgb01.z0, rgb01.y0, rgb01.x0 } : float3{ rgb01.x0, rgb01.y0, rgb01.z0 };
	const float3 rgb1 = is_BGR ? float3{ rgb01.z1, rgb01.y1, rgb01.x1 } : float3{ rgb01.x1, rgb01.y1, rgb01.z1 };
	const float a0 = rgb01.w0;
	const float a1 = rgb01.w1;

	constexpr bool is_limited = true;
	float3 yuv0 = RGB2YUV(rgb0, halfWidth * 2, height, in_max, out_max, is_limited);
	float3 yuv1 = RGB2YUV(rgb1, halfWidth * 2, height, in_max, out_max, is_limited);

	float y0 = yuv0.x, y1 = yuv1.x;
	float u = yuv0.y, v = yuv0.z;
	const float scale = out_max / in_max;
	float alpha0 = a0 * scale;
	float alpha1 = a1 * scale;

	dst[(y * halfWidth + x) * 2 + 0] = static_cast<uint16_t>(y0);
	dst[(y * halfWidth + x) * 2 + 1] = static_cast<uint16_t>(y1);
	dst[((y + height) * halfWidth + x) * 2 + 0] = static_cast<uint16_t>(u);
	dst[((y + height) * halfWidth + x) * 2 + 1] = static_cast<uint16_t>(v);
	dst[height * halfWidth * 4 + (y * halfWidth + x) * 2 + 0] = static_cast<uint16_t>(alpha0);
	dst[height * halfWidth * 4 + (y * halfWidth + x) * 2 + 1] = static_cast<uint16_t>(alpha1);
}

template<typename T, imageFormat format>
static cudaError_t launchYUYVToRGB( void* input, T* output, size_t width, size_t height, float in_max, float out_max, cudaStream_t stream)
{
	if( !input || !output || !width || !height )
		return cudaErrorInvalidValue;

	const int  halfWidth = width / 2;	// two pixels are output at once
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(halfWidth, blockDim.x), iDivUp(height, blockDim.y));

	YUYVToRGBA<T, format><<<gridDim, blockDim, 0, stream>>>((uchar4*)input, output, halfWidth, height, in_max, out_max);

	return CUDA(cudaGetLastError());
}

template<typename T, typename T2, bool is_BGR>
static cudaError_t launchConvertToUYVA( T* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, cudaStream_t stream)
{
	if( !input || !output || !width || !height )
		return cudaErrorInvalidValue;

	const int  halfWidth = width / 2;	// two pixels are output at once
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(halfWidth, blockDim.x), iDivUp(height, blockDim.y));

	ConvertToUYVA<T2, is_BGR><<<gridDim, blockDim, 0, stream>>>((T2*)input, output, halfWidth, height, in_max, out_max);

	return CUDA(cudaGetLastError());
}

template<typename T, typename T2, bool is_BGR>
static cudaError_t launchConvertToPA16( T* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, cudaStream_t stream)
{
	if( !input || !output || !width || !height )
		return cudaErrorInvalidValue;

	const int  halfWidth = width / 2;	// two pixels are output at once
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(halfWidth, blockDim.x), iDivUp(height, blockDim.y));

	ConvertToPA16<T2, is_BGR><<<gridDim, blockDim, 0, stream>>>((T2*)input, output, halfWidth, height, in_max, out_max);

	return CUDA(cudaGetLastError());
}

// cudaYUYVToRGB (uchar3)
cudaError_t cudaYUYVToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_YUYV>(input, (uchar6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaYUYVToRGB (float3)
cudaError_t cudaYUYVToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_YUYV>(input, (float6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaYUYVToRGBA (uchar4)
cudaError_t cudaYUYVToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_YUYV>(input, (uchar8*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaYUYVToRGBA (float4)
cudaError_t cudaYUYVToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_YUYV>(input, (float8*)output, width, height, 255.0f, 255.0f, stream);
}

//-----------------------------------------------------------------------------------

// cudaUYVYToRGB (uchar3)
cudaError_t cudaUYVYToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_UYVY>(input, (uchar6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVYToRGB (float3)
cudaError_t cudaUYVYToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_UYVY>(input, (float6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVYToRGBA (uchar4)
cudaError_t cudaUYVYToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_UYVY>(input, (uchar8*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVYToRGBA (float4)
cudaError_t cudaUYVYToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_UYVY>(input, (float8*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVAToRGB (uchar3)
cudaError_t cudaUYVAToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_UYVA>(input, (uchar6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVAToRGB (float3)
cudaError_t cudaUYVAToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_UYVA>(input, (float6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVAToRGBA (uchar4)
cudaError_t cudaUYVAToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_UYVA>(input, (uchar8*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaUYVAToRGBA (float4)
cudaError_t cudaUYVAToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_UYVA>(input, (float8*)output, width, height, 255.0f, 255.0f, stream);
}

//-----------------------------------------------------------------------------------

// cudaYVYUToRGB (uchar3)
cudaError_t cudaYVYUToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar6, IMAGE_YVYU>(input, (uchar6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaYUYVToRGB (float3)
cudaError_t cudaYVYUToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float6, IMAGE_YVYU>(input, (float6*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaYUYVToRGBA (uchar4)
cudaError_t cudaYVYUToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<uchar8, IMAGE_YVYU>(input, (uchar8*)output, width, height, 255.0f, 255.0f, stream);
}

// cudaYUYVToRGBA (float4)
cudaError_t cudaYVYUToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream )
{
	return launchYUYVToRGB<float8, IMAGE_YVYU>(input, (float8*)output, width, height, 255.0f, 255.0f, stream);
}



//-------------------------------------------------------------------------------------
// GRAY/RGB/RGBA/BGR/BGRA to YUV(+A)
//-------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------

// GRAY/RGB/RGBA/BGR/BGRA to UYVA 4:2:2:4.
cudaError_t cudaConvertToUYVA( uint8_t* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	return launchConvertToUYVA<uint8_t, uchar2, false>(input, output, width, height, in_max, out_max, stream);
}
cudaError_t cudaConvertToUYVA( float* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	return launchConvertToUYVA<float, float2, false>(input, output, width, height, in_max, out_max, stream);
}
cudaError_t cudaConvertToUYVA( uchar3* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToUYVA<uchar3, uchar6, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToUYVA<uchar3, uchar6, false>(input, output, width, height, in_max, out_max, stream);
	}
}
cudaError_t cudaConvertToUYVA( float3* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToUYVA<float3, float6, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToUYVA<float3, float6, false>(input, output, width, height, in_max, out_max, stream);
	}
}
cudaError_t cudaConvertToUYVA( uchar4* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToUYVA<uchar4, uchar8, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToUYVA<uchar4, uchar8, false>(input, output, width, height, in_max, out_max, stream);
	}
}
cudaError_t cudaConvertToUYVA( float4* input, uint8_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToUYVA<float4, float8, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToUYVA<float4, float8, false>(input, output, width, height, in_max, out_max, stream);
	}
}

cudaError_t cudaConvertToUYVA( void* input,  uint8_t* output, size_t width, size_t height, imageFormat format, float in_max, float out_max, cudaStream_t stream )
{
	cudaError_t err;

	switch(format) {
	case IMAGE_GRAY8:
		err = cudaConvertToUYVA((uint8_t *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_GRAY32F:
		err = cudaConvertToUYVA((float *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGB8:
		err = cudaConvertToUYVA((uchar3 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGB32F:
		err = cudaConvertToUYVA((float3 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGBA8:
		err = cudaConvertToUYVA((uchar4 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGBA32F:
		err = cudaConvertToUYVA((float4 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_BGR8:
		err = cudaConvertToUYVA((uchar3 *)input, output, width, height, in_max, out_max, true, stream);
		break;
	case IMAGE_BGR32F:
		err = cudaConvertToUYVA((float3 *)input, output, width, height, in_max, out_max, true, stream);
		break;
	case IMAGE_BGRA8:
		err = cudaConvertToUYVA((uchar4 *)input, output, width, height, in_max, out_max, true, stream);
		break;
	case IMAGE_BGRA32F:
		err = cudaConvertToUYVA((float4 *)input, output, width, height, in_max, out_max, true, stream);
		break;

	default:
		LogError(LOG_CUDA "cudaConvertToUYVA() -- invalid image format '%s'\n", imageFormatToStr(format));
		LogError(LOG_CUDA "                       supported formats are:\n");
		LogError(LOG_CUDA "                           * gray8, gray32f\n");
		LogError(LOG_CUDA "                           * rgb8, bgr8, rgb32f, bgr32f\n");
		LogError(LOG_CUDA "                           * rgba8, bgra8, rgba32f, bgra32f\n");

		err = cudaErrorInvalidValue;
	}

	return err;
}

//-----------------------------------------------------------------------------------

// GRAY/RGB/RGBA/BGR/BGRA to YUV+A PA16 4:2:2:4.
cudaError_t cudaConvertToPA16( uint8_t* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	return launchConvertToPA16<uint8_t, uchar2, false>(input, output, width, height, in_max, out_max, stream);
}
cudaError_t cudaConvertToPA16( float* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	return launchConvertToPA16<float, float2, false>(input, output, width, height, in_max, out_max, stream);
}
cudaError_t cudaConvertToPA16( uchar3* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToPA16<uchar3, uchar6, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToPA16<uchar3, uchar6, false>(input, output, width, height, in_max, out_max, stream);
	}
}
cudaError_t cudaConvertToPA16( float3* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToPA16<float3, float6, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToPA16<float3, float6, false>(input, output, width, height, in_max, out_max, stream);
	}
}
cudaError_t cudaConvertToPA16( uchar4* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToPA16<uchar4, uchar8, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToPA16<uchar4, uchar8, false>(input, output, width, height, in_max, out_max, stream);
	}
}
cudaError_t cudaConvertToPA16( float4* input, uint16_t* output, size_t width, size_t height, float in_max, float out_max, bool is_BGR, cudaStream_t stream )
{
	if (is_BGR) {
		return launchConvertToPA16<float4, float8, true>(input, output, width, height, in_max, out_max, stream);
	} else {
		return launchConvertToPA16<float4, float8, false>(input, output, width, height, in_max, out_max, stream);
	}
}

cudaError_t cudaConvertToPA16( void* input,  uint16_t* output, size_t width, size_t height, imageFormat format, float in_max, float out_max, cudaStream_t stream )
{
	cudaError_t err;

	switch(format) {
	case IMAGE_GRAY8:
		err = cudaConvertToPA16((uint8_t *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_GRAY32F:
		err = cudaConvertToPA16((float *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGB8:
		err = cudaConvertToPA16((uchar3 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGB32F:
		err = cudaConvertToPA16((float3 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGBA8:
		err = cudaConvertToPA16((uchar4 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_RGBA32F:
		err = cudaConvertToPA16((float4 *)input, output, width, height, in_max, out_max, false, stream);
		break;
	case IMAGE_BGR8:
		err = cudaConvertToPA16((uchar3 *)input, output, width, height, in_max, out_max, true, stream);
		break;
	case IMAGE_BGR32F:
		err = cudaConvertToPA16((float3 *)input, output, width, height, in_max, out_max, true, stream);
		break;
	case IMAGE_BGRA8:
		err = cudaConvertToPA16((uchar4 *)input, output, width, height, in_max, out_max, true, stream);
		break;
	case IMAGE_BGRA32F:
		err = cudaConvertToPA16((float4 *)input, output, width, height, in_max, out_max, true, stream);
		break;

	default:
		LogError(LOG_CUDA "cudaConvertToPA16() -- invalid image format '%s'\n", imageFormatToStr(format));
		LogError(LOG_CUDA "                       supported formats are:\n");
		LogError(LOG_CUDA "                           * gray8, gray32f\n");
		LogError(LOG_CUDA "                           * rgb8, bgr8, rgb32f, bgr32f\n");
		LogError(LOG_CUDA "                           * rgba8, bgra8, rgba32f, bgra32f\n");

		err = cudaErrorInvalidValue;
	}

	return err;
}
