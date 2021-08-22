/*
 * Copyright (c) 2021, edgecraft. All rights reserved.
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

#include "cudaWarp-YA-fisheye.h"
#include "cudaFilterMode.cuh"


// XY(output) -> 3D position w/ rotation.
inline __device__ float3 conv_2Dto3D_rotated(float cx, float cy, float fov, glm::quat q_view)
{
	// 2D -> 3D.
	// right-handed system. x: right(->), y: down(|v), z: far(X).
	float3 po = {
		.x = cx * fov,
		.y = cy * fov,
		.z = 1.0f,
	};

	// pan, tilt, roll.
	glm::vec3 p_org(po.x, po.y, po.z);
	glm::vec3 p_rot_tmp = q_view * p_org;
	float3 p_rot = {
		p_rot_tmp.x,
		p_rot_tmp.y,
		p_rot_tmp.z,
	};

	// normalized sphere. r = 1.0.
	float3 p_sph = normalize(p_rot);

	return p_sph;
}

// 3D position -> 2D position.
inline __device__ float2 conv_3Dto2D(float3 p_sph, float k, em_YA_FISHEYE_lens_spec lens_type)
{
	// XYZ -> theta_x, theta_z.
	float theta_x = atan2f(p_sph.y, p_sph.x);
	float theta_z = acosf(p_sph.z);

	// 3D -> 2D.
	float r = f_lens_radius_f(theta_z, k, lens_type);
	float tx = r * __cosf(theta_x);
	float ty = r * __sinf(theta_x);

	return float2{tx, ty};
}
inline __device__ float2 conv_3Dto2D_panorama(float3 p_sph)
{
	// XYZ -> theta_x, theta_z. for input panorama.
	float theta_x = atan2f(-p_sph.x, -p_sph.z);
	float theta_z = acosf(-p_sph.y);
	if (theta_x < 0.0f) theta_x += (2.0f * (float)M_PI);

	// 3D -> 2D. for input panorama.
	constexpr float pi_2_inv = 1.0f / (2.0f * (float)M_PI);
	constexpr float pi_inv = 1.0f / (float)M_PI;
	float tx = theta_x * pi_2_inv;
	float ty = theta_z * pi_inv;

	return float2{tx, ty};
}

// -> XY(input). with adjustment of lens center.
inline __device__ float2 conv_toUV(float2 p, float aspect, float width, float height, float xcenter, float ycenter)
{
	// float u = ((p.x * 0.5f * aspect) + 0.5f) * width;
	// float v = ((p.y * 0.5f) + 0.5f) * height;
	// u += xcenter;
	// v += ycenter;
	float u = __fmaf_rn(__fmaf_rn(p.x, 0.5f * aspect, 0.5f), width , xcenter);
	float v = __fmaf_rn(__fmaf_rn(p.y, 0.5f         , 0.5f), height, ycenter);

	return float2{u, v};
}
inline __device__ float2 conv_toUV_panorama(float2 p, float width, float height)
{
	// -> XY(input). with adjustment of lens center.
	float u = p.x * (width  - 2.0f) + 0.0f;	// TODO: (W - 2) < x <= (W - 1): Bi-linear between (W - 2) and W(=0).
	float v = p.y * (height - 1.0f);

	return float2{u, v};
}

// check over edge.
inline __device__ bool is_over_edge(float u, float v, float w, float h)
{
	bool over_edge = (
		( u < 0.0f )
		|| ( v < 0.0f )
		|| ( u > w - 1.0f )
		|| ( v > h - 1.0f )
	);

	return over_edge;
}

// cudaYAFisheye
template<typename T, typename S>
__global__ void cudaYAFisheye( T* input, S* output, st_YA_FISHEYE_param YA_fisheye_prm )
{
	const int2 uv_out = make_int2(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y);

	if( uv_out.x >= YA_fisheye_prm.oW || uv_out.y >= YA_fisheye_prm.oH )
		return;

	const int iW = YA_fisheye_prm.iW;
	const int iH = YA_fisheye_prm.iH;
	const int oW = YA_fisheye_prm.oW;
	const int oH = YA_fisheye_prm.oH;
	const float iW_f = iW;
	const float iH_f = iH;
	const float oW_f = oW;
	const float oH_f = oH;

	const float fov      = YA_fisheye_prm.v_fov_half_tan;
	const float k      = YA_fisheye_prm.lens_radius_scale;

	// convert to cartesian coordinates
	// const float cx = ((uv_out.x / oW_f) - 0.5f) * 2.0f * YA_fisheye_prm.oAspect;
	// const float cy = ((uv_out.y / oH_f) - 0.5f) * 2.0f;
	const float cx = __fmaf_rn(__fdividef(uv_out.x, oW_f), 2.0f, -1.0f) * YA_fisheye_prm.oAspect;
	const float cy = __fmaf_rn(__fdividef(uv_out.y, oH_f), 2.0f, -1.0f);

	// XY(output) -> 3D position w/ rotation.
	float3 p_sph = conv_2Dto3D_rotated(cx, cy, fov, YA_fisheye_prm.quat_view);

	float u;
	float v;
	bool over_edge;
	if (YA_fisheye_prm.panorama_back) {
		// for input panorama.
		// 3D position -> 2D position.
		float2 txy = conv_3Dto2D_panorama(p_sph);

		// -> XY(input). with adjustment of lens center.
		float2 uv = conv_toUV_panorama(txy, iW_f, iH_f);
		u = uv.x;
		v = uv.y;

		over_edge = is_over_edge(u, v, iW_f, iH_f);

	} else {
		// for input fisheye.
		// 3D position -> 2D position.
		float2 txy = conv_3Dto2D(p_sph, k, YA_fisheye_prm.lens_type);

		// -> XY(input). with adjustment of lens center.
		float2 uv = conv_toUV(txy, YA_fisheye_prm.iAspect_inv, iW_f, iH_f, YA_fisheye_prm.xcenter, YA_fisheye_prm.ycenter);
		u = uv.x;
		v = uv.y;

		bool negative_position = (YA_fisheye_prm.lens_type == em_ls_normal && p_sph.z <= 0.0f);
		over_edge = (is_over_edge(u, v, iW_f, iH_f) || negative_position);
	}

	// sampling pixel.
	auto get_pixel = [](
		auto input, auto u, auto v, auto iW, auto iH, auto oW, auto oH,
		auto scale, auto max_value, auto filter)
		-> auto {
		decltype(*input + 0) pix;
		switch (filter) {
		case FILTER_LINEAR:	// Bi-linear. 3x3 filter.
			pix = cudaFilterPixel<FILTER_LINEAR>(input, u, v, iW, iH, oW, oH, scale, max_value);
			break;
		case FILTER_CUBIC:	// Bi-cubic. 5x5 filter.
			pix = cudaFilterPixel<FILTER_CUBIC>(input, u, v, iW, iH, oW, oH, scale, max_value);
			break;
		case FILTER_SPLINE36:	// Spline36. 7x7 filter.
			pix = cudaFilterPixel<FILTER_SPLINE36>(input, u, v, iW, iH, oW, oH, scale, max_value);
			break;
		case FILTER_LANCZOS4:	// slowest. Lanczos4. 9x9 filter.
			pix = cudaFilterPixel<FILTER_LANCZOS4>(input, u, v, iW, iH, oW, oH, scale, max_value);
			break;
		case FILTER_POINT:	// fastest. nearest.
		default:
			pix = cudaFilterPixel<FILTER_POINT>(input, u, v, iW, iH, oW, oH, scale, max_value);
		}
		return pix;
	};

	constexpr float2 scale = { 1.0f, 1.0f };
	constexpr float max_value = 255.0f;
	T pix_in = !over_edge
		? get_pixel(input, u, v, iW, iH, oW, oH, scale, max_value, YA_fisheye_prm.filter_mode)
		: cast_vec<T>(0.0f);

	float3 pix_out_tmp = cast_vec<float3>(pix_in);

	S pix_out = cast_vec<S>(make_float4(pix_out_tmp, 255.0f));

	output[uv_out.y * oW + uv_out.x] = pix_out;
}


// cudaWarpYAFisheye
template<typename T, typename S>
inline cudaError_t cudaWarpYAFisheye__( T* input, S* output, st_YA_FISHEYE_param YA_fisheye_prm, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( YA_fisheye_prm.iW == 0 || YA_fisheye_prm.iH == 0 || YA_fisheye_prm.oW == 0 || YA_fisheye_prm.oH == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(YA_fisheye_prm.oW,blockDim.x), iDivUp(YA_fisheye_prm.oH,blockDim.y));

	cudaYAFisheye<T, S><<<gridDim, blockDim, 0, stream>>>(input, output, YA_fisheye_prm);

	return CUDA(cudaGetLastError());
}
#define FUNC_CUDA_WARP_YA_FISHEYE(T, S) \
cudaError_t cudaWarpYAFisheye( T* input, S* output, st_YA_FISHEYE_param YA_fisheye_prm, cudaStream_t stream ) \
{ \
	return cudaWarpYAFisheye__<T, S>( input, output, YA_fisheye_prm, stream ); \
}

// cudaWarpYAFisheye (uint8 grayscale)
FUNC_CUDA_WARP_YA_FISHEYE(uint8_t, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE(float, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE(uchar3, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE(uchar4, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE(float3, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE(float4, uint8_t);

// cudaWarpYAFisheye (float grayscale)
FUNC_CUDA_WARP_YA_FISHEYE(uint8_t, float);
FUNC_CUDA_WARP_YA_FISHEYE(float, float);
FUNC_CUDA_WARP_YA_FISHEYE(uchar3, float);
FUNC_CUDA_WARP_YA_FISHEYE(uchar4, float);
FUNC_CUDA_WARP_YA_FISHEYE(float3, float);
FUNC_CUDA_WARP_YA_FISHEYE(float4, float);

// cudaWarpYAFisheye (uchar3)
FUNC_CUDA_WARP_YA_FISHEYE(uint8_t, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE(float, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE(uchar3, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE(uchar4, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE(float3, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE(float4, uchar3);

// cudaWarpYAFisheye (uchar4)
FUNC_CUDA_WARP_YA_FISHEYE(uint8_t, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE(float, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE(uchar3, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE(uchar4, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE(float3, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE(float4, uchar4);

// cudaWarpYAFisheye (float3)
FUNC_CUDA_WARP_YA_FISHEYE(uint8_t, float3);
FUNC_CUDA_WARP_YA_FISHEYE(float, float3);
FUNC_CUDA_WARP_YA_FISHEYE(uchar3, float3);
FUNC_CUDA_WARP_YA_FISHEYE(uchar4, float3);
FUNC_CUDA_WARP_YA_FISHEYE(float3, float3);
FUNC_CUDA_WARP_YA_FISHEYE(float4, float3);

// cudaWarpYAFisheye (float4)
FUNC_CUDA_WARP_YA_FISHEYE(uint8_t, float4);
FUNC_CUDA_WARP_YA_FISHEYE(float, float4);
FUNC_CUDA_WARP_YA_FISHEYE(uchar3, float4);
FUNC_CUDA_WARP_YA_FISHEYE(uchar4, float4);
FUNC_CUDA_WARP_YA_FISHEYE(float3, float4);
FUNC_CUDA_WARP_YA_FISHEYE(float4, float4);

#undef FUNC_CUDA_WARP_YA_FISHEYE
