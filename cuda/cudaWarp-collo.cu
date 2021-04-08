/*
 * Copyright (c) 2021, Hiroshi Yamamoto. All rights reserved.
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

#include "cudaWarp.h"
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
inline __device__ float2 conv_3Dto2D(float3 p_sph, float k, em_COLLO_lens_spec lens_type)
{
	// XYZ -> theta_x, theta_z.
	float theta_x = atan2f(p_sph.y, p_sph.x);
	float theta_z = acosf(p_sph.z);

	// 3D -> 2D.
	float r = f_lens_radius(theta_z, k, lens_type);
	float tx = r * __cosf(theta_x);
	float ty = r * __sinf(theta_x);

	return float2{tx, ty};
}
inline __device__ float2 conv_3Dto2D_panorama(float3 p_sph)
{
	// XYZ -> theta_x, theta_z. for input panorama.
	float theta_x = atan2f(-p_sph.x, -p_sph.z);
	float theta_z = acosf(-p_sph.y);
	if (theta_x < 0.0f) theta_x += (2.0f * M_PI);

	// 3D -> 2D. for input panorama.
	float tx = theta_x / (2.0f * M_PI);
	float ty = theta_z / M_PI;

	return float2{tx, ty};
}

// -> XY(input). with adjustment of lens center.
inline __device__ float2 conv_toUV(float2 p, float aspect, float width, float height, float xcenter, float ycenter)
{
	float u = ((p.x * 0.5f / aspect) + 0.5f) * width;
	float v = ((p.y * 0.5f) + 0.5f) * height;
	u += xcenter;
	v += ycenter;

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

// cudaCollo
template<typename T, typename T_HiReso, typename Tpano, typename S>
__global__ void cudaCollo( T* input, T_HiReso* input_HiReso, Tpano* input_panorama, S* output, st_COLLO_param collo_prm )
{
	const int2 uv_out = make_int2(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y);

	if( uv_out.x >= collo_prm.oW || uv_out.y >= collo_prm.oH )
		return;

	// auto is_panorama = [&]() -> bool {
	// 	return (collo_prm.projection_mode == em_COLLO_projection_mode::PANORAMA);
	// };

	const int iW = collo_prm.iW;
	const int iH = collo_prm.iH;
	const int iW_HiReso = collo_prm.iW_HiReso;
	const int iH_HiReso = collo_prm.iH_HiReso;
	const int panoW = collo_prm.panoW;
	const int panoH = collo_prm.panoH;
	const int oW = collo_prm.oW;
	const int oH = collo_prm.oH;
	const float iW_f = iW;
	const float iH_f = iH;
	const float iW_HiReso_f = iW_HiReso;
	const float iH_HiReso_f = iH_HiReso;
	const float panoW_f = panoW;
	const float panoH_f = panoH;
	const float oW_f = oW;
	const float oH_f = oH;

	const float fov = collo_prm.v_fov_half_tan;
	const float k = collo_prm.lens_radius_scale;
	
	// convert to cartesian coordinates
	const float cx = ((uv_out.x / oW_f) - 0.5f) * 2.0f * collo_prm.oAspect;	
	const float cy = ((uv_out.y / oH_f) - 0.5f) * 2.0f;

	// XY(output) -> 3D position w/ rotation.
	float3 p_sph = conv_2Dto3D_rotated(cx, cy, fov, collo_prm.quat_view);

	// 3D position -> 2D position.
	float2 txy = conv_3Dto2D(p_sph, k, collo_prm.lens_type);

	// -> XY(input). with adjustment of lens center.
	float2 uv = conv_toUV(txy, collo_prm.iAspect, iW_f, iH_f, collo_prm.xcenter, collo_prm.ycenter);
	float u = uv.x;
	float v = uv.y;

	// -> XY(input_HiReso). with adjustment of lens center.
	float2 uv_HiReso = conv_toUV(txy, collo_prm.iAspect_HiReso, iW_HiReso_f, iH_HiReso_f, collo_prm.xcenter_HiReso, collo_prm.ycenter_HiReso);
	float u_HiReso = uv_HiReso.x;
	float v_HiReso = uv_HiReso.y;

	bool negative_position = (collo_prm.lens_type == em_ls_normal && p_sph.z <= 0.0f);
	bool over_edge = (is_over_edge(u, v, iW_f, iH_f) || negative_position);
	bool over_edge_HiReso = (is_over_edge(u_HiReso, v_HiReso, iW_HiReso_f, iH_HiReso_f) || negative_position);

	// panorama.
	float u_pano;
	float v_pano;
	bool over_edge_pano = false;
	if (collo_prm.overlay_panorama) {
		// rotate background only.
		// XY(output) -> 3D position w/ rotation.
		float3 p_sph_back = conv_2Dto3D_rotated(cx, cy, fov, collo_prm.quat_view_back);

		if (collo_prm.panorama_back) {
			// for input panorama.
			// 3D position -> 2D position.
			float2 txy_pano = conv_3Dto2D_panorama(p_sph_back);

			// -> XY(input). with adjustment of lens center.
			float2 uv_pano = conv_toUV_panorama(txy_pano, panoW_f, panoH_f);
			u_pano = uv_pano.x;
			v_pano = uv_pano.y;

			over_edge_pano = is_over_edge(u_pano, v_pano, panoW_f, panoH_f);

		} else {
			// for input fisheye.
			// 3D position -> 2D position.
			float2 txy_pano = conv_3Dto2D(p_sph_back, k, collo_prm.lens_type);

			// -> XY(input). with adjustment of lens center.
			float2 uv_pano = conv_toUV(txy_pano, collo_prm.panoAspect, panoW_f, panoH_f, collo_prm.xcenter, collo_prm.ycenter);
			u_pano = uv_pano.x;
			v_pano = uv_pano.y;

			bool negative_position_back = (collo_prm.lens_type == em_ls_normal && p_sph_back.z <= 0.0f);
			over_edge_pano = (is_over_edge(u_pano, v_pano, panoW_f, panoH_f) || negative_position_back);
		}
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

	float2 scale = { 1.0f, 1.0f };
	float max_value = 255.0f;
	T pix_in = !over_edge
		? get_pixel(input, u, v, iW, iH, oW, oH, scale, max_value, collo_prm.filter_mode)
		: cast_vec<T>(0.0f);

	T_HiReso pix_in_HiReso = !over_edge_HiReso
		? get_pixel(input_HiReso, u_HiReso, v_HiReso, iW_HiReso, iH_HiReso, oW, oH, scale, max_value, collo_prm.filter_mode)
		: cast_vec<T_HiReso>(0.0f);

	float4 pix_bg;
	if (collo_prm.overlay_panorama) {
		Tpano pix_pano = !over_edge_pano
			? get_pixel(input_panorama, u_pano, v_pano, panoW, panoH, oW, oH, scale, max_value, collo_prm.filter_mode)
			: cast_vec<Tpano>(0.0f);
		pix_bg = cast_vec<float4>(pix_pano);
	} else {
		pix_bg = cast_vec<float4>(collo_prm.bg_color);
	}
	float a = collo_prm.alpha_blend ? alpha(make_float4(pix_in)) / 255.0f : 1.0f;
	S pix_out = cast_vec<S>(cast_vec<float4>(pix_in_HiReso * a) + (pix_bg * (1.0f - a)));

	output[uv_out.y * oW + uv_out.x] = pix_out;
}


// cudaWarpCollo
template<typename T, typename T_HiReso, typename Tpano, typename S> inline cudaError_t cudaWarpCollo__( T* input, T_HiReso* input_HiReso, Tpano* input_panorama, S* output, st_COLLO_param collo_prm )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( collo_prm.iW == 0 || collo_prm.iH == 0 || collo_prm.oW == 0 || collo_prm.oH == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(collo_prm.oW,blockDim.x), iDivUp(collo_prm.oH,blockDim.y));

	cudaCollo<T, T_HiReso, Tpano, S><<<gridDim, blockDim>>>(input, input_HiReso, input_panorama, output, collo_prm);

	return CUDA(cudaGetLastError());
}
#define FUNC_CUDA_WARP_COLLO(T, S) \
cudaError_t cudaWarpCollo( T* input, uchar3* input_HiReso, uchar3* input_panorama, S* output, st_COLLO_param collo_prm ) \
{ \
	return cudaWarpCollo__<T, uchar3, uchar3, S>( input, input_HiReso, input_panorama, output, collo_prm ); \
}

// cudaWarpCollo (uint8 grayscale)
// FUNC_CUDA_WARP_COLLO(uint8_t, uint8_t);
// FUNC_CUDA_WARP_COLLO(float, uint8_t);
// FUNC_CUDA_WARP_COLLO(uchar3, uint8_t);
// FUNC_CUDA_WARP_COLLO(uchar4, uint8_t);
// FUNC_CUDA_WARP_COLLO(float3, uint8_t);
// FUNC_CUDA_WARP_COLLO(float4, uint8_t);

// cudaWarpCollo (float grayscale)
// FUNC_CUDA_WARP_COLLO(uint8_t, float);
FUNC_CUDA_WARP_COLLO(float, float);
FUNC_CUDA_WARP_COLLO(uchar3, float);
// FUNC_CUDA_WARP_COLLO(uchar4, float);
// FUNC_CUDA_WARP_COLLO(float3, float);
FUNC_CUDA_WARP_COLLO(float4, float);

// cudaWarpCollo (uchar3)
// FUNC_CUDA_WARP_COLLO(uint8_t, uchar3);
FUNC_CUDA_WARP_COLLO(float, uchar3);
FUNC_CUDA_WARP_COLLO(uchar3, uchar3);
// FUNC_CUDA_WARP_COLLO(uchar4, uchar3);
// FUNC_CUDA_WARP_COLLO(float3, uchar3);
FUNC_CUDA_WARP_COLLO(float4, uchar3);

// cudaWarpCollo (uchar4)
// FUNC_CUDA_WARP_COLLO(uint8_t, uchar4);
// FUNC_CUDA_WARP_COLLO(float, uchar4);
// FUNC_CUDA_WARP_COLLO(uchar3, uchar4);
// FUNC_CUDA_WARP_COLLO(uchar4, uchar4);
// FUNC_CUDA_WARP_COLLO(float3, uchar4);
// FUNC_CUDA_WARP_COLLO(float4, uchar4);

// cudaWarpCollo (float3)
// FUNC_CUDA_WARP_COLLO(uint8_t, float3);
// FUNC_CUDA_WARP_COLLO(float, float3);
// FUNC_CUDA_WARP_COLLO(uchar3, float3);
// FUNC_CUDA_WARP_COLLO(uchar4, float3);
// FUNC_CUDA_WARP_COLLO(float3, float3);
// FUNC_CUDA_WARP_COLLO(float4, float3);

// cudaWarpCollo (float4)
// FUNC_CUDA_WARP_COLLO(uint8_t, float4);
FUNC_CUDA_WARP_COLLO(float, float4);
FUNC_CUDA_WARP_COLLO(uchar3, float4);
// FUNC_CUDA_WARP_COLLO(uchar4, float4);
// FUNC_CUDA_WARP_COLLO(float3, float4);
FUNC_CUDA_WARP_COLLO(float4, float4);

#undef FUNC_CUDA_WARP_COLLO
