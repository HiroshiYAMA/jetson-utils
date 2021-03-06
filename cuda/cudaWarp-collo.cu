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

#include "cudaWarp-collo.h"
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

// cudaCollo
template<typename T, typename Tmask, typename T_HiReso, typename Tpano, typename S>
__global__ void cudaCollo(
	T* input, Tmask* mask, T_HiReso* input_HiReso, Tpano* input_panorama,
	S* output, S *output_fg, S *output_bg, S *output_mask,
	st_COLLO_param collo_prm )
{
	const int2 uv_out = make_int2(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y);

	if( uv_out.x >= collo_prm.oW || uv_out.y >= collo_prm.oH )
		return;

	// auto is_panorama = [&]() -> bool {
	// 	return (collo_prm.projection_mode == em_COLLO_projection_mode::PANORAMA);
	// };

#ifdef MODEL_OUTPUT_FGR_PHA
	const int iW = collo_prm.iW;
	const int iH = collo_prm.iH;
#endif
	const int iW_HiReso = collo_prm.iW_HiReso;
	const int iH_HiReso = collo_prm.iH_HiReso;
	const int panoW = collo_prm.panoW;
	const int panoH = collo_prm.panoH;
	const int oW = collo_prm.oW;
	const int oH = collo_prm.oH;
	const int mW = collo_prm.mW;
	const int mH = collo_prm.mH;
#ifdef MODEL_OUTPUT_FGR_PHA
	const float iW_f = iW;
	const float iH_f = iH;
#endif
	const float iW_HiReso_f = iW_HiReso;
	const float iH_HiReso_f = iH_HiReso;
	const float panoW_f = panoW;
	const float panoH_f = panoH;
	const float oW_f = oW;
	const float oH_f = oH;
	const float mW_f = mW;
	const float mH_f = mH;

	const float fov      = collo_prm.v_fov_half_tan;
	const float fov_back = collo_prm.v_fov_half_tan_back;
	const float k      = collo_prm.lens_radius_scale;
	const float k_back = collo_prm.lens_radius_scale_back;

	// convert to cartesian coordinates
	// const float cx = ((uv_out.x / oW_f) - 0.5f) * 2.0f * collo_prm.oAspect;
	// const float cy = ((uv_out.y / oH_f) - 0.5f) * 2.0f;
	const float cx = __fmaf_rn(__fdividef(uv_out.x, oW_f), 2.0f, -1.0f) * collo_prm.oAspect;
	const float cy = __fmaf_rn(__fdividef(uv_out.y, oH_f), 2.0f, -1.0f);

	// XY(output) -> 3D position w/ rotation.
	float3 p_sph = conv_2Dto3D_rotated(cx, cy, fov, collo_prm.quat_view);

	// 3D position -> 2D position.
	float2 txy = conv_3Dto2D(p_sph, k, collo_prm.lens_type);

#ifdef MODEL_OUTPUT_FGR_PHA
	// -> XY(input). with adjustment of lens center.
	float2 uv = conv_toUV(txy, collo_prm.iAspect_inv, iW_f, iH_f, collo_prm.xcenter, collo_prm.ycenter);
	float u = uv.x;
	float v = uv.y;
#endif

	// -> XY(input_HiReso). with adjustment of lens center.
	float2 uv_HiReso = conv_toUV(txy, collo_prm.iAspect_HiReso_inv, iW_HiReso_f, iH_HiReso_f, collo_prm.xcenter_HiReso, collo_prm.ycenter_HiReso);
	float u_HiReso = uv_HiReso.x;
	float v_HiReso = uv_HiReso.y;

	// -> XY(mask). with adjustment of lens center.
	float2 uv_mask = conv_toUV(txy, collo_prm.mAspect_inv, mW_f, mH_f, collo_prm.xcenter_mask, collo_prm.ycenter_mask);
	float u_mask = uv_mask.x;
	float v_mask = uv_mask.y;

	bool negative_position = (collo_prm.lens_type == em_ls_normal && p_sph.z <= 0.0f);
#ifdef MODEL_OUTPUT_FGR_PHA
	bool over_edge = (is_over_edge(u, v, iW_f, iH_f) || negative_position);
#endif
	bool over_edge_HiReso = (is_over_edge(u_HiReso, v_HiReso, iW_HiReso_f, iH_HiReso_f) || negative_position);
	bool over_edge_mask = (is_over_edge(u_mask, v_mask, mW_f, mH_f) || negative_position);

	// panorama.
	float u_pano;
	float v_pano;
	bool over_edge_pano = false;
	if (collo_prm.overlay_panorama) {
		// rotate background only.
		// XY(output) -> 3D position w/ rotation.
		float3 p_sph_back = conv_2Dto3D_rotated(cx, cy, fov_back, collo_prm.quat_view_back);

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
			float2 txy_pano = conv_3Dto2D(p_sph_back, k_back, collo_prm.lens_type_back);

			// -> XY(input). with adjustment of lens center.
			float2 uv_pano = conv_toUV(txy_pano, collo_prm.panoAspect_inv, panoW_f, panoH_f, collo_prm.xcenter, collo_prm.ycenter);
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

	constexpr float2 scale = { 1.0f, 1.0f };
	constexpr float max_value = 255.0f;
#ifdef MODEL_OUTPUT_FGR_PHA
	bool hi_reso = (collo_prm.HiReso || collo_prm.rgba);
	T pix_in = !over_edge && !hi_reso
		? get_pixel(input, u, v, iW, iH, oW, oH, scale, max_value, collo_prm.filter_mode)
		: cast_vec<T>(0.0f);
#endif

	Tmask pix_mask = !over_edge_mask && !collo_prm.rgba
		? get_pixel(mask, u_mask, v_mask, mW, mH, oW, oH, scale, max_value, collo_prm.filter_mode)
		: cast_vec<Tmask>(0.0f);

#ifdef MODEL_OUTPUT_FGR_PHA
	T_HiReso pix_in_HiReso = !over_edge_HiReso && hi_reso
#else
	T_HiReso pix_in_HiReso = !over_edge_HiReso
#endif
		? get_pixel(input_HiReso, u_HiReso, v_HiReso, iW_HiReso, iH_HiReso, oW, oH, scale, max_value, collo_prm.filter_mode)
		: cast_vec<T_HiReso>(0.0f);

#ifdef MODEL_OUTPUT_FGR_PHA
	float3 pix_fg = hi_reso
		? cast_vec<float3>(pix_in_HiReso)
		: cast_vec<float3>(pix_in);
#else
	float3 pix_fg = cast_vec<float3>(pix_in_HiReso);
#endif

	float3 pix_bg;
	if (collo_prm.overlay_panorama) {
		Tpano pix_pano = !over_edge_pano
			? get_pixel(input_panorama, u_pano, v_pano, panoW, panoH, oW, oH, scale, max_value, collo_prm.filter_mode)
			: cast_vec<Tpano>(0.0f);
		pix_bg = cast_vec<float3>(pix_pano);
	} else {
		pix_bg = cast_vec<float3>(collo_prm.bg_color);
	}
	constexpr float num255_inv = 1.0f / 255.0f;
	float a = collo_prm.rgba
		? collo_prm.alpha_blend ? alpha(make_float4(pix_in_HiReso)) * num255_inv : 1.0f
		: collo_prm.alpha_blend ? pix_mask * num255_inv : 1.0f;
	S pix_out = cast_vec<S>(make_float4((pix_fg * a) + (pix_bg * (1.0f - a)), 255.0f));

	output[uv_out.y * oW + uv_out.x] = pix_out;
	//
	if (collo_prm.camera_work) {
		output_fg[uv_out.y * oW + uv_out.x] = cast_vec<S>(pix_fg);
		output_bg[uv_out.y * oW + uv_out.x] = cast_vec<S>(pix_bg);
		output_mask[uv_out.y * oW + uv_out.x] = cast_vec<S>(pix_mask);
	}
}


// cudaWarpCollo
template<typename T, typename Tmask, typename T_HiReso, typename Tpano, typename S>
inline cudaError_t cudaWarpCollo__(
	T* input, Tmask* mask, T_HiReso* input_HiReso, Tpano* input_panorama,
	S* output, S *output_fg, S *output_bg, S *output_mask,
	st_COLLO_param collo_prm, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( collo_prm.iW == 0 || collo_prm.iH == 0 || collo_prm.oW == 0 || collo_prm.oH == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(collo_prm.oW,blockDim.x), iDivUp(collo_prm.oH,blockDim.y));

	cudaCollo<T, Tmask, T_HiReso, Tpano, S><<<gridDim, blockDim, 0, stream>>>(
		input, mask, input_HiReso, input_panorama,
		output, output_fg, output_bg, output_mask,
		collo_prm);

	return CUDA(cudaGetLastError());
}
#define FUNC_CUDA_WARP_COLLO(T, S) \
cudaError_t cudaWarpCollo( T* input, float* mask, uchar4* input_HiReso, uchar4* input_panorama, \
	S* output, S *output_fg, S *output_bg, S *output_mask, \
	st_COLLO_param collo_prm, cudaStream_t stream ) \
{ \
	return cudaWarpCollo__<T, float, uchar4, uchar4, S>( input, mask, input_HiReso, input_panorama, \
		output, output_fg, output_bg, output_mask, \
		collo_prm, stream ); \
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
// FUNC_CUDA_WARP_COLLO(uchar3, float);
FUNC_CUDA_WARP_COLLO(uchar4, float);
// FUNC_CUDA_WARP_COLLO(float3, float);
FUNC_CUDA_WARP_COLLO(float4, float);

// cudaWarpCollo (uchar3)
// FUNC_CUDA_WARP_COLLO(uint8_t, uchar3);
// FUNC_CUDA_WARP_COLLO(float, uchar3);
// FUNC_CUDA_WARP_COLLO(uchar3, uchar3);
// FUNC_CUDA_WARP_COLLO(uchar4, uchar3);
// FUNC_CUDA_WARP_COLLO(float3, uchar3);
// FUNC_CUDA_WARP_COLLO(float4, uchar3);

// cudaWarpCollo (uchar4)
// FUNC_CUDA_WARP_COLLO(uint8_t, uchar4);
FUNC_CUDA_WARP_COLLO(float, uchar4);
// FUNC_CUDA_WARP_COLLO(uchar3, uchar4);
FUNC_CUDA_WARP_COLLO(uchar4, uchar4);
// FUNC_CUDA_WARP_COLLO(float3, uchar4);
FUNC_CUDA_WARP_COLLO(float4, uchar4);

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
// FUNC_CUDA_WARP_COLLO(uchar3, float4);
FUNC_CUDA_WARP_COLLO(uchar4, float4);
// FUNC_CUDA_WARP_COLLO(float3, float4);
FUNC_CUDA_WARP_COLLO(float4, float4);

#undef FUNC_CUDA_WARP_COLLO
