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


// cudaCollo
template<typename T, typename S>
__global__ void cudaCollo( T* input, S* output, st_COLLO_param collo_prm )
{
	const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
				               blockDim.y * blockIdx.y + threadIdx.y);
						   
	if( uv_out.x >= collo_prm.oW || uv_out.y >= collo_prm.oH )
		return;
	
	const int iW  = collo_prm.iW;
	const int iH = collo_prm.iH;
	const int oW  = collo_prm.oW;
	const int oH = collo_prm.oH;
	const float iW_f  = iW;
	const float iH_f = iH;
	const float oW_f  = oW;
	const float oH_f = oH;

	const float fov = collo_prm.v_fov_half_tan;
	const float k = collo_prm.lens_radius_scale;
	
	// convert to cartesian coordinates
	const float cx = ((uv_out.x / oW_f) - 0.5f) * 2.0f * collo_prm.oAspect;	
	const float cy = ((uv_out.y / oH_f) - 0.5f) * 2.0f;

	// 2D -> 3D.
	// right-handed system. x: right(->), y: down(|v), z: far(X).
	float3 po = {
		.x = cx * fov,
		.y = cy * fov,
		.z = 1.0f,
	};

	// roll(inner).
	float3 p_roll = rotZ(po, collo_prm.rot.roll);

	// pan, tilt.
	float3 p_rot = p_roll;
	if (collo_prm.absolute) {
		p_rot = rotX(p_rot, -collo_prm.target_y * M_PI / 4.0f);	// X. tilt.
		p_rot = rotY(p_rot, collo_prm.target_x * M_PI / 4.0f);	// Y. pan.
		p_rot = rotZ(p_rot, collo_prm.rot.z);	// Z. outer roll.
	} else {
		p_rot = rotX(p_rot, collo_prm.rot.x);	// X. tilt.
		p_rot = rotY(p_rot, collo_prm.rot.y);	// Y. pan.
		p_rot = rotZ(p_rot, collo_prm.rot.z);	// Z. outer roll.
	}

	// normalized sphere. r = 1.0.
	float3 p_sph = normalize(p_rot);

	// XYZ -> theta_x, theta_z.
	float theta_x = atan2f(p_sph.y, p_sph.x);
	float theta_z = acosf(p_sph.z);

	// 3D -> 2D.
	float r = f_lens_radius(theta_z, k);
	float tx = r * __cosf(theta_x);
	float ty = r * __sinf(theta_x);

	// -> XY(input). with adjustment of lens center.
	float u = ((tx * 0.5f / collo_prm.iAspect) + 0.5f) * iW_f;
	float v = ((ty * 0.5f) + 0.5f) * iH_f;
	u += collo_prm.xcenter;
	v += collo_prm.ycenter;

	if( u < 0.0f ) u = 0.0f;
	if( v < 0.0f ) v = 0.0f;

	if( u > iW_f - 1.0f ) u = iW_f - 1.0f;
	if( v > iH_f - 1.0f ) v = iH_f - 1.0f;
	
	// sampling pixel.
	float2 scale = { 1.0f, 1.0f };
	float max_value = 255.0f;
	switch (collo_prm.filter_mode) {
	case FILTER_LINEAR:	// Bi-linear. 3x3 filter.
		output[uv_out.y * oW + uv_out.x] = cast_vec<S>(cudaFilterPixel<FILTER_LINEAR>(input, u, v, iW, iH, oW, oH, scale, max_value));
		break;
	case FILTER_CUBIC:	// Bi-cubic. 5x5 filter.
		output[uv_out.y * oW + uv_out.x] = cast_vec<S>(cudaFilterPixel<FILTER_CUBIC>(input, u, v, iW, iH, oW, oH, scale, max_value));
		break;
	case FILTER_SPLINE36:	// Spline36. 7x7 filter.
		output[uv_out.y * oW + uv_out.x] = cast_vec<S>(cudaFilterPixel<FILTER_SPLINE36>(input, u, v, iW, iH, oW, oH, scale, max_value));
		break;
	case FILTER_LANCZOS4:	// slowest. Lanczos4. 9x9 filter.
		output[uv_out.y * oW + uv_out.x] = cast_vec<S>(cudaFilterPixel<FILTER_LANCZOS4>(input, u, v, iW, iH, oW, oH, scale, max_value));
		break;
	case FILTER_POINT:	// fastest. nearest.
	default:
		output[uv_out.y * oW + uv_out.x] = cast_vec<S>(cudaFilterPixel<FILTER_POINT>(input, u, v, iW, iH, oW, oH, scale, max_value));
	}
}


// cudaWarpCollo
template<typename T, typename S> inline cudaError_t cudaWarpCollo__( T* input, S* output, st_COLLO_param collo_prm )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( collo_prm.iW == 0 || collo_prm.iH == 0 || collo_prm.oW == 0 || collo_prm.oH == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(collo_prm.oW,blockDim.x), iDivUp(collo_prm.oH,blockDim.y));

	cudaCollo<T, S><<<gridDim, blockDim>>>(input, output, collo_prm);

	return CUDA(cudaGetLastError());
}
#define FUNC_CUDA_WARP_COLLO(T, S) \
cudaError_t cudaWarpCollo( T* input, S* output, st_COLLO_param collo_prm ) { return cudaWarpCollo__<T, S>( input, output, collo_prm ); }

// cudaWarpCollo (uint8 grayscale)
FUNC_CUDA_WARP_COLLO(uint8_t, uint8_t);
FUNC_CUDA_WARP_COLLO(float, uint8_t);
FUNC_CUDA_WARP_COLLO(uchar3, uint8_t);
FUNC_CUDA_WARP_COLLO(uchar4, uint8_t);
FUNC_CUDA_WARP_COLLO(float3, uint8_t);
FUNC_CUDA_WARP_COLLO(float4, uint8_t);

// cudaWarpCollo (float grayscale)
FUNC_CUDA_WARP_COLLO(uint8_t, float);
FUNC_CUDA_WARP_COLLO(float, float);
FUNC_CUDA_WARP_COLLO(uchar3, float);
FUNC_CUDA_WARP_COLLO(uchar4, float);
FUNC_CUDA_WARP_COLLO(float3, float);
FUNC_CUDA_WARP_COLLO(float4, float);

// cudaWarpCollo (uchar3)
FUNC_CUDA_WARP_COLLO(uint8_t, uchar3);
FUNC_CUDA_WARP_COLLO(float, uchar3);
FUNC_CUDA_WARP_COLLO(uchar3, uchar3);
FUNC_CUDA_WARP_COLLO(uchar4, uchar3);
FUNC_CUDA_WARP_COLLO(float3, uchar3);
FUNC_CUDA_WARP_COLLO(float4, uchar3);

// cudaWarpCollo (uchar4)
FUNC_CUDA_WARP_COLLO(uint8_t, uchar4);
FUNC_CUDA_WARP_COLLO(float, uchar4);
FUNC_CUDA_WARP_COLLO(uchar3, uchar4);
FUNC_CUDA_WARP_COLLO(uchar4, uchar4);
FUNC_CUDA_WARP_COLLO(float3, uchar4);
FUNC_CUDA_WARP_COLLO(float4, uchar4);

// cudaWarpCollo (float3)
FUNC_CUDA_WARP_COLLO(uint8_t, float3);
FUNC_CUDA_WARP_COLLO(float, float3);
FUNC_CUDA_WARP_COLLO(uchar3, float3);
FUNC_CUDA_WARP_COLLO(uchar4, float3);
FUNC_CUDA_WARP_COLLO(float3, float3);
FUNC_CUDA_WARP_COLLO(float4, float3);

// cudaWarpCollo (float4)
FUNC_CUDA_WARP_COLLO(uint8_t, float4);
FUNC_CUDA_WARP_COLLO(float, float4);
FUNC_CUDA_WARP_COLLO(uchar3, float4);
FUNC_CUDA_WARP_COLLO(uchar4, float4);
FUNC_CUDA_WARP_COLLO(float3, float4);
FUNC_CUDA_WARP_COLLO(float4, float4);

#undef FUNC_CUDA_WARP_COLLO
