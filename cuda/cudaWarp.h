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

#ifndef __CUDA_WARP_H__
#define __CUDA_WARP_H__


#include "cudaUtility.h"


/**
 * Apply 2x3 affine warp to an 8-bit fixed-point RGBA image.
 * The 2x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpAffine( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
					   const float transform[2][3], bool transform_inverted=false );


/**
 * Apply 2x3 affine warp to an 32-bit floating-point RGBA image.
 * The 2x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpAffine( float4* input, float4* output, uint32_t width, uint32_t height,
					   const float transform[2][3], bool transform_inverted=false );


/**
 * Apply 3x3 perspective warp to an 8-bit fixed-point RGBA image.
 * The 3x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpPerspective( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
					        const float transform[3][3], bool transform_inverted=false );


/**
 * Apply 3x3 perspective warp to an 32-bit floating-point RGBA image.
 * The 3x3 matrix transform is in row-major order (transform[row][column])
 * If the transform has already been inverted, set transform_inverted to true.
 * @ingroup warping
 */
cudaError_t cudaWarpPerspective( float4* input, float4* output, uint32_t width, uint32_t height,
					        const float transform[3][3], bool transform_inverted=false );


/**
 * Apply in-place instrinsic lens distortion correction to an 8-bit fixed-point RGBA image.
 * Pinhole camera model with radial (barrel) distortion and tangential distortion.
 * @ingroup warping
 */
cudaError_t cudaWarpIntrinsic( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
						 const float2& focalLength, const float2& principalPoint, const float4& distortion );
											  

/**
 * Apply in-place instrinsic lens distortion correction to 32-bit floating-point RGBA image.
 * Pinhole camera model with radial (barrel) distortion and tangential distortion.
 * @ingroup warping
 */
cudaError_t cudaWarpIntrinsic( float4* input, float4* output, uint32_t width, uint32_t height,
						 const float2& focalLength, const float2& principalPoint, const float4& distortion );
											  

/**
 * Apply fisheye lens dewarping to an 8-bit fixed-point RGBA image.
 * @param[in] focus focus of the lens (in mm).
 * @ingroup warping
 */
cudaError_t cudaWarpFisheye( uchar4* input, uchar4* output, uint32_t width, uint32_t height, float focus );


/**
 * Apply fisheye lens dewarping to a 32-bit floating-point RGBA image.
 * @param[in] focus focus of the lens (in mm).
 * @ingroup warping
 */
cudaError_t cudaWarpFisheye( float4* input, float4* output, uint32_t width, uint32_t height, float focus );


#include <math.h>

// A.1. glm ライブラリーの一部機能は CPP を定義して制御できるようになっています。
#define GLM_FORCE_CUDA
#define GLM_FORCE_CXX14
#define GLM_FORCE_INLNIE
#define GLM_SWIZZLE
// 1. glm ライブラリーの基本機能が含まれます。
#include <glm/glm.hpp>
// 2. glm ライブラリーの拡張機能のうち既に仕様が安定した機能が glm/gtc に含まれます。
// #include <glm/gtc/constants.hpp>
// 3. glm ライブラリーの拡張機能のうち試験的に実装されている機能が glm/gtx に含まれます。
// #include <glm/gtx/color_space.hpp>
// 4. glm ライブラリーの拡張機能をひとまとめに取り込みたい場合には glm/ext.hpp を使います。
#include <glm/ext.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

// dgrees <--> radians.
template<typename T> constexpr inline __host__ __device__ T RAD(T d) { return static_cast<T>(d * M_PI / 180.0); }
template<typename T> constexpr inline __host__ __device__ T DEG(T r) { return static_cast<T>(r * 180.0 / M_PI); }

// rotation. X, Y, Z.
template<typename T, typename S> inline __host__ __device__ T rotX(T p, S th)
{
	T p_rot = p;

	p_rot.y = cos(th) * p.y - sin(th) * p.z;
	p_rot.z = sin(th) * p.y + cos(th) * p.z;

	return p_rot;
}
template<typename T, typename S> inline __host__ __device__ T rotY(T p, S th)
{
	T p_rot = p;

	p_rot.x =  cos(th) * p.x + sin(th) * p.z;
	p_rot.z = -sin(th) * p.x + cos(th) * p.z;

	return p_rot;
}
template<typename T, typename S> inline __host__ __device__ T rotZ(T p, S th)
{
	T p_rot = p;

	p_rot.x = cos(th) * p.x - sin(th) * p.y;
	p_rot.y = sin(th) * p.x + cos(th) * p.y;

	return p_rot;
}

template<typename T> constexpr inline __host__ __device__ T f_Equidistant(T theta, T k) { return k * theta; };
template<typename T> constexpr inline __host__ __device__ T f_Equisolid_angle(T theta, T k) { return static_cast<T>(k * 2.0 * sin(theta / 2.0)); };
template<typename T> constexpr inline __host__ __device__ T f_Orthographic(T theta, T k) { return static_cast<T>(k * sin(theta)); };
template<typename T> constexpr inline __host__ __device__ T f_Stereographic(T theta, T k) { return static_cast<T>(k * 2.0 * tan(theta / 2.0)); };
template<typename T> constexpr inline __host__ __device__ T f_Rectilinear(T theta, T k) { return static_cast<T>(k * tan(theta)); };
//
template<typename T> constexpr inline __host__ __device__ T f_lens_radius(T theta, T k) { return f_Equisolid_angle(theta, k); };

constexpr auto COLLO_LENS_TBL_NUM = 36;
struct st_COLLO_lens_table {
	int num;	// number of data.
	float step;	// step of degrees or radians.
	float data[COLLO_LENS_TBL_NUM];	// radius of each dgrees or radians.
};

struct st_COLLO_rotation {
	float x;
	float y;
	float z;
	float roll;
};

enum class em_COLLO_projection_mode : int {
	NORMAL,
	PANORAMA,
};

struct st_COLLO_param {
	// input.
	uint32_t iW;
	uint32_t iH;
	float iAspect;

	// input(panorama).
	uint32_t panoW;
	uint32_t panoH;
	float panoAspect;

	// output.
	uint32_t oW;
	uint32_t oH;
	float oAspect;
	float v_fov_half_tan;

	// lens spec.
	float xcenter;
	float ycenter;
	float lens_radius_scale;
	st_COLLO_lens_table lens_tbl;

	// rotaion.
	st_COLLO_rotation rot;
	glm::quat quat_view;

	// pixel sampling filter.
	int filter_mode;

	// // projection mode.
	// em_COLLO_projection_mode projection_mode;

	// overlay panorama background.
	bool overlay_panorama;
};

/**
 * Apply COLLO to an 8-bit fixed-point RGBA/RGB/GRAY image.
 * Apply COLLO to a 32-bit floating-point RGBA/RGB/GRAY image.
 * @param[in] focus focus of the lens (in mm).
 * @ingroup warping
 */
#define FUNC_CUDA_WARP_COLLO_HEADER(T, S) \
cudaError_t cudaWarpCollo( T* input, uchar3* input_panorama, S* output, st_COLLO_param collo_prm );

// cudaWarpCollo (uint8 grayscale)
FUNC_CUDA_WARP_COLLO_HEADER(uint8_t, uint8_t);
FUNC_CUDA_WARP_COLLO_HEADER(float, uint8_t);
FUNC_CUDA_WARP_COLLO_HEADER(uchar3, uint8_t);
FUNC_CUDA_WARP_COLLO_HEADER(uchar4, uint8_t);
FUNC_CUDA_WARP_COLLO_HEADER(float3, uint8_t);
FUNC_CUDA_WARP_COLLO_HEADER(float4, uint8_t);

// cudaWarpCollo (float grayscale)
FUNC_CUDA_WARP_COLLO_HEADER(uint8_t, float);
FUNC_CUDA_WARP_COLLO_HEADER(float, float);
FUNC_CUDA_WARP_COLLO_HEADER(uchar3, float);
FUNC_CUDA_WARP_COLLO_HEADER(uchar4, float);
FUNC_CUDA_WARP_COLLO_HEADER(float3, float);
FUNC_CUDA_WARP_COLLO_HEADER(float4, float);

// cudaWarpCollo (uchar3)
FUNC_CUDA_WARP_COLLO_HEADER(uint8_t, uchar3);
FUNC_CUDA_WARP_COLLO_HEADER(float, uchar3);
FUNC_CUDA_WARP_COLLO_HEADER(uchar3, uchar3);
FUNC_CUDA_WARP_COLLO_HEADER(uchar4, uchar3);
FUNC_CUDA_WARP_COLLO_HEADER(float3, uchar3);
FUNC_CUDA_WARP_COLLO_HEADER(float4, uchar3);

// cudaWarpCollo (uchar4)
FUNC_CUDA_WARP_COLLO_HEADER(uint8_t, uchar4);
FUNC_CUDA_WARP_COLLO_HEADER(float, uchar4);
FUNC_CUDA_WARP_COLLO_HEADER(uchar3, uchar4);
FUNC_CUDA_WARP_COLLO_HEADER(uchar4, uchar4);
FUNC_CUDA_WARP_COLLO_HEADER(float3, uchar4);
FUNC_CUDA_WARP_COLLO_HEADER(float4, uchar4);

// cudaWarpCollo (float3)
FUNC_CUDA_WARP_COLLO_HEADER(uint8_t, float3);
FUNC_CUDA_WARP_COLLO_HEADER(float, float3);
FUNC_CUDA_WARP_COLLO_HEADER(uchar3, float3);
FUNC_CUDA_WARP_COLLO_HEADER(uchar4, float3);
FUNC_CUDA_WARP_COLLO_HEADER(float3, float3);
FUNC_CUDA_WARP_COLLO_HEADER(float4, float3);

// cudaWarpCollo (float4)
FUNC_CUDA_WARP_COLLO_HEADER(uint8_t, float4);
FUNC_CUDA_WARP_COLLO_HEADER(float, float4);
FUNC_CUDA_WARP_COLLO_HEADER(uchar3, float4);
FUNC_CUDA_WARP_COLLO_HEADER(uchar4, float4);
FUNC_CUDA_WARP_COLLO_HEADER(float3, float4);
FUNC_CUDA_WARP_COLLO_HEADER(float4, float4);

#undef FUNC_CUDA_WARP_COLLO_HEADER

#endif
