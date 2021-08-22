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

#ifndef __CUDA_WARP_YA_FISHEYE_H__
#define __CUDA_WARP_YA_FISHEYE_H__


#include "cudaUtility.h"

#include <math.h>

#include "GLM.h"

// dgrees <--> radians.
template<typename T> constexpr inline __host__ __device__ T RAD(T d) { return static_cast<T>(d * M_PI / 180.0); }
template<typename T> constexpr inline __host__ __device__ T DEG(T r) { return static_cast<T>(r * 180.0 / M_PI); }
//
// fast version for CUDA.
inline __device__ float RAD_f(float d) { return d * (float)M_PI / 180.0f; }
inline __device__ float DEG_f(float r) { return r * 180.0f / (float)M_PI; }

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
//
// fast version for CUDA.
template<typename T> inline __device__ T rotX_f(T p, float th)
{
	T p_rot = p;

	p_rot.y = __cosf(th) * p.y - __sinf(th) * p.z;
	p_rot.z = __sinf(th) * p.y + __cosf(th) * p.z;

	return p_rot;
}
template<typename T> inline __device__ T rotY_f(T p, float th)
{
	T p_rot = p;

	p_rot.x =  __cosf(th) * p.x + __sinf(th) * p.z;
	p_rot.z = -__sinf(th) * p.x + __cosf(th) * p.z;

	return p_rot;
}
template<typename T> inline __device__ T rotZ_f(T p, float th)
{
	T p_rot = p;

	p_rot.x = __cosf(th) * p.x - __sinf(th) * p.y;
	p_rot.y = __sinf(th) * p.x + __cosf(th) * p.y;

	return p_rot;
}

enum class em_YA_FISHEYE_lens_spec : int {
	NORMAL,						// y = f * tan(theta).
	FISHEYE_EQUIDISTANT,		// y = f * theta.
	FISHEYE_EQUISOLID_ANGLE,	// y = f * 2 * sin(theta / 2).
	FISHEYE_ORTHOGRAPHIC,		// y = f * sin(theta).
	FISHEYE_STEREOGRAPHIC,		// y = f * 2 * tan(theta / 2).
};
constexpr auto em_ls_normal      = em_YA_FISHEYE_lens_spec::NORMAL;
constexpr auto em_ls_equidistant = em_YA_FISHEYE_lens_spec::FISHEYE_EQUIDISTANT;
constexpr auto em_ls_equi_angle  = em_YA_FISHEYE_lens_spec::FISHEYE_EQUISOLID_ANGLE;
constexpr auto em_ls_ortho       = em_YA_FISHEYE_lens_spec::FISHEYE_ORTHOGRAPHIC;
constexpr auto em_ls_st_graphic  = em_YA_FISHEYE_lens_spec::FISHEYE_STEREOGRAPHIC;

template<typename T> constexpr inline __host__ __device__ T f_Equidistant(T theta, T k)     { return k * theta; };
template<typename T> constexpr inline __host__ __device__ T f_Equisolid_angle(T theta, T k) { return static_cast<T>(k * 2.0 * sin(theta * 0.5)); };
template<typename T> constexpr inline __host__ __device__ T f_Orthographic(T theta, T k)    { return static_cast<T>(k * sin(theta)); };
template<typename T> constexpr inline __host__ __device__ T f_Stereographic(T theta, T k)   { return static_cast<T>(k * 2.0 * tan(theta * 0.5)); };
template<typename T> constexpr inline __host__ __device__ T f_Rectilinear(T theta, T k)     { return static_cast<T>(k * tan(theta)); };
//
template<typename T> constexpr inline __host__ __device__ T f_lens_radius(T theta, T k, em_YA_FISHEYE_lens_spec lens_type)
{
	switch (lens_type) {
	case em_ls_normal:
		return f_Rectilinear(theta, k);
	case em_ls_equidistant:
		return f_Equidistant(theta, k);
	case em_ls_equi_angle:
		return f_Equisolid_angle(theta, k);
	case em_ls_ortho:
		return f_Orthographic(theta, k);
	case em_ls_st_graphic:
		return f_Stereographic(theta, k);
	default:
		return f_Equisolid_angle(theta, k);
	}
};
//
// fast version for CUDA.
inline __device__ float f_Equidistant_f(float theta, float k)     { return k * theta; }
inline __device__ float f_Equisolid_angle_f(float theta, float k) { return k * 2.0f * __sinf(theta * 0.5f); }
inline __device__ float f_Orthographic_f(float theta, float k)    { return k * __sinf(theta); }
inline __device__ float f_Stereographic_f(float theta, float k)   { return k * 2.0f * __tanf(theta * 0.5f); }
inline __device__ float f_Rectilinear_f(float theta, float k)     { return k * __tanf(theta); }
//
inline __device__ float f_lens_radius_f(float theta, float k, em_YA_FISHEYE_lens_spec lens_type)
{
	switch (lens_type) {
	case em_ls_normal:
		return f_Rectilinear_f(theta, k);
	case em_ls_equidistant:
		return f_Equidistant_f(theta, k);
	case em_ls_equi_angle:
		return f_Equisolid_angle_f(theta, k);
	case em_ls_ortho:
		return f_Orthographic_f(theta, k);
	case em_ls_st_graphic:
		return f_Stereographic_f(theta, k);
	default:
		return f_Equisolid_angle_f(theta, k);
	}
}

constexpr auto YA_FISHEYE_LENS_TBL_NUM = 36;
struct st_YA_FISHEYE_lens_table {
	int num;	// number of data.
	float step;	// step of degrees or radians.
	float data[YA_FISHEYE_LENS_TBL_NUM];	// radius of each dgrees or radians.
};

struct st_YA_FISHEYE_rotation {
	float x;
	float y;
	float z;
	float roll;
};

struct st_YA_FISHEYE_param {
	// input.
	uint32_t iW;
	uint32_t iH;
	float iAspect;
	float iAspect_inv;

	// output.
	uint32_t oW;
	uint32_t oH;
	float oAspect;
	float oAspect_inv;
	float v_fov_half_tan;

	// lens spec.
	float xcenter;
	float ycenter;
	float lens_radius_scale;
	st_YA_FISHEYE_lens_table lens_tbl;
	em_YA_FISHEYE_lens_spec lens_type;

	// rotaion.
	// st_YA_FISHEYE_rotation rot;
	glm::quat quat_view;

	// pixel sampling filter.
	int filter_mode;

	// (background) input image is panorama.
	bool panorama_back;
};

/**
 * Apply YA_FISHEYE to an 8-bit fixed-point RGBA/RGB/GRAY image.
 * Apply YA_FISHEYE to a 32-bit floating-point RGBA/RGB/GRAY image.
 * @param[in] focus focus of the lens (in mm).
 * @ingroup warping
 */
#define FUNC_CUDA_WARP_YA_FISHEYE_HEADER(T, S) \
cudaError_t cudaWarpYAFisheye( T* input, S* output, st_YA_FISHEYE_param YA_fisheye_prm, cudaStream_t stream = NULL );

// cudaWarpYAFisheye (uint8 grayscale)
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uint8_t, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar3, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar4, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float3, uint8_t);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float4, uint8_t);

// cudaWarpYAFisheye (float grayscale)
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uint8_t, float);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float, float);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar3, float);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar4, float);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float3, float);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float4, float);

// cudaWarpYAFisheye (uchar3)
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uint8_t, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar3, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar4, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float3, uchar3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float4, uchar3);

// cudaWarpYAFisheye (uchar4)
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uint8_t, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar3, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar4, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float3, uchar4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float4, uchar4);

// cudaWarpYAFisheye (float3)
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uint8_t, float3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float, float3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar3, float3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar4, float3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float3, float3);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float4, float3);

// cudaWarpYAFisheye (float4)
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uint8_t, float4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float, float4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar3, float4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(uchar4, float4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float3, float4);
FUNC_CUDA_WARP_YA_FISHEYE_HEADER(float4, float4);

#undef FUNC_CUDA_WARP_YA_FISHEYE_HEADER

#endif
