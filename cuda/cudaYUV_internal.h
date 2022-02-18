/*
 * Copyright (c) 2022, edgecraft. All rights reserved.
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

#ifndef __CUDA_YUV_CONVERT_INTERNAL_H
#define __CUDA_YUV_CONVERT_INTERNAL_H

#include "cudaUtility.h"
#include "cudaVector.h"

// 色空間/係数 Kr, Kg, Kb.
struct st_ColorSpaceCoef
{
	float Kr;
	float Kg;
	float Kb;
};
constexpr st_ColorSpaceCoef color_space_coef_601 = { 0.299, 0.587, 0.114 };
constexpr st_ColorSpaceCoef color_space_coef_709 = { 0.2126, 0.7152, 0.0722 };
constexpr st_ColorSpaceCoef color_space_coef_2020 = { 0.2627, 0.678, 0.0593 };

// matrix YUV -> RGB.
#define make_mtx_yuv2rgb(cs_coef) { \
	{ 1,	0,												(1 - cs_coef.Kr) }, \
	{ 1,	-(cs_coef.Kb * (1 - cs_coef.Kb) / cs_coef.Kg),	-(cs_coef.Kr * (1 - cs_coef.Kr) / cs_coef.Kg) }, \
	{ 1,	(1 - cs_coef.Kb),								0 }, \
}
const __device__ float mtx_601_yuv2rgb[3][3] = make_mtx_yuv2rgb(color_space_coef_601);
const __device__ float mtx_709_yuv2rgb[3][3] = make_mtx_yuv2rgb(color_space_coef_709);
const __device__ float mtx_2020_yuv2rgb[3][3] = make_mtx_yuv2rgb(color_space_coef_2020);

// matrix RGB -> YUV.
#define make_mtx_rgb2yuv(cs_coef) { \
	{ cs_coef.Kr,						cs_coef.Kg,						cs_coef.Kb }, \
	{ -(cs_coef.Kr/(1 - cs_coef.Kb)),	-(cs_coef.Kg/(1 - cs_coef.Kb)),	1 }, \
	{ 1,								-(cs_coef.Kg/(1 - cs_coef.Kr)),	-(cs_coef.Kb/(1 - cs_coef.Kr)) }, \
}
const __device__ float mtx_601_rgb2yuv[3][3] = make_mtx_rgb2yuv(color_space_coef_601);
const __device__ float mtx_709_rgb2yuv[3][3] = make_mtx_rgb2yuv(color_space_coef_709);
const __device__ float mtx_2020_rgb2yuv[3][3] = make_mtx_rgb2yuv(color_space_coef_2020);

struct st_ColorSpaceRangeCoef
{
	float Yrange;
	float Crange;
	float Yrange_inv;
	float Crange_inv;
	float Ybias;
	float Cbias;
};
struct st_ColorSpaceRange
{
	float Ymax;
	float Ymin;
	float Cmax;
	float Cmin;
};
const __device__ st_ColorSpaceRange color_space_range_limited = { 235, 16, 240, 16 };	// limited range.
const __device__ st_ColorSpaceRange color_space_range_full = { 255, 0, 255, 0 };	// full range.
inline __device__ __host__ st_ColorSpaceRangeCoef gen_color_space_range_coef(const st_ColorSpaceRange &cs_range, float range_MAX)
{
	constexpr float range_MAX_base = 255;

	float scale = (range_MAX + 1) / (range_MAX_base + 1);
	st_ColorSpaceRange cs_range_scaled = {
		cs_range.Ymax * scale,
		cs_range.Ymin * scale,
		cs_range.Cmax * scale,
		cs_range.Cmin * scale,
	};

	st_ColorSpaceRangeCoef cs_range_coef = {
		range_MAX / (cs_range_scaled.Ymax - cs_range_scaled.Ymin),
		range_MAX / ((cs_range_scaled.Cmax - cs_range_scaled.Cmin) / 2),
		(cs_range_scaled.Ymax - cs_range_scaled.Ymin) / range_MAX,
		((cs_range_scaled.Cmax - cs_range_scaled.Cmin) / 2) / range_MAX,
		cs_range_scaled.Ymin,
		(range_MAX + 1) / 2,
	};

	return cs_range_coef;
}
inline __device__ __host__ st_ColorSpaceRangeCoef gen_color_space_range_coef_limited(float range_MAX)
{
	return gen_color_space_range_coef(color_space_range_limited, range_MAX);
}
inline __device__ __host__ st_ColorSpaceRangeCoef gen_color_space_range_coef_full(float range_MAX)
{
	return gen_color_space_range_coef(color_space_range_full, range_MAX);
}


inline __device__ __host__ float3 mul_mtx33_vec3(const float3 &vec_in, const float mtx[3][3])
{
	float3 vec_out;
	vec_out.x = mtx[0][0] * vec_in.x + mtx[0][1] * vec_in.y + mtx[0][2] * vec_in.z;
	vec_out.y = mtx[1][0] * vec_in.x + mtx[1][1] * vec_in.y + mtx[1][2] * vec_in.z;
	vec_out.z = mtx[2][0] * vec_in.x + mtx[2][1] * vec_in.y + mtx[2][2] * vec_in.z;

	return vec_out;
}



//-----------------------------------------------------------------------------------
// YUV to RGB colorspace conversion
// in: [0, scale_in].
// out: [0, scale_out].
//-----------------------------------------------------------------------------------
inline __device__ __host__ float3 YUV2RGB(const float3 &yuv, const float mtx[3][3], const float3 &d_range_inv, const float3 &bias, float scale = 1.0f)
{
	float3 yuv_tmp = (yuv - bias) * d_range_inv;

	float3 rgb = mul_mtx33_vec3(yuv_tmp, mtx);

	rgb = rgb * scale;

	return rgb;
}
inline __device__ __host__ float3 YUV2RGB(const float3 &yuv, const float mtx[3][3], const st_ColorSpaceRangeCoef &range_coef, float scale = 1.0f)
{
	float3 d_range_inv = {
		range_coef.Yrange_inv,
		range_coef.Crange_inv,
		range_coef.Crange_inv,
	};
	float3 bias = {
		range_coef.Ybias,
		range_coef.Cbias,
		range_coef.Cbias,
	};

	float3 rgb = YUV2RGB(yuv, mtx, d_range_inv, bias, scale);

	return rgb;
}
// limited range.
inline __device__ __host__ float3 YUV2RGB_limited(const float3 &yuv, const float mtx[3][3], float scale_in = 255.0f, float scale_out = 255.0f)
{
	auto range_coef_in = gen_color_space_range_coef_limited(scale_in);
	float scale = scale_out / scale_in;

	float3 rgb = YUV2RGB(yuv, mtx, range_coef_in, scale);
	rgb = clamp(rgb, 0.0f, scale_out);

	return rgb;
}
inline __device__ __host__ float3 YUV2RGB_601_limited(const float3 &yuv, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return YUV2RGB_limited(yuv, mtx_601_yuv2rgb, scale_in, scale_out);
}
inline __device__ __host__ float3 YUV2RGB_709_limited(const float3 &yuv, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return YUV2RGB_limited(yuv, mtx_709_yuv2rgb, scale_in, scale_out);
}
inline __device__ __host__ float3 YUV2RGB_2020_limited(const float3 &yuv, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return YUV2RGB_limited(yuv, mtx_2020_yuv2rgb, scale_in, scale_out);
}
// full range.
inline __device__ __host__ float3 YUV2RGB_full(const float3 &yuv, const float mtx[3][3], float scale_in = 255.0f, float scale_out = 255.0f)
{
	auto range_coef_in = gen_color_space_range_coef_full(scale_in);
	float scale = scale_out / scale_in;

	float3 rgb = YUV2RGB(yuv, mtx, range_coef_in, scale);
	rgb = clamp(rgb, 0.0f, scale_out);

	return rgb;
}
inline __device__ __host__ float3 YUV2RGB_601_full(const float3 &yuv, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return YUV2RGB_full(yuv, mtx_601_yuv2rgb, scale_in, scale_out);
}
inline __device__ __host__ float3 YUV2RGB_709_full(const float3 &yuv, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return YUV2RGB_full(yuv, mtx_709_yuv2rgb, scale_in, scale_out);
}
inline __device__ __host__ float3 YUV2RGB_2020_full(const float3 &yuv, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return YUV2RGB_full(yuv, mtx_2020_yuv2rgb, scale_in, scale_out);
}



//-----------------------------------------------------------------------------------
// RGB to YUV colorspace conversion
// in: [0, scale_in].
// out: [0, scale_out].
//-----------------------------------------------------------------------------------
inline __device__ __host__ float3 RGB2YUV(const float3 &rgb, const float mtx[3][3], const float3 &d_range, const float3 &bias, float scale = 1.0f)
{
	float3 rgb_tmp = rgb * scale;

	float3 yuv = mul_mtx33_vec3(rgb_tmp, mtx);

	yuv = yuv * d_range + bias;

	return yuv;
}
inline __device__ __host__ float3 RGB2YUV(const float3 &rgb, const float mtx[3][3], const st_ColorSpaceRangeCoef &range_coef, float scale = 1.0f)
{
	float3 d_range = {
		range_coef.Yrange,
		range_coef.Crange,
		range_coef.Crange,
	};
	float3 bias = {
		range_coef.Ybias,
		range_coef.Cbias,
		range_coef.Cbias,
	};

	float3 yuv = RGB2YUV(rgb, mtx, d_range, bias, scale);

	return yuv;
}
// limited range.
inline __device__ __host__ float3 RGB2YUV_limited(const float3 &rgb, const float mtx[3][3], float scale_in = 255.0f, float scale_out = 255.0f)
{
	auto range_coef_out = gen_color_space_range_coef_limited(scale_out);
	float scale = scale_out / scale_in;

	float3 yuv = RGB2YUV(rgb, mtx, range_coef_out, scale);
	yuv = clamp(yuv, 0.0f, scale_out);

	return yuv;
}
inline __device__ __host__ float3 RGB2YUV_601_limited(const float3 &rgb, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return RGB2YUV_limited(rgb, mtx_601_rgb2yuv, scale_in, scale_out);
}
inline __device__ __host__ float3 RGB2YUV_709_limited(const float3 &rgb, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return RGB2YUV_limited(rgb, mtx_709_rgb2yuv, scale_in, scale_out);
}
inline __device__ __host__ float3 RGB2YUV_2020_limited(const float3 &rgb, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return RGB2YUV_limited(rgb, mtx_2020_rgb2yuv, scale_in, scale_out);
}
// full range.
inline __device__ __host__ float3 RGB2YUV_full(const float3 &rgb, const float mtx[3][3], float scale_in = 255.0f, float scale_out = 255.0f)
{
	auto range_coef_out = gen_color_space_range_coef_full(scale_out);
	float scale = scale_out / scale_in;

	float3 yuv = RGB2YUV(rgb, mtx, range_coef_out, scale);
	yuv = clamp(yuv, 0.0f, scale_out);

	return yuv;
}
inline __device__ __host__ float3 RGB2YUV_601_full(const float3 &rgb, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return RGB2YUV_full(rgb, mtx_601_rgb2yuv, scale_in, scale_out);
}
inline __device__ __host__ float3 RGB2YUV_709_full(const float3 &rgb, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return RGB2YUV_full(rgb, mtx_709_rgb2yuv, scale_in, scale_out);
}
inline __device__ __host__ float3 RGB2YUV_2020_full(const float3 &rgb, float scale_in = 255.0f, float scale_out = 255.0f)
{
	return RGB2YUV_full(rgb, mtx_2020_rgb2yuv, scale_in, scale_out);
}

#endif
