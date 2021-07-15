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

#include "cudaBlur.h"



// gpuBlur. horizontal.
template<typename T, typename S, int blur_num>
__global__ void gpuBlur_h( T* input, S* output, int width, int height, float max_value )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x >= width || dst_y >= height) return;

	const auto scale = 1.0f / blur_num;
	const int pos_x[9] = {
		::max(dst_x - 4, 0),
		::max(dst_x - 3, 0),
		::max(dst_x - 2, 0),
		::max(dst_x - 1, 0),
		dst_x,
		::min(dst_x + 1, width - 1),
		::min(dst_x + 2, width - 1),
		::min(dst_x + 3, width - 1),
		::min(dst_x + 4, width - 1),
	};
	const float4 px[9] = {
		blur_num < 9 ? float4{} : make_float4(input[dst_y * width + pos_x[0]]),
		blur_num < 7 ? float4{} : make_float4(input[dst_y * width + pos_x[1]]),
		blur_num < 5 ? float4{} : make_float4(input[dst_y * width + pos_x[2]]),
		blur_num < 3 ? float4{} : make_float4(input[dst_y * width + pos_x[3]]),
		make_float4(input[dst_y * width + pos_x[4]]),
		blur_num < 3 ? float4{} : make_float4(input[dst_y * width + pos_x[5]]),
		blur_num < 5 ? float4{} : make_float4(input[dst_y * width + pos_x[6]]),
		blur_num < 7 ? float4{} : make_float4(input[dst_y * width + pos_x[7]]),
		blur_num < 9 ? float4{} : make_float4(input[dst_y * width + pos_x[8]]),
	};
	const S p = cast_vec<S>(clamp((px[0] + px[1] + px[2] + px[3] + px[4] + px[5] + px[6] + px[7] + px[8]) * scale, 0.0f, max_value));
	output[dst_y * width + dst_x] = p;
}

// gpuBlur. vertical.
template<typename T, typename S, int blur_num>
__global__ void gpuBlur_v( T* input, S* output, int width, int height, float max_value )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x >= width || dst_y >= height) return;

	const auto scale = 1.0f / blur_num;
	const int pos_y[9] = {
		::max(dst_y - 4, 0),
		::max(dst_y - 3, 0),
		::max(dst_y - 2, 0),
		::max(dst_y - 1, 0),
		dst_y,
		::min(dst_y + 1, height - 1),
		::min(dst_y + 2, height - 1),
		::min(dst_y + 3, height - 1),
		::min(dst_y + 4, height - 1),
	};
	const float4 py[9] = {
		blur_num < 9 ? float4{} : make_float4(input[pos_y[0] * width + dst_x]),
		blur_num < 7 ? float4{} : make_float4(input[pos_y[1] * width + dst_x]),
		blur_num < 5 ? float4{} : make_float4(input[pos_y[2] * width + dst_x]),
		blur_num < 3 ? float4{} : make_float4(input[pos_y[3] * width + dst_x]),
		make_float4(input[pos_y[4] * width + dst_x]),
		blur_num < 3 ? float4{} : make_float4(input[pos_y[5] * width + dst_x]),
		blur_num < 5 ? float4{} : make_float4(input[pos_y[6] * width + dst_x]),
		blur_num < 7 ? float4{} : make_float4(input[pos_y[7] * width + dst_x]),
		blur_num < 9 ? float4{} : make_float4(input[pos_y[8] * width + dst_x]),
	};
	const S p = cast_vec<S>(clamp((py[0] + py[1] + py[2] + py[3] + py[4] + py[5] + py[6] + py[7] + py[8]) * scale, 0.0f, max_value));
	output[dst_y * width + dst_x] = p;
}

// launchBlur
template<typename T, typename S>
static cudaError_t launchBlur(
	T* input, S* tmp_buf, S* output,
	size_t width, size_t height, int blur_type, float max_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	switch (blur_type) {
	case BlurType::BLUR_3X3:
		gpuBlur_h<T, S, 3><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuBlur_v<S, S, 3><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value);
		break;
	case BlurType::BLUR_5X5:
		gpuBlur_h<T, S, 5><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuBlur_v<S, S, 5><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value);
		break;
	case BlurType::BLUR_7X7:
		gpuBlur_h<T, S, 7><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuBlur_v<S, S, 7><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value);
		break;
	case BlurType::BLUR_9X9:
		gpuBlur_h<T, S, 9><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuBlur_v<S, S, 9><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value);
		break;
	default:
		gpuBlur_h<T, S, 1><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuBlur_v<S, S, 1><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value);
	}

	return CUDA(cudaGetLastError());
}

#define FUNC_CUDA_BLUR(T, S) \
cudaError_t cudaBlur( T* input, S* tmp_buf, S* output, size_t width, size_t height, int blur_type, float max_value, cudaStream_t stream ) \
{ \
	return launchBlur<T, S>(input, tmp_buf, output, width, height, blur_type, max_value, stream); \
}

// cudaBlur (uint8 grayscale)
FUNC_CUDA_BLUR(uint8_t, uint8_t);
FUNC_CUDA_BLUR(float, uint8_t);
FUNC_CUDA_BLUR(uchar3, uint8_t);
FUNC_CUDA_BLUR(uchar4, uint8_t);
FUNC_CUDA_BLUR(float3, uint8_t);
FUNC_CUDA_BLUR(float4, uint8_t);

// cudaBlur (float grayscale)
FUNC_CUDA_BLUR(uint8_t, float);
FUNC_CUDA_BLUR(float, float);
FUNC_CUDA_BLUR(uchar3, float);
FUNC_CUDA_BLUR(uchar4, float);
FUNC_CUDA_BLUR(float3, float);
FUNC_CUDA_BLUR(float4, float);

// cudaBlur (uchar3)
FUNC_CUDA_BLUR(uint8_t, uchar3);
FUNC_CUDA_BLUR(float, uchar3);
FUNC_CUDA_BLUR(uchar3, uchar3);
FUNC_CUDA_BLUR(uchar4, uchar3);
FUNC_CUDA_BLUR(float3, uchar3);
FUNC_CUDA_BLUR(float4, uchar3);

// cudaBlur (uchar4)
FUNC_CUDA_BLUR(uint8_t, uchar4);
FUNC_CUDA_BLUR(float, uchar4);
FUNC_CUDA_BLUR(uchar3, uchar4);
FUNC_CUDA_BLUR(uchar4, uchar4);
FUNC_CUDA_BLUR(float3, uchar4);
FUNC_CUDA_BLUR(float4, uchar4);

// cudaBlur (float3)
FUNC_CUDA_BLUR(uint8_t, float3);
FUNC_CUDA_BLUR(float, float3);
FUNC_CUDA_BLUR(uchar3, float3);
FUNC_CUDA_BLUR(uchar4, float3);
FUNC_CUDA_BLUR(float3, float3);
FUNC_CUDA_BLUR(float4, float3);

// cudaBlur (float4)
FUNC_CUDA_BLUR(uint8_t, float4);
FUNC_CUDA_BLUR(float, float4);
FUNC_CUDA_BLUR(uchar3, float4);
FUNC_CUDA_BLUR(uchar4, float4);
FUNC_CUDA_BLUR(float3, float4);
FUNC_CUDA_BLUR(float4, float4);

#undef FUNC_CUDA_BLUR

//-----------------------------------------------------------------------------------
cudaError_t cudaBlur(
	void* input,  void* tmp_buf, void* output,
	size_t width, size_t height, imageFormat format, int blur_type, cudaStream_t stream )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaBlur((uchar3*)input, (uchar3*)tmp_buf, (uchar3*)output, width, height, blur_type, 255.0f, stream);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaBlur((uchar4*)input, (uchar4*)tmp_buf, (uchar4*)output, width, height, blur_type, 255.0f, stream);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaBlur((float3*)input, (float3*)tmp_buf, (float3*)output, width, height, blur_type, FLT_MAX, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaBlur((float4*)input, (float4*)tmp_buf, (float4*)output, width, height, blur_type, FLT_MAX, stream);
	else if( format == IMAGE_GRAY8 )
		return cudaBlur((uint8_t*)input, (uint8_t*)tmp_buf, (uint8_t*)output, width, height, blur_type, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaBlur((float*)input, (float*)tmp_buf, (float*)output, width, height, blur_type, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaBlur() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
