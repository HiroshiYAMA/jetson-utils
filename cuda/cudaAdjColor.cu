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

#include "cudaAdjColor.h"

// gpuAdjColor.
template<typename T, typename S>
__global__ void gpuAdjColor( T* input, S* output, int width, int height,
							 float sat, float gain, float contrast,
							 float max_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float4 pi = make_float4(input[y * width + x]);

	float3 rgb = make_float3(pi);
	rgb /= 255.0f;
	rgb = applyColorAdjustment(rgb, sat, gain, contrast);
	rgb *= 255.0f;

	S po;
	if (imageFormatFromType<T>() == IMAGE_RGBA8
		|| imageFormatFromType<T>() == IMAGE_RGBA32F
		|| imageFormatFromType<T>() == IMAGE_BGRA8
		|| imageFormatFromType<T>() == IMAGE_BGRA32F) {
		float4 rgba = make_float4(rgb, pi.w);
		po = cast_vec<S>(rgba);
	} else {
		po = cast_vec<S>(rgb);
	}

	output[y * width + x] = po;
}

// launchResize
template<typename T, typename S>
static cudaError_t launchResize( T* input, S* output, size_t width, size_t height,
								 float sat, float gain, float contrast,
								 float max_value, cudaStream_t stream )
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

	gpuAdjColor<T, S><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, sat, gain, contrast, max_value);

	return CUDA(cudaGetLastError());
}

#define FUNC_CUDA_RESIZE(T, S) \
cudaError_t cudaAdjColor( T* input, S* output, size_t width, size_t height, \
						  float sat, float gain, float contrast, \
						  float max_value, cudaStream_t stream ) \
{ \
	return launchResize<T, S>(input, output, width, height, sat, gain, contrast, max_value, stream); \
}

// cudaAdjColor (uint8 grayscale)
FUNC_CUDA_RESIZE(uint8_t, uint8_t);
FUNC_CUDA_RESIZE(float, uint8_t);
FUNC_CUDA_RESIZE(uchar3, uint8_t);
FUNC_CUDA_RESIZE(uchar4, uint8_t);
FUNC_CUDA_RESIZE(float3, uint8_t);
FUNC_CUDA_RESIZE(float4, uint8_t);

// cudaAdjColor (float grayscale)
FUNC_CUDA_RESIZE(uint8_t, float);
FUNC_CUDA_RESIZE(float, float);
FUNC_CUDA_RESIZE(uchar3, float);
FUNC_CUDA_RESIZE(uchar4, float);
FUNC_CUDA_RESIZE(float3, float);
FUNC_CUDA_RESIZE(float4, float);

// cudaAdjColor (uchar3)
FUNC_CUDA_RESIZE(uint8_t, uchar3);
FUNC_CUDA_RESIZE(float, uchar3);
FUNC_CUDA_RESIZE(uchar3, uchar3);
FUNC_CUDA_RESIZE(uchar4, uchar3);
FUNC_CUDA_RESIZE(float3, uchar3);
FUNC_CUDA_RESIZE(float4, uchar3);

// cudaAdjColor (uchar4)
FUNC_CUDA_RESIZE(uint8_t, uchar4);
FUNC_CUDA_RESIZE(float, uchar4);
FUNC_CUDA_RESIZE(uchar3, uchar4);
FUNC_CUDA_RESIZE(uchar4, uchar4);
FUNC_CUDA_RESIZE(float3, uchar4);
FUNC_CUDA_RESIZE(float4, uchar4);

// cudaAdjColor (float3)
FUNC_CUDA_RESIZE(uint8_t, float3);
FUNC_CUDA_RESIZE(float, float3);
FUNC_CUDA_RESIZE(uchar3, float3);
FUNC_CUDA_RESIZE(uchar4, float3);
FUNC_CUDA_RESIZE(float3, float3);
FUNC_CUDA_RESIZE(float4, float3);

// cudaAdjColor (float4)
FUNC_CUDA_RESIZE(uint8_t, float4);
FUNC_CUDA_RESIZE(float, float4);
FUNC_CUDA_RESIZE(uchar3, float4);
FUNC_CUDA_RESIZE(uchar4, float4);
FUNC_CUDA_RESIZE(float3, float4);
FUNC_CUDA_RESIZE(float4, float4);

#undef FUNC_CUDA_RESIZE

//-----------------------------------------------------------------------------------
cudaError_t cudaAdjColor( void* input, void* output, size_t width, size_t height, imageFormat format,
						  float sat, float gain, float contrast, cudaStream_t stream )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaAdjColor((uchar3*)input, (uchar3*)output, width, height, sat, gain, contrast, 255.0f, stream);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaAdjColor((uchar4*)input, (uchar4*)output, width, height, sat, gain, contrast, 255.0f, stream);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaAdjColor((float3*)input, (float3*)output, width, height, sat, gain, contrast, FLT_MAX, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaAdjColor((float4*)input, (float4*)output, width, height, sat, gain, contrast, FLT_MAX, stream);
	else if( format == IMAGE_GRAY8 )
		return cudaAdjColor((uint8_t*)input, (uint8_t*)output, width, height, sat, gain, contrast, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaAdjColor((float*)input, (float*)output, width, height, sat, gain, contrast, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaAdjColor() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
