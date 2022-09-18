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

#include "cudaSpill.h"



template <typename T>
__global__ void gpuSpill( T* input, T* output, int width, int height, em_COLOR_ADJ_SPILL_MODE spill_mode )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px = input[ y * width + x ];

	output[ y * width + x ] = apply_spill_pix(px, spill_mode);
}

template<typename T>
static cudaError_t launchSpill( T* input, T* output, size_t  width, size_t height, em_COLOR_ADJ_SPILL_MODE spill_mode, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0  )
		return cudaErrorInvalidValue;

	// launch kernel
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuSpill<T><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, spill_mode);

	return CUDA(cudaGetLastError());
}



//-----------------------------------------------------------------------------------
cudaError_t cudaSpill( uchar3* input, uchar3* output, size_t width, size_t height, em_COLOR_ADJ_SPILL_MODE spill_mode, cudaStream_t stream )
{
	return launchSpill<uchar3>(input, output, width, height, spill_mode, stream);
}
cudaError_t cudaSpill( uchar4* input, uchar4* output, size_t width, size_t height, em_COLOR_ADJ_SPILL_MODE spill_mode, cudaStream_t stream )
{
	return launchSpill<uchar4>(input, output, width, height, spill_mode, stream);
}
cudaError_t cudaSpill( float3* input, float3* output, size_t width, size_t height, em_COLOR_ADJ_SPILL_MODE spill_mode, cudaStream_t stream )
{
	return launchSpill<float3>(input, output, width, height, spill_mode, stream);
}
cudaError_t cudaSpill( float4* input, float4* output, size_t width, size_t height, em_COLOR_ADJ_SPILL_MODE spill_mode, cudaStream_t stream )
{
	return launchSpill<float4>(input, output, width, height, spill_mode, stream);
}

cudaError_t cudaSpill( void* input, void* output, size_t width, size_t height, imageFormat format, em_COLOR_ADJ_SPILL_MODE spill_mode, cudaStream_t stream )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaSpill((uchar3*)input, (uchar3*)output, width, height, spill_mode, stream);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaSpill((uchar4*)input, (uchar4*)output, width, height, spill_mode, stream);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaSpill((float3*)input, (float3*)output, width, height, spill_mode, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaSpill((float4*)input, (float4*)output, width, height, spill_mode, stream);

	LogError(LOG_CUDA "cudaSpill() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                   supported formats are:\n");
	LogError(LOG_CUDA "                       * rgb8, bgr8\n");
	LogError(LOG_CUDA "                       * rgba8, bgra8\n");
	LogError(LOG_CUDA "                       * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                       * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
