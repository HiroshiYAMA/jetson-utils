/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "cudaResize.h"
#include "cudaFilterMode.cuh"



// gpuResize. nearest.
template<typename T, typename S>
__global__ void gpuResize_nearest( float2 scale, T* input, int iWidth, int iHeight, S* output, int oWidth, int oHeight )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		const S px = cast_vec<S>(cudaFilterPixel<FILTER_POINT>(input, dst_x, dst_y, iWidth, iHeight, oWidth, oHeight, scale));
		output[dst_y * oWidth + dst_x] = px;
	}
}

// gpuResize. linear.
template <typename T, typename S>
__global__ void gpuResize_linear( float2 scale, T* input, int iWidth, int iHeight, S* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		S out = cast_vec<S>(cudaFilterPixel<FILTER_LINEAR>(input, dst_x, dst_y, iWidth, iHeight, oWidth, oHeight, scale, max_value));
		output[dst_y * oWidth + dst_x] = out;
	}
}

// gpuResize. cubic.
template <typename T, typename S>
__global__ void gpuResize_cubic( float2 scale, T* input, int iWidth, int iHeight, S* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		S out = cast_vec<S>(cudaFilterPixel<FILTER_CUBIC>(input, dst_x, dst_y, iWidth, iHeight, oWidth, oHeight, scale, max_value));
		output[dst_y * oWidth + dst_x] = out;
	}
}

// gpuResize. area.
template <typename T, typename S>
__global__ void gpuResize_area( float2 scale, T* input, int iWidth, int iHeight, S* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		S out = cast_vec<S>(cudaFilterPixel<FILTER_AREA>(input, dst_x, dst_y, iWidth, iHeight, oWidth, oHeight, scale, max_value));
		output[dst_y * oWidth + dst_x] = out;
	}
}

// gpuResize. lanczos4.
template <typename T, typename S>
__global__ void gpuResize_lanczos4( float2 scale, T* input, int iWidth, int iHeight, S* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		S out = cast_vec<S>(cudaFilterPixel<FILTER_LANCZOS4>(input, dst_x, dst_y, iWidth, iHeight, oWidth, oHeight, scale, max_value));
		output[dst_y * oWidth + dst_x] = out;
	}
}

// gpuResize. spline36.
template <typename T, typename S>
__global__ void gpuResize_spline36( float2 scale, T* input, int iWidth, int iHeight, S* output, int oWidth, int oHeight, float max_value )
{
	const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

	if (dst_x < oWidth && dst_y < oHeight)
	{
		S out = cast_vec<S>(cudaFilterPixel<FILTER_SPLINE36>(input, dst_x, dst_y, iWidth, iHeight, oWidth, oHeight, scale, max_value));
		output[dst_y * oWidth + dst_x] = out;
	}
}

// launchResize
template<typename T, typename S>
static cudaError_t launchResize( T* input, size_t inputWidth, size_t inputHeight,
				             S* output, size_t outputWidth, size_t outputHeight, int mode, float max_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							     float(inputHeight) / float(outputHeight) );

	// launch kernel
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if ((inputWidth == outputWidth && inputHeight == outputHeight) || mode == static_cast<int>(InterpolationFlags::INTER_NEAREST)) {
		gpuResize_nearest<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight);

	} else {
		if (mode == static_cast<int>(InterpolationFlags::INTER_LINEAR)) {
			gpuResize_linear<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);

		} else if (mode == static_cast<int>(InterpolationFlags::INTER_CUBIC)) {
			gpuResize_cubic<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);

		} else if (mode == static_cast<int>(InterpolationFlags::INTER_AREA)) {
			if (inputWidth < outputWidth || inputHeight < outputHeight) {
				gpuResize_linear<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
			} else {
				gpuResize_area<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
			}

		} else if (mode == static_cast<int>(InterpolationFlags::INTER_LANCZOS4)) {
			gpuResize_lanczos4<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);

		} else if (mode == static_cast<int>(InterpolationFlags::INTER_SPLINE36)) {
			gpuResize_spline36<T, S><<<gridDim, blockDim, 0, stream>>>(scale, input, inputWidth, inputHeight, output, outputWidth, outputHeight, max_value);
		}
	}

	return CUDA(cudaGetLastError());
}

#define FUNC_CUDA_RESIZE(T, S) \
cudaError_t cudaResize( T* input, size_t inputWidth, size_t inputHeight, S* output, size_t outputWidth, size_t outputHeight, int mode, float max_value, cudaStream_t stream ) \
{ \
	return launchResize<T, S>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, mode, max_value, stream); \
}

// cudaResize (uint8 grayscale)
FUNC_CUDA_RESIZE(uint8_t, uint8_t);
FUNC_CUDA_RESIZE(float, uint8_t);
FUNC_CUDA_RESIZE(uchar3, uint8_t);
FUNC_CUDA_RESIZE(uchar4, uint8_t);
FUNC_CUDA_RESIZE(float3, uint8_t);
FUNC_CUDA_RESIZE(float4, uint8_t);

// cudaResize (float grayscale)
FUNC_CUDA_RESIZE(uint8_t, float);
FUNC_CUDA_RESIZE(float, float);
FUNC_CUDA_RESIZE(uchar3, float);
FUNC_CUDA_RESIZE(uchar4, float);
FUNC_CUDA_RESIZE(float3, float);
FUNC_CUDA_RESIZE(float4, float);

// cudaResize (uchar3)
FUNC_CUDA_RESIZE(uint8_t, uchar3);
FUNC_CUDA_RESIZE(float, uchar3);
FUNC_CUDA_RESIZE(uchar3, uchar3);
FUNC_CUDA_RESIZE(uchar4, uchar3);
FUNC_CUDA_RESIZE(float3, uchar3);
FUNC_CUDA_RESIZE(float4, uchar3);

// cudaResize (uchar4)
FUNC_CUDA_RESIZE(uint8_t, uchar4);
FUNC_CUDA_RESIZE(float, uchar4);
FUNC_CUDA_RESIZE(uchar3, uchar4);
FUNC_CUDA_RESIZE(uchar4, uchar4);
FUNC_CUDA_RESIZE(float3, uchar4);
FUNC_CUDA_RESIZE(float4, uchar4);

// cudaResize (float3)
FUNC_CUDA_RESIZE(uint8_t, float3);
FUNC_CUDA_RESIZE(float, float3);
FUNC_CUDA_RESIZE(uchar3, float3);
FUNC_CUDA_RESIZE(uchar4, float3);
FUNC_CUDA_RESIZE(float3, float3);
FUNC_CUDA_RESIZE(float4, float3);

// cudaResize (float4)
FUNC_CUDA_RESIZE(uint8_t, float4);
FUNC_CUDA_RESIZE(float, float4);
FUNC_CUDA_RESIZE(uchar3, float4);
FUNC_CUDA_RESIZE(uchar4, float4);
FUNC_CUDA_RESIZE(float3, float4);
FUNC_CUDA_RESIZE(float4, float4);

#undef FUNC_CUDA_RESIZE

//-----------------------------------------------------------------------------------
cudaError_t cudaResize( void* input,  size_t inputWidth,  size_t inputHeight,
				    void* output, size_t outputWidth, size_t outputHeight, imageFormat format, int mode, cudaStream_t stream )
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return cudaResize((uchar3*)input, inputWidth, inputHeight, (uchar3*)output, outputWidth, outputHeight, mode, 255.0f, stream);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return cudaResize((uchar4*)input, inputWidth, inputHeight, (uchar4*)output, outputWidth, outputHeight, mode, 255.0f, stream);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return cudaResize((float3*)input, inputWidth, inputHeight, (float3*)output, outputWidth, outputHeight, mode, FLT_MAX, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return cudaResize((float4*)input, inputWidth, inputHeight, (float4*)output, outputWidth, outputHeight, mode, FLT_MAX, stream);
	else if( format == IMAGE_GRAY8 )
		return cudaResize((uint8_t*)input, inputWidth, inputHeight, (uint8_t*)output, outputWidth, outputHeight, mode, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaResize((float*)input, inputWidth, inputHeight, (float*)output, outputWidth, outputHeight, mode, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaResize() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
