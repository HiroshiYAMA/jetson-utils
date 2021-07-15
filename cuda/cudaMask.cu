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

 #include "cudaMask.h"



// gpuMask.
// * Mask an image on the GPU (supports RGB/BGR, RGBA/BGRA)
template<typename T, typename S>
__global__ void gpuMask(T *input, S *mask, T *output, size_t width, size_t height, float4 bg, float2 range)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	T pix_src = input[y * width + x];
	S pix_mask = mask[y * width + x];
	T pix_bg = cast_vec<T>(bg);
	// T pix_dst = (pix_mask > S(0)) ? pix_src : pix_bg;
	float alpha = pix_mask / range.y;
	T pix_dst = cast_vec<T>(pix_src * alpha + pix_bg * (1.0f - alpha));

	output[y * width + x] = pix_dst;
}

// launchMask
// * Mask an image on the GPU (supports RGB/BGR, RGBA/BGRA)
template<typename T, typename S>
static cudaError_t launchMask(T *input, S *mask, T *output, size_t width, size_t height, float bg_color[3], float2 range, cudaStream_t stream)
{
	if( !input || !mask || !output )
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

	const auto bg = make_vec<float4>(bg_color[0], bg_color[1], bg_color[2], 255.0f);
	gpuMask<T, S><<<gridDim, blockDim, 0, stream>>>(input, mask, output, width, height, bg, range);

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaMask(void *input, void *mask, void *output, size_t width, size_t height,
    imageFormat format, imageFormat format_mask, float bg_color[3], float2 range, cudaStream_t stream)
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<uchar3, uchar>((uchar3 *)input, (uchar *)mask, (uchar3 *)output, width, height, bg_color, range, stream);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<uchar3, float>((uchar3 *)input, (float *)mask, (uchar3 *)output, width, height, bg_color, range, stream);
		}
	}
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<uchar4, uchar>((uchar4 *)input, (uchar *)mask, (uchar4 *)output, width, height, bg_color, range, stream);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<uchar4, float>((uchar4 *)input, (float *)mask, (uchar4 *)output, width, height, bg_color, range, stream);
		}
	}
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<float3, uchar>((float3 *)input, (uchar *)mask, (float3 *)output, width, height, bg_color, range, stream);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<float3, float>((float3 *)input, (float *)mask, (float3 *)output, width, height, bg_color, range, stream);
		}
	}
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<float4, uchar>((float4 *)input, (uchar *)mask, (float4 *)output, width, height, bg_color, range, stream);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<float4, float>((float4 *)input, (float *)mask, (float4 *)output, width, height, bg_color, range, stream);
		}
	}

	LogError(LOG_CUDA "cudaMask() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
cudaError_t cudaMask(void *input, void *mask, void *output, size_t width, size_t height,
    imageFormat format, imageFormat format_mask, float bg_color[3], cudaStream_t stream)
{
	return cudaMask(input, mask, output, width, height, format, format_mask, bg_color, float2{0, 255}, stream);
}