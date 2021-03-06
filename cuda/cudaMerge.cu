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

 #include "cudaMerge.h"



// gpuMerge.
// * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from some single color planes(using GRAY format))
template<typename T, int CH>
__global__ void gpuMerge(T *input0, T *input1, T *input2, T *input3, T *output, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p0 = input0[y * width + x];
	const T p1 = input1[y * width + x];
	const T p2 = input2[y * width + x];
	const T p3 = (CH == 4) ? input3[y * width + x] : T(0);

	output[(y * width + x) * CH + 0] = p0;
	output[(y * width + x) * CH + 1] = p1;
	output[(y * width + x) * CH + 2] = p2;
	if (CH == 4) output[(y * width + x) * CH + 3] = p3;
}

// gpuMerge.
// * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from 3 colors and alpha plane)
template<typename T, typename S, typename R>
__global__ void gpuMerge(T *input_color, S *input_alpha, R *output, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	T pix_color = input_color[y * width + x];
	S pix_alpha = input_alpha[y * width + x];
	R pix = make_vec<R>(pix_color.x, pix_color.y, pix_color.z, pix_alpha);

	output[y * width + x] = pix;
}

// launchMerge
// * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from some single color planes(using GRAY format))
template<typename T, int CH>
static cudaError_t launchMerge(T **input, T *output, size_t width, size_t height, cudaStream_t stream)
{
	if( !input[0] || !input[1] || !input[2] || (CH == 4 ? !input[3] : false) || !output )
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

	gpuMerge<T, CH><<<gridDim, blockDim, 0, stream>>>(input[0], input[1], input[2], (CH == 4) ? input[3] : nullptr, output, width, height);

	return CUDA(cudaGetLastError());
}

// launchMerge
// * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from 3 colors and alpha plane)
template<typename T, typename S, typename R>
static cudaError_t launchMerge(T *input_color, S *input_alpha, R *output, size_t width, size_t height, cudaStream_t stream)
{
	if( !input_color || !input_alpha || !output )
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

	gpuMerge<T, S, R><<<gridDim, blockDim, 0, stream>>>(input_color, input_alpha, output, width, height);

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaMerge(void **input, void *output, size_t width, size_t height, imageFormat format, cudaStream_t stream)
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
		return launchMerge<uchar, 3>((uchar **)input, (uchar *)output, width, height, stream);
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return launchMerge<uchar, 4>((uchar **)input, (uchar *)output, width, height, stream);
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
		return launchMerge<float, 3>((float **)input, (float *)output, width, height, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return launchMerge<float, 4>((float **)input, (float *)output, width, height, stream);

	LogError(LOG_CUDA "cudaMerge() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgb8, bgr8\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}

cudaError_t cudaMerge(void *input_color, void *input_alpha, void *output, size_t width, size_t height, imageFormat format, cudaStream_t stream)
{
	if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
		return launchMerge<uchar3, uchar, uchar4>((uchar3 *)input_color, (uchar *)input_alpha, (uchar4 *)output, width, height, stream);
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
		return launchMerge<float3, float, float4>((float3 *)input_color, (float *)input_alpha, (float4 *)output, width, height, stream);

	LogError(LOG_CUDA "cudaMerge() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
