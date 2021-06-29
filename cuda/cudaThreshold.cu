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

 #include "cudaThreshold.h"



// gpuThreshold. binary.
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
__global__ void gpuThreshold_binary(T *input, T *output, size_t width, size_t height, float threshold, float max_value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p_src = input[y * width + x];
	const T p_dst = (float(p_src) > threshold) ? T(max_value) : T(0);

	output[y * width + x] = p_dst;
}

// gpuThreshold. binary_inv.
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
__global__ void gpuThreshold_binary_inv(T *input, T *output, size_t width, size_t height, float threshold, float max_value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p_src = input[y * width + x];
	const T p_dst = (float(p_src) > threshold) ? T(0) : T(max_value);

	output[y * width + x] = p_dst;
}

// gpuThreshold. trunc.
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
__global__ void gpuThreshold_trunc(T *input, T *output, size_t width, size_t height, float threshold, float max_value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p_src = input[y * width + x];
	const T p_dst = (float(p_src) > threshold) ? T(threshold) : p_src;

	output[y * width + x] = p_dst;
}

// gpuThreshold. tozero.
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
__global__ void gpuThreshold_tozero(T *input, T *output, size_t width, size_t height, float threshold, float max_value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p_src = input[y * width + x];
	const T p_dst = (float(p_src) > threshold) ? p_src : T(0);

	output[y * width + x] = p_dst;
}

// gpuThreshold. tozero_inv.
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
__global__ void gpuThreshold_tozero_inv(T *input, T *output, size_t width, size_t height, float threshold, float max_value)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T p_src = input[y * width + x];
	const T p_dst = (float(p_src) > threshold) ? T(0) : p_src;

	output[y * width + x] = p_dst;
}

// launchThreshold
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
static cudaError_t launchThreshold(T *input, T *output, size_t width, size_t height,
	float threshold, float max_value, int mode, cudaStream_t stream)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(64, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	if (mode == static_cast<int>(BinarizationFlags::THRESH_BINARY_INV)) {
		gpuThreshold_binary_inv<T><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, threshold, max_value);
	} else if (mode == static_cast<int>(BinarizationFlags::THRESH_TRUNC)) {
		gpuThreshold_trunc<T><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, threshold, max_value);
	} else if (mode == static_cast<int>(BinarizationFlags::THRESH_TOZERO)) {
		gpuThreshold_tozero<T><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, threshold, max_value);
	} else if (mode == static_cast<int>(BinarizationFlags::THRESH_TOZERO_INV)) {
		gpuThreshold_tozero_inv<T><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, threshold, max_value);
	} else {
		gpuThreshold_binary<T><<<gridDim, blockDim, 0, stream>>>(input, output, width, height, threshold, max_value);
	}

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaThreshold(void* input, void* output, size_t width, size_t height, imageFormat format,
    float threshold, float max_value, int mode, cudaStream_t stream)
{
	if( format == IMAGE_GRAY8 )
		return launchThreshold<uchar>((uchar *)input, (uchar *)output, width, height, threshold, max_value, mode, stream);
	else if( format == IMAGE_GRAY32F )
		return launchThreshold<float>((float *)input, (float *)output, width, height, threshold, max_value, mode, stream);

	LogError(LOG_CUDA "cudaThreshold() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");

	return cudaErrorInvalidValue;
}
