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

#include "cudaMorphology.h"



__device__ float min_centered_array(const float *ary, int num, int max_num)
{
	int i_start = (max_num / 2) - (num / 2);
	float val = ary[i_start];

	for (int i = i_start + 1, cnt = 1; cnt < num; i++, cnt++) {
		if (val > ary[i]) val = ary[i];
	}

	return val;
}

__device__ float max_centered_array(const float *ary, int num, int max_num)
{
	int i_start = (max_num / 2) - (num / 2);
	float val = ary[i_start];

	for (int i = i_start + 1, cnt = 1; cnt < num; i++, cnt++) {
		if (val < ary[i]) val = ary[i];
	}

	return val;
}

template<typename T> __device__ T rescale(T v, float scale, float oft_in, float oft_out, float min_val, float max_val)
{
	return clamp((v - oft_in) * scale + oft_out, min_val, max_val);
}

// gpuErosion. horizontal.
template<typename T, typename S, int morphology_num>
__global__ void gpuErosion_h( T* input, S* output, int width, int height, float max_value )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x >= width || dst_y >= height) return;

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
	const float px[9] = {
		morphology_num < 9 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[0]]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[1]]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[2]]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[3]]),
		static_cast<float>(input[dst_y * width + pos_x[4]]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[5]]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[6]]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[7]]),
		morphology_num < 9 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[8]]),
	};

	const S p = cast_vec<S>(clamp(min_centered_array(px, morphology_num, 9), 0.0f, max_value));
	output[dst_y * width + dst_x] = p;
}

// gpuErosion. vertical.
template<typename T, typename S, int morphology_num>
__global__ void gpuErosion_v( T* input, S* output, int width, int height, float max_value, bool is_normalize, float scale, float2 range_in, float2 range_out )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x >= width || dst_y >= height) return;

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
	const float py[9] = {
		morphology_num < 9 ? 0.0f : static_cast<float>(input[pos_y[0] * width + dst_x]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[pos_y[1] * width + dst_x]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[pos_y[2] * width + dst_x]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[pos_y[3] * width + dst_x]),
		static_cast<float>(input[pos_y[4] * width + dst_x]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[pos_y[5] * width + dst_x]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[pos_y[6] * width + dst_x]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[pos_y[7] * width + dst_x]),
		morphology_num < 9 ? 0.0f : static_cast<float>(input[pos_y[8] * width + dst_x]),
	};

	float p_tmp = min_centered_array(py, morphology_num, 9);
	p_tmp = is_normalize ? rescale(p_tmp, scale, range_in.x, range_out.x, range_out.x, range_out.y) : p_tmp;
	const S p = cast_vec<S>(clamp(p_tmp, 0.0f, max_value));
	output[dst_y * width + dst_x] = p;
}

// gpuDilation. horizontal.
template<typename T, typename S, int morphology_num>
__global__ void gpuDilation_h( T* input, S* output, int width, int height, float max_value )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x >= width || dst_y >= height) return;

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
	const float px[9] = {
		morphology_num < 9 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[0]]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[1]]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[2]]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[3]]),
		static_cast<float>(input[dst_y * width + pos_x[4]]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[5]]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[6]]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[7]]),
		morphology_num < 9 ? 0.0f : static_cast<float>(input[dst_y * width + pos_x[8]]),
	};

	const S p = cast_vec<S>(clamp(max_centered_array(px, morphology_num, 9), 0.0f, max_value));
	output[dst_y * width + dst_x] = p;
}

// gpuDilation. vertical.
template<typename T, typename S, int morphology_num>
__global__ void gpuDilation_v( T* input, S* output, int width, int height, float max_value, bool is_normalize, float scale, float2 range_in, float2 range_out )
{
	const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (dst_x >= width || dst_y >= height) return;

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
	const float py[9] = {
		morphology_num < 9 ? 0.0f : static_cast<float>(input[pos_y[0] * width + dst_x]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[pos_y[1] * width + dst_x]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[pos_y[2] * width + dst_x]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[pos_y[3] * width + dst_x]),
		static_cast<float>(input[pos_y[4] * width + dst_x]),
		morphology_num < 3 ? 0.0f : static_cast<float>(input[pos_y[5] * width + dst_x]),
		morphology_num < 5 ? 0.0f : static_cast<float>(input[pos_y[6] * width + dst_x]),
		morphology_num < 7 ? 0.0f : static_cast<float>(input[pos_y[7] * width + dst_x]),
		morphology_num < 9 ? 0.0f : static_cast<float>(input[pos_y[8] * width + dst_x]),
	};

	float p_tmp = max_centered_array(py, morphology_num, 9);
	p_tmp = is_normalize ? rescale(p_tmp, scale, range_in.x, range_out.x, range_out.x, range_out.y) : p_tmp;
	const S p = cast_vec<S>(clamp(p_tmp, 0.0f, max_value));
	output[dst_y * width + dst_x] = p;
}

// launchErosion
template<typename T, typename S>
static cudaError_t launchErosion(
	T* input, S* tmp_buf, S* output,
	size_t width, size_t height, int morphology_type, const float2 &range_in, const float2 &range_out, float max_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const bool is_normalize = !(range_in.x == -1.0f && range_in.y == -1.0f && range_out.x == -1.0f && range_out.y == -1.0f);
	const float scale  = is_normalize ? (range_out.y - range_out.x) / (range_in.y - range_in.x) : 0.0f;

	// launch kernel
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	switch (morphology_type) {
	case MorphologyType::MORPHOLOGY_3X3:
		gpuErosion_h<T, S, 3><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuErosion_v<S, S, 3><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	case MorphologyType::MORPHOLOGY_5X5:
		gpuErosion_h<T, S, 5><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuErosion_v<S, S, 5><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	case MorphologyType::MORPHOLOGY_7X7:
		gpuErosion_h<T, S, 7><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuErosion_v<S, S, 7><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	case MorphologyType::MORPHOLOGY_9X9:
		gpuErosion_h<T, S, 9><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuErosion_v<S, S, 9><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	default:
		gpuErosion_h<T, S, 1><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuErosion_v<S, S, 1><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
	}

	return CUDA(cudaGetLastError());
}

// launchDilation
template<typename T, typename S>
static cudaError_t launchDilation(
	T* input, S* tmp_buf, S* output,
	size_t width, size_t height, int morphology_type, const float2 &range_in, const float2 &range_out, float max_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const bool is_normalize = !(range_in.x == -1.0f && range_in.y == -1.0f && range_out.x == -1.0f && range_out.y == -1.0f);
	const float scale  = is_normalize ? (range_out.y - range_out.x) / (range_in.y - range_in.x) : 0.0f;

	// launch kernel
#ifdef JETSON
	const dim3 blockDim(32, 8);
#else
	const dim3 blockDim(64, 8);
#endif
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	switch (morphology_type) {
	case MorphologyType::MORPHOLOGY_3X3:
		gpuDilation_h<T, S, 3><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuDilation_v<S, S, 3><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	case MorphologyType::MORPHOLOGY_5X5:
		gpuDilation_h<T, S, 5><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuDilation_v<S, S, 5><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	case MorphologyType::MORPHOLOGY_7X7:
		gpuDilation_h<T, S, 7><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuDilation_v<S, S, 7><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	case MorphologyType::MORPHOLOGY_9X9:
		gpuDilation_h<T, S, 9><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuDilation_v<S, S, 9><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
		break;
	default:
		gpuDilation_h<T, S, 1><<<gridDim, blockDim, 0, stream>>>(input, tmp_buf, width, height, max_value);
		gpuDilation_v<S, S, 1><<<gridDim, blockDim, 0, stream>>>(tmp_buf, output, width, height, max_value, is_normalize, scale, range_in, range_out);
	}

	return CUDA(cudaGetLastError());
}

#define FUNC_CUDA_EROSION(T, S) \
cudaError_t cudaErosion( T* input, S* tmp_buf, S* output, size_t width, size_t height, int morphology_type, float max_value, cudaStream_t stream ) \
{ \
	return launchErosion<T, S>(input, tmp_buf, output, width, height, morphology_type, float2{ -1.0f, -1.0f }, float2{ -1.0f, -1.0f }, max_value, stream); \
}
#define FUNC_CUDA_DILATION(T, S) \
cudaError_t cudaDilation( T* input, S* tmp_buf, S* output, size_t width, size_t height, int morphology_type, float max_value, cudaStream_t stream ) \
{ \
	return launchDilation<T, S>(input, tmp_buf, output, width, height, morphology_type, float2{ -1.0f, -1.0f }, float2{ -1.0f, -1.0f }, max_value, stream); \
}
#define FUNC_CUDA_EROSION_NORMALIZE(T, S) \
cudaError_t cudaErosionNormalize( T* input, S* tmp_buf, S* output, size_t width, size_t height, int morphology_type, const float2 &range_in, const float2 &range_out, float max_value, cudaStream_t stream ) \
{ \
	return launchErosion<T, S>(input, tmp_buf, output, width, height, morphology_type, range_in, range_out, max_value, stream); \
}
#define FUNC_CUDA_DILATION_NORMALIZE(T, S) \
cudaError_t cudaDilationNormalize( T* input, S* tmp_buf, S* output, size_t width, size_t height, int morphology_type, const float2 &range_in, const float2 &range_out, float max_value, cudaStream_t stream ) \
{ \
	return launchDilation<T, S>(input, tmp_buf, output, width, height, morphology_type, range_in, range_out, max_value, stream); \
}

// cudaErosion (uint8 grayscale)
FUNC_CUDA_EROSION(uint8_t, uint8_t);
FUNC_CUDA_EROSION(float, uint8_t);
FUNC_CUDA_EROSION_NORMALIZE(uint8_t, uint8_t);
FUNC_CUDA_EROSION_NORMALIZE(float, uint8_t);

// cudaErosion (float grayscale)
FUNC_CUDA_EROSION(uint8_t, float);
FUNC_CUDA_EROSION(float, float);
FUNC_CUDA_EROSION_NORMALIZE(uint8_t, float);
FUNC_CUDA_EROSION_NORMALIZE(float, float);

// cudaDilation (uint8 grayscale)
FUNC_CUDA_DILATION(uint8_t, uint8_t);
FUNC_CUDA_DILATION(float, uint8_t);
FUNC_CUDA_DILATION_NORMALIZE(uint8_t, uint8_t);
FUNC_CUDA_DILATION_NORMALIZE(float, uint8_t);

// cudaDilation (float grayscale)
FUNC_CUDA_DILATION(uint8_t, float);
FUNC_CUDA_DILATION(float, float);
FUNC_CUDA_DILATION_NORMALIZE(uint8_t, float);
FUNC_CUDA_DILATION_NORMALIZE(float, float);

#undef FUNC_CUDA_EROSION
#undef FUNC_CUDA_DILATION
#undef FUNC_CUDA_EROSION_NORMALIZE
#undef FUNC_CUDA_DILATION_NORMALIZE

//-----------------------------------------------------------------------------------
cudaError_t cudaErosion(
	void* input,  void* tmp_buf, void* output,
	size_t width, size_t height, imageFormat format, int morphology_type, cudaStream_t stream )
{
	if( format == IMAGE_GRAY8 )
		return cudaErosion((uint8_t*)input, (uint8_t*)tmp_buf, (uint8_t*)output, width, height, morphology_type, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaErosion((float*)input, (float*)tmp_buf, (float*)output, width, height, morphology_type, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaErosion() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");

	return cudaErrorInvalidValue;
}
cudaError_t cudaErosionNormalize(
	void* input,  void* tmp_buf, void* output,
	size_t width, size_t height, imageFormat format, int morphology_type, const float2 &range_in, const float2 &range_out, cudaStream_t stream )
{
	if( format == IMAGE_GRAY8 )
		return cudaErosionNormalize((uint8_t*)input, (uint8_t*)tmp_buf, (uint8_t*)output, width, height, morphology_type, range_in, range_out, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaErosionNormalize((float*)input, (float*)tmp_buf, (float*)output, width, height, morphology_type, range_in, range_out, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaErosionNormalize() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");

	return cudaErrorInvalidValue;
}

cudaError_t cudaDilation(
	void* input,  void* tmp_buf, void* output,
	size_t width, size_t height, imageFormat format, int morphology_type, cudaStream_t stream )
{
	if( format == IMAGE_GRAY8 )
		return cudaDilation((uint8_t*)input, (uint8_t*)tmp_buf, (uint8_t*)output, width, height, morphology_type, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaDilation((float*)input, (float*)tmp_buf, (float*)output, width, height, morphology_type, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaDilation() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");

	return cudaErrorInvalidValue;
}
cudaError_t cudaDilationNormalize(
	void* input,  void* tmp_buf, void* output,
	size_t width, size_t height, imageFormat format, int morphology_type, const float2 &range_in, const float2 &range_out, cudaStream_t stream )
{
	if( format == IMAGE_GRAY8 )
		return cudaDilationNormalize((uint8_t*)input, (uint8_t*)tmp_buf, (uint8_t*)output, width, height, morphology_type, range_in, range_out, 255.0f, stream);
	else if( format == IMAGE_GRAY32F )
		return cudaDilationNormalize((float*)input, (float*)tmp_buf, (float*)output, width, height, morphology_type, range_in, range_out, FLT_MAX, stream);

	LogError(LOG_CUDA "cudaDilationNormalize() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");

	return cudaErrorInvalidValue;
}
