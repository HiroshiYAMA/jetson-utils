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

// launchThreshold
// * Binarize an image on the GPU (supports grayscale)
template<typename T>
static cudaError_t launchThreshold(T *input, T *output, size_t width, size_t height,
	float threshold, float max_value, int mode)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	if (mode == static_cast<int>(BinarizationFlags::THRESH_BINARY_INV)) {
		gpuThreshold_binary_inv<T><<<gridDim, blockDim>>>(input, output, width, height, threshold, max_value);
	} else {
		gpuThreshold_binary<T><<<gridDim, blockDim>>>(input, output, width, height, threshold, max_value);
	}

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaThreshold(void* input, void* output, size_t width, size_t height, imageFormat format,
    float threshold, float max_value, int mode)
{
	if( format == IMAGE_GRAY8 )
		return launchThreshold<uchar>((uchar *)input, (uchar *)output, width, height, threshold, max_value, mode);
	else if( format == IMAGE_GRAY32F )
		return launchThreshold<float>((float *)input, (float *)output, width, height, threshold, max_value, mode);

	LogError(LOG_CUDA "cudaThreshold() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * gray8\n");
	LogError(LOG_CUDA "                    * gray32f\n");

	return cudaErrorInvalidValue;
}
