#include "cudaMask.h"



// gpuMask.
// * Mask an image on the GPU (supports RGB/BGR, RGBA/BGRA)
template<typename T, typename S>
__global__ void gpuMask(T *input, S *mask, T *output, size_t width, size_t height, float4 bg)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	T pix_src = input[y * width + x];
	S pix_mask = mask[y * width + x];
	T pix_bg = cast_vec<T>(bg);
	T pix_dst = (pix_mask > S(0)) ? pix_src : pix_bg;

	output[y * width + x] = pix_dst;
}

// launchMask
// * Mask an image on the GPU (supports RGB/BGR, RGBA/BGRA)
template<typename T, typename S>
static cudaError_t launchMask(T *input, S *mask, T *output, size_t width, size_t height, float bg_color[3])
{
	if( !input || !mask || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	const auto bg = make_vec<float4>(bg_color[0], bg_color[1], bg_color[2], 255.0f);
	gpuMask<T, S><<<gridDim, blockDim>>>(input, mask, output, width, height, bg);

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
cudaError_t cudaMask(void *input, void *mask, void *output, size_t width, size_t height,
    imageFormat format, imageFormat format_mask, float bg_color[3])
{
	if( format == IMAGE_RGB8 || format == IMAGE_BGR8 ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<uchar3, uchar>((uchar3 *)input, (uchar *)mask, (uchar3 *)output, width, height, bg_color);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<uchar3, float>((uchar3 *)input, (float *)mask, (uchar3 *)output, width, height, bg_color);
		}
	}
	else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<uchar4, uchar>((uchar4 *)input, (uchar *)mask, (uchar4 *)output, width, height, bg_color);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<uchar4, float>((uchar4 *)input, (float *)mask, (uchar4 *)output, width, height, bg_color);
		}
	}
	else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<float3, uchar>((float3 *)input, (uchar *)mask, (float3 *)output, width, height, bg_color);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<float3, float>((float3 *)input, (float *)mask, (float3 *)output, width, height, bg_color);
		}
	}
	else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F ) {
		if ( format_mask == IMAGE_GRAY8 ) {
			return launchMask<float4, uchar>((float4 *)input, (uchar *)mask, (float4 *)output, width, height, bg_color);
		} else if ( format_mask == IMAGE_GRAY32F ) {
			return launchMask<float4, float>((float4 *)input, (float *)mask, (float4 *)output, width, height, bg_color);
		}
	}

	LogError(LOG_CUDA "cudaMask() -- invalid image format '%s'\n", imageFormatToStr(format));
	LogError(LOG_CUDA "                supported formats are:\n");
	LogError(LOG_CUDA "                    * rgba8, bgra8\n");
	LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

	return cudaErrorInvalidValue;
}
