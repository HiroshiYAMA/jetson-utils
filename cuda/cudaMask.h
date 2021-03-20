#ifndef __CUDA_MASK_H__
#define __CUDA_MASK_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"
#include <vector>


/**
 * Mask an image on the GPU (supports RGB/BGR, RGBA/BGRA)
 * @ingroup mask
 */
cudaError_t cudaMask(void *input, void *mask, void *output, size_t width, size_t height,
    imageFormat format, imageFormat format_mask, float bg_color[3], float2 range);
cudaError_t cudaMask(void *input, void *mask, void *output, size_t width, size_t height,
    imageFormat format, imageFormat format_mask, float bg_color[3]);

#endif
