#ifndef __CUDA_THRESHOLD_H__
#define __CUDA_THRESHOLD_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"

enum class BinarizationFlags : int {
    THRESH_BINARY     = 0,
    THRESH_BINARY_INV = 1,
    THRESH_TRUNC      = 2,
    THRESH_TOZERO     = 3,
    THRESH_TOZERO_INV = 4,
    THRESH_MAX        = 5,
};


/**
 * Binarize an image on the GPU (supports grayscale)
 * @ingroup threshold
 */
cudaError_t cudaThreshold(void* input, void* output, size_t width, size_t height, imageFormat format,
    float threshold, float max_value, int mode);

#endif

