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

#ifndef __CUDA_MORPHOLOGY_H__
#define __CUDA_MORPHOLOGY_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "cudaMath.h"
#include "imageFormat.h"

enum MorphologyType {
    MORPHOLOGY_1X1,
    MORPHOLOGY_3X3,
    MORPHOLOGY_5X5,
    MORPHOLOGY_7X7,
    MORPHOLOGY_9X9,
    MORPHOLOGY_MAX,
};

#define FUNC_CUDA_EROSION_HEADER(T, S, M) \
cudaError_t cudaErosion( \
    T* input, S* tmp_buf, S* output, \
    size_t width, size_t height, int morphology_type, float max_value = M, cudaStream_t stream = NULL );
#define FUNC_CUDA_DILATION_HEADER(T, S, M) \
cudaError_t cudaDilation( \
    T* input, S* tmp_buf, S* output, \
    size_t width, size_t height, int morphology_type, float max_value = M, cudaStream_t stream = NULL );

/**
 * Erosion a uint8 grayscale image on the GPU.
 * @ingroup morphology
 */
FUNC_CUDA_EROSION_HEADER(uint8_t, uint8_t, 255.0f);
FUNC_CUDA_EROSION_HEADER(float, uint8_t, 255.0f);

/**
 * Erosion a floating-point grayscale image on the GPU.
 * @ingroup morphology
 */
FUNC_CUDA_EROSION_HEADER(uint8_t, float, FLT_MAX);
FUNC_CUDA_EROSION_HEADER(float, float, FLT_MAX);

/**
 * Dilation a uint8 grayscale image on the GPU.
 * @ingroup morphology
 */
FUNC_CUDA_DILATION_HEADER(uint8_t, uint8_t, 255.0f);
FUNC_CUDA_DILATION_HEADER(float, uint8_t, 255.0f);

/**
 * Dilation a floating-point grayscale image on the GPU.
 * @ingroup morphology
 */
FUNC_CUDA_DILATION_HEADER(uint8_t, float, FLT_MAX);
FUNC_CUDA_DILATION_HEADER(float, float, FLT_MAX);

#undef FUNC_CUDA_EROSION_HEADER
#undef FUNC_CUDA_DILATION_HEADER

/**
 * Erosion an image on the GPU (supports grayscale, RGB/BGR, RGBA/BGRA)
 * @ingroup morphology
 */
cudaError_t cudaErosion(
    void* input,  void* tmp_buf, void* output,
    size_t width, size_t height, imageFormat format, int morphology_type, cudaStream_t stream = NULL );

/**
 * Dilation an image on the GPU (supports grayscale, RGB/BGR, RGBA/BGRA)
 * @ingroup morphology
 */
cudaError_t cudaDilation(
    void* input,  void* tmp_buf, void* output,
    size_t width, size_t height, imageFormat format, int morphology_type, cudaStream_t stream = NULL );

#endif
