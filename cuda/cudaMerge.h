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

#ifndef __CUDA_MERGE_H__
#define __CUDA_MERGE_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"
#include <vector>


/**
 * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from some single color planes(using GRAY format))
 * @ingroup merge
 */
cudaError_t cudaMerge(void **input, void *output, size_t width, size_t height, imageFormat format, cudaStream_t stream = NULL);

/**
 * Merge an image on the GPU (supports RGB/BGR, RGBA/BGRA from 3 colors and alpha plane)
 * @ingroup merge
 */
#define FUNC_CUDA_MERGE_HEADER(T, S, R) \
cudaError_t cudaMerge(T* input_color, S* input_alpha, R* output, size_t width, size_t height, cudaStream_t stream = NULL);

FUNC_CUDA_MERGE_HEADER(uchar3, uint8_t, uchar4);
FUNC_CUDA_MERGE_HEADER(uchar3, float, uchar4);
FUNC_CUDA_MERGE_HEADER(uchar4, uint8_t, uchar4);
FUNC_CUDA_MERGE_HEADER(uchar4, float, uchar4);
FUNC_CUDA_MERGE_HEADER(float3, uint8_t, float4);
FUNC_CUDA_MERGE_HEADER(float3, float, float4);
FUNC_CUDA_MERGE_HEADER(float4, uint8_t, float4);
FUNC_CUDA_MERGE_HEADER(float4, float, float4);

#undef FUNC_CUDA_MERGE_HEADER

cudaError_t cudaMerge(void *input_color, void *input_alpha, void *output, size_t width, size_t height, imageFormat format, cudaStream_t stream = NULL);

#endif
