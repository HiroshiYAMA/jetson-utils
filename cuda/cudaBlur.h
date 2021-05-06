/*
 * Copyright (c) 2021, hogehoge. All rights reserved.
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

#ifndef __CUDA_BLUR_H__
#define __CUDA_BLUR_H__


#include "cudaUtility.h"
#include "cudaVector.h"
#include "imageFormat.h"

enum BlurType {
    BLUR_1X1,
    BLUR_3X3,
    BLUR_5X5,
    BLUR_7X7,
    BLUR_9X9,
    BLUR_MAX,
};

#define FUNC_CUDA_BLUR_HEADER(T, S, M) \
cudaError_t cudaBlur( \
    T* input, S* tmp_buf, S* output, \
    size_t width, size_t height, int blur_type, float max_value = M, cudaStream_t stream = NULL );

/**
 * Blur a uint8 grayscale image on the GPU.
 * @ingroup blur
 */
FUNC_CUDA_BLUR_HEADER(uint8_t, uint8_t, 255.0f);
FUNC_CUDA_BLUR_HEADER(float, uint8_t, 255.0f);
FUNC_CUDA_BLUR_HEADER(uchar3, uint8_t, 255.0f);
FUNC_CUDA_BLUR_HEADER(uchar4, uint8_t, 255.0f);
FUNC_CUDA_BLUR_HEADER(float3, uint8_t, 255.0f);
FUNC_CUDA_BLUR_HEADER(float4, uint8_t, 255.0f);

/**
 * Blur a floating-point grayscale image on the GPU.
 * @ingroup blur
 */
FUNC_CUDA_BLUR_HEADER(uint8_t, float, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float, float, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(uchar3, float, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(uchar4, float, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float3, float, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float4, float, FLT_MAX);

/**
 * Blur a uchar3 RGB/BGR image on the GPU.
 * @ingroup blur
 */
FUNC_CUDA_BLUR_HEADER(uint8_t, uchar3, 255.0f);
FUNC_CUDA_BLUR_HEADER(float, uchar3, 255.0f);
FUNC_CUDA_BLUR_HEADER(uchar3, uchar3, 255.0f);
FUNC_CUDA_BLUR_HEADER(uchar4, uchar3, 255.0f);
FUNC_CUDA_BLUR_HEADER(float3, uchar3, 255.0f);
FUNC_CUDA_BLUR_HEADER(float4, uchar3, 255.0f);

/**
 * Blur a float3 RGB/BGR image on the GPU.
 * @ingroup blur
 */
FUNC_CUDA_BLUR_HEADER(uint8_t, float3, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float, float3, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(uchar3, float3, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(uchar4, float3, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float3, float3, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float4, float3, FLT_MAX);

/**
 * Blur a uchar4 RGBA/BGRA image on the GPU.
 * @ingroup blur
 */
FUNC_CUDA_BLUR_HEADER(uint8_t, uchar4, 255.0f);
FUNC_CUDA_BLUR_HEADER(float, uchar4, 255.0f);
FUNC_CUDA_BLUR_HEADER(uchar3, uchar4, 255.0f);
FUNC_CUDA_BLUR_HEADER(uchar4, uchar4, 255.0f);
FUNC_CUDA_BLUR_HEADER(float3, uchar4, 255.0f);
FUNC_CUDA_BLUR_HEADER(float4, uchar4, 255.0f);

/**
 * Blur a float4 RGBA/BGRA image on the GPU.
 * @ingroup blur
 */
FUNC_CUDA_BLUR_HEADER(uint8_t, float4, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float, float4, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(uchar3, float4, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(uchar4, float4, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float3, float4, FLT_MAX);
FUNC_CUDA_BLUR_HEADER(float4, float4, FLT_MAX);

#undef FUNC_CUDA_BLUR_HEADER

/**
 * Blur an image on the GPU (supports grayscale, RGB/BGR, RGBA/BGRA)
 * @ingroup blur
 */
cudaError_t cudaBlur(
    void* input,  void* tmp_buf, void* output,
    size_t width, size_t height, imageFormat format, int blur_type, cudaStream_t stream = NULL );

#endif

