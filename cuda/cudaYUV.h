/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_YUV_CONVERT_H
#define __CUDA_YUV_CONVERT_H


#include "cudaUtility.h"
#include "imageFormat.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV I420 4:2:0 planar to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a YUV I420 planar image to RGB uchar3.
 */
cudaError_t cudaI420ToRGB(void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL);

/**
 * Convert a YUV I420 planar image to RGB float3.
 */
cudaError_t cudaI420ToRGB(void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL);

/**
 * Convert a YUV I420 planar image to RGBA uchar4.
 */
cudaError_t cudaI420ToRGBA(void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL);

/**
 * Convert a YUV I420 planar image to RGB float4.
 */
cudaError_t cudaI420ToRGBA(void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL);

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV YV12 4:2:0 planar to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a YUV YV12 planar image to RGB uchar3.
 */
cudaError_t cudaYV12ToRGB(void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL);

/**
 * Convert a YUV YV12 planar image to RGB float3.
 */
cudaError_t cudaYV12ToRGB(void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL);

/**
 * Convert a YUV YV12 planar image to RGBA uchar4.
 */
cudaError_t cudaYV12ToRGBA(void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL);

/**
 * Convert a YUV YV12 planar image to RGB float4.
 */
cudaError_t cudaYV12ToRGBA(void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL);

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name RGB to YUV I420 4:2:0 planar
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert an RGB uchar3 buffer into YUV I420 planar.
 */
cudaError_t cudaRGBToI420( uchar3* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an RGB float3 buffer into YUV I420 planar.
 */
cudaError_t cudaRGBToI420( float3* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an RGBA uchar4 buffer into YUV I420 planar.
 */
cudaError_t cudaRGBAToI420( uchar4* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an RGBA float4 buffer into YUV I420 planar.
 */
cudaError_t cudaRGBAToI420( float4* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name RGB to YUV YV12 4:2:0 planar
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert an RGB uchar3 buffer into YUV YV12 planar.
 */
cudaError_t cudaRGBToYV12( uchar3* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an RGB float3 buffer into YUV YV12 planar.
 */
cudaError_t cudaRGBToYV12( float3* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an RGBA uchar4 buffer into YUV YV12 planar.
 */
cudaError_t cudaRGBAToYV12( uchar4* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an RGBA float4 buffer into YUV YV12 planar.
 */
cudaError_t cudaRGBAToYV12( float4* input, void* output, size_t width, size_t height, cudaStream_t stream = NULL );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV YUYV 4:2:2 packed to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a YUYV 422 packed image into RGB uchar3.
 */
cudaError_t cudaYUYVToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a YUYV 422 packed image into RGB float3.
 */
cudaError_t cudaYUYVToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a YUYV 422 packed image into RGBA uchar4.
 */
cudaError_t cudaYUYVToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a YUYV 422 packed image into RGBA float4.
 */
cudaError_t cudaYUYVToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV YVYU 4:2:2 packed to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a YVYU 422 packed image into RGB uchar3.
 */
cudaError_t cudaYVYUToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a YVYU 422 packed image into RGB float3.
 */
cudaError_t cudaYVYUToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a YVYU 422 packed image into RGBA uchar4.
 */
cudaError_t cudaYVYUToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a YVYU 422 packed image into RGBA float4.
 */
cudaError_t cudaYVYUToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV UYVY 4:2:2 packed to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a UYVY 422 packed image into RGB uchar3.
 */
cudaError_t cudaUYVYToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVY 422 packed image into RGB float3.
 */
cudaError_t cudaUYVYToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVY 422 packed image into RGBA uchar4.
 */
cudaError_t cudaUYVYToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVY 422 packed image into RGBA float4.
 */
cudaError_t cudaUYVYToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVA 4224 packed image into RGB uchar3.
 */
cudaError_t cudaUYVAToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVA 4224 packed image into RGB float3.
 */
cudaError_t cudaUYVAToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVA 4224 packed image into RGBA uchar4.
 */
cudaError_t cudaUYVAToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert a UYVA 4224 packed image into RGBA float4.
 */
cudaError_t cudaUYVAToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL );

///@}


// GRAY/RGB/RGBA/BGR/BGRA to UYVA 4:2:2:4.
cudaError_t cudaConvertToUYVA( uint8_t* input, uint8_t* output, size_t width, size_t height, float in_max = 255.0f, float out_max = 255.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToUYVA( float* input, uint8_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 255.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToUYVA( uchar3* input, uint8_t* output, size_t width, size_t height, float in_max = 255.0f, float out_max = 255.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToUYVA( float3* input, uint8_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 255.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToUYVA( uchar4* input, uint8_t* output, size_t width, size_t height, float in_max = 255.0f, float out_max = 255.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToUYVA( float4* input, uint8_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 255.0f, bool is_BGR = false, cudaStream_t stream = NULL );
// GRAY to UYVA 4:2:2:4.
// cudaError_t cudaGRAY32FToUYVA( float* input, uint8_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 255.0f, cudaStream_t stream = NULL );
cudaError_t cudaConvertToUYVA( void* input,  uint8_t* output, size_t width, size_t height, imageFormat format, float in_max = 1.0f, float out_max = 255.0f, cudaStream_t stream = NULL );

// GRAY/RGB/RGBA/BGR/BGRA to YUV+A PA16 4:2:2:4.
cudaError_t cudaConvertToPA16( uint8_t* input, uint16_t* output, size_t width, size_t height, float in_max = 255.0f, float out_max = 65535.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToPA16( float* input, uint16_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 65535.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToPA16( uchar3* input, uint16_t* output, size_t width, size_t height, float in_max = 255.0f, float out_max = 65535.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToPA16( float3* input, uint16_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 65535.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToPA16( uchar4* input, uint16_t* output, size_t width, size_t height, float in_max = 255.0f, float out_max = 65535.0f, bool is_BGR = false, cudaStream_t stream = NULL );
cudaError_t cudaConvertToPA16( float4* input, uint16_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 65535.0f, bool is_BGR = false, cudaStream_t stream = NULL );
// GRAY to YUV+A PA16 4:2:2:4.
// cudaError_t cudaGRAY32FToPA16( float* input, uint16_t* output, size_t width, size_t height, float in_max = 1.0f, float out_max = 65535.0f, cudaStream_t stream = NULL );
cudaError_t cudaConvertToPA16( void* input,  uint16_t* output, size_t width, size_t height, imageFormat format, float in_max = 1.0f, float out_max = 65535.0f, cudaStream_t stream = NULL );


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV NV12 4:2:0 to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert an NV12 texture (semi-planar 4:2:0) to RGB uchar3 format.
 * NV12 = 8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling.
 */
cudaError_t cudaNV12ToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an NV12 texture (semi-planar 4:2:0) to RGB float3 format.
 * NV12 = 8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling.
 */
cudaError_t cudaNV12ToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an NV12 texture (semi-planar 4:2:0) to RGBA uchar4 format.
 * NV12 = 8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling.
 */
cudaError_t cudaNV12ToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL );

/**
 * Convert an NV12 texture (semi-planar 4:2:0) to RGBA float4 format.
 * NV12 = 8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling.
 */
cudaError_t cudaNV12ToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL );

// P010_10LE. like a 10bit version NV12.
cudaError_t cudaP010_10LEToRGB( void* input, uchar3* output, size_t width, size_t height, cudaStream_t stream = NULL );
cudaError_t cudaP010_10LEToRGB( void* input, float3* output, size_t width, size_t height, cudaStream_t stream = NULL );
cudaError_t cudaP010_10LEToRGBA( void* input, uchar4* output, size_t width, size_t height, cudaStream_t stream = NULL );
cudaError_t cudaP010_10LEToRGBA( void* input, float4* output, size_t width, size_t height, cudaStream_t stream = NULL );

///@}

#endif

