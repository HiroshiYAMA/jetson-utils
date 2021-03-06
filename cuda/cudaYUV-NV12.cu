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

#include "cudaYUV.h"
#include "cudaVector.h"

#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define FIXED_DECIMAL_POINT             24
#define FIXED_POINT_MULTIPLIER          1.0f
#define FIXED_COLOR_COMPONENT_MASK      0xffffffff


//-----------------------------------------------------------------------------------
// YUV to RGB colorspace conversion
//-----------------------------------------------------------------------------------
static inline __device__ float clamp( float x )	{ return fminf(fmaxf(x, 0.0f), 255.0f); }

// YUV2RGB
template<typename T>
static inline __device__ T YUV2RGB(const uint3& yuvi)
{
	const float luma = yuvi.x;
	const float u    = yuvi.y - 128.0f;
	const float v    = yuvi.z - 128.0f;
	constexpr float s = 1.0f / 256.0f * 255.0f;	// TODO clamp for uchar output?

#if 1
	return make_vec<T>(clamp((luma + 1.402f * v) * s),
				    clamp((luma - 0.344f * u - 0.714f * v) * s),
				    clamp((luma + 1.772f * u) * s), 255);
#else
	return make_vec<T>(clamp((luma + 1.140f * v) * s),
				    clamp((luma - 0.395f * u - 0.581f * v) * s),
				    clamp((luma + 2.032f * u) * s), 255);
#endif
}


//-----------------------------------------------------------------------------------
// NV12 to RGB
//-----------------------------------------------------------------------------------
template<typename T>
__global__ void NV12ToRGB(uint32_t* srcImage, size_t nSourcePitch,
                          T* dstImage,        size_t nDestPitch,
                          uint32_t width,     uint32_t height)
{
	int x, y;
	int x_even;
	uint32_t processingPitch;
	uint8_t *srcImageU8 = (uint8_t *)srcImage;
	uint3 yuv10;

	processingPitch = nSourcePitch;

	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y *  blockDim.y       +  threadIdx.y;
	x_even = x & ~1;

	if( x >= width )
		return; //x = width - 1;

	if( y >= height )
		return; // y = height - 1;

	yuv10.x = srcImageU8[y * processingPitch + x    ];

	uint32_t chromaOffset    = processingPitch * height;
	int y_chroma = y >> 1;

	if (y & 1)  // odd scanline ?
	{
		uint32_t chromaCb;
		uint32_t chromaCr;

		chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x_even    ];
		chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x_even + 1];

		if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
		{
			chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x_even    ] + 1) >> 1;
			chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x_even + 1] + 1) >> 1;
		}

		yuv10.y = chromaCb;
		yuv10.z = chromaCr;
	}
	else
	{
		yuv10.y = srcImageU8[chromaOffset + y_chroma * processingPitch + x_even    ];
		yuv10.z = srcImageU8[chromaOffset + y_chroma * processingPitch + x_even + 1];
	}

	// YUV to RGB transformation conversion
	dstImage[y * width + x]     = YUV2RGB<T>(yuv10);
}


template<typename T> 
static cudaError_t launchNV12ToRGB( void* srcDev, T* dstDev, size_t width, size_t height, cudaStream_t stream )
{
	if( !srcDev || !dstDev )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const size_t srcPitch = width * sizeof(uint8_t);
	const size_t dstPitch = width * sizeof(T);
	
#ifdef JETSON
	const dim3 blockDim(32, 8, 1);
#else
	const dim3 blockDim(64, 8, 1);
#endif
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y), 1);

	NV12ToRGB<T><<<gridDim, blockDim, 0, stream>>>( (uint32_t*)srcDev, srcPitch, dstDev, dstPitch, width, height );
	
	return CUDA(cudaGetLastError());
}

// cudaNV12ToRGB (uchar3)
cudaError_t cudaNV12ToRGB( void* srcDev, uchar3* destDev, size_t width, size_t height, cudaStream_t stream )
{
	return launchNV12ToRGB<uchar3>(srcDev, destDev, width, height, stream);
}

// cudaNV12ToRGB (float3)
cudaError_t cudaNV12ToRGB( void* srcDev, float3* destDev, size_t width, size_t height, cudaStream_t stream )
{
	return launchNV12ToRGB<float3>(srcDev, destDev, width, height, stream);
}

// cudaNV12ToRGBA (uchar4)
cudaError_t cudaNV12ToRGBA( void* srcDev, uchar4* destDev, size_t width, size_t height, cudaStream_t stream )
{
	return launchNV12ToRGB<uchar4>(srcDev, destDev, width, height, stream);
}

// cudaNV12ToRGBA (float4)
cudaError_t cudaNV12ToRGBA( void* srcDev, float4* destDev, size_t width, size_t height, cudaStream_t stream )
{
	return launchNV12ToRGB<float4>(srcDev, destDev, width, height, stream);
}


#if 0
// cudaNV12SetupColorspace
cudaError_t cudaNV12SetupColorspace( float hue )
{
	const float hueSin = sin(hue);
	const float hueCos = cos(hue);

	float hueCSC[9];

	const bool itu601 = false;

	if( itu601 /*CSC == ITU601*/)
	{
		//CCIR 601
		hueCSC[0] = 1.1644f;
		hueCSC[1] = hueSin * 1.5960f;
		hueCSC[2] = hueCos * 1.5960f;
		hueCSC[3] = 1.1644f;
		hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
		hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
		hueCSC[6] = 1.1644f;
		hueCSC[7] = hueCos *  2.0172f;
		hueCSC[8] = hueSin * -2.0172f;
	}
	else /*if(CSC == ITU709)*/
	{
		//CCIR 709
		hueCSC[0] = 1.0f;
		hueCSC[1] = hueSin * 1.57480f;
		hueCSC[2] = hueCos * 1.57480f;
		hueCSC[3] = 1.0;
		hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
		hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);
		hueCSC[6] = 1.0f;
		hueCSC[7] = hueCos *  1.85560f;
		hueCSC[8] = hueSin * -1.85560f;
	}


	if( CUDA_FAILED(cudaMemcpyToSymbol(constHueColorSpaceMat, hueCSC, sizeof(float) * 9)) )
		return cudaErrorInvalidSymbol;

	uint32_t cudaAlpha = ((uint32_t)0xff<< 24);

	if( CUDA_FAILED(cudaMemcpyToSymbol(constAlpha, &cudaAlpha, sizeof(uint32_t))) )
		return cudaErrorInvalidSymbol;

	nv12ColorspaceSetup = true;
	return cudaSuccess;
}
#endif

