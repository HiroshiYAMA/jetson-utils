/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_FILTER_MODE_H__
#define __CUDA_FILTER_MODE_H__


#include "cudaUtility.h"


/**
 * Enumeration of interpolation filtering modes.
 * @see cudaFilterModeFromStr() and cudaFilterModeToStr()
 * @ingroup cudaFilter
 */
enum cudaFilterMode
{
	FILTER_POINT	= 0,	 /**< Nearest-neighbor sampling */
	FILTER_LINEAR	= 1,	 /**< Bilinear filtering */
    FILTER_CUBIC	= 2,	// Bi-Cubic.
    FILTER_AREA		= 3,	// Area.
    FILTER_LANCZOS4	= 4,	// Lanczos4.
    // FILTER_LINEAR_EXACT	= 5,	// dummy.
    FILTER_SPLINE36	= 6,	// Spline36.
    FILTER_MAX		= 7,	// dummy.
};

/**
 * Parse a cudaFilterMode enum from a string.
 * @returns The parsed cudaFilterMode, or default_value on error.
 * @ingroup cudaFilter
 */
cudaFilterMode cudaFilterModeFromStr( const char* filter, cudaFilterMode default_value=FILTER_LINEAR );

/**
 * Convert a cudaFilterMode enum to a string.
 * @ingroup cudaFilter
 */
const char* cudaFilterModeToStr( cudaFilterMode filter );


/**
 * Enumeration of image layout formats.
 * @ingroup cudaFilter
 */
enum cudaDataFormat
{
	FORMAT_HWC,	/**< Height * Width * Channels (packed format) */
	FORMAT_CHW,	/**< Channels * Width * Height (DNN format) */
	
	/**< Default format (HWC) */
	FORMAT_DEFAULT = FORMAT_HWC
};

						
#endif


