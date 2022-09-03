/*
 * Copyright (c) 2022, edgecraft. All rights reserved.
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

#pragma once

#include "cudaUtility.h"

enum em_COLOR_ADJ_SPILL_MODE {
	COLOR_ADJ_SPILL_NONE,
	COLOR_ADJ_SPILL_GREEN,
	COLOR_ADJ_SPILL_BLUE
};

template<typename T>	// T: float3, float4, uchar3, uchar4.
inline __device__ T apply_spill_pix(T px, em_COLOR_ADJ_SPILL_MODE spill_mode)
{
	switch (spill_mode) {
	case COLOR_ADJ_SPILL_GREEN:
		{
			float r = px.x;
			float g = px.y;
			float b = px.z;
			float th = (r + b) / 2.0f;
			if (g > th) g = th;
			px.y = g;
		}
		break;
	case COLOR_ADJ_SPILL_BLUE:
		{
			float r = px.x;
			float g = px.y;
			float b = px.z;
			float th = (r + g) / 2.0f;
			if (b > th) b = th;
			px.z = b;
		}
		break;
	case COLOR_ADJ_SPILL_NONE:
	default:
		break;
	}

	return px;
}
