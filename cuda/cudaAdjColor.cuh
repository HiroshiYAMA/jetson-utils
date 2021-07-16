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

#pragma once

#include "cudaAdjColor.h"



// convert RGB -> HSV(cylinder model).
inline __device__ float3 RGB2HSV( const float3 &rgb )
{
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;

    float max = r > g ? r : g;
    max = max > b ? max : b;
    float min = r < g ? r : g;
    min = min < b ? min : b;
    float h = max - min;
    if (h > 0.0f) {
        if (max == r) {
            h = (g - b) / h;
            if (h < 0.0f) {
                h += 6.0f;
            }
        } else if (max == g) {
            h = 2.0f + (b - r) / h;
        } else {
            h = 4.0f + (r - g) / h;
        }
    }
    h /= 6.0f;
    float s = (max - min);
    if (max != 0.0f)
        s /= max;
    float v = max;

    return float3{h, s, v};
}

// convert HSV(cylinder model) -> RGB.
inline __device__ float3 HSV2RGB( const float3 &hsv )
{
    float h = hsv.x;
    float s = hsv.y;
    float v = hsv.z;

    float r = v;
    float g = v;
    float b = v;
    if (s > 0.0f) {
        h *= 6.0f;
        int i = (int) h;
        float f = h - (float) i;
        switch (i) {
            default:
            case 0:
                g *= 1 - s * (1 - f);
                b *= 1 - s;
                break;
            case 1:
                r *= 1 - s * f;
                b *= 1 - s;
                break;
            case 2:
                r *= 1 - s;
                b *= 1 - s * (1 - f);
                break;
            case 3:
                r *= 1 - s;
                g *= 1 - s * f;
                break;
            case 4:
                r *= 1 - s * (1 - f);
                g *= 1 - s;
                break;
            case 5:
                g *= 1 - s;
                b *= 1 - s * f;
                break;
        }
    }

    return float3{r, g, b};
}

// apply saturation, luminance, contrast.
inline __device__ float3 applyColorAdjustment(const float3 &rgb_src, float sat, float gain, float contrast)
{
    float3 hsv = RGB2HSV(rgb_src);

    hsv.y *= sat;
    hsv.z *= gain;
    hsv.z -= 0.5f;
    hsv.z *= contrast;
    hsv.z += 0.5f;
    hsv = clamp(hsv, 0.0f, 1.0f);

    float3 rgb_dst = HSV2RGB(hsv);
    rgb_dst = clamp(rgb_dst, 0.0f, 1.0f);

    return rgb_dst;
}
