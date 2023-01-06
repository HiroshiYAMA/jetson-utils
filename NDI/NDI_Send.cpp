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

#include "NDI_Send.h"

#include "logging.h"
#define LOG_NDI_SEND "[NDI send] "

#include <strings.h>

#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaResize.h"


// constructor
ndiSend::ndiSend( const videoOptions& options ) : videoOutput(options)
{
	mStreaming = true;

	mOptions.deviceType = videoOptions::DEVICE_NDI;
}


// destructor
ndiSend::~ndiSend()
{
	CUDA_FREE_MAPPED(img_pre, false);
	for (int i = 0; i < IMG_NUM; i++) CUDA_FREE_MAPPED(img[i], true);

	// Because one buffer is in flight we need to make sure that there is no chance that we might free it before
	// NDI is done with it. You can ensure this either by sending another frame, or just by sending a frame with
	// a NULL pointer.
	NDIlib_send_send_video_async_v2(pNDI_send, NULL);

	// Destroy the NDI sender
	NDIlib_send_destroy(pNDI_send);
}


// Create
ndiSend* ndiSend::Create( const videoOptions& options )
{
	auto ndi_send = new ndiSend(options);
	if (!ndi_send) {
		LogError(LOG_NDI_SEND "Cannot create instance.\n");
		return nullptr;
	}

	auto &opt = ndi_send->mOptions;

	// Create an NDI source that is clocked to the video.
	ndi_send->NDI_send_create_desc.p_ndi_name = opt.resource.location.c_str();

	// We create the NDI sender
	ndi_send->pNDI_send = NDIlib_send_create(&(ndi_send->NDI_send_create_desc));
	if (!ndi_send->pNDI_send) {
		LogError(LOG_NDI_SEND "ERROR!! NDIlib_send_create.\n");
		delete ndi_send;
		return nullptr;
	}

	auto &frm = ndi_send->NDI_video_frame;
	frm.xres = opt.width;
	frm.yres = opt.height;
#ifdef NDI_SEND_PA16
	frm.FourCC = NDIlib_FourCC_type_PA16;
	frm.line_stride_in_bytes = opt.width * 1 * sizeof(uint16_t);
#else
	frm.FourCC = NDIlib_FourCC_type_RGBA;
	frm.line_stride_in_bytes = opt.width * 4;
	// frm.FourCC = NDIlib_FourCC_type_UYVA;
	// // frm.FourCC = NDIlib_FourCC_type_UYVY;
	// frm.line_stride_in_bytes = opt.width * 2;
#endif
	frm.frame_rate_N = opt.frameRateNum;
	frm.frame_rate_D = opt.frameRateDenom;
	frm.frame_format_type = NDIlib_frame_format_type_progressive;

	if( !cudaAllocMapped(&(ndi_send->img_pre), make_int2(frm.xres, frm.yres * 3), false) )	// PA16. 3 planes of Y, UV, Alpha.
	{
		LogError(LOG_NDI_SEND "failed to allocate CUDA memory for image(pre) (%ux%u)\n", frm.xres, frm.yres);
		return nullptr;
	}
	for (int i = 0; i < IMG_NUM; i++) {
		if( !cudaAllocMapped(&(ndi_send->img[i]), make_int2(frm.xres, frm.yres * 3), true) )	// PA16. 3 planes of Y, UV, Alpha.
		{
			LogError(LOG_NDI_SEND "failed to allocate CUDA memory for image[] (%ux%u)\n", frm.xres, frm.yres);
			return nullptr;
		}
	}

	return ndi_send;
}


// Render
bool ndiSend::Render( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	if( !image || width == 0 || height == 0 )
		return false;

	if( mOptions.width != width || mOptions.height != height )
	{
		if( mOptions.width != 0 || mOptions.height != 0 )
			LogWarning(LOG_NDI_SEND "ndiSend::Render() -- warning, input dimensions (%ux%u) are different than expected (%ux%u)\n", width, height, mOptions.width, mOptions.height);

		mOptions.width  = width;
		mOptions.height = height;
	}

	auto copy_img = [&](auto image) -> void {
#ifdef NDI_SEND_PA16
		// [0, 255] -> [0, 65535].
		// cuda RGBA8 -> PA16.
		cudaConvertToPA16(image, img_pre, width, height, 255.0f, 65535.0f);
		cudaMemcpyAsync(img[idx_back], img_pre, width * height * 3 * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
#else
		// [0, 255] -> [0, 255].
		// // cuda RGBA8 -> UYVA.
		// cudaConvertToUYVA(image, (uint8_t *)img_pre, width, height);
		// cudaMemcpyAsync(img[idx_back], img_pre, width * height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
		// cuda RGBA8 -> RGBA8.
		cudaMemcpyAsync(img[idx_back], image, width * height * 4 * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
#endif
	};

	switch (format) {
	case IMAGE_GRAY8:
		copy_img((uint8_t *)image);
		break;
	case IMAGE_GRAY32F:
		copy_img((float *)image);
		break;
	case IMAGE_RGB8: case IMAGE_BGR8:
		copy_img((uchar3 *)image);
		break;
	case IMAGE_RGB32F: case IMAGE_BGR32F:
		copy_img((float3 *)image);
		break;
	case IMAGE_RGBA8: case IMAGE_BGRA8:
		copy_img((uchar4 *)image);
		break;
	case IMAGE_RGBA32F: case IMAGE_BGRA32F:
		copy_img((float4 *)image);
		break;
	default:
		LogError(LOG_NDI_SEND "ndiSend::Render() -- invalid image format '%s'\n", imageFormatToStr(format));
		LogError(LOG_NDI_SEND "                     supported formats are:\n");
		LogError(LOG_NDI_SEND "                         * gray8\n");
		LogError(LOG_NDI_SEND "                         * gray32f\n");
		LogError(LOG_NDI_SEND "                         * rgb8, bgr8\n");
		LogError(LOG_NDI_SEND "                         * rgba8, bgra8\n");
		LogError(LOG_NDI_SEND "                         * rgb32f, bgr32f\n");
		LogError(LOG_NDI_SEND "                         * rgba32f, bgra32f\n");
	}

	// const bool substreams_success = videoOutput::Render(image, width, height, format);

	// CUDA(cudaStreamSynchronize(mStream));
	
	// We now submit the frame asynchronously. This means that this call will return immediately and the
	// API will "own" the memory location until there is a synchronozing event. A synchronouzing event is
	// one of : NDIlib_send_send_video_async, NDIlib_send_send_video, NDIlib_send_destroy
	NDI_video_frame.p_data = (uint8_t*)img[idx_front];
	NDIlib_send_send_video_async_v2(pNDI_send, &NDI_video_frame);

	idx_front = (idx_front + 1) & 1;
	idx_back = (idx_front + 1) & 1;

	// return substreams_success;
	return true;
}
