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


// constructor
ndiSend::ndiSend( const videoOptions& options ) : videoOutput(options)
{
	if (!NDIlib_initialize())
	{
		LogError(LOG_NDI_SEND "Cannot run NDI.");
		exit(EXIT_FAILURE);
	}

	mStreaming = true;

	mOptions.deviceType = videoOptions::DEVICE_NDI;
}


// destructor
ndiSend::~ndiSend()
{
	// Because one buffer is in flight we need to make sure that there is no chance that we might free it before
	// NDI is done with it. You can ensure this either by sending another frame, or just by sending a frame with
	// a NULL pointer.
	NDIlib_send_send_video_async_v2(pNDI_send, NULL);

	// Destroy the NDI sender
	NDIlib_send_destroy(pNDI_send);

	// Not required, but nice
	NDIlib_destroy();
}


// Create
ndiSend* ndiSend::Create( const videoOptions& options )
{
	auto ndi_send = new ndiSend(options);
	if (!ndi_send) {
		LogError(LOG_NDI_SEND "Cannot create instance.");
		return nullptr;
	}

	auto &opt = ndi_send->mOptions;

	// Create an NDI source that is clocked to the video.
	ndi_send->NDI_send_create_desc.p_ndi_name = opt.resource.location.c_str();

	// We create the NDI sender
	ndi_send->pNDI_send = NDIlib_send_create(&(ndi_send->NDI_send_create_desc));
	if (!ndi_send->pNDI_send) {
		LogError(LOG_NDI_SEND "ERROR!! NDIlib_send_create.");
		return nullptr;
	}

	auto &frm = ndi_send->NDI_video_frame;
	frm.xres = opt.width;
	frm.yres = opt.height;
	// frm.FourCC = NDIlib_FourCC_type_BGRA;
	// frm.FourCC = NDIlib_FourCC_type_RGBA;
	// frm.FourCC = NDIlib_FourCC_type_UYVA;
	frm.FourCC = NDIlib_FourCC_type_PA16;
	frm.line_stride_in_bytes = opt.width * 1 * sizeof(uint16_t);
	frm.frame_rate_N = opt.frameRateNum;
	frm.frame_rate_D = opt.frameRateDenom;
	frm.frame_format_type = NDIlib_frame_format_type_progressive;

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

	// const bool substreams_success = videoOutput::Render(image, width, height, format);

	// CUDA(cudaStreamSynchronize(mStream));
	
	// We now submit the frame asynchronously. This means that this call will return immediately and the
	// API will "own" the memory location until there is a synchronozing event. A synchronouzing event is
	// one of : NDIlib_send_send_video_async, NDIlib_send_send_video, NDIlib_send_destroy
	NDI_video_frame.p_data = (uint8_t*)image;
	NDIlib_send_send_video_async_v2(pNDI_send, &NDI_video_frame);

	// return substreams_success;
	return true;
}
