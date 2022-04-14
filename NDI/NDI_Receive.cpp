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

#include "NDI_Receive.h"

#include "cudaColorspace.h"

#include "logging.h"
#define LOG_NDI_RECV "[NDI receive] "

#include <strings.h>

bool signal_recieved_NDI_recv = false;

// constructor
ndiReceive::ndiReceive( const videoOptions& options ) : videoSource(options)
{
	mOptions.deviceType = videoOptions::DEVICE_NDI;

	mFrameCount = 0;
	mBufferOUT.SetThreaded(false);
}


// destructor
ndiReceive::~ndiReceive()
{
	// stop thread. NDI receive.
	this->StopThread();

	// Destroy the receiver
	NDIlib_recv_destroy(pNDI_recv);
}


// Run
void ndiReceive::Run()
{
	LogInfo(LOG_NDI_RECV "Start Thread. NDI receive.\n");
	while (!signal_recieved_NDI_recv && mThreadStarted) {
		checkBuffer();
	}
	LogInfo(LOG_NDI_RECV "Stop Thread. NDI receive.\n");
}

// Create
ndiReceive* ndiReceive::Create( const videoOptions& options )
{
	// create NDI receice instance
	auto ndi_recv = new ndiReceive(options);
	if (!ndi_recv) {
		LogError(LOG_NDI_RECV "Cannot create instance.\n");
		return nullptr;
	}

	auto &opt = ndi_recv->mOptions;

	// We create the NDI sender
	ndi_recv->NDI_recv_create_desc.bandwidth = NDIlib_recv_bandwidth_highest;
// #ifdef NDI_SEND_PA16
// 	ndi_recv->NDI_recv_create_desc.color_format = NDIlib_recv_color_format_best;	// to receive P216, PA16.
// #else
	ndi_recv->NDI_recv_create_desc.color_format = NDIlib_recv_color_format_RGBX_RGBA;
// #endif
	ndi_recv->pNDI_recv = NDIlib_recv_create_v3(&ndi_recv->NDI_recv_create_desc);
	if (!ndi_recv->pNDI_recv) {
		printf("pNDI_recv = NULL\n");
		delete ndi_recv;
		return nullptr;
	}

	// Connect to our sources
	NDIlib_source_t p_source;
	p_source.p_ndi_name = opt.resource.location.c_str();
	NDIlib_recv_connect(ndi_recv->pNDI_recv, &p_source);

	ndi_recv->mStreaming = true;

	// disable looping for cameras
	opt.loop = 0;

	// pre fetch video frame spec.
	auto &frm = ndi_recv->NDI_video_frame;
	LogInfo(LOG_NDI_RECV "pre fetch one video frame.\n");
	while (!signal_recieved_NDI_recv) {
		if (NDIlib_recv_capture_v2(ndi_recv->pNDI_recv, &frm, nullptr, nullptr, 1000) == NDIlib_frame_type_video) {
			// update options.
			opt.width = frm.xres;
			opt.height = frm.yres;
			opt.frameRateNum = frm.frame_rate_N;
			opt.frameRateDenom = frm.frame_rate_D;
			opt.frameRate = float(opt.frameRateNum) / opt.frameRateDenom;

			LogInfo(LOG_NDI_RECV "size = %d x %d, fps = %d/%d, [%4s]\n", opt.width, opt.height, opt.frameRateNum, opt.frameRateDenom, (char *)(&frm.FourCC));

			// Free the data 
			NDIlib_recv_free_video_v2(ndi_recv->pNDI_recv, &frm);

			break;
		}
	}

	// start thread. NDI receive.
	ndi_recv->StartThread();

	return ndi_recv;
}


// checkBuffer
void ndiReceive::checkBuffer()
{
	if( !pNDI_recv )
		return;

	void* nextBuffer = nullptr;

	// receice NDI.
	size_t NDIsize = 0;
	while (!signal_recieved_NDI_recv) {
		if (NDIlib_recv_capture_v2(pNDI_recv, &NDI_video_frame, nullptr, nullptr, 1000) == NDIlib_frame_type_video) {
			switch (NDI_video_frame.FourCC) {
			case NDIlib_FourCC_type_RGBX:
			case NDIlib_FourCC_type_RGBA:
				mFormatIN = imageFormat::IMAGE_RGBA8;
				NDIsize = NDI_video_frame.data_size_in_bytes * NDI_video_frame.yres;
				break;

			// case NDIlib_FourCC_video_type_P216:
			// case NDIlib_FourCC_video_type_PA16:
			default:
				LogError(LOG_NDI_RECV "not support FourCC\n");
				goto checkBuffer_end;
			}

			break;
		}
	}

	// make sure ringbuffer is allocated
	if( !mBufferIN.Alloc(mOptions.numBuffers, NDIsize, RingBuffer::ZeroCopy) )
	{
		LogError(LOG_NDI_RECV "ndiReceive -- failed to allocate %u buffers (%zu bytes each)\n", mOptions.numBuffers, NDIsize);
		goto checkBuffer_end;
	}

	// copy to next ringbuffer
	nextBuffer = mBufferIN.Peek(RingBuffer::Write);

	if( !nextBuffer )
	{
		LogError(LOG_NDI_RECV "ndiReceive -- failed to retrieve next ringbuffer for writing\n");
		goto checkBuffer_end;
	}

	memcpy(nextBuffer, NDI_video_frame.p_data, NDIsize);
	mBufferIN.Next(RingBuffer::Write);
	mFrameCount++;
	mWaitEvent.Wake();

checkBuffer_end:

	// Free the data 
	NDIlib_recv_free_video_v2(pNDI_recv, &NDI_video_frame);

	return;
}


// Capture
bool ndiReceive::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// confirm the camera is streaming
	if( !mStreaming )
		return false;

	// wait until a new frame is recieved
	if( !mWaitEvent.Wait(timeout) )
		return false;

	// get the latest ringbuffer
	void* latestIN = mBufferIN.Next(RingBuffer::ReadLatestOnce);

	if( !latestIN )
		return false;

	// allocate ringbuffer for colorspace conversion
	const size_t rgbBufferSize = imageFormatSize(format, GetWidth(), GetHeight());

	if( !mBufferOUT.Alloc(mOptions.numBuffers, rgbBufferSize, mOptions.zeroCopy ? RingBuffer::ZeroCopy : 0) )
	{
		LogError(LOG_NDI_RECV "ndiReceive -- failed to allocate %u buffers (%zu bytes each)\n", mOptions.numBuffers, rgbBufferSize);
		return false;
	}

	// perform colorspace conversion
	void* nextRGB = mBufferOUT.Next(RingBuffer::Write);

	if( CUDA_FAILED(cudaConvertColor(latestIN, mFormatIN, nextRGB, format, GetWidth(), GetHeight(), make_float2(0,255), mStream)) )
	{
		LogError(LOG_NDI_RECV "ndiReceive::Capture() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_NDI_RECV "                        supported formats are:\n");
		LogError(LOG_NDI_RECV "                            * rgb8\n");
		LogError(LOG_NDI_RECV "                            * rgba8\n");
		LogError(LOG_NDI_RECV "                            * rgb32f\n");
		LogError(LOG_NDI_RECV "                            * rgba32f\n");

		return false;
	}

	*output = nextRGB;
	return true;
}
