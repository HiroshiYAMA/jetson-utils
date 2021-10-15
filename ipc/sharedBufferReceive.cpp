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

#include "sharedBufferReceive.h"
#include "cudaColorspace.h"
#include "cudaMappedMemory.h"

#include "filesystem.h"

#include <sstream>
#include <unistd.h>
#include <string.h>
#include <strings.h>



// constructor
sharedBufferReceive::sharedBufferReceive( const videoOptions& options ) : videoSource(options)
	, sb({
		.id = {
			.shm_header = "/" + options.resource.location + "_shm_header",
			.shm_body   = "/" + options.resource.location + "_shm_body",
			.sem        = "/" + options.resource.location + "_sem",
	    },
	    .img_info = nullptr,
		})
{
	mEOS        = false;
	mFrameCount = 0;
	mFormatSharedBuffer = IMAGE_GRAY8;
	mBufSizeReceive = 0;

	sb_buf_receive = nullptr;
	sb_buf_out = nullptr;

	mOptions.deviceType = videoOptions::DEVICE_SHAREDBUFFER;
}


// destructor
sharedBufferReceive::~sharedBufferReceive()
{
	mEOS = true;
	Close();
}


// Create
sharedBufferReceive* sharedBufferReceive::Create( const videoOptions& options )
{
	sharedBufferReceive* ptr = new sharedBufferReceive(options);

	if(!ptr) return nullptr;

	if(!ptr->init()) {
		LogError(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive -- failed to create receiver for %s\n", ptr->mOptions.resource.string.c_str());
		return nullptr;
	}

	return ptr;
}


// Create
sharedBufferReceive* sharedBufferReceive::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;

	opt.resource = resource;
	opt.ioType   = videoOptions::INPUT;

	return Create(opt);
}


// init
bool sharedBufferReceive::init()
{
	const URI& uri = GetResource();

	// first, check that the shared buffer exists
	if( uri.protocol == "sb" )
	{
		const std::string path_shm_header = "/dev/shm/" + std::string(sb.id.shm_header.c_str() + 1);
		const std::string path_shm_body = "/dev/shm/" + std::string(sb.id.shm_body.c_str() + 1);
		const std::string path_sem = "/dev/shm/sem." + std::string(sb.id.sem.c_str() + 1);
		if( !fileExists(path_shm_header) || !fileExists(path_shm_body) || !fileExists(path_sem) )
		{
			LogError(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive -- couldn't find files '%s'\n",
				(path_shm_header + ", " + path_shm_body + " or " + path_sem).c_str());
			return false;
		}
	} else {
		LogError(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive -- isn't shared buffer '%s'\n", uri.protocol.c_str());
		return false;
	}

	LogInfo(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive -- creating receiver for %s\n", uri.location.c_str());

	// get shared buffer parameters.
    sb.img_info = SharedBuffer::Create(sb.id);
	if (sb.img_info == nullptr) return false;

    SharedBuffer::st_IMAGE_INFO_HEADER header;
    sb.img_info->ReceiveHeader(header);
	std::cout << header << std::endl;

	mFormatSharedBuffer = imageFormatFromDataType(header.data_type, header.channel);

	mOptions.width = header.width;
	mOptions.height = header.height;
    mBufSizeReceive = header.line_offset * header.height;

	return true;
}


imageFormat sharedBufferReceive::imageFormatFromDataType(SB_DATA_TYPE data_type, uint32_t channel) {
		switch (channel) {
		case 1:
			switch (data_type) {
			case SB_DATA_TYPE::UINT_TYPE_8:
			case SB_DATA_TYPE::INT_TYPE_8:          return IMAGE_GRAY8;
			case SB_DATA_TYPE::FLOAT_TYPE_BINARY32: return IMAGE_GRAY32F;
			}
			break;
		case 3:
			switch (data_type) {
			case SB_DATA_TYPE::UINT_TYPE_8:
			case SB_DATA_TYPE::INT_TYPE_8:          return IMAGE_RGB8;
			case SB_DATA_TYPE::FLOAT_TYPE_BINARY32: return IMAGE_RGB32F;
			}
			break;
		case 4:
			switch (data_type) {
			case SB_DATA_TYPE::UINT_TYPE_8:
			case SB_DATA_TYPE::INT_TYPE_8:          return IMAGE_RGBA8;
			case SB_DATA_TYPE::FLOAT_TYPE_BINARY32: return IMAGE_RGBA32F;
			}
			break;
		}

		// no support format.
		LogError(LOG_SHARED_BUFFER_RECEIVE "not support (%s).\n", SharedBuffer::imageInfoDataTypeToString(data_type).c_str());
		LogError(LOG_SHARED_BUFFER_RECEIVE "  support format: uint8_t, uchar3, uchar4, float, float3, float4.\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "  fall back format: uchar4.\n");
		return IMAGE_RGBA8;
	};


// Capture
bool sharedBufferReceive::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// confirm the stream is open
	if( !mStreaming || mEOS )
	{
		if( !Open() )
			return false;
	}

	if (sb.img_info == nullptr) return false;

	auto w = mOptions.width;
	auto h = mOptions.height;

	if (sb_buf_receive == nullptr) {
		if( !cudaAllocMapped(&sb_buf_receive, mBufSizeReceive) )
		{
			LogError(LOG_SHARED_BUFFER_RECEIVE "failed to allocate CUDA memory for sb_buf_receive (%ux%u)\n", w, h);
			return false;
		}
	}
	if (sb_buf_out == nullptr) {
		if( !cudaAllocMapped(&sb_buf_out, imageFormatSize(format, w, h), mOptions.zeroCopy) )
		{
			LogError(LOG_SHARED_BUFFER_RECEIVE "failed to allocate CUDA memory for sb_buf_out (%ux%u)\n", w, h);
			return false;
		}
	}

	// capture frame.
	if (!sb.img_info->ReceiveBuf(static_cast<uint8_t *>(sb_buf_receive), mBufSizeReceive)) {
		LogError(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive::Capture() -- couldn't receive\n");
		return false;
	}

	// ***32F: input range is normalized [0, 1].
	if( CUDA_FAILED(cudaConvertColor(sb_buf_receive, mFormatSharedBuffer, sb_buf_out, format, w, h, make_float2(0,1), mStream)) )
	{
		LogError(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive::Capture() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_SHARED_BUFFER_RECEIVE "                         supported formats are:\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "                             * gray8\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "                             * gray32f\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "                             * rgb8\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "                             * rgba8\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "                             * rgb32f\n");
		LogError(LOG_SHARED_BUFFER_RECEIVE "                             * rgba32f\n");
		return false;
	}

	*output = sb_buf_out;
	return true;
}


// Open
bool sharedBufferReceive::Open()
{
	if( mEOS )
	{
		LogWarning(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive -- end of stream (EOS) has been reached, stream has been closed\n");
		return false;
	}

	if( mStreaming )
		return true;

	mStreaming = true;
	return true;
}


// Close
void sharedBufferReceive::Close()
{
	if( !mStreaming && !mEOS )  // if EOS was set, the pipeline is actually open
		return;

	sb.img_info.reset();
	CUDA_FREE_HOST(sb_buf_receive);
	CUDA_FREE_MAPPED(sb_buf_out, mOptions.zeroCopy);
	mStreaming = false;
	LogInfo(LOG_SHARED_BUFFER_RECEIVE "sharedBufferReceive -- pipeline stopped\n");
}
