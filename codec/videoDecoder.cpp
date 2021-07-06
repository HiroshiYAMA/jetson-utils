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

#include "videoDecoder.h"
#include "cudaColorspace.h"
#include "cudaMappedMemory.h"

#include "filesystem.h"

#include <sstream>
#include <unistd.h>
#include <string.h>
#include <strings.h>



// supported image file extensions
const char* videoDecoder::SupportedExtensions[] = { "mkv", "mp4", "qt",
										"flv", "avi", "h264",
										"h265", "mov", "webm", NULL };

bool videoDecoder::IsSupportedExtension( const char* ext )
{
	if( !ext )
		return false;

	uint32_t extCount = 0;

	while(true)
	{
		if( !SupportedExtensions[extCount] )
			break;

		if( strcasecmp(SupportedExtensions[extCount], ext) == 0 )
			return true;

		extCount++;
	}

	return false;
}


// constructor
videoDecoder::videoDecoder( const videoOptions& options ) : videoSource(options)
{
	mEOS        = false;
	mLoopCount  = 1;
	mFrameCount = 0;
	mFrameCount_Max = 0;
	mFormatYUV  = IMAGE_NV12;

	video_ptr = std::make_unique<cv::VideoCapture>();
	video_buf_NV12 = nullptr;
	video_buf_RGBA = nullptr;
}


// destructor
videoDecoder::~videoDecoder()
{
	Close();
}


// Create
videoDecoder* videoDecoder::Create( const videoOptions& options )
{
	videoDecoder* dec = new videoDecoder(options);

	if( !dec )
		return NULL;

	if( !dec->init() )
	{
		LogError(LOG_VIDEO_DECODER "videoDecoder -- failed to create decoder for %s\n", dec->mOptions.resource.string.c_str());
		return NULL;
	}

	return dec;
}


// Create
videoDecoder* videoDecoder::Create( const URI& resource, videoOptions::Codec codec )
{
	videoOptions opt;

	opt.resource = resource;
	opt.codec    = codec;
	opt.ioType   = videoOptions::INPUT;

	return Create(opt);
}


// init
bool videoDecoder::init()
{
	// first, check that the file exists
	if( mOptions.resource.protocol == "file" )
	{
		if( !fileExists(mOptions.resource.location) )
		{
			LogError(LOG_VIDEO_DECODER "videoDecoder -- couldn't find file '%s'\n", mOptions.resource.location.c_str());
			return false;
		}
	}

	LogInfo(LOG_VIDEO_DECODER "videoDecoder -- creating decoder for %s\n", mOptions.resource.location.c_str());

	// get file parameters.
	video_ptr->open(mOptions.resource.location, cv::CAP_FFMPEG);
	mFrameCount_Max = int32_t(video_ptr->get(cv::CAP_PROP_FRAME_COUNT));
	mOptions.width = int(video_ptr->get(cv::CAP_PROP_FRAME_WIDTH));
	mOptions.height = int(video_ptr->get(cv::CAP_PROP_FRAME_HEIGHT));
	mOptions.frameRate = float(video_ptr->get(cv::CAP_PROP_FPS));

	constexpr auto pixel_depth = 8;
	constexpr auto convert_rgb = false;	// NV12.
	video_ptr->set(cv::CAP_PROP_CONVERT_RGB, pixel_depth * 10.0 + static_cast<double>(convert_rgb));

	if( !cudaAllocMapped(&video_buf_NV12, make_int2(mOptions.width, mOptions.height * 3 / 2)) )
	{
		LogError(LOG_VIDEO_DECODER "failed to allocate CUDA memory for video_buf_NV12 (%ux%u)\n", mOptions.width, mOptions.height * 3 / 2);
		return false;
	}
	if( !cudaAllocMapped(&video_buf_RGBA, make_int2(mOptions.width, mOptions.height)) )
	{
		LogError(LOG_VIDEO_DECODER "failed to allocate CUDA memory for video_buf_RGBA (%ux%u)\n", mOptions.width, mOptions.height);
		return false;
	}

	// build pipeline string
	if( !buildLaunchStr() )
	{
		LogError(LOG_VIDEO_DECODER "videoDecoder -- failed to build pipeline string\n");
		return false;
	}

	// (re)open.
	video_ptr.reset();
	video_ptr = std::make_unique<cv::VideoCapture>(mLaunchStr, cv::CAP_GSTREAMER);

	return true;
}


// buildLaunchStr
bool videoDecoder::buildLaunchStr()
{
	std::ostringstream ss;

	// determine the requested protocol to use
	const URI& uri = GetResource();

	if( uri.protocol == "file" )
	{
		ss << "filesrc location=" << mOptions.resource.location << " ! ";

		if( uri.extension == "mkv" || uri.extension == "webm" )
			ss << "matroskademux ! ";
		else if( uri.extension == "mp4" || uri.extension == "qt" || uri.extension == "mov" )
			ss << "qtdemux ! ";
		else if( uri.extension == "flv" )
			ss << "flvdemux ! ";
		else if( uri.extension == "avi" )
			ss << "avidemux ! ";
		else if( uri.extension != "h264" && uri.extension != "h265" )
		{
			LogError(LOG_VIDEO_DECODER "videoDecoder -- unsupported video file extension (%s)\n", uri.extension.c_str());
			LogError(LOG_VIDEO_DECODER "              supported video extensions are:\n");
			LogError(LOG_VIDEO_DECODER "                 * mkv, webm\n");
			LogError(LOG_VIDEO_DECODER "                 * mp4, qt, mov\n");
			LogError(LOG_VIDEO_DECODER "                 * flv\n");
			LogError(LOG_VIDEO_DECODER "                 * avi\n");
			LogError(LOG_VIDEO_DECODER "                 * h264, h265\n");

			return false;
		}

		ss << "queue ! ";

		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "h265parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
			ss << "mpegvideoparse ! ";
		else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
			ss << "mpeg4videoparse ! ";

		mOptions.deviceType = videoOptions::DEVICE_FILE;
	}
	else
	{
		LogError(LOG_VIDEO_DECODER "videoDecoder -- unsupported protocol (%s)\n", uri.protocol.c_str());
		LogError(LOG_VIDEO_DECODER "              supported protocols are:\n");
		LogError(LOG_VIDEO_DECODER "                 * file://\n");

		return false;
	}

	if( mOptions.codec == videoOptions::CODEC_H264 )
		// ss << "omxh264dec ! ";
		ss << "nvv4l2decoder ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		// ss << "omxh265dec ! ";
		ss << "nvv4l2decoder ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		// ss << "omxvp8dec ! ";
		ss << "nvv4l2decoder ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		// ss << "omxvp9dec ! ";
		ss << "nvv4l2decoder ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
		// ss << "omxmpeg2videodec ! ";
		ss << "nvv4l2decoder ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
		// ss << "omxmpeg4videodec ! ";
		ss << "nvv4l2decoder ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		ss << "nvjpegdec ! ";
	// else if( mOptions.codec == videoOptions::CODEC_QTRLE )
	// 	ss << "avdec_qtrle ! videoconvert ! video/x-raw, format=(string)RGBA ! ";
	else
	{
		LogError(LOG_VIDEO_DECODER "videoDecoder -- unsupported codec requested (%s)\n", videoOptions::CodecToStr(mOptions.codec));
		LogError(LOG_VIDEO_DECODER "              supported decoder codecs are:\n");
		LogError(LOG_VIDEO_DECODER "                 * h264\n");
		LogError(LOG_VIDEO_DECODER "                 * h265\n");
		LogError(LOG_VIDEO_DECODER "                 * vp8\n");
		LogError(LOG_VIDEO_DECODER "                 * vp9\n");
		LogError(LOG_VIDEO_DECODER "                 * mpeg2\n");
		LogError(LOG_VIDEO_DECODER "                 * mpeg4\n");
		LogError(LOG_VIDEO_DECODER "                 * mjpeg\n");
		// LogError(LOG_VIDEO_DECODER "                 * qtrle\n");

		return false;
	}

	ss << "nvvideoconvert ! video/x-raw,format=NV12 ! ";

	// add the app sink
	ss << "appsink";
	// ss << "appsink sync=false";

	mLaunchStr = ss.str();

	LogInfo(LOG_VIDEO_DECODER "videoDecoder -- pipeline string:\n");
	LogInfo(LOG_VIDEO_DECODER "%s\n", mLaunchStr.c_str());

	return true;
}


// Capture
bool videoDecoder::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// check EOS.
	mEOS = video_ptr->get(cv::CAP_PROP_POS_FRAMES) >= (mFrameCount_Max - 1);

	// confirm the stream is open
	if( !mStreaming || mEOS )
	{
		if( !Open() )
			return false;
	}

	// capture frame.
	do { *video_ptr >> vidoe_img; } while (vidoe_img.empty());

	// // allocate ringbuffer for colorspace conversion
	const size_t yuvBufferSize = imageFormatSize(mFormatYUV, GetWidth(), GetHeight());
	const size_t rgbBufferSize = imageFormatSize(format, GetWidth(), GetHeight());

	cudaMemcpyAsync(video_buf_NV12, vidoe_img.data, yuvBufferSize, cudaMemcpyHostToDevice, mStream);
	if( CUDA_FAILED(cudaConvertColor(video_buf_NV12, mFormatYUV, video_buf_RGBA, format, GetWidth(), GetHeight(), make_float2(0,255), mStream)) )
	{
		LogError(LOG_VIDEO_DECODER "videoDecoder::Capture() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_VIDEO_DECODER "                         supported formats are:\n");
		LogError(LOG_VIDEO_DECODER "                             * rgb8\n");
		LogError(LOG_VIDEO_DECODER "                             * rgba8\n");
		LogError(LOG_VIDEO_DECODER "                             * rgb32f\n");
		LogError(LOG_VIDEO_DECODER "                             * rgba32f\n");

		return false;
	}

	*output = video_buf_RGBA;
	return true;
}


// Open
bool videoDecoder::Open()
{
	if( mEOS )
	{
		if( isLooping() )
		{
			video_ptr.reset();
			video_ptr = std::make_unique<cv::VideoCapture>(mLaunchStr, cv::CAP_GSTREAMER);

			LogWarning(LOG_VIDEO_DECODER "videoDecoder -- seeking stream to beginning (loop %zu of %i)\n", mLoopCount+1, mOptions.loop);

			mLoopCount++;
			mEOS = false;
		}
		else
		{
			LogWarning(LOG_VIDEO_DECODER "videoDecoder -- end of stream (EOS) has been reached, stream has been closed\n");
			return false;
		}
	}

	if( mStreaming )
		return true;

	mStreaming = true;
	return true;
}


// Close
void videoDecoder::Close()
{
	if( !mStreaming && !mEOS )  // if EOS was set, the pipeline is actually open
		return;

	video_ptr.reset();
	CUDA_FREE_HOST(video_buf_NV12);
	CUDA_FREE_HOST(video_buf_RGBA);
	mStreaming = false;
	LogInfo(LOG_VIDEO_DECODER "videoDecoder -- pipeline stopped\n");
}
