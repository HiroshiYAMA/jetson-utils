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

#include "videoEncoder.h"

#include "filesystem.h"
#include "timespec.h"
#include "logging.h"

#include "cudaColorspace.h"
#include "cudaResize.h"
#include "cudaMappedMemory.h"
#include "imageFormat.h"

#include <sstream>
#include <string.h>
#include <strings.h>
#include <unistd.h>


// supported video file extensions
const char* videoEncoder::SupportedExtensions[] = { "mkv", "mp4", "qt",
										"flv", "avi", "h264",
										"h265", NULL };

bool videoEncoder::IsSupportedExtension( const char* ext )
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
videoEncoder::videoEncoder( const videoOptions& options ) : videoOutput(options)
{
	rec_buf_BGR = nullptr;
}


// destructor
videoEncoder::~videoEncoder()
{
	Close();
}


// Create
videoEncoder* videoEncoder::Create( const videoOptions& options )
{
	videoEncoder* enc = new videoEncoder(options);

	if( !enc )
		return NULL;

	if( !enc->init() )
	{
		LogError(LOG_VIDEO_ENCODER "videoEncoder -- failed to create encoder engine\n");
		return NULL;
	}

	return enc;
}


// Create
videoEncoder* videoEncoder::Create( const URI& resource, videoOptions::Codec codec )
{
	videoOptions opt;

	opt.resource = resource;
	opt.codec    = codec;
	opt.ioType   = videoOptions::OUTPUT;

	return Create(opt);
}


// init
bool videoEncoder::init()
{
	// check for default codec
	if( mOptions.codec == videoOptions::CODEC_UNKNOWN )
	{
		LogWarning(LOG_VIDEO_ENCODER "videoEncoder -- codec not specified, defaulting to H.264\n");
		mOptions.codec = videoOptions::CODEC_H264;
	}

	// check if default framerate is needed
	if( mOptions.frameRate <= 0 )
		mOptions.frameRate = 30;

	// build pipeline string
	if( !buildLaunchStr() )
	{
		LogError(LOG_VIDEO_ENCODER "videoEncoder -- failed to build pipeline string\n");
		return false;
	}

	// open.
	const int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
	rec.open(mLaunchStr, cv::CAP_GSTREAMER, fourcc, mOptions.frameRate, cv::Size(mOptions.width, mOptions.height));

	if( !cudaAllocMapped(&rec_buf_BGR, make_int2(mOptions.width, mOptions.height)) )
	{
		LogError(LOG_VIDEO_ENCODER "failed to allocate CUDA memory for rec_buf_BGR (%ux%u)\n", mOptions.width, mOptions.height);
		return false;
	}
	rec_img = cv::Mat(cv::Size(mOptions.width, mOptions.height), CV_8UC3, rec_buf_BGR);

	return true;
}


// buildLaunchStr
bool videoEncoder::buildLaunchStr()
{
	std::ostringstream ss;

	// setup appsrc input element
	ss << "appsrc ! ";

	// set default bitrate (if needed)
	if( mOptions.bitRate == 0 )
		mOptions.bitRate = 4000000;

	// determine the requested protocol to use
	const URI& uri = GetResource();

	ss << "video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=NV12 ! ";

	if( mOptions.codec == videoOptions::CODEC_H264 )
		// ss << "omxh264enc bitrate=" << mOptions.bitRate << " profile=high ! video/x-h264 !  ";	// TODO:  investigate quality-level setting
		ss << "nvvideoconvert ! nvv4l2h264enc bitrate=" << mOptions.bitRate << " profile=High ! video/x-h264 !  ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		// ss << "omxh265enc bitrate=" << mOptions.bitRate << " ! video/x-h265 ! ";
		ss << "nvvideoconvert ! nvv4l2h265enc bitrate=" << mOptions.bitRate << " ! video/x-h265 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		// ss << "omxvp8enc bitrate=" << mOptions.bitRate << " ! video/x-vp8 ! ";
		ss << "vp8enc bitrate=" << mOptions.bitRate << " ! video/x-vp8 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		// ss << "omxvp9enc bitrate=" << mOptions.bitRate << " ! video/x-vp9 ! ";
		ss << "vp9enc bitrate=" << mOptions.bitRate << " ! video/x-vp9 ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		// ss << "nvjpegenc ! image/jpeg ! ";
		ss << "avenc_mjpeg ! image/jpeg ! ";
	else
	{
		LogError(LOG_VIDEO_ENCODER "videoEncoder -- unsupported codec requested (%s)\n", videoOptions::CodecToStr(mOptions.codec));
		LogError(LOG_VIDEO_ENCODER "              supported encoder codecs are:\n");
		LogError(LOG_VIDEO_ENCODER "                 * h264\n");
		LogError(LOG_VIDEO_ENCODER "                 * h265\n");
		LogError(LOG_VIDEO_ENCODER "                 * vp8\n");
		LogError(LOG_VIDEO_ENCODER "                 * vp9\n");
		LogError(LOG_VIDEO_ENCODER "                 * mjpeg\n");
	}

	if( uri.protocol == "file" )
	{
		if( uri.extension == "mkv" )
		{
			ss << "matroskamux ! ";
		}
		else if( uri.extension == "flv" )
		{
			ss << "flvmux ! ";
		}
		else if( uri.extension == "avi" )
		{
			if( mOptions.codec == videoOptions::CODEC_H265 || mOptions.codec == videoOptions::CODEC_VP9 )
			{
				LogError(LOG_VIDEO_ENCODER "videoEncoder -- AVI format doesn't support codec %s\n", videoOptions::CodecToStr(mOptions.codec));
				LogError(LOG_VIDEO_ENCODER "              supported AVI codecs are:\n");
				LogError(LOG_VIDEO_ENCODER "                 * h264\n");
				LogError(LOG_VIDEO_ENCODER "                 * vp8\n");
				LogError(LOG_VIDEO_ENCODER "                 * mjpeg\n");

				return false;
			}

			ss << "avimux ! ";
		}
		else if( uri.extension == "mp4" || uri.extension == "qt" )
		{
			if( mOptions.codec == videoOptions::CODEC_H264 )
				ss << "h264parse ! ";
			else if( mOptions.codec == videoOptions::CODEC_H265 )
				ss << "h265parse ! ";

			ss << "qtmux ! ";
		}
		else if( uri.extension != "h264" && uri.extension != "h265" )
		{
			printf(LOG_VIDEO_ENCODER "videoEncoder -- unsupported video file extension (%s)\n", uri.extension.c_str());
			printf(LOG_VIDEO_ENCODER "              supported video extensions are:\n");
			printf(LOG_VIDEO_ENCODER "                 * mkv\n");
			printf(LOG_VIDEO_ENCODER "                 * mp4, qt\n");
			printf(LOG_VIDEO_ENCODER "                 * flv\n");
			printf(LOG_VIDEO_ENCODER "                 * avi\n");
			printf(LOG_VIDEO_ENCODER "                 * h264, h265\n");

			return false;
		}

		ss << "filesink location=" << uri.location;

		mOptions.deviceType = videoOptions::DEVICE_FILE;
	}

	mLaunchStr = ss.str();

	LogInfo(LOG_VIDEO_ENCODER "videoEncoder -- pipeline launch string:\n");
	LogInfo(LOG_VIDEO_ENCODER "%s\n", mLaunchStr.c_str());

	return true;
}


// encodeBGR
bool videoEncoder::encodeBGR()
{
	// confirm the stream is open
	if( !mStreaming )
	{
		if( !Open() )
			return false;
	}

	rec << rec_img;

	return true;
}


// Render
bool videoEncoder::Render( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	if( !image || width == 0 || height == 0 )
		return false;

	if( mOptions.width != width || mOptions.height != height )
	{
		if( mOptions.width != 0 || mOptions.height != 0 )
			LogWarning(LOG_VIDEO_ENCODER "videoEncoder::Render() -- warning, input dimensions (%ux%u) are different than expected (%ux%u)\n", width, height, mOptions.width, mOptions.height);

		mOptions.width  = width;
		mOptions.height = height;
	}

	// error checking / return
	bool enc_success = false;

	auto render_end = [&]() -> bool {
		return enc_success;
	};

	if( CUDA_FAILED(cudaConvertColor(image, format, rec_buf_BGR, IMAGE_BGR8, width, height, float2{0, 255}, mStream)) )
	{
		LogError(LOG_VIDEO_ENCODER "videoEncoder::Render() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_VIDEO_ENCODER "                        supported formats are:\n");
		LogError(LOG_VIDEO_ENCODER "                            * rgb8\n");
		LogError(LOG_VIDEO_ENCODER "                            * rgba8\n");
		LogError(LOG_VIDEO_ENCODER "                            * rgb32f\n");
		LogError(LOG_VIDEO_ENCODER "                            * rgba32f\n");

		enc_success = false;
		render_end();
	}

	// CUDA(cudaDeviceSynchronize());	// TODO replace with cudaStream?
	CUDA(cudaStreamSynchronize(mStream));

	// encode YUV buffer
	// enc_success = encodeYUV(nextYUV, i420Size);
	enc_success = encodeBGR();

	// render sub-streams
	render_end();
}


// Open
bool videoEncoder::Open()
{
	if( mStreaming )
		return true;

	mStreaming = true;
	return true;
}


// Close
void videoEncoder::Close()
{
	if( !mStreaming )
		return;

	mStreaming = false;
	CUDA_FREE_HOST(rec_buf_BGR);
	LogInfo(LOG_VIDEO_ENCODER "videoEncoder -- pipeline stopped\n");
}
