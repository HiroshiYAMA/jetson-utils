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

#ifndef __VODEO_DECODER_H__
#define __VODEO_DECODER_H__

#include "videoSource.h"

#include "logging.h"
#define LOG_VIDEO_DECODER "[video decoder] "

#include <opencv2/opencv.hpp>



class videoDecoder : public videoSource
{
public:
	/**
	 * Create a decoder from the provided video options.
	 */
	static videoDecoder* Create( const videoOptions& options );

	/**
	 * Create a decoder instance from from a path and optional videoOptions.
	 */
	static videoDecoder* Create( const URI& resource, videoOptions::Codec codec );

	/**
	 * Destructor
	 */
	~videoDecoder();

	/**
	 * Capture the next decoded frame.
	 * @see videoSource::Capture()
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }

	/**
	 * Capture the next decoded frame.
	 * @see videoSource::Capture()
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Open the stream.
	 * @see videoSource::Open()
	 */
	virtual bool Open();

	/**
	 * Close the stream.
	 * @see videoSource::Close()
	 */
	virtual void Close();

	/**
	 * Return true if End Of Stream (EOS) has been reached.
	 * In the context of videoDecoder, EOS means that playback
	 * has reached the end of the file, and looping is either
	 * disabled or all loops have already been run.  In the case
	 * of RTP/RTSP, it means that the stream has terminated.
	 */
	inline bool IsEOS() const				{ return mEOS; }
	//
	void GoTo1stFrame()
	{
		LogInfo(LOG_VIDEO_DECODER "videoDecoder -- Go to 1st frame.\n");
		mEOS = true;
		mStreaming = true;
	}

	virtual void Pause() {}
	virtual void Start() {}
	virtual void NextFrame() {}

	/**
	 * Return the interface type (videoDecoder::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of videoDecoder class.
	 */
	static const uint32_t Type = (1 << 6);

	/**
	 * String array of supported video file extensions, terminated
	 * with a NULL sentinel value.  The supported extension are:
	 *
	 *    - MKV
	 *    - MP4 / QT
	 *    - AVI
	 *    - FLV
	 *
	 * @see IsSupportedExtension() to check a string against this list.
	 */
	static const char* SupportedExtensions[];

	/**
	 * Return true if the extension is in the list of SupportedExtensions.
	 * @param ext string containing the extension to be checked (should not contain leading dot)
	 * @see SupportedExtensions for the list of supported video file extensions.
	 */
	static bool IsSupportedExtension( const char* ext );

protected:
	videoDecoder( const videoOptions& options );

	bool buildLaunchStr();

	bool init();

	inline bool isLooping() const { return (mOptions.loop < 0) || ((mOptions.loop > 0) && (mLoopCount < mOptions.loop)); }

	std::string	mLaunchStr;
	bool		mEOS;
	size_t		mLoopCount;
	size_t		mFrameCount;
	size_t		mFrameCount_Max;
	imageFormat	mFormatYUV;

	std::unique_ptr<cv::VideoCapture> video_ptr;
	cv::Mat vidoe_img;
	uint8_t *video_buf_NV12;
	uchar4 *video_buf_RGBA;
};

#endif
