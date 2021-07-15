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

#ifndef __VIDEO_ENCODER_H__
#define __VIDEO_ENCODER_H__

#include "videoOutput.h"

#include "logging.h"
#define LOG_VIDEO_ENCODER "[video encoder] "

#include <opencv2/opencv.hpp>


class videoEncoder : public videoOutput
{
public:
	/**
	 * Create an encoder from the provided video options.
	 */
	static videoEncoder* Create( const videoOptions& options );

	/**
	 * Create an encoder instance from resource URI and codec.
	 */
	static videoEncoder* Create( const URI& resource, videoOptions::Codec codec );

	/**
	 * Destructor
	 */
	~videoEncoder();

	/**
	 * Encode the next frame.
	 * @see videoOutput::Render()
	 */
	template<typename T> bool Render( T* image, uint32_t width, uint32_t height )		{ return Render((void**)image, width, height, imageFormatFromType<T>()); }

	/**
	 * Encode the next frame.
	 * @see videoOutput::Render()
	 */
	virtual bool Render( void* image, uint32_t width, uint32_t height, imageFormat format );

	/**
	 * Open the stream.
	 * @see videoOutput::Open()
	 */
	virtual bool Open();

	/**
	 * Close the stream.
	 * @see videoOutput::Open()
	 */
	virtual void Close();

	/**
	 * Return the interface type (videoEncoder::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of videoEncoder class.
	 */
	static const uint32_t Type = (1 << 7);

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
	videoEncoder( const videoOptions& options );

	bool init();

	bool buildLaunchStr();

	bool encodeBGR();

	std::string  mCapsStr;
	std::string  mLaunchStr;
	std::string  mOutputPath;
	std::string  mOutputIP;
	uint16_t     mOutputPort;

	cv::VideoWriter rec;
	cv::Mat rec_img;
	uchar3 *rec_buf_BGR;
};


#endif
