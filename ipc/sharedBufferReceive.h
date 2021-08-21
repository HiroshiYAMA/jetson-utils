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

#include "SharedBuffer.h"
#include "videoSource.h"

#include "logging.h"
#define LOG_SHARED_BUFFER_RECEIVE "[shared buffer receive] "



class sharedBufferReceive : public videoSource
{
public:
	using SB_DATA_TYPE = SharedBuffer::em_IMAGE_INFO_DATA_TYPE;
	static imageFormat imageFormatFromDataType(SB_DATA_TYPE data_type, uint32_t channel);

	/**
	 * Create a receiver from the provided video options.
	 */
	static sharedBufferReceive* Create( const videoOptions& options );

	/**
	 * Create a receiver instance from from a resource and optional videoOptions.
	 */
	static sharedBufferReceive* Create( const char* resource, const videoOptions& options=videoOptions() );

	/**
	 * Destructor
	 */
	~sharedBufferReceive();

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
	 * In the context of sharedBufferReceive, EOS means that 
	 * the stream has terminated.
	 */
	inline bool IsEOS() const				{ return mEOS; }
	//
	virtual void GoTo1stFrame() {}
	virtual void Pause() {}
	virtual void Start() {}
	virtual void NextFrame() {}

	/**
	 * Return the interface type (sharedBufferReceive::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of sharedBufferReceive class.
	 */
	static const uint32_t Type = (1 << 8);

protected:
	sharedBufferReceive( const videoOptions& options );

	bool init();

	bool		mEOS;
	size_t		mFrameCount;
	imageFormat	mFormatSharedBuffer;
	size_t		mBufSizeReceive;

	st_SharedBuffer sb;

	void *sb_buf_receive;
	void *sb_buf_out;
};
