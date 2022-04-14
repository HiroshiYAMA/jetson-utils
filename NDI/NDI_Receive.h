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

#ifndef __NDI_RECEIVE_H__
#define __NDI_RECEIVE_H__

#include <string>

#include "Thread.h"
#include "Event.h"
#include "RingBuffer.h"

#include "videoSource.h"

#include <Processing.NDI.Lib.h>
// #include <Processing.NDI.Advanced.h>

extern bool signal_recieved_NDI_recv;

class ndiReceive : public videoSource, public Thread
{
public:
	/**
	 * Create an decoder from the provided video options.
	 */
	static ndiReceive* Create( const videoOptions& options );

	/**
	 * Destructor
	 */
	~ndiReceive();

	// /**
	//  * Open the stream.
	//  * @see videoSource::Open()
	//  */
	// virtual bool Open();

	// /**
	//  * Close the stream.
	//  * @see videoSource::Close()
	//  */
	// virtual void Close();

	/**
	 * Capture the next image frame from NDI.
	 * @see videoSource::Capture
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX ) { return Capture((void**)image, imageFormatFromType<T>(), timeout); }

	/**
	 * Capture the next image frame from NDI.
	 * @see videoSource::Capture
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX ) override;

	/**
	 * Return the interface type (ndiReceive::Type)
	 */
	virtual inline uint32_t GetType() const override{ return Type; }

	/**
	 * Unique type identifier of ndiReceive class.
	 */
	static const uint32_t Type = (1 << 10);

	void GoTo1stFrame() override {}
	virtual void Pause() override {}
	virtual void Start() override {}
	virtual void NextFrame() override {}

private:
	ndiReceive( const videoOptions& options );

	void Run() override;
	void checkBuffer();

	size_t mFrameCount;

	imageFormat mFormatIN;

	RingBuffer mBufferIN;
	RingBuffer mBufferOUT;

	Event mWaitEvent;

	NDIlib_recv_create_v3_t NDI_recv_create_desc;
	NDIlib_recv_instance_t pNDI_recv;
	NDIlib_video_frame_v2_t NDI_video_frame;
};

#endif
