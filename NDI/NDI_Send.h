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
 
#ifndef __NDI_SEND_H_
#define __NDI_SEND_H_


#include "videoOutput.h"

#include <Processing.NDI.Lib.h>
// #include <Processing.NDI.Advanced.h>

class ndiSend : public videoOutput
{
public:
	/**
	 * Create an encoder from the provided video options.
	 */
	static ndiSend* Create( const videoOptions& options );

	/**
	 * Destructor
	 */
	~ndiSend();

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

	// /**
	//  * Open the stream.
	//  * @see videoOutput::Open()
	//  */
	// virtual bool Open();

	// /**
	//  * Close the stream.
	//  * @see videoOutput::Close()
	//  */
	// virtual void Close();

	/**
	 * Return the interface type (ndiSend::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of ndiSend class.
	 */
	static const uint32_t Type = (1 << 11);

protected:
	ndiSend( const videoOptions& options );

	NDIlib_send_create_t NDI_send_create_desc;
	NDIlib_send_instance_t pNDI_send;
	NDIlib_video_frame_v2_t NDI_video_frame;

	static constexpr auto IMG_NUM = 2;
	int idx_front = 0;
	int idx_back = (idx_front + 1) & 1;
	uint16_t* img[IMG_NUM] = { NULL };	// uint16_t. PA16.
};

#endif
