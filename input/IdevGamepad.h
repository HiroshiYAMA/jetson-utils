/*
 * Copyright (c) 2023, edgecraft. All rights reserved.
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

#include <cstdint>

// Interface class.
class IGamepadDevice
{
public:
	// Poll the device for updates.
	virtual bool Poll( uint32_t timeout = 0 ) = 0;

	// Is Gamepad Attached.
	virtual bool IsAttached() const = 0;

	// Get Axis.
	virtual int16_t GetAxis_Left_X() const = 0;
	virtual int16_t GetAxis_Left_Y() const = 0;
	virtual int16_t GetAxis_Right_X() const = 0;
	virtual int16_t GetAxis_Right_Y() const = 0;
	virtual int16_t GetAxis_Trigger_L() const = 0;
	virtual int16_t GetAxis_Trigger_R() const = 0;

	// Is Axis Motion.
	virtual bool IsAxisMotion() const = 0;

	// Get Button.
	virtual uint8_t GetButton_A() const = 0;
	virtual uint8_t GetButton_B() const = 0;
	virtual uint8_t GetButton_X() const = 0;
	virtual uint8_t GetButton_Y() const = 0;
	virtual uint8_t GetButton_Back() const = 0;
	virtual uint8_t GetButton_Guide() const = 0;
	virtual uint8_t GetButton_Start() const = 0;
	virtual uint8_t GetButton_Stick_L() const = 0;
	virtual uint8_t GetButton_Stick_R() const = 0;
	virtual uint8_t GetButton_Shoulder_L() const = 0;
	virtual uint8_t GetButton_Shoulder_R() const = 0;
	virtual uint8_t GetButton_Dpad_U() const = 0;
	virtual uint8_t GetButton_Dpad_D() const = 0;
	virtual uint8_t GetButton_Dpad_L() const = 0;
	virtual uint8_t GetButton_Dpad_R() const = 0;

	// Is Button Down/Up.
	virtual bool IsButtonDown() const = 0;
	virtual bool IsButtonUp() const = 0;
};
