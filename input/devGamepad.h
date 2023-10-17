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

#ifndef __DEV_GAMEPAD_H__
#define __DEV_GAMEPAD_H__

#include <SDL2/SDL.h>

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "IdevGamepad.h"

/**
 * Gamepad device
 * @ingroup input
 */
class GamepadDevice : public IGamepadDevice
{
public:
	/**
	 * Create device
	 */
	static std::unique_ptr<GamepadDevice> Create( const char* device="hogehoge USB Gamepad" );

	// constructor
	GamepadDevice();

	/**
	 * Destructor
	 */
	virtual ~GamepadDevice();

	/**
	 * Poll the device for updates
	 */
	bool Poll( uint32_t timeout=0 ) override;

	// Open 1st device.
	void Open1stDevice();

	// Is Gamepad Attached.
	bool IsAttached() const override { return SDL_GameControllerGetAttached(Gamepad); }

	// Get Axis.
	int16_t GetAxis(SDL_GameControllerAxis axis) const {
		return SDL_GameControllerGetAxis(Gamepad, axis);
	}
	int16_t GetAxis_Left_X() const override { return GetAxis(SDL_CONTROLLER_AXIS_LEFTX); }
	int16_t GetAxis_Left_Y() const override { return GetAxis(SDL_CONTROLLER_AXIS_LEFTY); }
	int16_t GetAxis_Right_X() const override { return GetAxis(SDL_CONTROLLER_AXIS_RIGHTX); }
	int16_t GetAxis_Right_Y() const override { return GetAxis(SDL_CONTROLLER_AXIS_RIGHTY); }
	int16_t GetAxis_Trigger_L() const override { return GetAxis(SDL_CONTROLLER_AXIS_TRIGGERLEFT); }
	int16_t GetAxis_Trigger_R() const override { return GetAxis(SDL_CONTROLLER_AXIS_TRIGGERRIGHT); }

	// Is Axis Motion.
	bool IsAxisMotion() const override { return axis_motion; }

	// Get Button.
	uint8_t GetButton(SDL_GameControllerButton button) const {
		return SDL_GameControllerGetButton(Gamepad, button);
	}
	uint8_t GetButton_A() const override { return GetButton(SDL_CONTROLLER_BUTTON_A); }
	uint8_t GetButton_B() const override { return GetButton(SDL_CONTROLLER_BUTTON_B); }
	uint8_t GetButton_X() const override { return GetButton(SDL_CONTROLLER_BUTTON_X); }
	uint8_t GetButton_Y() const override { return GetButton(SDL_CONTROLLER_BUTTON_Y); }
	uint8_t GetButton_Back() const override { return GetButton(SDL_CONTROLLER_BUTTON_BACK); }
	uint8_t GetButton_Guide() const override { return GetButton(SDL_CONTROLLER_BUTTON_GUIDE); }
	uint8_t GetButton_Start() const override { return GetButton(SDL_CONTROLLER_BUTTON_START); }
	uint8_t GetButton_Stick_L() const override { return GetButton(SDL_CONTROLLER_BUTTON_LEFTSTICK); }
	uint8_t GetButton_Stick_R() const override { return GetButton(SDL_CONTROLLER_BUTTON_RIGHTSTICK); }
	uint8_t GetButton_Shoulder_L() const override { return GetButton(SDL_CONTROLLER_BUTTON_LEFTSHOULDER); }
	uint8_t GetButton_Shoulder_R() const override { return GetButton(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER); }
	uint8_t GetButton_Dpad_U() const override { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_UP); }
	uint8_t GetButton_Dpad_D() const override { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_DOWN); }
	uint8_t GetButton_Dpad_L() const override { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_LEFT); }
	uint8_t GetButton_Dpad_R() const override { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_RIGHT); }

	// Is Button Down/Up.
	bool IsButtonDown() const override { return button_down; }
	bool IsButtonUp() const override { return button_up; }


protected:
	SDL_GameController *Gamepad;

	SDL_Event event;
	bool axis_motion;
	bool button_down;
	bool button_up;
};

#endif
