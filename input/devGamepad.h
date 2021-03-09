/*
 * Copyright (c) 2021, ほげ. All rights reserved.
 */

#ifndef __DEV_GAMEPAD_H__
#define __DEV_GAMEPAD_H__

#include <SDL2/SDL.h>

#include <iostream>
#include <memory>
#include <vector>
#include <string>

/**
 * Gamepad device
 * @ingroup input
 */
class GamepadDevice
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
	bool Poll( uint32_t timeout=0 );

	// Get Axis.
	int16_t GetAxis(SDL_GameControllerAxis axis) {
		return SDL_GameControllerGetAxis(Gamepad, axis);
	}
	int16_t GetAxis_Left_X() { return GetAxis(SDL_CONTROLLER_AXIS_LEFTX); }
	int16_t GetAxis_Left_Y() { return GetAxis(SDL_CONTROLLER_AXIS_LEFTY); }
	int16_t GetAxis_Right_X() { return GetAxis(SDL_CONTROLLER_AXIS_RIGHTX); }
	int16_t GetAxis_Right_Y() { return GetAxis(SDL_CONTROLLER_AXIS_RIGHTY); }
	int16_t GetAxis_Trigger_L() { return GetAxis(SDL_CONTROLLER_AXIS_TRIGGERLEFT); }
	int16_t GetAxis_Trigger_R() { return GetAxis(SDL_CONTROLLER_AXIS_TRIGGERRIGHT); }

	// Get Button.
	uint8_t GetButton(SDL_GameControllerButton button) {
		return SDL_GameControllerGetButton(Gamepad, button);
	}
	uint8_t GetButton_A() { return GetButton(SDL_CONTROLLER_BUTTON_A); }
	uint8_t GetButton_B() { return GetButton(SDL_CONTROLLER_BUTTON_B); }
	uint8_t GetButton_X() { return GetButton(SDL_CONTROLLER_BUTTON_X); }
	uint8_t GetButton_Y() { return GetButton(SDL_CONTROLLER_BUTTON_Y); }
	uint8_t GetButton_Back() { return GetButton(SDL_CONTROLLER_BUTTON_BACK); }
	uint8_t GetButton_Guide() { return GetButton(SDL_CONTROLLER_BUTTON_GUIDE); }
	uint8_t GetButton_Start() { return GetButton(SDL_CONTROLLER_BUTTON_START); }
	uint8_t GetButton_Stick_L() { return GetButton(SDL_CONTROLLER_BUTTON_LEFTSTICK); }
	uint8_t GetButton_Stick_R() { return GetButton(SDL_CONTROLLER_BUTTON_RIGHTSTICK); }
	uint8_t GetButton_Shoulder_L() { return GetButton(SDL_CONTROLLER_BUTTON_LEFTSHOULDER); }
	uint8_t GetButton_Shoulder_R() { return GetButton(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER); }
	uint8_t GetButton_Dpad_U() { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_UP); }
	uint8_t GetButton_Dpad_D() { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_DOWN); }
	uint8_t GetButton_Dpad_L() { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_LEFT); }
	uint8_t GetButton_Dpad_R() { return GetButton(SDL_CONTROLLER_BUTTON_DPAD_RIGHT); }


protected:
	SDL_GameController *Gamepad;
};

#endif
