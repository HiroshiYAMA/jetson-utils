/*
 * Copyright (c) 2021, ほげ. All rights reserved.
 */

#include "devGamepad.h"
// #include "devInput.h"

#include "logging.h"


// Open 1st device.
void GamepadDevice::Open1stDevice()
{
	auto num = SDL_NumJoysticks();
	LogInfo("Num of Joystick: %d\n", num);
	Gamepad = NULL;
	event = {};
	axis_motion = false;
	button_down = false;
	button_up   = false;

	for (int i = 0; i < num; i++) {
		if (SDL_IsGameController(i)) {
			Gamepad = SDL_GameControllerOpen(i);
			if (Gamepad == NULL) {
				LogError("Could not open gamecontroller %i: %s\n", i, SDL_GetError());
			}

			SDL_GameControllerEventState(SDL_ENABLE);
			break;	// first gamepad.
		}
	}
}

// constructor
GamepadDevice::GamepadDevice()
{
	Open1stDevice();
}


// destructor
GamepadDevice::~GamepadDevice()
{
	SDL_GameControllerClose(Gamepad);
}


// Create
std::unique_ptr<GamepadDevice> GamepadDevice::Create( const char* name )
{
	auto gpad = std::make_unique<GamepadDevice>();
	printf("======== Gamepad(%s) ========\n",
		SDL_GameControllerGetAttached(gpad->Gamepad) ? "Attached" : "***** Removed *****");

	auto name_str = SDL_GameControllerName(gpad->Gamepad);
	auto map_str = SDL_GameControllerMapping(gpad->Gamepad);

	if (name_str == NULL || map_str == NULL) {
		LogError("gamepad -- can't opened device\n");
		// return std::move(nullptr);
	} else {
		LogSuccess("gamepad -- opened device %s:%s\n", name_str, map_str);

		// SDL_free(const_cast<char *>(name_str));
		// SDL_free(const_cast<char *>(map_str));
	}

	return std::move(gpad);
}


// Poll
bool GamepadDevice::Poll( uint32_t timeout )
{
	axis_motion = false;
	button_down = false;
	button_up   = false;

	while (SDL_PollEvent(&event)) {
		switch (event.type) {
		case SDL_CONTROLLERAXISMOTION:
			axis_motion = true;
			break;
		case SDL_CONTROLLERBUTTONDOWN:
			button_down = true;
			break;
		case SDL_CONTROLLERBUTTONUP:
			button_up = true;
			break;
		case SDL_CONTROLLERDEVICEADDED:
			LogInfo("*** Gamepad Attached ***\n");
			Open1stDevice();
			break;
		case SDL_CONTROLLERDEVICEREMOVED:
			LogInfo("*** Gamepad Removed ***\n");
			// SDL_GameControllerClose(Gamepad);
			break;
		default:
			;
		}
		// LogInfo("[GamePad]: Event: %d\n", event.type);
    }

	return true;	
}
