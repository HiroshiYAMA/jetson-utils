/*
 * Copyright (c) 2021, ほげ. All rights reserved.
 */

#include "devGamepad.h"
// #include "devInput.h"

#include "logging.h"


// constructor
GamepadDevice::GamepadDevice()
{
	auto num = SDL_NumJoysticks();
	LogInfo("Num of Joystick: %d\n", num);

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


// destructor
GamepadDevice::~GamepadDevice()
{
	SDL_GameControllerClose(Gamepad);
}


// Create
std::unique_ptr<GamepadDevice> GamepadDevice::Create( const char* name )
{
	auto gpad = std::make_unique<GamepadDevice>();

	auto name_str = SDL_GameControllerName(gpad->Gamepad);
	auto map_str = SDL_GameControllerMapping(gpad->Gamepad);

	if (name_str == NULL || map_str == NULL) {
		LogError("gamepad -- can't opened device\n");
		return std::move(nullptr);
	}

	LogSuccess("gamepad -- opened device %s:%s\n", name_str, map_str);

	// SDL_free(const_cast<char *>(name_str));
	// SDL_free(const_cast<char *>(map_str));

	return std::move(gpad);
}


// Poll
bool GamepadDevice::Poll( uint32_t timeout )
{
	SDL_GameControllerUpdate();

	return true;	
}
