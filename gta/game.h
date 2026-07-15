#pragma once
#include <random>
#include "framework.h"
#include "nativeCaller.h"
#include "natives.h"
#include "ipc.h"
#include "vertex_buffers.h"
#include "input.h"

static std::random_device RD;
static std::mt19937 Gen(RD());
static std::uniform_real_distribution<float> UniformRandom(0.0f, 1.0f);

const static Hash ENTITY_XF = 3003014393;
const static Vector3 AIRPORT = {.x = -1161.462f, .y = -2584.786f, .z = 13.505f };
const static Vector3 HIGHWAY = {.x = -704.8778f, .y = -2111.786,  .z = 13.51563f};

// Clears wanted level
// Runs every tick
inline static void ClearWanted()
{
	if (PLAYER::GET_PLAYER_WANTED_LEVEL(_PLAYER) > 0)
	{
		PLAYER::SET_PLAYER_WANTED_LEVEL(_PLAYER, 0, false);
		PLAYER::SET_PLAYER_WANTED_LEVEL_NOW(_PLAYER, false);
	}
}

// Forces the camera to be aligned with the car and face the front
// Runs every tick
inline static void CenterCamera()
{

	nativeInit(0x28B022A17B068A3A); // FORCE_BONNET_CAMERA_RELATIVE_HEADING_AND_PITCH
	nativePush(0);
	nativePush(0);
	nativeCall();
	CAM::SET_GAMEPLAY_CAM_RELATIVE_HEADING(0.0f);
	CAM::SET_GAMEPLAY_CAM_RELATIVE_PITCH(-10.0f, 1.0f);
}

// Creates a car at a given position and sets the player into it
// There's a lot of idiosyncracies from the RAGE engine API in here, and it's better to just not think about it
// Runs once at the beginning of every training episode
inline static void InitializePlayerDrivingPosition(Vector3 Position)
{
	float Heading = 360 * UniformRandom(Gen);
	STREAMING::SET_VEHICLE_POPULATION_BUDGET(0);
	PLAYER::SET_EVERYONE_IGNORE_PLAYER(_PLAYER, true);
	PLAYER::SET_POLICE_IGNORE_PLAYER(_PLAYER, true);
	auto V = _VEHICLE;
	ENTITY::DELETE_ENTITY(&V);
	ENTITY::SET_ENTITY_COORDS(_PED, Position.x, Position.y, Position.z, false, false, false, true);
	STREAMING::REQUEST_MODEL(ENTITY_XF);
	while (!STREAMING::HAS_MODEL_LOADED(ENTITY_XF)) WAIT(0);
	PED::SET_PED_INTO_VEHICLE(_PED, VEHICLE::CREATE_VEHICLE(ENTITY_XF, Position.x, Position.y, Position.z, Heading, true, false), -1);
	//STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(ENTITY_XF);
}


// Sets the position and heading of the car the player is in
// Runs once at the beginning of every training episode
inline static void ResetPlayerDrivingPosition(Vector3 Position, float Heading)
{
	auto V = _VEHICLE;
	ENTITY::SET_ENTITY_COORDS(V, Position.x, Position.y, Position.z, false, false, false, true);
	ENTITY::SET_ENTITY_HEADING(V, Heading);
	WAIT(0);
	VEHICLE::SET_VEHICLE_FIXED(V);
}


// Initializes everything for a training episode (car position, orientation, etc.)
// Runs once at the beginning of every training episode
inline static void Reset()
{
	float Heading = 360 * UniformRandom(Gen);
	ResetPlayerDrivingPosition(HIGHWAY, Heading);
}

static Flags FLAGS;

// Code to run every tick (i.e. frame) of the game
// Checks to see if the python script is running, if so initializes the training episode, checks to see if the car crashed and needs to reset, reads and writes to synchronized memory, etc.
void OnTick()
{
	static RequestLockedMemory<GameState> GameStateMemory("GameState");
	static GameState GameState{};
	auto VEHICLE = _VEHICLE;

	if (VEHICLE == NULL || FLAGS.GetFlag(RESET))
	{
		Reset();
		FLAGS.SetFlag(RESET, false);
	}
	else if (VEHICLE != NULL)
	{
		//SendKeypress();
		CenterCamera();

		bool Collided = ENTITY::HAS_ENTITY_COLLIDED_WITH_ANYTHING(VEHICLE);
		auto Velocity = ENTITY::GET_ENTITY_VELOCITY(VEHICLE);
		auto Forward = ENTITY::GET_ENTITY_FORWARD_VECTOR(VEHICLE);
		auto Reward = (Forward.x * Velocity.x) + (Forward.y * Velocity.y) + (Forward.z * Velocity.z);

		GameState.set_reward(Reward);
		GameState.set_collided(Collided);

		if (FLAGS.GetFlag(UNSTUCK)) GameStateMemory.ReadFlag.Set();
		GameStateMemory = GameState;
	}
}


void ScriptMain()
{
	InitializePlayerDrivingPosition(HIGHWAY);
	while (!VSConstants::Data.IsInitialized()) WAIT(0);
	FLAGS.SetFlag(BEGIN_TRAINING, true);
	while (true)
	{
		Ray::UpdateAll();
		OnTick();
		WAIT(0);
	}
}