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

inline static void ClearWanted()
{
	if (PLAYER::GET_PLAYER_WANTED_LEVEL(_PLAYER) > 0)
	{
		PLAYER::SET_PLAYER_WANTED_LEVEL(_PLAYER, 0, false);
		PLAYER::SET_PLAYER_WANTED_LEVEL_NOW(_PLAYER, false);
	}
}

inline static void CenterCamera()
{

	nativeInit(0x28B022A17B068A3A); // FORCE_BONNET_CAMERA_RELATIVE_HEADING_AND_PITCH
	nativePush(0);
	nativePush(0);
	nativeCall();
	CAM::SET_GAMEPLAY_CAM_RELATIVE_HEADING(0.0f);
	CAM::SET_GAMEPLAY_CAM_RELATIVE_PITCH(-10.0f, 1.0f);
}


// It's better to just not think about it
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

inline static void ResetPlayerDrivingPosition(Vector3 Position, float Heading)
{
	auto V = _VEHICLE;
	ENTITY::SET_ENTITY_COORDS(V, Position.x, Position.y, Position.z, false, false, false, true);
	ENTITY::SET_ENTITY_HEADING(V, Heading);
	VEHICLE::SET_VEHICLE_FIXED(V);
}

inline static void Reset()
{
	float Heading = 360 * UniformRandom(Gen);
	ResetPlayerDrivingPosition(HIGHWAY, Heading);
}

static Flags FLAGS;

void OnTick()
{
	static RequestLockedMemory<GameState, REQUEST_GAME_STATE> GameStateMemory("GameState");
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

		if (FLAGS.GetFlag(UNSTUCK)) FLAGS.SetFlag(REQUEST_GAME_STATE, true);
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
		OnTick();
		WAIT(0);
	}
}