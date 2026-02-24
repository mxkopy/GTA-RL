#pragma once
#include <random>
#include "framework.h"
#include "nativeCaller.h"
#include "natives.h"
#include "ipc.h"
#include "vertexbuffers.h"

static std::random_device RD;
static std::mt19937 Gen(RD());
static std::uniform_real_distribution<float> UniformRandom(0.0f, 1.0f);

const static Hash ENTITY_XF = 3003014393;
const static Vector3 AIRPORT = {.x = -1161.462f, .y = -2584.786f, .z = 13.505f };
const static Vector3 HIGHWAY = {.x = -704.8778f, .y = -2111.786,  .z = 13.51563f};


#define _PLAYER PLAYER::PLAYER_ID()
#define _PED PLAYER::GET_PLAYER_PED(_PLAYER)
#define _LAST_VEHICLE PED::GET_VEHICLE_PED_IS_IN(_PED, true)
#define _VEHICLE PED::GET_VEHICLE_PED_IS_IN(_PED, false)


inline static void ClearTraffic()
{
	Vector3 PlayerCoords = ENTITY::GET_ENTITY_COORDS(_PLAYER, true);
	GAMEPLAY::CLEAR_AREA_OF_VEHICLES(PlayerCoords.x, PlayerCoords.y, PlayerCoords.z, 10000.0f, false, false, false, false, false);
	//GAMEPLAY::CLEAR_AREA_OF_COPS(0, 0, 0, 10000.0f, false);
	//GAMEPLAY::CLEAR_AREA_OF_PEDS(0, 0, 0, 10000.0f, false);
}

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
	nativeInit(0x28B022A17B068A3A);
	nativePush(0);
	nativePush(0);
	nativeCall();
	//CAM::SET_GAMEPLAY_CAM_RELATIVE_HEADING(0);
	//CAM::SET_GAMEPLAY_CAM_RELATIVE_PITCH(0, 0);
	nativeInit(0x48608C3464F58AB4);
	nativePush(0);
	nativePush(0);
	nativePush(1);
	nativeCall();
}

// It's better to just not think about it
inline static void ResetPlayerDrivingPosition(Vector3 Position, float Heading)
{
	auto V = _VEHICLE;
	ENTITY::DELETE_ENTITY(&V);
	ENTITY::SET_ENTITY_COORDS(_PED, Position.x, Position.y, Position.z, false, false, false, true);
	STREAMING::REQUEST_MODEL(ENTITY_XF);
	while (!STREAMING::HAS_MODEL_LOADED(ENTITY_XF)) WAIT(0);
	PED::SET_PED_INTO_VEHICLE(_PED, VEHICLE::CREATE_VEHICLE(ENTITY_XF, Position.x, Position.y, Position.z, Heading, true, false), -1);
	STREAMING::SET_MODEL_AS_NO_LONGER_NEEDED(ENTITY_XF);
}

inline static void Reset()
{
	float Heading = UniformRandom(Gen);
	PLAYER::SET_EVERYONE_IGNORE_PLAYER(_PLAYER, true);
	PLAYER::SET_POLICE_IGNORE_PLAYER(_PLAYER, true);
	ResetPlayerDrivingPosition(HIGHWAY, Heading);
	ClearTraffic();
}

static Flags FLAGS;

void OnTick()
{
	static RequestLockedMemory<GameState, REQUEST_GAME_STATE> GameStateMemory("GameState");
	static GameState GameState{};

	if (_VEHICLE != NULL)
	{
		CenterCamera();

		bool Collided = ENTITY::HAS_ENTITY_COLLIDED_WITH_ANYTHING(_VEHICLE);
		auto Velocity = ENTITY::GET_ENTITY_VELOCITY(_VEHICLE);
		auto Forward = (Eigen::Map<Vector3f>) VSConstantBuffers::F;

		GameState.set_collided(Collided);
		GameState.mutable_camera_direction()->set_x(Forward[0]);
		GameState.mutable_camera_direction()->set_y(Forward[1]);
		GameState.mutable_camera_direction()->set_z(Forward[2]);
		GameState.mutable_velocity()->set_x(Velocity.x);
		GameState.mutable_velocity()->set_y(Velocity.y);
		GameState.mutable_velocity()->set_z(Velocity.z);
		GameStateMemory = GameState;

		if (Collided) Reset();
	}
	else
	{
		Reset();
	}
}


void ScriptMain()
{
	while (!VSConstantBuffers::Data.IsInitialized()) WAIT(0);
	FLAGS.SetFlag(BEGIN_TRAINING, true);
	while (true)
	{
		OnTick();
		WAIT(0);
	}
}