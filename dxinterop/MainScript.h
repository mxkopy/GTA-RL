#include "framework.h"

static std::random_device RD;
static std::mt19937 Gen(RD());
static std::uniform_real_distribution<float> UniformRandom(0.0f, 1.0f);



static void ClearTraffic()
{
	GAMEPLAY::CLEAR_AREA_OF_VEHICLES(0, 0, 0, 10000.0f, false, false, false, false, false);
	//GAMEPLAY::CLEAR_AREA_OF_COPS(0, 0, 0, 10000.0f, false);
	//GAMEPLAY::CLEAR_AREA_OF_PEDS(0, 0, 0, 10000.0f, false);
}

static void ClearWanted(Player Player)
{
	if (PLAYER::GET_PLAYER_WANTED_LEVEL(Player) > 0)
	{
		PLAYER::SET_PLAYER_WANTED_LEVEL(Player, 0, false);
		PLAYER::SET_PLAYER_WANTED_LEVEL_NOW(Player, false);
	}
}

static void CenterCamera()
{
	CAM::SET_GAMEPLAY_CAM_RELATIVE_HEADING(0.0f);
	CAM::SET_GAMEPLAY_CAM_RELATIVE_PITCH(0.0f, 0.0f);
}


const static Vector3 AIRPORT(-1161.462f, -2584.786f, 13.505f);
const static Vector3 HIGHWAY(-704.8778f, -2111.786f, 13.51563f);
const static Vector3 HIGHWAY_DIRECTION(-0.7894784f, -0.6133158f, 0.02382357f);

static void Reset(Player Player, Vehicle Vehicle)
{
	PLAYER::SET_EVERYONE_IGNORE_PLAYER(Player, true);
	PLAYER::SET_POLICE_IGNORE_PLAYER(Player, true);
	
	ENTITY::SET_ENTITY_COORDS(Vehicle, HIGHWAY.x, HIGHWAY.y, HIGHWAY.z, 1, 0, 0, 1);
	ENTITY::SET_ENTITY_HEADING(Vehicle, UniformRandom(Gen));

	VEHICLE::SET_VEHICLE_FIXED(Vehicle);
	VEHICLE::SET_VEHICLE_ENGINE_HEALTH(Vehicle, 1000.0f);

}

static void OnTick()
{
	static Player Player = PLAYER::PLAYER_ID();
	static Ped Ped = PLAYER::GET_PLAYER_PED(Player);
	static Vehicle Vehicle = PED::GET_VEHICLE_PED_IS_IN(Ped, false);

	if (Vehicle != 0)
	{
		bool Collided = ENTITY::HAS_ENTITY_COLLIDED_WITH_ANYTHING(Vehicle);

	}
}