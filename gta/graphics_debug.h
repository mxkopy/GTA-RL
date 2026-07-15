#pragma once
#include "framework.h"
#include "vertex_buffers.h"


// Controls ingame debug rays with path tracing
// Converts screen-space coordinates to a direction and origin in world-space, casts a ray along that direction, and writes the result to a RayCast protobuf object stored in memory
struct Ray
{
    // Stores a Protobuf object in memory containing screen space coordinates, origin, world-space direction
    StructuredMemory<RayCast> Memory;
    RayCast Data{};

    Vector3 Collision;

    Ray(string Tagname) : Memory("Ray" + Tagname) 
    {}

    Vector3f ComputeDirection()
    {
        auto Axes = (const Matrix3f&)VSConstants::Axes;
        return Axes * Vector3f(Data.x(), Data.y(), 1);
    }

    // Casts a ray in world-space indicated by the screen-space coordinates stored by the StructuredMemory<RayCast> object
    void Cast()
    {
        Data = Memory;
        BOOL Hit;
        Vector3 Normal;
        Entity EntityHit;
        Vector3f V = 1000.0f * ComputeDirection();
        Eigen::Map<Vector3f> P = VSConstants::P;
        auto RaycastHandle = WORLDPROBE::_CAST_RAY_POINT_TO_POINT(P[0], P[1], P[2], P[0] + V[0], P[1] + V[1], P[2] + V[2], 511, NULL, 7);
        WORLDPROBE::_GET_RAYCAST_RESULT(RaycastHandle, &Hit, &Collision, &Normal, &EntityHit);
    }

    void Update()
    {
        Data.mutable_collision()->set_x(Collision.x);
        Data.mutable_collision()->set_y(Collision.y);
        Data.mutable_collision()->set_z(Collision.z);
        Memory = Data;
    }

    // Displays the rays ingame
    Vector3 Debug()
    {
        Eigen::Map<Vector3f> P = VSConstants::P;
        Eigen::Map<Vector3f> F = VSConstants::F;
        Eigen::Map<Vector3f> L = VSConstants::L;

        GRAPHICS::DRAW_LINE(
            P[0] + F[0],
            P[1] + F[1],
            P[2] + F[2],
            Collision.x,
            Collision.y,
            Collision.z,
            Data.r(),
            Data.g(),
            Data.b(),
            Data.a()
        );
    }

    static void UpdateAll()
    {
        static Ray Rays[] = {
            Ray("A")
        };
        for (auto& Ray : Rays)
        {
            Ray.Cast();
            Ray.Update();
            Ray.Debug();
        }
    }
};

