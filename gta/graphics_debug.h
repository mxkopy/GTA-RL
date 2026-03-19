#pragma once
#include "framework.h"
#include "vertex_buffers.h"

struct Ray
{
    StructuredMemory<RayCast> Memory;
    RayCast Data{};

    Ray(float X, float Y, string Tagname) : Memory("Ray" + Tagname) 
    {
        Data.set_x(X);
        Data.set_y(Y);
    }

    Vector3f ComputeDirection()
    {
        auto Axes = (const Matrix3f&)VSConstants::Axes;
        return Axes * Vector3f(Data.x(), Data.y(), 1);
    }

    static Vector3 Cast(Eigen::Map<Vector3f> P, Vector3f V)
    {
        Vector3 Collision;
        BOOL Hit;
        Vector3 Normal;
        Entity EntityHit;
        auto RaycastHandle = WORLDPROBE::_CAST_RAY_POINT_TO_POINT(P[0], P[1], P[2], P[0] + V[0], P[1] + V[1], P[2] + V[2], 511, NULL, 7);
        WORLDPROBE::_GET_RAYCAST_RESULT(RaycastHandle, &Hit, &Collision, &Normal, &EntityHit);
        return Collision;
    }

    void ComputeCollision()
    {
        auto V = 1000.0f * ComputeDirection();
        auto P = (Eigen::Map<Vector3f>) VSConstants::P;
        Data.mutable_position()->set_x(P[0]);
        Data.mutable_position()->set_y(P[1]);
        Data.mutable_position()->set_z(P[2]);
        auto Collision = Cast(P, V);
        Data.mutable_collision() -> set_x(Collision.x);
        Data.mutable_collision() -> set_y(Collision.y);
        Data.mutable_collision() -> set_z(Collision.z);
        Data.set_nearclip(0.15);
        Data.set_farclip(10003.815);
        //Data.set_nearclip(CAM::_0xA03502FC581F7D9B());
        //Data.set_farclip(CAM::_0x9780F32BCAF72431());
        //Data.set_nearclip(CAM::_0xD0082607100D7193());
        //Data.set_farclip(CAM::_0xDFC8CBC606FDB0FC());
        Memory = Data;
    }

    static void Update()
    {
        static Ray Rays[] = {
            Ray(0.0, 0.0, "A"),
        };
        //DEBUG_ENTER;
        //BREAKPOINT;
        for (auto& Ray : Rays) Ray.ComputeCollision();
    }
};