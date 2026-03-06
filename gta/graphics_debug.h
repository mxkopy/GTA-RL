#pragma once
#include "framework.h"
#include "vertex_buffers.h"

//float NearClip = CAM::_0xD0082607100D7193();
//float FarClip = CAM::_0xDFC8CBC606FDB0FC();

struct Ray
{
    UINT C, R;
    StructuredMemory<Vec3f> Memory;
    Vec3f Data{};

    Ray(UINT C, UINT R) : C(C), R(R), Memory("Ray" + std::to_string(C) + "_" + std::to_string(R)) {}

    Eigen::Vector3f ComputeDirection() const
    {
        float VW = VSConstants::VW, VH = VSConstants::VH, SW = VSConstants::SW, SH = VSConstants::SH;
        float X = 2 * (float(C) / VW) - (SW / VW);
        float Y = 2 * (float(R) / VH) - (SH / VH);
        float Z = 1;
        return (Matrix3f&)VSConstants::Axes * Eigen::Vector3f(X, Y, Z);
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
        auto Collision = Cast(VSConstants::P, V);
        Data.set_x(Collision.x);
        Data.set_y(Collision.y);
        Data.set_z(Collision.z);
        Memory = Data;
    }

    operator Vector3f () const
    {
        return Vector3f(Data.x(), Data.y(), Data.z());
    }

    static void Update()
    {
        float VW = VSConstants::VW, VH = VSConstants::VH, SW = VSConstants::SW, SH = VSConstants::SH;
        static Ray Rays[] = {
            Ray(SW / 4, SH / 4),
            Ray((SW / 4) + (SW / 2), SH / 4),
            Ray(SW / 4, (SH / 4) + (SH / 2)),
            Ray((SW / 4) + (SW / 2), (SH / 4) + (SH / 2))
        };
        for (auto& Ray : Rays) Ray.ComputeCollision();
    }
};