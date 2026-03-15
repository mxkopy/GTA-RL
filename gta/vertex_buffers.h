#pragma once
#include "framework.h"
#include "ipc.h"

// Vertex Shader Constant Buffers
// The vertex shader stage converts vertex information into pixel information 
// As such it contains useful camera parameters (world position, world direction, FOV, etc)
// This struct implements methods to retrieve this information via the DX11 API
struct VSConstants
{
    inline static StructuredMemory<VertexShaderConstants> VSMemory{ "VSConstants" };
    inline static VertexShaderConstants Data{};

    #define MAP_VSB_TO_EIGEN_TYPE(NAME, MAP_TYPE, VSB_INDEX, VSB_OFFSET) \
        inline static struct { \
            operator const Eigen::Map<MAP_TYPE> () const { \
                auto Floats = (float*) Data.constant_buffers().Get(VSB_INDEX).data(); \
                return Eigen::Map<MAP_TYPE>(Floats + VSB_OFFSET); \
            } \
        } NAME;

    #define MAP_VSB_OFFSET(NAME, VSB_INDEX, VSB_OFFSET) \
        inline static struct { \
            operator const float () const { \
                auto Floats = (float*) Data.constant_buffers().Get(VSB_INDEX).data(); \
                return Floats[VSB_OFFSET]; \
            } \
        } NAME;

    MAP_VSB_TO_EIGEN_TYPE(P, Vector3f, 2, 12); // Camera position
    MAP_VSB_TO_EIGEN_TYPE(F, Vector3f, 2, 16); // Camera forward vector
    MAP_VSB_TO_EIGEN_TYPE(R, Vector3f, 2, 28); // Right, left, down & up planes defining the frustrum 
    MAP_VSB_TO_EIGEN_TYPE(L, Vector3f, 2, 32); // 
    MAP_VSB_TO_EIGEN_TYPE(D, Vector3f, 2, 36); //
    MAP_VSB_TO_EIGEN_TYPE(U, Vector3f, 2, 40); //

    MAP_VSB_OFFSET(VW, 2, 20); // View width
    MAP_VSB_OFFSET(VH, 2, 21); // View height
    MAP_VSB_OFFSET(SW, 2, 60); // Screen width
    MAP_VSB_OFFSET(SH, 2, 61); // Screen height

    // Anonymous named struct to calculate the camera's rotation matrix
    inline static struct {
        operator const Matrix3f& () const {
            static Matrix3f M;
            Eigen::Map<Vector3f> 
                R = VSConstants::R, 
                L = VSConstants::L, 
                D = VSConstants::D, 
                U = VSConstants::U, 
                Z = VSConstants::F;
            auto X = (R - L).stableNormalized();
            auto Y = (U - D).stableNormalized();
            M << X, Y, Z;
            return M;
        }
    } Axes;

    static void Update(ComPtr<ID3D11DeviceContext> DeviceContext)
    {
        // Set index range of vertex buffers to be retrieved 
        // Anything past VB 3 seems to crash the game and is probably unrelated to the camera parameters
        constexpr size_t L = 0, R = 3;
        constexpr size_t N = R - L;

        static array<ID3D11Buffer*, N> Buffers = {};
        static array<ID3D11Buffer*, N> StagingBuffers = {};
        static array<D3D11_MAPPED_SUBRESOURCE, N> Subresources = {};
        static array<D3D11_BUFFER_DESC, N> BufferDescs = {};
        static bool Initialized = false;
        
        DeviceContext->VSGetConstantBuffers(L, N, Buffers.data());

        if (!Initialized)
        {
            for (auto i = 0; i < N; i++) {
                Buffers[i]->GetDesc(&BufferDescs[i]);
                BufferDescs[i].Usage = D3D11_USAGE_STAGING;
                BufferDescs[i].BindFlags = NULL;
                BufferDescs[i].CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
                Device->CreateBuffer(&BufferDescs[i], NULL, &StagingBuffers[i]);
                Data.add_constant_buffers();
            }
            Initialized = true;
        }

        for (auto i = 0; i < N; i++) DeviceContext->CopyResource(StagingBuffers[i], Buffers[i]);
        for (auto i = 0; i < N; i++) ERR(DeviceContext->Map(StagingBuffers[i], 0, D3D11_MAP_READ, NULL, &Subresources[i]));
        for (auto i = 0; i < N; i++) Data.set_constant_buffers(i, Subresources[i].pData, Subresources[i].DepthPitch);
        for (auto i = 0; i < N; i++) DeviceContext->Unmap(StagingBuffers[N - i - 1], 0);
        Data.set_nearclip(CAM::_0xA03502FC581F7D9B());
        Data.set_farclip(CAM::_0x9780F32BCAF72431());
        //Data.set_nearclip(CAM::_0xD0082607100D7193());
        //Data.set_farclip(CAM::_0xDFC8CBC606FDB0FC());
        VSMemory = Data;
    }
};
