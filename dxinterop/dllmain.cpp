// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "framework.h"
#include "ipc.h"

using Microsoft::WRL::ComPtr;
using std::string;
using std::unordered_map;
using Eigen::Vector3f;
using Eigen::Matrix3f;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Util

#define STRINGIFY(x) #x

static ComPtr<IDXGISwapChain> SwapChain;
static DXGI_SWAP_CHAIN_DESC SwapChainDesc;
static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> DeviceContext;
static void** DeviceContextVirtualTable;

static RECT winRect;
static void UPDATE_WINDOW_RECT(HWND window) 
{
    if (!GetWindowRect(window, &winRect)) throw std::system_error(S_SERDST, std::system_category());
}

static void GetDeviceAndContextFromSwapChain(void* chain) {
    SwapChain = (IDXGISwapChain*) chain;
    ERR(SwapChain->GetDesc(&SwapChainDesc));
    ERR(SwapChain->GetDevice(__uuidof(ID3D11Device), &Device));
    Device->GetImmediateContext(&DeviceContext);
    DeviceContextVirtualTable = (void**)*(void**)DeviceContext.Get();
}

static bool DidSwapChainUpdate(HWND window) {
    static int winH, winW;
    UPDATE_WINDOW_RECT(window);
    bool winSizeChanged = (winH != winRect.bottom - winRect.top) || (winW != winRect.right - winRect.left);
    winH = winRect.bottom - winRect.top;
    winW = winRect.right - winRect.left;
    return winSizeChanged;
}

void DEBUG_TEXTURE2D(ComPtr<ID3D11Texture2D> Texture, const char* Name = "DEBUG") 
{
    D3D11_TEXTURE2D_DESC TextureDesc;
    Texture->GetDesc(&TextureDesc);
    LOG(Name);
    LOG("Shape          " << TextureDesc.Width << ", " << TextureDesc.Height);
    LOG("Format         " << TextureDesc.Format);
    LOG("BindFlags:     " << TextureDesc.BindFlags);
    LOG("MiscFlags      " << TextureDesc.MiscFlags);
    LOG("SampleCount:   " << TextureDesc.SampleDesc.Count);
    LOG("SampleQuality: " << TextureDesc.SampleDesc.Quality);
    LOG("MipLevels:     " << TextureDesc.MipLevels);
    LOG("ArraySize:     " << TextureDesc.ArraySize);
    LOG("Usage:         " << TextureDesc.Usage);
    LOG("");
}

void DEBUG_CONSTANT_BUFFER(ComPtr<ID3D11Buffer> ConstantBuffer, const char* Name = "DEBUG")
{
    D3D11_BUFFER_DESC BufferDesc;
    ConstantBuffer->GetDesc(&BufferDesc);
    LOG(Name);
    LOG("ByteWidth: " << BufferDesc.ByteWidth);
    LOG("StructureByteStride: " << BufferDesc.StructureByteStride);
    LOG("Usage:               " << BufferDesc.Usage);
    LOG("MiscFlags:           " << BufferDesc.MiscFlags);
    LOG("BindFlags:           " << BufferDesc.BindFlags);
    LOG("CPUAccessFlags:      " << BufferDesc.CPUAccessFlags);
    LOG("");
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Compute Shader


DXGI_FORMAT GetDepthFormatFromDepthStencilFormat(DXGI_FORMAT Format)
{
    switch (Format)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    }
    throw std::system_error(E_NOTIMPL, std::system_category());
}

DXGI_FORMAT GetStencilFormatFromDepthStencilFormat(DXGI_FORMAT Format)
{
    switch (Format)
    {
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    }
    throw std::system_error(E_NOTIMPL, std::system_category());
}

template<typename T>
concept IsD3D11View = std::is_base_of_v<ID3D11View, T>;

template<IsD3D11View ViewType>
void GetTextureFromView(ComPtr<ViewType> View, ComPtr<ID3D11Texture2D>& Texture)
{
    ComPtr<ID3D11Resource> Resource;
    View->GetResource(&Resource);
    ERR(Resource.As(&Texture));
}

template<IsD3D11View ViewType>
void GetTextureFromView(ComPtr<ViewType> View, ComPtr<ID3D11Texture2D>& Texture, D3D11_TEXTURE2D_DESC* TextureDesc)
{
    GetTextureFromView(View, Texture);
    Texture->GetDesc(TextureDesc);
}

template<IsD3D11View ViewType>
void GetTextureFromView(ComPtr<ViewType> View, D3D11_TEXTURE2D_DESC* TextureDesc)
{
    ComPtr<ID3D11Texture2D> Texture;
    GetTextureFromView(View, Texture);
    Texture->GetDesc(TextureDesc);
}

struct VSConstantBuffers
{
    VSConstantBuffers() = default;

    inline static StructuredMemory<ByteBuffers> VSMemory{ string("VSConstantBuffers") };
    inline static ByteBuffers VSBuffers{};

#define MAP_VSB_TO_EIGEN_TYPE(NAME, MAP_TYPE, VSB_INDEX, VSB_OFFSET) \
    struct { \
        operator const Eigen::Map<MAP_TYPE> () const { \
            auto Floats = (float*) VSBuffers.data(VSB_INDEX).data(); \
            return Eigen::Map<MAP_TYPE>(Floats + VSB_OFFSET); \
        } \
    } NAME;

#define MAP_VSB_OFFSET(NAME, VSB_INDEX, VSB_OFFSET) \
    struct { \
        operator const float () const { \
            auto Floats = (float*) VSBuffers.data(VSB_INDEX).data(); \
            return Floats[VSB_OFFSET]; \
        } \
    } NAME;

    MAP_VSB_TO_EIGEN_TYPE(P, Vector3f, 2, 12);
    MAP_VSB_TO_EIGEN_TYPE(R, Vector3f, 2, 28);
    MAP_VSB_TO_EIGEN_TYPE(L, Vector3f, 2, 32);
    MAP_VSB_TO_EIGEN_TYPE(D, Vector3f, 2, 36);
    MAP_VSB_TO_EIGEN_TYPE(U, Vector3f, 2, 40);

    MAP_VSB_OFFSET(VW, 2, 20);
    MAP_VSB_OFFSET(VH, 2, 21);
    MAP_VSB_OFFSET(SW, 2, 60);
    MAP_VSB_OFFSET(SH, 2, 61);

    template<size_t N>
    using VSBufferArray = array<ComPtr<ID3D11Buffer>, N>;

    template<size_t Start, size_t N>
    static void Update(ComPtr<ID3D11DeviceContext> DeviceContext)
    {
        array<ID3D11Buffer*, N> Buffers{};
        DeviceContext->VSGetConstantBuffers(Start, N, Buffers.data());

        static array<ComPtr<ID3D11Buffer>, N> StagingBuffers = {};
        static array<D3D11_MAPPED_SUBRESOURCE, N> Subresources = {};
        static bool Initialized = false;

        if (!Initialized)
        {
            array<D3D11_BUFFER_DESC, N> BufferDescs = {};

            for (auto i = 0; i < N; i++) {
                Buffers[i]->GetDesc(&BufferDescs[i]);
                BufferDescs[i].Usage = D3D11_USAGE_STAGING;
                BufferDescs[i].BindFlags = NULL;
                BufferDescs[i].CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
                Device->CreateBuffer(&BufferDescs[i], NULL, StagingBuffers[i].GetAddressOf());
                i++;
            }
            Initialized = true;
        }
        
        VSBuffers.Clear();
        for (auto i = 0; i < N; i++) DeviceContext->CopyResource(StagingBuffers[i].Get(), Buffers[i]);
        for (auto i = 0; i < N; i++) DeviceContext->Map(StagingBuffers[i].Get(), 0, D3D11_MAP_WRITE_DISCARD, NULL, &Subresources[i]);
        for (auto i = 0; i < N; i++) VSBuffers.set_data(i, Subresources[i].pData, Subresources[i].DepthPitch);
        for (auto i = 0; i < N; i++) DeviceContext->Unmap(StagingBuffers[N - i - 1].Get(), 0);
        VSMemory = VSBuffers;
    }

};

static VSConstantBuffers VSBs{};

static Matrix3f Axes()
{
    Matrix3f M;
    Eigen::Map<Vector3f> R = VSBs.R, L = VSBs.L, D = VSBs.D, U = VSBs.U;
    auto X = (R - L).stableNormalized();
    auto Y = (U - D).stableNormalized();
    auto Z = (R + L).stableNormalized();
    M << X, Y, Z;
    return M;
}

//float NearClip = CAM::_0xD0082607100D7193();
//float FarClip = CAM::_0xDFC8CBC606FDB0FC();

struct Ray
{
    UINT C, R;
    StructuredMemory<Vec3f> Memory;
    Vec3f Data{};

    Ray(UINT C, UINT R) : C(C), R(R), Memory("Ray" + std::to_string(C) + "_" + std::to_string(R))
    {}

    Eigen::Vector3f ComputeDirection() const
    {
        float VW = VSBs.VW, VH = VSBs.VH, SW = VSBs.SW, SH = VSBs.SH;
        float X = 2 * (float(C) / VW) - (SW / VW);
        float Y = 2 * (float(R) / VH) - (SH / VH);
        float Z = 1;
        return Axes() * Eigen::Vector3f(X, Y, Z);
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
        auto Collision = Cast(VSBs.P, V);
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
        float VW = VSBs.VW, VH = VSBs.VH, SW = VSBs.SW, SH = VSBs.SH;
        static Ray Rays[] = {
            Ray(SW / 4, SH / 4),
            Ray((SW / 4) + (SW / 2), SH / 4),
            Ray(SW / 4, (SH / 4) + (SH / 2)),
            Ray((SW / 4) + (SW / 2), (SH / 4) + (SH / 2))
        };
        for (auto& Ray : Rays) Ray.ComputeCollision();
    }
};


struct CameraTransforms
{
    inline static ComPtr<ID3D11Buffer> LastMatrixBuffer;
    inline static ComPtr<ID3D11Buffer> CurrentMatrixBuffer;

    static void SetupMatrixBuffers()
    {
        D3D11_BUFFER_DESC BufferDesc = { 0 };
        BufferDesc.ByteWidth = sizeof(float) * 40;
        BufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        BufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ERR(Device->CreateBuffer(&BufferDesc, NULL, LastMatrixBuffer.GetAddressOf()));
        ERR(Device->CreateBuffer(&BufferDesc, NULL, CurrentMatrixBuffer.GetAddressOf()));
    }

    static void Update(ComPtr<ID3D11DeviceContext> DeviceContext)
    {   
        auto A = Axes();
        auto P = (Eigen::Map<Vector3f>) VSBs.P;

        DeviceContext->CopyResource(LastMatrixBuffer.Get(), CurrentMatrixBuffer.Get());

        D3D11_MAPPED_SUBRESOURCE CurrentMatrixSubresource;
        DeviceContext->Map(CurrentMatrixBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, NULL, &CurrentMatrixSubresource);
        float* MXData = (float*)CurrentMatrixSubresource.pData;

        for (int i = 0; i < 3; i++)
        {
            MXData[0 + i] = A(0, i);
            MXData[4 + i] = A(1, i);
            MXData[8 + i] = A(2, i);
            MXData[12 + i] = P[i];
        }

        DeviceContext->Unmap(CurrentMatrixBuffer.Get(), 0);
        DeviceContext->Flush();
    }

    CameraTransforms() = default;

};


struct DepthStencilComputeShader
{
    inline static ComPtr<ID3D11ComputeShader> ComputeShader;

    inline static ComPtr<ID3D11Texture2D> LastDepthTexture;
    inline static ComPtr<ID3D11Texture2D> VelocityDepthTexture;

    inline static ComPtr<ID3D11ShaderResourceView> DepthSRV;
    inline static ComPtr<ID3D11ShaderResourceView> LastDepthSRV;

    inline static ComPtr<ID3D11UnorderedAccessView> DepthUAV;
    inline static ComPtr<ID3D11UnorderedAccessView> VelocityDepthUAV;

    inline static CameraTransforms Camera;

    static void SetupDepthTexture
    (

        ComPtr<ID3D11Texture2D>& DepthStencilTexture,
        D3D11_TEXTURE2D_DESC& DepthStencilTextureDesc

    ) {
        D3D11_TEXTURE2D_DESC TextureDesc;
        TextureDesc.Width = DepthStencilTextureDesc.Width;
        TextureDesc.Height = DepthStencilTextureDesc.Height;
        TextureDesc.MipLevels = 1;
        TextureDesc.ArraySize = 1;
        TextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        TextureDesc.SampleDesc.Count = 1;
        TextureDesc.SampleDesc.Quality = 0;
        TextureDesc.Usage = D3D11_USAGE_DEFAULT;
        TextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        TextureDesc.CPUAccessFlags = 0;
        TextureDesc.MiscFlags = 0;
        ERR(Device->CreateTexture2D(&TextureDesc, NULL, VelocityDepthTexture.GetAddressOf()));
        ERR(Device->CreateTexture2D(&DepthStencilTextureDesc, NULL, &LastDepthTexture));
    }

    static void SetupDepthSRV
    (

        ComPtr<ID3D11Texture2D>& DepthStencilTexture,
        D3D11_TEXTURE2D_DESC& DepthTextureDesc

    ) {
        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        SRVDesc.Format = GetDepthFormatFromDepthStencilFormat(DepthTextureDesc.Format);
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        SRVDesc.Texture2D.MostDetailedMip = 0;
        SRVDesc.Texture2D.MipLevels = -1;
        ERR(Device->CreateShaderResourceView(DepthStencilTexture.Get(), &SRVDesc, DepthSRV.GetAddressOf()));
        ERR(Device->CreateShaderResourceView(LastDepthTexture.Get(), &SRVDesc, LastDepthSRV.GetAddressOf()));
    }

    
    static void SetupTextureUAV
    (

        ComPtr<ID3D11UnorderedAccessView>& UAV,
        ComPtr<ID3D11Texture2D>& Texture

    ) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
        D3D11_TEXTURE2D_DESC TextureDesc;
        Texture->GetDesc(&TextureDesc);
        UAVDesc.Format = TextureDesc.Format;
        UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        UAVDesc.Texture2D.MipSlice = 0;
        ERR(Device->CreateUnorderedAccessView(Texture.Get(), &UAVDesc, UAV.GetAddressOf()));
    }


    static void SetupResources(ComPtr<ID3D11DepthStencilView>& DepthStencilView)
    {
        ComPtr<ID3D11Texture2D> DepthStencilTexture;
        D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
        GetTextureFromView(DepthStencilView, DepthStencilTexture, &DepthStencilTextureDesc);
        SetupDepthTexture(DepthStencilTexture, DepthStencilTextureDesc);
        SetupDepthSRV(DepthStencilTexture, DepthStencilTextureDesc);
        SetupTextureUAV(VelocityDepthUAV, VelocityDepthTexture);
    }

    static void CreateComputeShader()
    {
        ComPtr<ID3DBlob> ShaderBlob;
        ComPtr<ID3DBlob> ErrorBlob;

        // Velocity buffer implementation:
        // 
        // Multiply the pixel NDCs by the inverse of the previous frame's camera matrix, 
        // then multiply that by the current frame's camera matrix 

        const char Shader[] =
            #include "ComputeShader.hlsl.cppliteral"
        ;

        HRESULT HR = D3DCompile
        (
            Shader,                             // SrcData
            sizeof(Shader),                     // SrcDataSize
            NULL,                               // SourceName
            NULL,                               // Defines
            D3D_COMPILE_STANDARD_FILE_INCLUDE,  // Include 
            "main",                             // EntryPoint
            "cs_5_0",                           // Target
            NULL,                               // Flags1
            NULL,                               // Flags2
            ShaderBlob.GetAddressOf(),          // Code
            ErrorBlob.GetAddressOf()            // ErrorMsgs
        );

        if (HR != S_OK) logfile << std::string((char*)ErrorBlob->GetBufferPointer(), ErrorBlob->GetBufferSize());
        ERR(HR);
        ERR(Device->CreateComputeShader(ShaderBlob->GetBufferPointer(), ShaderBlob->GetBufferSize(), NULL, ComputeShader.GetAddressOf()));
    }

    static void RunComputeShader(ComPtr<ID3D11DeviceContext> DeviceContext)
    {
        static const size_t N_CBF = 2;
        static const size_t N_SRV = 2;
        static const size_t N_UAV = 1;
        ID3D11Buffer* CBuffers[N_CBF] = { Camera.CurrentMatrixBuffer.Get(), Camera.LastMatrixBuffer.Get() };
        ID3D11ShaderResourceView* ShaderResourceViews[N_SRV] = { DepthSRV.Get(), LastDepthSRV.Get() };
        ID3D11UnorderedAccessView* UnorderedAccessViews[N_UAV] = { VelocityDepthUAV.Get() };
        DeviceContext->CSSetShader(ComputeShader.Get(), NULL, NULL);
        DeviceContext->CSSetConstantBuffers(0, N_CBF, CBuffers);
        DeviceContext->CSSetShaderResources(0, N_SRV, ShaderResourceViews);
        DeviceContext->CSSetUnorderedAccessViews(0, N_UAV, UnorderedAccessViews, NULL);
        D3D11_TEXTURE2D_DESC DepthTextureDesc;
        LastDepthTexture->GetDesc(&DepthTextureDesc);
        DeviceContext->Dispatch((DepthTextureDesc.Width + 32) / 32, (DepthTextureDesc.Height + 32) / 32, 1);
        DeviceContext->Flush();
        DeviceContext->CSSetShader(nullptr, nullptr, 0);
        ID3D11UnorderedAccessView* NullUAV[N_UAV] = { nullptr };
        ID3D11Buffer* NullCB[N_CBF] = { nullptr, nullptr };
        ID3D11ShaderResourceView* NullSRV[N_SRV] = { nullptr, nullptr };
        DeviceContext->CSSetUnorderedAccessViews(0, N_UAV, NullUAV, nullptr);
        DeviceContext->CSSetShaderResources(0, N_SRV, NullSRV);
        DeviceContext->CSSetConstantBuffers(0, N_CBF, NullCB);
        ComPtr<ID3D11Texture2D> DSVTexture;
        ComPtr<ID3D11Resource> DSVResource;
        DepthSRV->GetResource(&DSVResource);
        ERR(DSVResource.As(&DSVTexture));
        DeviceContext->CopyResource(LastDepthTexture.Get(), DSVTexture.Get());
        DeviceContext->Flush();
    }

    DepthStencilComputeShader() = default;
    DepthStencilComputeShader(ComPtr<ID3D11DepthStencilView> DepthStencilView) 
    {
        SetupResources(DepthStencilView);
        CreateComputeShader();
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// CUDA


cudaChannelFormatDesc CudaChannelFormatFromDXGIFormat(DXGI_FORMAT Format)
{
    switch (Format)
    {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
        return { 8, 8, 8, 8, cudaChannelFormatKindUnsigned };
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return { 32, 32, 0, 0, cudaChannelFormatKindNone };
    case DXGI_FORMAT_R32G32_FLOAT:
        return { 32, 32, 0, 0, cudaChannelFormatKindFloat };
    case DXGI_FORMAT_R32_FLOAT:
        return { 32, 0, 0, 0, cudaChannelFormatKindFloat };
    case DXGI_FORMAT_R8_UINT:
        return { 8, 0, 0, 0, cudaChannelFormatKindUnsigned };
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        return { 32, 32, 32, 32, cudaChannelFormatKindFloat };
    }
    throw std::system_error(E_NOTIMPL, std::system_category());
}

UINT GetBytesPerPixelFromDXGIFormat(DXGI_FORMAT Format)
{
    cudaChannelFormatDesc ChannelFormat = CudaChannelFormatFromDXGIFormat(Format);
    return (ChannelFormat.x + ChannelFormat.y + ChannelFormat.z + ChannelFormat.w) / 8;
}

struct IPCCUDAArray
{
    inline static unordered_map<string, IPCCUDAArray*> Instances;

    void* CUDAMemory = nullptr;
    StructuredMemory<CUDAArrayObject> IPCMemory;

    IPCCUDAArray
    (
        cudaChannelFormatDesc ChannelFormat,
        cudaExtent Extent, 
        uint64_t BPP, 
        string Tagname
    ) : IPCMemory(Tagname)
    {
        uint64_t Pitch;
        cudaMallocPitch(&CUDAMemory, &Pitch, BPP * Extent.width, Extent.height);

        cudaIpcMemHandle_t Handle;
        cudaIpcGetMemHandle(&Handle, CUDAMemory);

        CUDAExtent E = {};

        E.set_width(Extent.width);
        E.set_height(Extent.height);
        E.set_depth(Extent.depth);

        CUDAChannelFormatDesc CF = {};
        CF.set_x(ChannelFormat.x);
        CF.set_y(ChannelFormat.y);
        CF.set_z(ChannelFormat.z);
        CF.set_w(ChannelFormat.w);
        CF.set_f(ChannelFormat.f);

        CUDAArrayObject Data = {};
        Data.set_handle(&Handle, sizeof(cudaIpcMemHandle_t));
        Data.set_allocated_formatdesc(&CF);
        Data.set_pitch(Pitch);
        Data.set_bpp(BPP);
        Data.set_allocated_extent(&E);

        IPCMemory = Data;
    }

    void CopyFrom(cudaArray_t CudaArray)
    {
        auto Data = static_cast<CUDAArrayObject>(IPCMemory);
        CUERR(cudaMemcpy2DFromArray(CUDAMemory, Data.pitch(), CudaArray, 0, 0, Data.bpp() * Data.extent().width(), Data.extent().height(), cudaMemcpyDefault));
    }

    virtual void Delete()
    {
        cudaFree(CUDAMemory);
    }

    static void DeleteAll()
    {
        for (auto const& [_, Array] : IPCCUDAArray::Instances) Array->Delete();
    }
};

template <typename T>
struct IPCCUDAD3D11GraphicsResource : IPCCUDAArray
{
    cudaGraphicsResource_t GraphicsResource;

    static cudaChannelFormatDesc Format(ComPtr<ID3D11Texture2D>& Texture)
    {
        D3D11_TEXTURE2D_DESC TextureDesc;
        Texture->GetDesc(&TextureDesc);
        return CudaChannelFormatFromDXGIFormat(TextureDesc.Format);
    }

    static cudaExtent Extent(ComPtr<ID3D11Texture2D>& Texture)
    {
        D3D11_TEXTURE2D_DESC TextureDesc;
        Texture->GetDesc(&TextureDesc);
        return {
            .width = TextureDesc.Width,
            .height = TextureDesc.Height,
            .depth = 1
        };
    }

    static UINT BPP(ComPtr<ID3D11Texture2D>& Texture)
    {
        D3D11_TEXTURE2D_DESC TextureDesc;
        Texture->GetDesc(&TextureDesc);
        return GetBytesPerPixelFromDXGIFormat(TextureDesc.Format);
    }

    IPCCUDAD3D11GraphicsResource
    (
        ComPtr<T>& D3D11Object, 
        string Tagname
    ) : IPCCUDAArray(Format(D3D11Object), Extent(D3D11Object), BPP(D3D11Object), Tagname)
    {
        CUERR(cudaGraphicsD3D11RegisterResource(&GraphicsResource, D3D11Object.Get(), cudaGraphicsRegisterFlagsNone)); // try other cudaGraphicsRegisterFlags here ?
    }

    void Update()
    {
        cudaArray_t MappedArray;
        CUERR(cudaGraphicsMapResources(1, &GraphicsResource));
        if constexpr (std::is_same_v<T, ID3D11Texture2D>)
        {
            CUERR(cudaGraphicsSubResourceGetMappedArray(&MappedArray, GraphicsResource, 0, 0));
            CopyFrom(MappedArray);
        }
        CUERR(cudaGraphicsUnmapResources(1, &GraphicsResource));
        CUERR(cudaDeviceSynchronize());
    }

    void Delete()
    {
        IPCCUDAArray::Delete();
        cudaGraphicsUnregisterResource(GraphicsResource);
    }
};

using CudaD3D11TextureArray = IPCCUDAD3D11GraphicsResource<ID3D11Texture2D>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Hooks


using ClearDepthStencilViewFunction = void(__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
static ClearDepthStencilViewFunction ClearDepthStencilView = NULL;

const unsigned int NUM_VSB = 3;

//static void DEBUG_ARROWS
//(
//
//    MemoryMappedFile<float> VSB[NUM_VSB]
//
//) {
//    const static int NumberOfArrows = 4;
//    static MemoryMappedFile<float> DebugArrows[NumberOfArrows];
//    static MemoryMappedFile<int> ArrowColors[NumberOfArrows];
//    static bool Initialized = false;
//    if (!Initialized)
//    {
//        for (int i = 0; i < NumberOfArrows; i++)
//        {
//            DebugArrows[i] = MemoryMappedFile<float>(6, "DebugArrow" + std::to_string(i));
//            ArrowColors[i] = MemoryMappedFile<int>(4, "DebugArrowColors" + std::to_string(i));
//        }
//        Initialized = true;
//    }
//
//    for (int i = 0; i < NumberOfArrows; i++)
//    {
//        GRAPHICS::DRAW_LINE(
//            DebugArrows[i][0],
//            DebugArrows[i][1],
//            DebugArrows[i][2],
//            DebugArrows[i][0] + DebugArrows[i][3],
//            DebugArrows[i][1] + DebugArrows[i][4],
//            DebugArrows[i][2] + DebugArrows[i][5],
//            ArrowColors[i][0],
//            ArrowColors[i][1],
//            ArrowColors[i][2],
//            ArrowColors[i][3]
//        );
//    }
//}

// Unfortunately there doesn't seem to be a straightforward way to read the depth stencil texture directly into cuda memory.
// (see the section for cudaGraphicsD3D11RegisterResource in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html)
// So we have to have a 'CUDA-staging texture' between the depth stencil texture w/ a CUDA-compatible format.
// We'll write to this texture using a compute shader, whose inputs we set up here.
// The input to the compute shader is an SRV (Shader Resource View) bound to the depth stencil's backing texture,
// and the output is a UAV (Unordered Access View) bound to the CUDA-staging texture. 


void ClearDepthStencilViewHook
(

    ID3D11DeviceContext* pDeviceContext, 
    ID3D11DepthStencilView* pDepthStencilView, 
    UINT clearFlags, 
    FLOAT depth, 
    UINT8 stencil

) {

    ComPtr<ID3D11DeviceContext> DeviceContext = pDeviceContext;
    ComPtr<ID3D11DepthStencilView> DepthStencilView = pDepthStencilView;

    static ID3D11DepthStencilView* SentinelDSV;
    static ID3D11DepthStencilView* BindDSV;
    static ID3D11DepthStencilView* LastDSV;
    static D3D11_DEPTH_STENCIL_DESC DepthStencilStateDesc;
    static bool DepthStencilEnabledLastFrame;

    ID3D11DepthStencilState* DepthStencilState;
    DeviceContext->OMGetDepthStencilState(&DepthStencilState, NULL);
    DepthStencilState->GetDesc(&DepthStencilStateDesc);

    // There's some strange game you have to play to copy out the depth texture at the right time 
    // It might be that the depth stencil texture gets cleared at the top of each present call, 
    // so getting the one for the current frame involves knowing which DSV's being cleared next

    if (SentinelDSV == nullptr && DepthStencilStateDesc.DepthEnable) SentinelDSV = pDepthStencilView;
    if (BindDSV == nullptr && pDepthStencilView == SentinelDSV && !DepthStencilStateDesc.DepthEnable && !DepthStencilEnabledLastFrame) BindDSV = LastDSV;

    //LOG(pDepthStencilView << " " << )

    // After we've got BindDSV we can start setting up all the IPC-related memory
    // For whatever reason, CUDA doesn't support mapping the depth buffer so there's
    // a bit of a process:
    // 
    // 1. Get the depth stencil view's texture and bind it to a shader resource view
    // 2. Create the CUDA-staging texture and bind it to an unordered access view 
    // 3. Allocate CUDA memory & create its IPC memory handle
    // 4. When appropriate, run the compute shader to write to (2) and map & copy (2) to (3) via CUDA

    static DepthStencilComputeShader ComputeShader;
    static CudaD3D11TextureArray* CudaArray;
    static bool Initialized;

    // 1 - 3
    if(BindDSV != nullptr && pDepthStencilView == BindDSV && CudaArray == nullptr)
    {
        LOG("setup");
        //LaunchDebugger();
        //DebugBreak();
        ComputeShader = DepthStencilComputeShader(DepthStencilView);
        CudaArray = new CudaD3D11TextureArray(DepthStencilComputeShader::VelocityDepthTexture, "DepthBuffer");
    }

    //static MemoryMappedFile<bool> UpdateSignal = MemoryMappedFile<bool>("RayCastUpdate");

    // 4 - 5
    if(pDepthStencilView == SentinelDSV && !DepthStencilStateDesc.DepthEnable && CudaArray != nullptr)
    {
        //GAMEPLAY::SET_GAME_PAUSED(true);
        VSConstantBuffers::Update<0, 3>(DeviceContext);
        CameraTransforms::Update(DeviceContext);
        DepthStencilComputeShader::RunComputeShader(DeviceContext);
        CudaArray -> Update();
        //GAMEPLAY::SET_GAME_PAUSED(false);
    }

    LastDSV                      = pDepthStencilView;
    DepthStencilEnabledLastFrame = DepthStencilStateDesc.DepthEnable;

    return ClearDepthStencilView(pDeviceContext, pDepthStencilView, clearFlags, depth, stencil);
}

static void HookClearDepthStencilView()
{
    ClearDepthStencilView = (ClearDepthStencilViewFunction)DeviceContextVirtualTable[53];
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach
    (
        &(PVOID&)ClearDepthStencilView,
        (PBYTE*)&ClearDepthStencilViewHook
    );
    DetourTransactionCommit();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Main / ScripthookV



static void presentCallback(void* chain) {

    static ComPtr<ID3D10Multithread> MultithreadContext;
    //static MemoryMappedFile<float> NearClipFarClip(2, "NearClipFarClip");

    GetDeviceAndContextFromSwapChain(chain);
    DidSwapChainUpdate(SwapChainDesc.OutputWindow);

    //DeviceContext -> VSGetConstantBuffers(0, )

    //NearClipFarClip[0] = CAM::_0xD0082607100D7193(); // NearClip
    //NearClipFarClip[1] = CAM::_0xDFC8CBC606FDB0FC(); // FarClip

    // DX11 will launch an amortized version of ClearDepthStencilView once in a while
    // and for whatever reason we do not want to hook that
    // It happens only occasionally, so it seems we can reliably detect if we have
    // the correct vtable entry by just checking that it hasn't changed 

    static void* LastVTEntry;
    static bool FoundPrimaryVTEntry;
    static bool Hooked;

    //LaunchDebugger();

    if (!Hooked)
    {
        DeviceContext.As(&MultithreadContext);
        MultithreadContext->SetMultithreadProtected(true);
        DeviceContextVirtualTable = (void**)*(void**)DeviceContext.Get();
        if (!FoundPrimaryVTEntry && LastVTEntry == DeviceContextVirtualTable[53])
        {
            HookClearDepthStencilView();
            //DebugBreak();
            Hooked = true;
            FoundPrimaryVTEntry = true;
        }
    }

    LastVTEntry = DeviceContextVirtualTable[53];


    // RTV hook related stuff (i.e. the getting displayed pixels)
    static CudaD3D11TextureArray* RenderTargetArray;
    static ComPtr<ID3D11RenderTargetView> RenderTargetView;
    static ComPtr<ID3D11Resource> RenderTargetResource;
    static ComPtr<ID3D11Texture2D> RenderTargetTexture;

    DeviceContext->OMGetRenderTargets(1, RenderTargetView.GetAddressOf(), NULL);

    if (RenderTargetTexture == nullptr)
    {
        GetTextureFromView(RenderTargetView, RenderTargetTexture);
        RenderTargetArray = new CudaD3D11TextureArray(RenderTargetTexture, "RGB");
    }
    else
    {
        RenderTargetArray -> Update();
    }
}

BOOL APIENTRY DllMain
(
    HMODULE hModule,                   
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
) {
    int DeviceCount = 0;
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        cudaGetDeviceCount(&DeviceCount);
        LOG("Attached! Cuda Device Count: " << DeviceCount << std::endl);
        presentCallbackRegister(presentCallback);
        break;
    case DLL_PROCESS_DETACH:
        presentCallbackUnregister(presentCallback);
        CudaD3D11TextureArray::DeleteAll();
        break;
    }
    return TRUE;
}

