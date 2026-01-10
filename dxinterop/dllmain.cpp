// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "framework.h"
#include "scripthookv_sdk/inc/main.h"

using Microsoft::WRL::ComPtr;
using std::string;
using std::unordered_map;
using namespace Eigen;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Util

static std::ofstream logfile("dxinterop.log");

#ifndef NDEBUG
#define LOG(msg) logfile << msg << std::endl
#else
#define LOG(msg)
#endif

static ComPtr<IDXGISwapChain> SwapChain;
static DXGI_SWAP_CHAIN_DESC SwapChainDesc;
static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> DeviceContext;
static void** DeviceContextVirtualTable;

HRESULT _ERR;
cudaError_t _CUERR;

#define ERR(CALL)\
_ERR = CALL;\
if(_ERR != S_OK){\
    LOG(#CALL << " returned error: " << _ERR);\
    throw std::system_error(_ERR, std::system_category());\
}

#define CUERR(CALL)\
_CUERR = CALL;\
if(_CUERR != cudaSuccess){\
    LOG(#CALL << " returned error: " << cudaGetErrorString(_CUERR));\
    throw std::system_error(_CUERR, std::system_category());\
}


template<typename T>
struct MemoryMappedFile
{
    HANDLE Handle = NULL;

    T* File = nullptr;

    MemoryMappedFile(size_t NumberOfElements, string Filename)
    {
        std::wstring t = std::wstring(Filename.begin(), Filename.end());
        Handle = CreateFileMapping(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            NumberOfElements * sizeof(T),
            t.c_str()
        );
        File = (T*) MapViewOfFile(Handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    }

    MemoryMappedFile(string Filename) : MemoryMappedFile(1, Filename)
    {}

    MemoryMappedFile() = default;

    T& operator[](size_t Index)
    {
        return File[Index];
    }

    void Delete()
    {
        UnmapViewOfFile(File);
        CloseHandle(Handle);
    }

    bool Flush()
    {
        return FlushFileBuffers(Handle);
    }

};


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
    }
    throw std::system_error(E_NOTIMPL, std::system_category());
}

UINT GetBytesPerPixelFromDXGIFormat(DXGI_FORMAT Format)
{
    cudaChannelFormatDesc ChannelFormat = CudaChannelFormatFromDXGIFormat(Format);
    return (ChannelFormat.x + ChannelFormat.y + ChannelFormat.z + ChannelFormat.w) / 8;
}

// Sets up CUDA memory 
// Creates two anonymous memory mapped files storing metadata & IPC handle respectively
struct IPCCUDAArray
{
    inline static unordered_map<string, IPCCUDAArray*> Instances;

    void* Memory = nullptr;
    cudaChannelFormatDesc                ChannelFormat = {};
    uint64_t                             BPP = {};
    uint64_t                             Pitch = {};
    cudaExtent                           Extent = {};
    cudaIpcMemHandle_t                   ArrayHandle = {};

    MemoryMappedFile<uint64_t>           ArrayFormatFile;
    MemoryMappedFile<cudaIpcMemHandle_t> ArrayHandleFile;

    IPCCUDAArray() = default;

    void Setup
    (
        cudaChannelFormatDesc InitialChannelFormat, 
        cudaExtent InitialExtent, 
        uint64_t InitialBPP,
        string Tag
    ) {
        ChannelFormat = InitialChannelFormat;
        Extent = InitialExtent;
        BPP = InitialBPP;
        CUERR(cudaMallocPitch(&Memory, &Pitch, BPP * Extent.width, Extent.height));
        CUERR(cudaIpcGetMemHandle(&ArrayHandle, Memory));
        ArrayFormatFile = MemoryMappedFile<uint64_t>(4, Tag+"Info");
        ArrayHandleFile = MemoryMappedFile<cudaIpcMemHandle_t>(Tag);
    }

    IPCCUDAArray(
        cudaChannelFormatDesc InitialChannelFormat,
        cudaExtent InitialExtent,
        uint64_t InitialBPP,
        string Tag
    ) {
        Setup(InitialChannelFormat, InitialExtent, InitialBPP, Tag);
    }

    void WriteInfo()
    {
        CUERR(cudaIpcGetMemHandle(&ArrayHandle, Memory));
        ArrayHandleFile[0] = ArrayHandle;
        ArrayFormatFile[0] = (ChannelFormat.x > 0) + (ChannelFormat.y > 0) + (ChannelFormat.z > 0) + (ChannelFormat.w > 0);
        ArrayFormatFile[1] = BPP;
        ArrayFormatFile[2] = Pitch;
        ArrayFormatFile[3] = Extent.height;
        ArrayHandleFile.Flush();
        ArrayFormatFile.Flush();
    }

    void CopyFrom(cudaArray_t CudaArray)
    {
        CUERR(cudaMemcpy2DFromArray(Memory, Pitch, CudaArray, 0, 0, BPP * Extent.width, Extent.height, cudaMemcpyDefault));
    }

    virtual void Delete()
    {
        ArrayHandleFile.Delete();
        ArrayFormatFile.Delete();
        cudaFree(Memory);
    }

    static void DeleteAll()
    {
        for (auto const& [_, Array] : IPCCUDAArray::Instances) Array->Delete();
    }
};

template <typename T>
struct IPCCUDAD3D11GraphicsResource : IPCCUDAArray
{
    cudaGraphicsResource_t GraphicsResource = {};
    
    IPCCUDAD3D11GraphicsResource<T>() = default;

    IPCCUDAD3D11GraphicsResource(ComPtr<ID3D11Texture2D>& Texture, string Tag)
    {
        D3D11_TEXTURE2D_DESC TextureDesc;
        Texture->GetDesc(&TextureDesc);
        ChannelFormat = CudaChannelFormatFromDXGIFormat(TextureDesc.Format);
        Extent.width = TextureDesc.Width;
        Extent.height = TextureDesc.Height;
        Extent.depth = 1;
        BPP = GetBytesPerPixelFromDXGIFormat(TextureDesc.Format);
        // try other cudaGraphicsRegisterFlags here ?
        CUERR(cudaGraphicsD3D11RegisterResource(&GraphicsResource, Texture.Get(), cudaGraphicsRegisterFlagsNone));
        Setup(ChannelFormat, Extent, BPP, Tag);
    }

    IPCCUDAD3D11GraphicsResource(ComPtr<ID3D11Buffer>& Buffer, string Tag)
    {
        D3D11_BUFFER_DESC BufferDesc;
        Buffer->GetDesc(&BufferDesc);
        ChannelFormat = {0};
        Extent.width = BufferDesc.ByteWidth;
        Extent.height = 1;
        Extent.depth = 1;
        BPP = 1;
        CUERR(cudaGraphicsD3D11RegisterResource(&GraphicsResource, Buffer.Get(), cudaGraphicsRegisterFlagsNone));
        Setup(ChannelFormat, Extent, BPP, Tag);
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
        WriteInfo();
    }

    void Delete() override
    {
        IPCCUDAArray::Delete();
        cudaGraphicsUnregisterResource(GraphicsResource);
    }
};

using CudaD3D11TextureArray = IPCCUDAD3D11GraphicsResource<ID3D11Texture2D>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Hooks

// Unfortunately there doesn't seem to be a straightforward way to read the depth stencil texture directly into cuda memory.
// (see the section for cudaGraphicsD3D11RegisterResource in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html)
// So we have to have a 'CUDA-staging texture' between the depth stencil texture w/ a CUDA-compatible format.
// We'll write to this texture using a compute shader, whose inputs we set up here.
// The input to the compute shader is an SRV (Shader Resource View) bound to the depth stencil's backing texture,
// and the output is a UAV (Unordered Access View) bound to the CUDA-staging texture. 

void SetupComputeShaderResources
(

    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    ComPtr<ID3D11ShaderResourceView>& ShaderResourceView,
    ComPtr<ID3D11Texture2D>& DepthTexture,
    ComPtr<ID3D11UnorderedAccessView>& UnorderedAccessView

) {
    ComPtr<ID3D11Texture2D> DepthStencilTexture;
    D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
    GetTextureFromView(DepthStencilView, DepthStencilTexture, &DepthStencilTextureDesc);

    D3D11_SHADER_RESOURCE_VIEW_DESC ShaderResourceViewDesc;
    ShaderResourceViewDesc.Format = GetDepthFormatFromDepthStencilFormat(DepthStencilTextureDesc.Format);
    ShaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    ShaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
    ShaderResourceViewDesc.Texture2D.MipLevels = -1;
    ERR(Device->CreateShaderResourceView(DepthStencilTexture.Get(), &ShaderResourceViewDesc, ShaderResourceView.GetAddressOf()));

    D3D11_TEXTURE2D_DESC DepthTextureDesc;
    DepthTextureDesc.Width = DepthStencilTextureDesc.Width;
    DepthTextureDesc.Height = DepthStencilTextureDesc.Height;
    DepthTextureDesc.MipLevels = 1;
    DepthTextureDesc.ArraySize = 1;
    DepthTextureDesc.Format = DXGI_FORMAT_R32_FLOAT;
    DepthTextureDesc.SampleDesc.Count = 1;
    DepthTextureDesc.SampleDesc.Quality = 0;
    DepthTextureDesc.Usage = D3D11_USAGE_DEFAULT;
    DepthTextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    DepthTextureDesc.CPUAccessFlags = 0;
    DepthTextureDesc.MiscFlags = 0;
    ERR(Device->CreateTexture2D(&DepthTextureDesc, NULL, DepthTexture.GetAddressOf()));

    D3D11_UNORDERED_ACCESS_VIEW_DESC UnorderedAccessViewDesc;
    UnorderedAccessViewDesc.Format = DepthTextureDesc.Format;
    UnorderedAccessViewDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    UnorderedAccessViewDesc.Texture2D.MipSlice = 0;
    ERR(Device->CreateUnorderedAccessView(DepthTexture.Get(), &UnorderedAccessViewDesc, UnorderedAccessView.GetAddressOf()));

}

void SetupComputeShaderResources
(

    ComPtr<ID3D11DeviceContext> DeviceContext,
    ComPtr<ID3D11DepthStencilView> DepthStencilView,
    ComPtr<ID3D11ShaderResourceView>& ShaderResourceView,
    ComPtr<ID3D11Texture2D>& DepthTextureBuffer,
    ComPtr<ID3D11UnorderedAccessView>& UnorderedAccessView

) {
    ComPtr<ID3D11Device> Device;
    DeviceContext->GetDevice(&Device);
    SetupComputeShaderResources(Device, DepthStencilView, ShaderResourceView, DepthTextureBuffer, UnorderedAccessView);
}

// Compute shader for converting DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS to DXGI_FORMAT_R32_FLOAT
// TODO: Add separate UAV for stencil
void CreateComputeShader
(
    ComPtr<ID3D11Device> Device, 
    ComPtr<ID3D11ComputeShader>& ComputeShader
) {

    const char Shader[] =
    R"(
        Texture2D<float3> InputTexture : register(t);
        RWTexture2D<float> OutputTexture : register(u);

        [numthreads(32, 32, 1)]
        void main(uint3 DTid : SV_DispatchThreadID)
        {
            OutputTexture[DTid.xy].r = InputTexture[DTid.xy].r;
        }
    )";

    ComPtr<ID3DBlob> ShaderBlob;
    ComPtr<ID3DBlob> ErrorBlob;

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

void RunComputeShader
(
    ComPtr<ID3D11DeviceContext> DeviceContext,
    ComPtr<ID3D11ComputeShader> ComputeShader,
    ComPtr<ID3D11ShaderResourceView> ShaderResourceView,
    ComPtr<ID3D11UnorderedAccessView> UnorderedAccessView,
    UINT X,
    UINT Y
) {


    DeviceContext->CSSetShader(ComputeShader.Get(), NULL, NULL);
    DeviceContext->CSSetShaderResources(0, 1, ShaderResourceView.GetAddressOf());
    DeviceContext->CSSetUnorderedAccessViews(0, 1, UnorderedAccessView.GetAddressOf(), NULL);

    DeviceContext->Dispatch(X, Y, 1);
    DeviceContext->Flush();

    DeviceContext->CSSetShader(nullptr, nullptr, 0);

    ID3D11UnorderedAccessView* NullUAV[1] = { nullptr };
    ID3D11ShaderResourceView* NullSRV[1] = { nullptr };

    DeviceContext->CSSetUnorderedAccessViews(0, 1, NullUAV, nullptr);
    DeviceContext->CSSetShaderResources(0, 1, NullSRV);

}

using ClearDepthStencilViewFunction = void(__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
static ClearDepthStencilViewFunction ClearDepthStencilView = NULL;

const unsigned int NUM_VSB = 3;

static void DRAW_RGB
(

    MemoryMappedFile<float> VSB[NUM_VSB]

) {

    Eigen::Vector3f CameraPosition(VSB[2][12], VSB[2][13], VSB[2][14]);
    Eigen::Vector3f CameraDirection(VSB[2][16], VSB[2][17], VSB[2][18]);
    Eigen::Vector3f P = CameraPosition + CameraDirection;

    static MemoryMappedFile<float> Rf(3, "RVector");
    static MemoryMappedFile<float> Gf(3, "GVector");
    static MemoryMappedFile<float> Bf(3, "BVector");

    Eigen::Vector3f R = P + Eigen::Vector3f(Rf[0], Rf[1], Rf[2]);
    Eigen::Vector3f G = P + Eigen::Vector3f(Gf[0], Gf[1], Gf[2]);
    Eigen::Vector3f B = P + Eigen::Vector3f(Bf[0], Bf[1], Bf[2]);

    GRAPHICS::DRAW_LINE(P[0], P[1], P[2], R[0], R[1], R[2], 255, 0, 0, 255);
    GRAPHICS::DRAW_LINE(P[0], P[1], P[2], G[0], G[1], G[2], 0, 255, 0, 255);
    GRAPHICS::DRAW_LINE(P[0], P[1], P[2], B[0], B[1], B[2], 0, 0, 255, 255);

}

static void DEBUG_VERTEXSHADER
(

    ComPtr<ID3D11Device> Device,
    ComPtr<ID3D11DeviceContext> DeviceContext

) {
    const unsigned int N = NUM_VSB;

    static bool Initialized = false;
    static MemoryMappedFile<float> VSBFiles[N];
    static MemoryMappedFile<uint64_t> VSBFileLengths[N];
    static ID3D11Buffer* StagingBuffers[N];

    D3D11_MAPPED_SUBRESOURCE Subresources[N] = {};
    ID3D11Buffer* ConstantBuffers[N] = {};
    DeviceContext->VSGetConstantBuffers(0, N, ConstantBuffers);

    if (!Initialized)
    {
        for (int i = 0; i < N; i++)
        {
            D3D11_BUFFER_DESC BufferDesc;
            ConstantBuffers[i]->GetDesc(&BufferDesc);
            BufferDesc.Usage = D3D11_USAGE_STAGING;
            BufferDesc.BindFlags = 0;
            BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
            Device->CreateBuffer(&BufferDesc, NULL, &StagingBuffers[i]);
            VSBFiles[i] = MemoryMappedFile<float>(BufferDesc.ByteWidth / sizeof(float), "VSB" + std::to_string(i));
            VSBFileLengths[i] = MemoryMappedFile<uint64_t>("VSB" + std::to_string(i) + "Length");
        }
        Initialized = true;
    }

    for (int i = 0; i < N; i++)
    {
        D3D11_BUFFER_DESC BufferDesc, StagingDesc;
        ConstantBuffers[i]->GetDesc(&BufferDesc);
        StagingBuffers[i]->GetDesc(&StagingDesc);
        VSBFileLengths[i][0] = BufferDesc.ByteWidth;
        VSBFileLengths[i].Flush();
        if (BufferDesc.ByteWidth != StagingDesc.ByteWidth)
        {
            StagingDesc.ByteWidth = BufferDesc.ByteWidth;
            StagingBuffers[i]->Release();
            Device->CreateBuffer(&StagingDesc, NULL, &StagingBuffers[i]);
        }
    }

    for (int i = 0; i < N; i++)
    {
        DeviceContext->CopyResource(StagingBuffers[i], ConstantBuffers[i]);
        DeviceContext->Flush();
    }

    for (int i = 0; i < N; i++) 
    {
        DeviceContext->Map(StagingBuffers[i], 0, D3D11_MAP_READ, NULL, &Subresources[i]);
    }

    for (int i = 0; i < N; i++)
    {
        memcpy(VSBFiles[i].File, Subresources[i].pData, Subresources[i].DepthPitch);
    }

    for (int i = 0; i < N; i++) 
    {
        DeviceContext->Unmap(StagingBuffers[i], 0);
    }

    DRAW_RGB(VSBFiles);

}

void ClearDepthStencilViewHook
(

    ID3D11DeviceContext* pDeviceContext, 
    ID3D11DepthStencilView* pDepthStencilView, 
    UINT clearFlags, 
    FLOAT depth, 
    UINT8 stencil

) {

    ComPtr<ID3D11DeviceContext> DeviceContext = pDeviceContext;
    static ComPtr<ID3D11DepthStencilView> DepthStencilView = pDepthStencilView;

    static ID3D11DepthStencilView* SentinelDSV;
    static ID3D11DepthStencilView* BindDSV;
    static ID3D11DepthStencilView* LastDSV;
    static ID3D11DepthStencilState* DepthStencilState;
    static D3D11_DEPTH_STENCIL_DESC DepthStencilStateDesc;
    static bool DepthStencilEnabledLastFrame;

    DeviceContext->OMGetDepthStencilState(&DepthStencilState, NULL);
    DepthStencilState->GetDesc(&DepthStencilStateDesc);

    // There's some strange game you have to play to bind the right depth stencil view and copy out the texture at the right time

    if (SentinelDSV == nullptr && DepthStencilStateDesc.DepthEnable) SentinelDSV = pDepthStencilView;
    if (BindDSV == nullptr && pDepthStencilView == SentinelDSV && !DepthStencilStateDesc.DepthEnable && !DepthStencilEnabledLastFrame) BindDSV = LastDSV;

    // After we've got BindDSV we can start setting up all the IPC-related memory
    // For whatever reason, CUDA doesn't support mapping the depth buffer so there's
    // a bit of a process:
    // 
    // 1. Get the depth stencil view's texture and bind it to a shader resource view
    // 2. Create the CUDA-staging texture and bind it to an unordered access view 
    // 3. Allocate CUDA memory & create its IPC memory handle
    // 4. When appropriate, run the compute shader to write to (2) and map & copy (2) to (3) via CUDA

    static CudaD3D11TextureArray CudaArray;
    static ComPtr<ID3D11ComputeShader> ComputeShader;

    static ComPtr<ID3D11Texture2D> DepthStencilTexture;
    static CD3D11_TEXTURE2D_DESC DepthStencilTextureDesc;

    static ComPtr<ID3D11ShaderResourceView> ShaderResourceView;
    static ComPtr<ID3D11Texture2D> DepthStencilBuffer;
    static ComPtr<ID3D11UnorderedAccessView> UnorderedAccessView;

    // 1 - 3
    if(BindDSV != nullptr && pDepthStencilView == BindDSV && CudaArray.Memory == nullptr)
    {
        LOG("setup");
        GetTextureFromView(DepthStencilView, DepthStencilTexture, &DepthStencilTextureDesc);
        SetupComputeShaderResources(DeviceContext, DepthStencilView, ShaderResourceView, DepthStencilBuffer, UnorderedAccessView);
        CreateComputeShader(Device, ComputeShader);
        CudaArray = CudaD3D11TextureArray(DepthStencilBuffer, "DepthBuffer");
    }

    // 4 - 5
    if(pDepthStencilView == SentinelDSV && !DepthStencilStateDesc.DepthEnable && CudaArray.Memory != nullptr)
    { 
        //DEBUG_VERTEXSHADER(Device, DeviceContext);
        RunComputeShader
        (
            DeviceContext, 
            ComputeShader,
            ShaderResourceView, 
            UnorderedAccessView, 
            (DepthStencilTextureDesc.Width + 32) / 32,
            (DepthStencilTextureDesc.Height + 32) / 32
        );
        CudaArray.Update();
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
    static MemoryMappedFile<float> NearClipFarClip(2, "NearClipFarClip");

    GetDeviceAndContextFromSwapChain(chain);
    DidSwapChainUpdate(SwapChainDesc.OutputWindow);

    //DeviceContext -> VSGetConstantBuffers(0, )

    NearClipFarClip[0] = CAM::_0xD0082607100D7193(); // NearClip
    NearClipFarClip[1] = CAM::_0xDFC8CBC606FDB0FC(); // FarClip

    //LOG()

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
    static CudaD3D11TextureArray RenderTargetArray;
    static ComPtr<ID3D11RenderTargetView> RenderTargetView;
    static ComPtr<ID3D11Resource> RenderTargetResource;
    static ComPtr<ID3D11Texture2D> RenderTargetTexture;

    DeviceContext->OMGetRenderTargets(1, RenderTargetView.GetAddressOf(), NULL);

    if (RenderTargetTexture == nullptr)
    {
        GetTextureFromView(RenderTargetView, RenderTargetTexture);
        RenderTargetArray = CudaD3D11TextureArray(RenderTargetTexture, "RGB");
    }
    else
    {
        RenderTargetArray.Update();
    }
    //DEBUG_SCREEN_COORDINATES();
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

