// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "framework.h"
#include "scripthookv_sdk/inc/main.h"

using Microsoft::WRL::ComPtr;
using std::string;
using std::unordered_map;
using Vector3f = Eigen::Vector3f;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Util



static std::ofstream logfile("dxinterop.log");

#define STRINGIFY(x) #x

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


namespace std {

    template<>
    struct hash<std::pair<UINT, UINT>>
    {
        size_t operator()(const std::pair<UINT, UINT>& P) const noexcept
        {
            auto L = std::hash<size_t>{}(P.first);
            auto R = std::hash<size_t>{}(P.second);
            return L ^ (R << 1);
        }
    };

    template <typename T>
    struct hash<MemoryMappedFile<T>>
    {
        size_t operator()(const MemoryMappedFile<T>& F) const noexcept
        {
            return std::hash<size_t>(F.File);
        }
    };
}


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

struct CameraTransforms
{
    inline static ComPtr<ID3D11Buffer> LastMatrixBuffer;
    inline static ComPtr<ID3D11Buffer> CurrentMatrixBuffer;

    inline static ComPtr<ID3D11Buffer> VSConstantBuffer;
    inline static ComPtr<ID3D11Buffer> VSConstantStagingBuffer;

    inline static const float MAGIC = 0.75;
    inline static MemoryMappedFile<float> scale = MemoryMappedFile<float>("scale");

    static void SetupVertexConstantBuffer()
    {
        ComPtr<ID3D11DeviceContext> DeviceContext;
        Device->GetImmediateContext(&DeviceContext);
        DeviceContext->VSGetConstantBuffers(2, 1, &VSConstantBuffer);
        D3D11_BUFFER_DESC BufferDesc;
        VSConstantBuffer->GetDesc(&BufferDesc);
        BufferDesc.Usage = D3D11_USAGE_STAGING;
        BufferDesc.BindFlags = NULL;
        BufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        Device->CreateBuffer(&BufferDesc, NULL, VSConstantStagingBuffer.GetAddressOf());
    }

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

    
    struct Ray
    {
        inline static float NearClip;
        inline static float FarClip;
        inline static float SW; // Screen Width
        inline static float SH; // Screen Height
        inline static float VW; // Viewport Width
        inline static float VH; // Viewport Height
        inline static Eigen::Matrix3f Axes;
        inline static Eigen::Vector3f P; // Camera Position
        using  PixelCoord = std::pair<UINT, UINT>;
        inline static std::unordered_map<PixelCoord, MemoryMappedFile<float>> RayData;


        UINT C, R;

        Eigen::Vector3f ComputeDirection() const
        {
            float X = 2 * (float(C) / VW) - (SW / VW);
            float Y = 2 * (float(R) / VH) - (SH / VH);
            float Z = 1;
            return Axes * Eigen::Vector3f(X, Y, Z);
        }

        void ComputeCollision() const
        {
            auto V = 1000.0f * ComputeDirection();
            Vector3 Collision;
            BOOL Hit;
            Vector3 Normal;
            Entity EntityHit;
            auto RaycastHandle = WORLDPROBE::_CAST_RAY_POINT_TO_POINT(P[0], P[1], P[2], P[0] + V[0], P[1] + V[1], P[2] + V[2], 511, NULL, 7);
            WORLDPROBE::_GET_RAYCAST_RESULT(RaycastHandle, &Hit, &Collision, &Normal, &EntityHit);
            float CollisionFloats[3] = { Collision.x, Collision.y, Collision.z };
            std::memcpy(RayData[PixelCoord(C, R)].File, CollisionFloats, sizeof(CollisionFloats) );
        }

        Ray(UINT C, UINT R) : C(C), R(R)
        {
            PixelCoord Coordinate = { C, R };
            string Tagname = "Ray" + std::to_string(C) + '_' + std::to_string(R);
            if (!RayData.contains(Coordinate)) RayData[Coordinate] = MemoryMappedFile<float>(3, Tagname);
            ComputeCollision();
        }

        static void Update()
        {
            Ray(Ray::SW / 4, Ray::SH / 4);
            Ray((Ray::SW / 4) + (Ray::SW / 2), Ray::SH / 4);
            Ray(Ray::SW / 4, (Ray::SH / 4) + (Ray::SH / 2));
            Ray((Ray::SW / 4) + (Ray::SW / 2), (Ray::SH / 4) + (Ray::SH / 2));
        }
    };

    static void Update()
    {   
        ComPtr<ID3D11DeviceContext> DeviceContext;
        Device->GetImmediateContext(&DeviceContext);

        // Get VertexShader constants (camera position & orientation, perspective matrix, etc)
        DeviceContext->VSGetConstantBuffers(2, 1, &VSConstantBuffer);
        // Copy VertexShader constants into CPU-accessible buffer
        DeviceContext->CopyResource(VSConstantStagingBuffer.Get(), VSConstantBuffer.Get());
        // Update previous matrix
        DeviceContext->CopyResource(LastMatrixBuffer.Get(), CurrentMatrixBuffer.Get());
        DeviceContext->Flush();

        // Access VertexShader constants via CPU
        D3D11_MAPPED_SUBRESOURCE VertexShaderSubresource;
        D3D11_MAPPED_SUBRESOURCE CurrentMatrixSubresource;
        DeviceContext->Map(VSConstantStagingBuffer.Get(), 0, D3D11_MAP_READ, NULL, &VertexShaderSubresource);
        DeviceContext->Map(CurrentMatrixBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, NULL, &CurrentMatrixSubresource);
        
        float* VSData = (float*)VertexShaderSubresource.pData;
        float* MXData = (float*)CurrentMatrixSubresource.pData;
        
        Eigen::Map<Vector3f> R(VSData + 28);
        Eigen::Map<Vector3f> L(VSData + 32);
        Eigen::Map<Vector3f> D(VSData + 36);
        Eigen::Map<Vector3f> U(VSData + 40);

        Eigen::Map<Vector3f> X(MXData + 0);
        Eigen::Map<Vector3f> Y(MXData + 4);
        Eigen::Map<Vector3f> Z(MXData + 8);
        Eigen::Map<Vector3f> P(MXData + 12);

        // Update current matrix & other relevant items 
        X = (R - L).stableNormalized();
        Y = (U - D).stableNormalized();
        Z = (R + L).stableNormalized();
        P = Eigen::Map<Vector3f>(VSData + 12);

        float& ScreenWidth = VSData[60];
        float& ScreenHeight = VSData[61];
        float NearClip = CAM::_0xD0082607100D7193();
        float FarClip = CAM::_0xDFC8CBC606FDB0FC();

        MXData[32] = NearClip;
        MXData[33] = FarClip;
        MXData[34] = ScreenWidth;
        MXData[35] = ScreenHeight;
        MXData[36] = VSData[20]; // Viewport width (px)
        MXData[37] = VSData[21]; // Viewport height (px)
        MXData[38] = scale[0];

        Ray::Axes << X, Y, Z;
        Ray::P << P;
        Ray::NearClip = NearClip;
        Ray::FarClip = FarClip;
        Ray::SW = ScreenWidth;
        Ray::SH = ScreenHeight;
        Ray::VW = VSData[20];
        Ray::VH = VSData[21];
        Ray::Update();

        // End CPU access
        DeviceContext->Unmap(VSConstantStagingBuffer.Get(), 0);
        DeviceContext->Unmap(LastMatrixBuffer.Get(), 0);
        DeviceContext->Flush();
    }

    CameraTransforms() = default;

    CameraTransforms(ComPtr<ID3D11Device> Device)
    {
        SetupVertexConstantBuffer();
        SetupMatrixBuffers();
    }

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

    void SetupDepthTexture
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

    void SetupDepthSRV
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

    
    void SetupTextureUAV
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


    void SetupResources(ComPtr<ID3D11DepthStencilView>& DepthStencilView)
    {
        ComPtr<ID3D11Texture2D> DepthStencilTexture;
        D3D11_TEXTURE2D_DESC DepthStencilTextureDesc;
        GetTextureFromView(DepthStencilView, DepthStencilTexture, &DepthStencilTextureDesc);
        SetupDepthTexture(DepthStencilTexture, DepthStencilTextureDesc);
        SetupDepthSRV(DepthStencilTexture, DepthStencilTextureDesc);
        SetupTextureUAV(VelocityDepthUAV, VelocityDepthTexture);
    }

    void CreateComputeShader()
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

    void RunComputeShader()
    {
        static const size_t N_CBF = 2;
        static const size_t N_SRV = 2;
        static const size_t N_UAV = 1;
        ComPtr<ID3D11DeviceContext> DeviceContext;
        Device->GetImmediateContext(&DeviceContext);
        Camera.Update();
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
        DepthStencilComputeShader::Camera = CameraTransforms(Device);
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


using ClearDepthStencilViewFunction = void(__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
static ClearDepthStencilViewFunction ClearDepthStencilView = NULL;

const unsigned int NUM_VSB = 3;

static void DEBUG_ARROWS
(

    MemoryMappedFile<float> VSB[NUM_VSB]

) {
    const static int NumberOfArrows = 4;
    static MemoryMappedFile<float> DebugArrows[NumberOfArrows];
    static MemoryMappedFile<int> ArrowColors[NumberOfArrows];
    static bool Initialized = false;
    if (!Initialized)
    {
        for (int i = 0; i < NumberOfArrows; i++)
        {
            DebugArrows[i] = MemoryMappedFile<float>(6, "DebugArrow" + std::to_string(i));
            ArrowColors[i] = MemoryMappedFile<int>(4, "DebugArrowColors" + std::to_string(i));
        }
        Initialized = true;
    }

    for (int i = 0; i < NumberOfArrows; i++)
    {
        GRAPHICS::DRAW_LINE(
            DebugArrows[i][0],
            DebugArrows[i][1],
            DebugArrows[i][2],
            DebugArrows[i][0] + DebugArrows[i][3],
            DebugArrows[i][1] + DebugArrows[i][4],
            DebugArrows[i][2] + DebugArrows[i][5],
            ArrowColors[i][0],
            ArrowColors[i][1],
            ArrowColors[i][2],
            ArrowColors[i][3]
        );
    }
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
            BufferDesc.BindFlags = NULL;
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

    DEBUG_ARROWS(VSBFiles);

}

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

    static CudaD3D11TextureArray CudaArray;
    static CudaD3D11TextureArray DeltaArray;
    static DepthStencilComputeShader ComputeShader;

    // 1 - 3
    if(BindDSV != nullptr && pDepthStencilView == BindDSV && CudaArray.Memory == nullptr)
    {
        LOG("setup");
        //LaunchDebugger();
        //DebugBreak();
        ComputeShader = DepthStencilComputeShader(DepthStencilView);
        CudaArray = CudaD3D11TextureArray(DepthStencilComputeShader::VelocityDepthTexture, "DepthBuffer");
    }

    static MemoryMappedFile<bool> UpdateSignal = MemoryMappedFile<bool>("RayCastUpdate");

    // 4 - 5
    if(pDepthStencilView == SentinelDSV && !DepthStencilStateDesc.DepthEnable && CudaArray.Memory != nullptr)
    {
        if (UpdateSignal[0]) {
            GAMEPLAY::SET_GAME_PAUSED(true);
            DEBUG_VERTEXSHADER(Device, DeviceContext);
            ComputeShader.RunComputeShader();
            CudaArray.Update();
            GAMEPLAY::SET_GAME_PAUSED(false);
            UpdateSignal[0] = false;
        }
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

