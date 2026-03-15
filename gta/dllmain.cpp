// dllmain.cpp : Defines the entry point for the DLL application.

#include "framework.h"
#include "ipc.h"
#include "vertex_buffers.h"
#include "cuda_ipc.h"
#include "graphics_debug.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Util

static void GetDeviceAndContextFromSwapChain(void* chain) {
    SwapChain = (IDXGISwapChain*) chain;
    ERR(SwapChain->GetDesc(&SwapChainDesc));
    ERR(SwapChain->GetDevice(__uuidof(ID3D11Device), &Device));
    Device->GetImmediateContext(&DeviceContext);
    DeviceContextVirtualTable = (void**)*(void**)DeviceContext.Get();
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
        auto A = (Matrix3f&) VSConstants::Axes;
        auto P = (Eigen::Map<Vector3f>) VSConstants::P;

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

        if (HR != S_OK) LOG(std::string((char*)ErrorBlob->GetBufferPointer(), ErrorBlob->GetBufferSize()));
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

#ifdef _WINDLL
#include "game.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Hooks


// Unfortunately there doesn't seem to be a straightforward way to read the depth stencil texture directly into cuda memory.
// (see the section for cudaGraphicsD3D11RegisterResource in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html)
// So we have to have a 'CUDA-staging texture' between the depth stencil texture w/ a CUDA-compatible format.
// We'll write to this texture using a compute shader, whose inputs we set up here.
// The input to the compute shader is an SRV (Shader Resource View) bound to the depth stencil's backing texture,
// and the output is a UAV (Unordered Access View) bound to the CUDA-staging texture. 


using ClearDepthStencilViewFunction = void(__thiscall*)(ID3D11DeviceContext*, ID3D11DepthStencilView*, UINT, FLOAT, UINT8);
static ClearDepthStencilViewFunction ClearDepthStencilView = NULL;

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

    // After we've got BindDSV we can start setting up all the IPC-related memory
    // For whatever reason, CUDA doesn't support mapping the depth buffer so there's
    // a bit of a process:
    // 
    // 1. Get the depth stencil view's texture and bind it to a shader resource view
    // 2. Create the CUDA-staging texture and bind it to an unordered access view 
    // 3. Allocate CUDA memory & create its IPC memory handle
    // 4. When appropriate, run the compute shader to write to (2) and map & copy (2) to (3) via CUDA

    static DepthStencilComputeShader ComputeShader;
    static CudaD3D11TextureArray CUDADepthArray;
    static bool Initialized;

    // 1 - 3
    if(BindDSV != nullptr && pDepthStencilView == BindDSV && CUDADepthArray.cuMemory == nullptr)
    {
        LOG("CUDA IPC setup");
        ComputeShader = DepthStencilComputeShader(DepthStencilView);
        CUDADepthArray = CudaD3D11TextureArray(DepthStencilComputeShader::VelocityDepthTexture);
        CUDADepthArray.Publish("Depth");
        CameraTransforms::SetupMatrixBuffers();
    }

    // 4
    if(pDepthStencilView == SentinelDSV && !DepthStencilStateDesc.DepthEnable && CUDADepthArray.cuMemory != nullptr)
    {
        //GAMEPLAY::SET_GAME_PAUSED(true);
        VSConstants::Update(DeviceContext);
        CameraTransforms::Update(DeviceContext);
        DepthStencilComputeShader::RunComputeShader(DeviceContext);
        CUDADepthArray.Update();
        //if( FLAGS.GetFlag(BEGIN_TRAINING) ){
        //    Ray::Update();
        //    FLAGS.SetFlag(RAYCASTS, true);
        //    FLAGS.WaitUntil(RAYCASTS, false);
        //}
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
    GetDeviceAndContextFromSwapChain(chain);

    // DX11 will launch an amortized version of ClearDepthStencilView once in a while
    // and for whatever reason we do not want to hook that
    // It happens only occasionally, so it seems we can reliably detect if we have
    // the correct vtable entry by just checking that it hasn't changed 

    static void* LastVTEntry;
    static bool FoundPrimaryVTEntry;
    static bool Hooked;

    if (!Hooked)
    {
        DeviceContext.As(&MultithreadContext);
        MultithreadContext->SetMultithreadProtected(true);
        DeviceContextVirtualTable = (void**)*(void**)DeviceContext.Get();
        if (!FoundPrimaryVTEntry && LastVTEntry == DeviceContextVirtualTable[53])
        {
            HookClearDepthStencilView();
            Hooked = true;
            FoundPrimaryVTEntry = true;
        }
    }

    LastVTEntry = DeviceContextVirtualTable[53];


    // RTV hook related stuff (i.e.  getting displayed pixels)
    static CudaD3D11TextureArray RenderTargetArray;
    static ComPtr<ID3D11RenderTargetView> RenderTargetView;
    static ComPtr<ID3D11Resource> RenderTargetResource;
    static ComPtr<ID3D11Texture2D> RenderTargetTexture;

    DeviceContext->OMGetRenderTargets(1, RenderTargetView.GetAddressOf(), NULL);

    if (RenderTargetTexture == nullptr)
    {
        GetTextureFromView(RenderTargetView, RenderTargetTexture);
        RenderTargetArray = CudaD3D11TextureArray(RenderTargetTexture);
        RenderTargetArray.Publish("RGB");
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
        scriptRegister(hModule, ScriptMain);
        break;
    case DLL_PROCESS_DETACH:
        scriptUnregister(ScriptMain);
        presentCallbackUnregister(presentCallback);
        CUDAPitchedArray::FreeAll();
        break;
    }
    return TRUE;
}
#else

int main(int argc, char* argv[])
{
    cudaChannelFormatDesc Format = {8, 8, 8, 8, cudaChannelFormatKind::cudaChannelFormatKindFloat};
    cudaExtent Extent = { 100, 100, 1 };
    UINT BPP = 4;

    IPCCUDAArray CUDAArray(Format, Extent, BPP, "Test");

    StructuredMemory<Vec3f> SM("VectorTest");
    Vec3f V;
    V.set_x(1.0);
    V.set_y(2.0);
    V.set_z(3.0);
    SM = V;
    auto U = static_cast<Vec3f>(SM);
    std::cout << U.x() << " " << U.y() << " " << U.z() << std::endl;

    RequestLockedMemory<Vec3f, 1> Reader("Test");
    while (true)
    {
        Vec3f V = static_cast<Vec3f>(Reader);
        std::cout << V.x() << std::endl;
    }

}
#endif