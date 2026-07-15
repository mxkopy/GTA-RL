#pragma once
#include <set>
#include "framework.h"
#include "ipc.h"

// Classes for sharing GPU data across processes

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

// Wrapper around CUDA Pitched Arrays
// Stores auxiliary information (bytes-per-pixel, extent, etc) so it can be read/written by other methods in the program
// Also publishes this information via IPC so it can be read by cuda libraries in other programs 
struct CUDAPitchedArray
{
    inline static set<CUDAPitchedArray*> Instances;

    void* cuMemory = nullptr;
    size_t ByteLength = 0;
    
    operator void*& () {
        return cuMemory;
    }

    cudaChannelFormatDesc ChannelFormat{};
    cudaExtent Extent{};
    size_t Pitch{};

    inline UINT BPP()
    {
        return (ChannelFormat.x + ChannelFormat.y + ChannelFormat.z + ChannelFormat.w) / CHAR_BIT;
    }

    // Copies memory from another cudaArray into the one stored by this object
    cudaError_t CopyFrom(cudaArray_t Array)
    {
        return cudaMemcpy2DFromArray(cuMemory, Pitch, Array, 0, 0, BPP() * Extent.width, Extent.height, cudaMemcpyDefault);
    }

    void Free()
    {
        cudaFree(cuMemory);
        Instances.erase(this);
    }

    static void FreeAll()
    {
        for (auto Array : CUDAPitchedArray::Instances) Array -> Free();
    }

    // Publishes the memory segment information (bytes per pixel, width, length, etc) to an IPC channel that makes it readable by other programs  
    void Publish(string Tagname)
    {
        StructuredMemory<CUDAPitchedArrayObject> Memory(Tagname);
        CUDAPitchedArrayObject Data;
        cudaIpcMemHandle_t Handle;
        cudaIpcGetMemHandle(&Handle, cuMemory);
        Data.set_pitch(Pitch);
        Data.set_handle(&Handle, sizeof(cudaIpcMemHandle_t));
        Data.mutable_extent()->set_width(Extent.width);
        Data.mutable_extent()->set_height(Extent.height);
        Data.mutable_extent()->set_depth(Extent.depth);
        Data.mutable_format()->set_x(ChannelFormat.x);
        Data.mutable_format()->set_y(ChannelFormat.y);
        Data.mutable_format()->set_z(ChannelFormat.z);
        Data.mutable_format()->set_w(ChannelFormat.w);
        Data.mutable_format()->set_f(ChannelFormat.f);
        Memory = Data;
    }

    CUDAPitchedArray(cudaChannelFormatDesc ChannelFormat, cudaExtent Extent) : ChannelFormat(ChannelFormat), Extent(Extent)
    {
        ByteLength = BPP() * Extent.width * Extent.height;
        CUERR(cudaMallocPitch(&cuMemory, &Pitch, ByteLength / Extent.height, Extent.height));
        Instances.insert(this);
    }

    CUDAPitchedArray() = default;

};


// Boilerplate to create a CUDA array from DirectX11 Texture object, and update it when the latter changes
template <typename T>
struct D3DBackedCUDAPitchedArray : CUDAPitchedArray
{
    cudaGraphicsResource_t GraphicsResource = NULL;

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

    void Update()
    {
        cudaArray_t MappedArray;
        CUERR(cudaGraphicsMapResources(1, &GraphicsResource));
        if constexpr (std::is_same_v<T, ID3D11Texture2D>)
        {
            CUERR(cudaGraphicsSubResourceGetMappedArray(&MappedArray, GraphicsResource, 0, 0));
            CUDAPitchedArray::CopyFrom(MappedArray);
        }
        CUERR(cudaGraphicsUnmapResources(1, &GraphicsResource));
        CUERR(cudaDeviceSynchronize());
    }

    D3DBackedCUDAPitchedArray(ComPtr<T>& D3D11Object) : CUDAPitchedArray(Format(D3D11Object), Extent(D3D11Object))
    {
        CUERR(cudaGraphicsD3D11RegisterResource(&GraphicsResource, D3D11Object.Get(), cudaGraphicsRegisterFlagsNone)); // try other cudaGraphicsRegisterFlags here ?
    }

    D3DBackedCUDAPitchedArray() = default;
};

using CudaD3D11TextureArray = D3DBackedCUDAPitchedArray<ID3D11Texture2D>;
