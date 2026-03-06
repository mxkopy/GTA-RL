#pragma once

//#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files

#include <windows.h>
#include <string>
#include <array>
#include <span>
#include <vector>
#include <set>
#include <unordered_map>
#include <type_traits>
#include <system_error>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <format>
#include <random>
#include <cstddef>
#include <mutex>
#include <assert.h>
#include <dxgi.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <d3d11shader.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <driver_types.h>
#include <wrl/client.h>
#include <iomanip>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Geometry>
#include "detours/detours.h"
#include "launch_debugger.h"
#include "amalgamated.pb.h"
#include "natives.h"

#ifdef DEBUG
static std::ofstream logfile("GTA-RL.log");
#define LOG(msg) logfile << msg << std::endl
#else
#define LOG(msg)
#endif

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

using Microsoft::WRL::ComPtr;
using std::string;
using std::set;
using std::unordered_map;
using Eigen::Vector3f;
using Eigen::Matrix3f;
using std::mutex;

static ComPtr<IDXGISwapChain> SwapChain;
static DXGI_SWAP_CHAIN_DESC SwapChainDesc;
static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> DeviceContext;
static void** DeviceContextVirtualTable;