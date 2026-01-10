#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>


#include <system_error>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <format>


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
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>

#include "detours.h"
#include "scripthookv_sdk/inc/natives.h"
#include "scripthookv_sdk/inc/main.h"
#include "launchDebugger.h"
