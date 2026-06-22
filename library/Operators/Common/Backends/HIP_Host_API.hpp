///////////////////////////////////////////////////////////////////////////////
//
// File: HIP_Host_API.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#define CHECK_LAST_HIPCUDA_ERROR()                                             \
    {                                                                          \
        hipError_t err = hipGetLastError();                                    \
        if (err != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "HIP Runtime Error at: " << __FILE__ << ":"           \
                      << __LINE__ << std::endl;                                \
            std::cerr << hipGetErrorString(err) << std::endl;                  \
            exit(0);                                                           \
        }                                                                      \
    }
#define CHECK_HIPCUDA_ERROR(err)                                               \
    if (err != hipSuccess)                                                     \
    {                                                                          \
        std::cerr << "HIP Runtime Error at: " << __FILE__ << ":" << __LINE__   \
                  << std::endl;                                                \
        std::cerr << hipGetErrorString(err) << std::endl;                      \
        exit(0);                                                               \
    }
// Helper to check HIP driver errors
#define CHECK_HIPCUDA_DRIVER_ERROR(err)                                        \
    if (err != hipSuccess)                                                     \
    {                                                                          \
        std::cerr << "HIP Driver API Error at: " << __FILE__ << ":"            \
                  << __LINE__ << std::endl;                                    \
        std::cerr << hipGetErrorString(err) << std::endl;                      \
        exit(0);                                                               \
    }
// Helper to check HIPRTC errors
#define CHECK_NEKRTC_ERROR(err)                                                \
    {                                                                          \
        if (err != HIPRTC_SUCCESS)                                             \
        {                                                                      \
            std::cerr << "HIP Runtime Error at: " << __FILE__ << ":"           \
                      << __LINE__ << std::endl;                                \
            std::cerr << "HIPRTC error: " << hiprtcGetErrorString(err)         \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    }
#define nekLaunchKernel hipModuleLaunchKernel
#define nekModuleLoadData hipModuleLoadData
#define nekModuleGetFunction hipModuleGetFunction
#define nekModuleUnload hipModuleUnload
#define NEKdevice hipDevice_t
#define NEKmodule hipModule_t
#define NEKfunction hipFunction_t
#define nekrtcProgram hiprtcProgram
#define nekrtcCreateProgram hiprtcCreateProgram
#define nekrtcDestroyProgram hiprtcDestroyProgram
#define nekrtcAddNameExpression hiprtcAddNameExpression
#define nekrtcGetLoweredName hiprtcGetLoweredName
#define nekrtcCompileProgram hiprtcCompileProgram
#define nekrtcGetCodeSize hiprtcGetCodeSize
#define nekrtcGetCode hiprtcGetCode

#if defined(NEKTAR_ENABLE_HIP)
template <unsigned int ndim> class hipcudaBlock
{
};

// Kernel launcher on a one-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// hipcudaBlock<1>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_1DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAM, ...)                             \
    {                                                                          \
        unsigned char *shmemptr = nullptr;                                     \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, SHMEMSIZE, STREAM>>>(                    \
            __VA_ARGS__, shmemptr, hipcudaBlock<1>());                         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a two-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// hipcudaBlock<2>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_2DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZEX, GRIDSIZEY,            \
                                      BLOCKSIZEX, BLOCKSIZEY, SHMEMSIZE,       \
                                      STREAM, ...)                             \
    {                                                                          \
        unsigned char *shmemptr = nullptr;                                     \
        dim3 GRIDSIZE(GRIDSIZEX, GRIDSIZEY, 1);                                \
        dim3 BLOCKSIZE(BLOCKSIZEX, BLOCKSIZEY, 1);                             \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, SHMEMSIZE, STREAM>>>(                    \
            __VA_ARGS__, shmemptr, hipcudaBlock<2>());                         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a three-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// hipcudaBlock<3>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_3DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ, \
                                      BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ,      \
                                      SHMEMSIZE, STREAM, ...)                  \
    {                                                                          \
        unsigned char *shmemptr = nullptr;                                     \
        dim3 GRIDSIZE(GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ);                        \
        dim3 BLOCKSIZE(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);                    \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, SHMEMSIZE, STREAM>>>(                    \
            __VA_ARGS__, shmemptr, hipcudaBlock<3>());                         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a one-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type hipcudaBlock<1>.
#define DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAM, ...)                     \
    {                                                                          \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, 0, STREAM>>>(__VA_ARGS__,                \
                                                   hipcudaBlock<1>());         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a two-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type hipcudaBlock<2>.
#define DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(                                 \
    KERNEL, GRIDSIZEX, GRIDSIZEY, BLOCKSIZEX, BLOCKSIZEY, STREAM, ...)         \
    {                                                                          \
        dim3 GRIDSIZE(GRIDSIZEX, GRIDSIZEY, 1);                                \
        dim3 BLOCKSIZE(BLOCKSIZEX, BLOCKSIZEY, 1);                             \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, 0, STREAM>>>(__VA_ARGS__,                \
                                                   hipcudaBlock<2>());         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }

// Kernel launcher on a three-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type hipcudaBlock<3>.
#define DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(                                 \
    KERNEL, GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ, BLOCKSIZEX, BLOCKSIZEY,           \
    BLOCKSIZEZ, STREAM, ...)                                                   \
    {                                                                          \
        dim3 GRIDSIZE(GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ);                        \
        dim3 BLOCKSIZE(BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ);                    \
        KERNEL<<<GRIDSIZE, BLOCKSIZE, 0, STREAM>>>(__VA_ARGS__,                \
                                                   hipcudaBlock<3>());         \
        CHECK_LAST_HIPCUDA_ERROR();                                            \
    }
#endif
