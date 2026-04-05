///////////////////////////////////////////////////////////////////////////////
//
// File: CUDA_HOST_API.hpp
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#define CHECK_LAST_HIPCUDA_ERROR()                                             \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                                \
            std::cerr << cudaGetErrorString(err) << std::endl;                 \
            exit(0);                                                           \
        }                                                                      \
    }
#define CHECK_HIPCUDA_ERROR(err)                                               \
    if (err != cudaSuccess)                                                    \
    {                                                                          \
        std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":" << __LINE__  \
                  << std::endl;                                                \
        std::cerr << cudaGetErrorString(err) << std::endl;                     \
        exit(0);                                                               \
    }
// Helper to check CUDA driver errors
#define CHECK_HIPCUDA_DRIVER_ERROR(err)                                        \
    {                                                                          \
        if (err != CUDA_SUCCESS)                                               \
        {                                                                      \
            const char *errStr;                                                \
            cuGetErrorString(err, &errStr);                                    \
            fprintf(stderr, "CUDA Driver API Error: %s %s %d\n", errStr,       \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    }
// Helper to check NVRTC errors
#define CHECK_NEKRTC_ERROR(err)                                                \
    {                                                                          \
        if (err != NVRTC_SUCCESS)                                              \
        {                                                                      \
            std::cerr << "CUDA Runtime Error at: " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;                                \
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(err)           \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
    }
#define nekCtxGetCurrent cuCtxGetCurrent
#define nekLaunchKernel cuLaunchKernel
#define nekModuleLoadData cuModuleLoadData
#define nekModuleGetFunction cuModuleGetFunction
#define nekModuleUnload cuModuleUnload
#define NEKdevice CUdevice
#define NEKmodule CUmodule
#define NEKfunction CUfunction
#define NEKcontext CUcontext
#define nekrtcProgram nvrtcProgram
#define nekrtcCreateProgram nvrtcCreateProgram
#define nekrtcDestroyProgram nvrtcDestroyProgram
#define nekrtcAddNameExpression nvrtcAddNameExpression
#define nekrtcGetLoweredName nvrtcGetLoweredName
#define nekrtcCompileProgram nvrtcCompileProgram
#define nekrtcGetCodeSize nvrtcGetPTXSize
#define nekrtcGetCode nvrtcGetPTX
