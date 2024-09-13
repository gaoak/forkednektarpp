///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondCUDAKernels.cuh
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void DirBndCondKernel(const unsigned int nsize,
                                 const int *__restrict__ mapPtr,
                                 const TData *__restrict__ inPtr,
                                 TData *__restrict__ outPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        outPtr[mapPtr[i]] = inPtr[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void DirBndCondKernel(const unsigned int nsize,
                                 const TData *__restrict__ signPtr,
                                 const int *__restrict__ mapPtr,
                                 const TData *__restrict__ inPtr,
                                 TData *__restrict__ outPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        outPtr[mapPtr[i]] = signPtr[i] * inPtr[i];
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void ParallelDirBndSignKernel(const unsigned int nsize,
                                         const int *__restrict__ signPtr,
                                         TData *__restrict__ outPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        outPtr[signPtr[i]] *= -1;
        i += blockDim.x * gridDim.x;
    }
}

template <typename TData>
__global__ void LocalDirBndCondKernel(const unsigned int nsize,
                                      const int *__restrict__ id0Ptr,
                                      const int *__restrict__ id1Ptr,
                                      const TData *__restrict__ signPtr,
                                      TData *__restrict__ outPtr)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    while (i < nsize)
    {
        outPtr[id0Ptr[i]] = outPtr[id1Ptr[i]] * signPtr[i];
        i += blockDim.x * gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DirBndCondKernel(const unsigned int nsize, const int *mapPtr,
                     const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DirBndCondKernel<<<gridSize, blockSize>>>(nsize, mapPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    DirBndCondKernel(const unsigned int nsize, const TData *signPtr,
                     const int *mapPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DirBndCondKernel<<<gridSize, blockSize>>>(nsize, signPtr, mapPtr, inPtr,
                                              outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    ParallelDirBndSignKernel(const unsigned int nsize, const int *signPtr,
                             TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    ParallelDirBndSignKernel<<<gridSize, blockSize>>>(nsize, signPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    LocalDirBndCondKernel(const unsigned int nsize, const int *id0Ptr,
                          const int *id1Ptr, const TData *signPtr,
                          TData *outPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    LocalDirBndCondKernel<<<gridSize, blockSize>>>(nsize, id0Ptr, id1Ptr,
                                                   signPtr, outPtr);
}

} // namespace Nektar::Operators::detail

#endif
