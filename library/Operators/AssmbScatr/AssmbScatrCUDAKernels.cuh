///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrCUDAKernels.cuh
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void AssembleKernel(const unsigned int nsize,
                               const int *__restrict__ assmbPtr,
                               const TData *__restrict__ signPtr,
                               const TData *__restrict__ inPtr,
                               TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        atomic_add<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
            outPtr + assmbPtr[i], signPtr[i] * inPtr[i]);
        i += stride;
    }
}

template <typename TData>
__global__ void AssembleKernel(const unsigned int nsize,
                               const int *__restrict__ assmbPtr,
                               const TData sign,
                               const TData *__restrict__ inPtr,
                               TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        atomic_add<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
            outPtr + assmbPtr[i], sign * inPtr[i]);
        i += stride;
    }
}

template <typename TData>
__global__ void AssembleKernel(const unsigned int nsize,
                               const int *__restrict__ assmbPtr,
                               const TData *__restrict__ inPtr,
                               TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        atomic_add<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(
            outPtr + assmbPtr[i], inPtr[i]);
        i += stride;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int nsize,
                                    const int *__restrict__ assmbPtr,
                                    const TData *__restrict__ signPtr,
                                    const TData *__restrict__ inPtr,
                                    TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        outPtr[i] = signPtr[i] * inPtr[assmbPtr[i]];
        i += stride;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int nsize,
                                    const int *__restrict__ assmbPtr,
                                    const TData sign,
                                    const TData *__restrict__ inPtr,
                                    TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        outPtr[i] = sign * inPtr[assmbPtr[i]];
        i += stride;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int nsize,
                                    const int *__restrict__ assmbPtr,
                                    const TData *__restrict__ inPtr,
                                    TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        outPtr[i] = inPtr[assmbPtr[i]];
        i += stride;
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData *signPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernel<TData>
        <<<gridSize, blockSize>>>(nsize, assmbPtr, signPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData sign, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernel<TData>
        <<<gridSize, blockSize>>>(nsize, assmbPtr, sign, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const int *assmbPtr,
                   const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernel<TData>
        <<<gridSize, blockSize>>>(nsize, assmbPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData *signPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernel<TData>
        <<<gridSize, blockSize>>>(nsize, assmbPtr, signPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData sign, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernel<TData>
        <<<gridSize, blockSize>>>(nsize, assmbPtr, sign, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                        const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernel<TData>
        <<<gridSize, blockSize>>>(nsize, assmbPtr, inPtr, outPtr);
}

} // namespace Nektar::Operators::detail

#endif
