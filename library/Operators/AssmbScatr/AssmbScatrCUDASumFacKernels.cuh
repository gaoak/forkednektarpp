///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrCUDASumFacKernels.cuh
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
__global__ void AssembleKernel(const unsigned int nsize,
                               const unsigned int offset,
                               const int *__restrict__ assmbPtr,
                               const TData *__restrict__ signPtr,
                               const TData *__restrict__ inPtr,
                               TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        unsigned int index = offset + i;
        atomicAdd(outPtr + assmbPtr[index], signPtr[index] * inPtr[index]);
        i += stride;
    }
}

template <typename TData>
__global__ void AssembleKernel(const unsigned int nsize,
                               const unsigned int offset,
                               const int *__restrict__ assmbPtr,
                               const TData sign,
                               const TData *__restrict__ inPtr,
                               TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        unsigned int index = offset + i;
        atomicAdd(outPtr + assmbPtr[index], sign * inPtr[index]);
        i += stride;
    }
}

template <typename TData>
__global__ void AssembleKernel(const unsigned int nsize,
                               const unsigned int offset,
                               const int *__restrict__ assmbPtr,
                               const TData *__restrict__ inPtr,
                               TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        unsigned int index = offset + i;
        atomicAdd(outPtr + assmbPtr[index], inPtr[index]);
        i += stride;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int nsize,
                                    const unsigned int offset,
                                    const int *__restrict__ assmbPtr,
                                    const TData *__restrict__ signPtr,
                                    const TData *__restrict__ inPtr,
                                    TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        unsigned int index = offset + i;
        outPtr[index]      = signPtr[index] * inPtr[assmbPtr[index]];
        i += stride;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int nsize,
                                    const unsigned int offset,
                                    const int *__restrict__ assmbPtr,
                                    const TData sign,
                                    const TData *__restrict__ inPtr,
                                    TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        unsigned int index = offset + i;
        outPtr[index]      = sign * inPtr[assmbPtr[index]];
        i += stride;
    }
}

template <typename TData>
__global__ void GlobalToLocalKernel(const unsigned int nsize,
                                    const unsigned int offset,
                                    const int *__restrict__ assmbPtr,
                                    const TData *__restrict__ inPtr,
                                    TData *__restrict__ outPtr)
{
    unsigned int i      = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while (i < nsize)
    {
        unsigned int index = offset + i;
        outPtr[index]      = inPtr[assmbPtr[index]];
        i += stride;
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const unsigned int offset,
                   const int *assmbPtr, const TData *signPtr,
                   const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernel<TData><<<gridSize, blockSize>>>(nsize, offset, assmbPtr,
                                                   signPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const unsigned int offset,
                   const int *assmbPtr, const TData sign, const TData *inPtr,
                   TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernel<TData>
        <<<gridSize, blockSize>>>(nsize, offset, assmbPtr, sign, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    AssembleKernel(const unsigned int nsize, const unsigned int offset,
                   const int *assmbPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AssembleKernel<TData>
        <<<gridSize, blockSize>>>(nsize, offset, assmbPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const unsigned int offset,
                        const int *assmbPtr, const TData *signPtr,
                        const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernel<TData><<<gridSize, blockSize>>>(nsize, offset, assmbPtr,
                                                        signPtr, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const unsigned int offset,
                        const int *assmbPtr, const TData sign,
                        const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernel<TData>
        <<<gridSize, blockSize>>>(nsize, offset, assmbPtr, sign, inPtr, outPtr);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    GlobalToLocalKernel(const unsigned int nsize, const unsigned int offset,
                        const int *assmbPtr, const TData *inPtr, TData *outPtr)
{
    const unsigned int blockSize = 256u;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    GlobalToLocalKernel<TData>
        <<<gridSize, blockSize>>>(nsize, offset, assmbPtr, inPtr, outPtr);
}

} // namespace Nektar::Operators::detail

#endif
