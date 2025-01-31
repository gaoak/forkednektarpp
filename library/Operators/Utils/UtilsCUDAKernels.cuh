///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsCUDAKernels.cuh
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

namespace Nektar
{

template <typename TData>
__global__ void interleaveKernel(const unsigned int VectorWidth,
                                 const unsigned int npts, TData *buffer,
                                 TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = npts * VectorWidth * metaBlock;

    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    localBarrier<NektarSpaces::CUDA>(CUDAblock());

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        unsigned int vecElem = idx % VectorWidth;
        unsigned int iElem   = idx / VectorWidth;
        inout[offset + idx]  = buffer[offset + vecElem * npts + iElem];
    }
}

template <typename TData>
__global__ void deInterleaveKernel(const unsigned int VectorWidth,
                                   const unsigned int npts, TData *buffer,
                                   TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = npts * VectorWidth * metaBlock;

    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    localBarrier<NektarSpaces::CUDA>(CUDAblock());

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        unsigned int vecElem = idx / npts;
        unsigned int iElem   = idx % npts;
        inout[offset + idx]  = buffer[offset + iElem * VectorWidth + vecElem];
    }
}

template <typename TData>
__global__ void BuildInterleaveMapKernel(const unsigned int npts,
                                         const unsigned int newVecWidth,
                                         const unsigned int offset,
                                         TData *deInterleaveMapPtr,
                                         TData *interleaveMapPtr, TData *buffer)
{
    const unsigned int metaBlock   = blockIdx.x;
    const unsigned int groupOffset = npts * newVecWidth * metaBlock;

    const unsigned int idx0   = threadIdx.x;
    const unsigned int stride = blockDim.x;

    for (unsigned int idx = idx0; idx < npts * newVecWidth; idx += stride)
    {
        buffer[groupOffset + idx] = offset + groupOffset + idx;
    }

    localBarrier<NektarSpaces::CUDA>(CUDAblock());

    for (unsigned int idx = idx0; idx < npts * newVecWidth; idx += stride)
    {
        unsigned int vecElem = idx % newVecWidth;
        unsigned int iElem   = idx / newVecWidth;
        deInterleaveMapPtr[groupOffset + idx] =
            buffer[groupOffset + vecElem * npts + iElem];
    }

    localBarrier<NektarSpaces::CUDA>(CUDAblock());

    for (unsigned int idx = idx0; idx < npts * newVecWidth; idx += stride)
    {
        interleaveMapPtr[deInterleaveMapPtr[groupOffset + idx]] =
            offset + groupOffset + idx;
    }
}

template <size_t VectorWidth, typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
interleave(const unsigned int numMetaBlocks, const unsigned int npts,
           TData *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
    cudaMalloc(&buffer, bufferSize);

    interleaveKernel<<<gridSize, blockSize>>>(VectorWidth, npts, buffer, inout);

    cudaFree(buffer);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleaveKernel<<<gridSize, blockSize>>>(VectorWidth, npts, buffer,
                                                inout);

    cudaFree(buffer);
}

template <typename ExecSpace>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
BuildInterleaveMap(const unsigned int numMetaBlocks, const unsigned int npts,
                   const unsigned int newVecWidth, const unsigned int offset,
                   int *deInterleaveMapPtr, int *interleaveMapPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * newVecWidth * numMetaBlocks * npts;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    BuildInterleaveMapKernel<<<gridSize, blockSize>>>(npts, newVecWidth, offset,
                                                      deInterleaveMapPtr,
                                                      interleaveMapPtr, buffer);

    cudaFree(buffer);
}

} // namespace Nektar

#endif
