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
                                 const unsigned int numMetaBlocks,
                                 const unsigned int npts, TData *buffer,
                                 TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = npts * VectorWidth * metaBlock;

    for (unsigned int idx = threadIdx.x; idx < npts * VectorWidth;
         idx += blockDim.x)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < npts * VectorWidth;
         idx += blockDim.x)
    {
        unsigned int vecElem = idx % VectorWidth;
        unsigned int iElem   = idx / VectorWidth;
        inout[offset + idx]  = buffer[offset + vecElem * npts + iElem];
    }
}

template <typename TData>
__global__ void deInterleaveKernel(const unsigned int VectorWidth,
                                   const unsigned int numMetaBlocks,
                                   const unsigned int npts, TData *buffer,
                                   TData *inout)
{
    const unsigned int metaBlock = blockIdx.x;
    const unsigned int offset    = npts * VectorWidth * metaBlock;

    for (unsigned int idx = threadIdx.x; idx < npts * VectorWidth;
         idx += blockDim.x)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < npts * VectorWidth;
         idx += blockDim.x)
    {
        unsigned int vecElem = idx / npts;
        unsigned int iElem   = idx % npts;
        inout[offset + idx]  = buffer[offset + iElem * VectorWidth + vecElem];
    }
}

template <typename TData>
__global__ void BuildInterleaveMapKernel(const unsigned int numMetaBlocks,
                                         const unsigned int npts,
                                         const unsigned int newVecWidth,
                                         const unsigned int offset,
                                         TData *deInterleaveMapPtr,
                                         TData *interleaveMapPtr, TData *buffer)
{
    // rewrite UtilsSYCL.hpp BuildInterleaveMapKernel into CUDA codes
    const unsigned int metaBlock   = blockIdx.x;
    const unsigned int groupOffset = npts * newVecWidth * metaBlock;

    for (unsigned int idx = threadIdx.x; idx < npts * newVecWidth;
         idx += blockDim.x)
    {
        buffer[groupOffset + idx] = offset + groupOffset + idx;
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < npts * newVecWidth;
         idx += blockDim.x)
    {
        unsigned int vecElem = idx % newVecWidth;
        unsigned int iElem   = idx / newVecWidth;
        deInterleaveMapPtr[groupOffset + idx] =
            buffer[groupOffset + vecElem * npts + iElem];
    }

    __syncthreads();

    for (unsigned int idx = threadIdx.x; idx < npts * newVecWidth;
         idx += blockDim.x)
    {
        interleaveMapPtr[deInterleaveMapPtr[groupOffset + idx]] =
            offset + groupOffset + idx;
    }
}

template <size_t VectorWidth, typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
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

    interleaveKernel<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks, npts,
                                              buffer, inout);

    cudaFree(buffer);
}

template <typename ExecSpace, typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    deInterleave(const unsigned int VectorWidth,
                 const unsigned int numMetaBlocks, const unsigned int npts,
                 TData *inout)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
    cudaMalloc(&buffer, bufferSize);

    deInterleaveKernel<<<gridSize, blockSize>>>(VectorWidth, numMetaBlocks,
                                                npts, buffer, inout);

    cudaFree(buffer);
}

template <typename ExecSpace>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::CUDA>::value,
                            void>::type
    BuildInterleaveMap(const unsigned int numMetaBlocks,
                       const unsigned int npts, const unsigned int newVecWidth,
                       const unsigned int offset, int *deInterleaveMapPtr,
                       int *interleaveMapPtr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * newVecWidth * numMetaBlocks * npts;

    int *buffer;
    cudaMalloc(&buffer, bufferSize);

    BuildInterleaveMapKernel<<<gridSize, blockSize>>>(
        numMetaBlocks, npts, newVecWidth, offset, deInterleaveMapPtr,
        interleaveMapPtr, buffer);

    cudaFree(buffer);
}

} // namespace Nektar

#endif
