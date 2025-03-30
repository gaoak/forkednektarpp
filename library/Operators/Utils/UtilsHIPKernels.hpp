///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsHIPKernels.hpp
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

#if defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)

namespace Nektar
{

template <typename TData>
__global__ void interleaveKernel(const unsigned int VectorWidth,
                                 const unsigned int npts, TData *buffer,
                                 TData *inout)
{
    interleaveKernel(VectorWidth, npts, buffer, inout, cudaBlock1D());
}

template <typename TData>
__global__ void deInterleaveKernel(const unsigned int VectorWidth,
                                   const unsigned int npts, TData *buffer,
                                   TData *inout)
{
    deInterleaveKernel(VectorWidth, npts, buffer, inout, cudaBlock1D());
}

template <typename TData>
__global__ void BuildInterleaveMapKernel(const unsigned int npts,
                                         const unsigned int newVecWidth,
                                         const unsigned int offset,
                                         TData *deInterleaveMapPtr,
                                         TData *interleaveMapPtr, TData *buffer)
{
    BuildInterleaveMapKernel(npts, newVecWidth, offset, deInterleaveMapPtr,
                             interleaveMapPtr, buffer, cudaBlock1D());
}

template <unsigned int VectorWidth, typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::HIP>,
                               void>::type
interleave(const unsigned int numMetaBlocks, const unsigned int npts,
           TData *inout)
{
    const unsigned int blockSize = NektarSpaces::HIP::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));
    interleaveKernel<<<gridSize, blockSize>>>(VectorWidth, npts, buffer, inout);
    CHECK_LAST_HIP_ERROR();
    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::HIP>,
                               void>::type
deInterleave(const unsigned int VectorWidth, const unsigned int numMetaBlocks,
             const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::HIP::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(TData) * VectorWidth * numMetaBlocks * npts;

    TData *buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));
    deInterleaveKernel<<<gridSize, blockSize>>>(VectorWidth, npts, buffer,
                                                inout);
    CHECK_LAST_HIP_ERROR();
    CHECK_HIP_ERROR(hipFree(buffer));
}

template <typename ExecSpace>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::HIP>,
                               void>::type
BuildInterleaveMap(const unsigned int numMetaBlocks, const unsigned int npts,
                   const unsigned int newVecWidth, const unsigned int offset,
                   int *deInterleaveMapPtr, int *interleaveMapPtr)
{
    const unsigned int blockSize = NektarSpaces::HIP::defaultBlockSize;
    const unsigned int gridSize  = numMetaBlocks;
    const unsigned int bufferSize =
        sizeof(int) * newVecWidth * numMetaBlocks * npts;

    int *buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));
    BuildInterleaveMapKernel<<<gridSize, blockSize>>>(npts, newVecWidth, offset,
                                                      deInterleaveMapPtr,
                                                      interleaveMapPtr, buffer);
    CHECK_LAST_HIP_ERROR();
    CHECK_HIP_ERROR(hipFree(buffer));
}

} // namespace Nektar

#endif
