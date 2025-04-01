///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsDeviceKernels.hpp
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

namespace Nektar
{

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)) ||                      \
    defined(NEKTAR_ENABLE_SYCL)
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void interleaveKernel(const unsigned int VectorWidth,
                                               const unsigned int npts,
                                               TData *buffer, TData *inout,
                                               const TthreadBlock &threadBlock)
{
    const unsigned int metaBlock = getBlockIdx(threadBlock);
    const unsigned int offset    = npts * VectorWidth * metaBlock;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        unsigned int vecElem = idx % VectorWidth;
        unsigned int iElem   = idx / VectorWidth;
        inout[offset + idx]  = buffer[offset + vecElem * npts + iElem];
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void deInterleaveKernel(
    const unsigned int VectorWidth, const unsigned int npts, TData *buffer,
    TData *inout, const TthreadBlock &threadBlock)
{
    const unsigned int metaBlock = getBlockIdx(threadBlock);
    const unsigned int offset    = npts * VectorWidth * metaBlock;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        buffer[offset + idx] = inout[offset + idx];
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < npts * VectorWidth; idx += stride)
    {
        unsigned int vecElem = idx / npts;
        unsigned int iElem   = idx % npts;
        inout[offset + idx]  = buffer[offset + iElem * VectorWidth + vecElem];
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BuildInterleaveMapKernel(
    const unsigned int npts, const unsigned int newVecWidth,
    const unsigned int offset, TData *deInterleaveMapPtr,
    TData *interleaveMapPtr, TData *buffer, const TthreadBlock &threadBlock)
{
    const unsigned int metaBlock   = getBlockIdx(threadBlock);
    const unsigned int groupOffset = npts * newVecWidth * metaBlock;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < npts * newVecWidth; idx += stride)
    {
        buffer[groupOffset + idx] = offset + groupOffset + idx;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < npts * newVecWidth; idx += stride)
    {
        unsigned int vecElem = idx % newVecWidth;
        unsigned int iElem   = idx / newVecWidth;
        deInterleaveMapPtr[groupOffset + idx] =
            buffer[groupOffset + vecElem * npts + iElem];
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < npts * newVecWidth; idx += stride)
    {
        interleaveMapPtr[deInterleaveMapPtr[groupOffset + idx]] =
            offset + groupOffset + idx;
    }
}
#endif

} // namespace Nektar

#include "Operators/Utils/UtilsDeviceOnHostKernels.hpp"
#include "Operators/Utils/UtilsHIPCUDAKernels.hpp"
#include "Operators/Utils/UtilsSYCLKernels.hpp"
