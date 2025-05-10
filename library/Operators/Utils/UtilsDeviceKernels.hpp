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
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void interleaveKernel(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *buffer, TData *inout,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t e = getBlockIdx(threadBlock); e < numElmtGroups;
         e += getBlockRange(threadBlock))
    {
        TData *bufferptr = buffer + npts * interleaveWidth * e;
        TData *inoutptr  = inout + npts * interleaveWidth * e;

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            bufferptr[idx] = inoutptr[idx];
        }

        localBarrier(threadBlock);

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            unsigned int vecElem = idx % interleaveWidth;
            unsigned int iElem   = idx / interleaveWidth;
            inoutptr[idx]        = bufferptr[vecElem * npts + iElem];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void deInterleaveKernel(
    const unsigned int interleaveWidth, const size_t numElmtGroups,
    const unsigned int npts, TData *buffer, TData *inout,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (size_t e = getBlockIdx(threadBlock); e < numElmtGroups;
         e += getBlockRange(threadBlock))
    {
        TData *bufferptr = buffer + npts * interleaveWidth * e;
        TData *inoutptr  = inout + npts * interleaveWidth * e;

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            bufferptr[idx] = inoutptr[idx];
        }

        localBarrier(threadBlock);

        for (unsigned int idx = idx0; idx < npts * interleaveWidth;
             idx += stride)
        {
            unsigned int vecElem = idx / npts;
            unsigned int iElem   = idx % npts;
            inoutptr[idx]        = bufferptr[iElem * interleaveWidth + vecElem];
        }
    }
}

template <bool APPEND = false, bool TRANSPOSE = false, typename TData>
NEK_DEVICE_INLINE static void MatVecKernel(const unsigned int ilane,
                                           const unsigned int nmTot,
                                           const TData *__restrict__ nodToMod,
                                           const TData *__restrict__ in,
                                           TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nmTot; ++i)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nmTot; ++j)
        {
            if constexpr (TRANSPOSE)
            {
                tmp += nodToMod[j * nmTot + i] * in[warpsize * j + ilane];
            }
            else
            {
                tmp += nodToMod[i * nmTot + j] * in[warpsize * j + ilane];
            }
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += tmp;
        }
        else
        {
            out[warpsize * i + ilane] = tmp;
        }
    }
}

template <bool APPEND = false, bool TRANSPOSE = false, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void MatVecQPKernel(const unsigned int nmTot,
                                             const TData *__restrict__ nodToMod,
                                             const TData *__restrict__ in,
                                             TData *__restrict__ out,
                                             const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nmTot; i += stride)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int j = 0u; j < nmTot; j++)
        {
            if constexpr (TRANSPOSE)
            {
                tmp += in[j] * nodToMod[j * nmTot + i];
            }
            else
            {
                tmp += in[j] * nodToMod[i * nmTot + j];
            }
        }

        if constexpr (APPEND)
        {
            out[i] += tmp;
        }
        else
        {
            out[i] = tmp;
        }
    }

    localBarrier(threadBlock);
}
#endif

} // namespace Nektar

#include "Operators/Utils/UtilsDeviceOnHostKernels.hpp"
#include "Operators/Utils/UtilsHIPCUDAKernels.hpp"
#include "Operators/Utils/UtilsSYCLKernels.hpp"
