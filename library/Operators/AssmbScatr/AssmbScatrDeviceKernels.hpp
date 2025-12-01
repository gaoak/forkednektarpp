///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrDeviceKernels.hpp
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

#include "Operators/Common/Spaces.hpp"

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)) ||                      \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
namespace Nektar::Operators::detail
{
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *__restrict__ GSInfo,
    const int *__restrict__ sign, TData *__restrict__ inoutptr,
    const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    const unsigned *offset = GSInfo + 1;
    const unsigned *ind    = GSInfo + nvals + 2;

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass            = 0;
        const unsigned start = offset[idx];
        const unsigned end   = offset[idx + 1];
        for (unsigned j = start; j < end; ++j)
        {
            ass += inoutptr[ind[j]] * sign[j];
        }
        for (unsigned j = start; j < end; ++j)
        {
            inoutptr[ind[j]] = ass * sign[j];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *index,
    const int *sign, TData *inoutptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int warpSize = NektarSpaces::Device::warpSize;

    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass = 0;
        // this will determine the offset for index and sign data
        // could be pre-caclculated and passed.
        unsigned cnt = 0;
        for (unsigned i = warpSize; i <= idx; i += warpSize)
        {
            cnt += nassemble[(i - warpSize) / warpSize * warpSize] * warpSize;
        }

        const unsigned i       = idx % warpSize;
        const unsigned nassemb = nassemble[idx];
        for (unsigned j = 0; j < nassemb; ++j)
        {
            ass += inoutptr[index[cnt + j * warpSize + i]] *
                   sign[cnt + j * warpSize + i];
        }

        for (unsigned j = 0; j < nassemb; ++j)
        {
            inoutptr[index[cnt + j * warpSize + i]] =
                ass * sign[cnt + j * warpSize + i];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *index,
    const unsigned *offset, const int *sign, TData *inoutptr,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int warpSize = NektarSpaces::Device::warpSize;

    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass              = 0;
        const unsigned ioffset = offset[idx];
        const unsigned nassemb = nassemble[idx];
        for (unsigned j = 0; j < nassemb; ++j)
        {
            const unsigned ind = ioffset + j * warpSize;
            ass += inoutptr[index[ind]] * sign[ind];
        }

        for (unsigned j = 0; j < nassemb; ++j)
        {
            const unsigned ind   = ioffset + j * warpSize;
            inoutptr[index[ind]] = ass * sign[ind];
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleScatrBndKernel(
    const unsigned nvals, const unsigned *GSInfo, const int *sign,
    TData *inoutptr, TData *bndptr, const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    const unsigned *offset = GSInfo + 1;
    const unsigned *ind    = GSInfo + nvals + 2;

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass  = 0;
        auto start = offset[idx];
        auto nidx  = ind[start];
        auto nbnd  = ind[start + 1];

        // cannot evalaute cnt at end of loop since may only accesss
        // loop once in GPU. So following hack just calculated cnt
        unsigned cnt = 0;
        for (unsigned i = 0; i < idx; ++i)
        {
            cnt += ind[offset[i]];
        }

        start += 2;
        // assemble values
        for (unsigned j = 0; j < nidx; ++j)
        {
            ass += inoutptr[ind[start + j]] * sign[cnt + j];
        }
        // copy assembled values back to local values
        for (unsigned j = 0; j < nidx; ++j)
        {
            inoutptr[ind[start + j]] = ass * sign[cnt + j];
        }
        // put assembled values into boudnary array
        start += nidx;
        for (unsigned j = 0; j < nbnd; ++j)
        {
            bndptr[ind[start + j]] = ass;
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleFromBndKernel(
    const unsigned nvals, const unsigned *GSInfo, const int *sign,
    const TData *bndptr, TData *inoutptr, const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    const unsigned *offset = GSInfo + 1;
    const unsigned *ind    = GSInfo + nvals + 2;

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        // cannot evalaute cnt at end of loop since may only accesss
        // loop once in GPU. So following hack just calculated cnt
        unsigned cnt = 0;
        for (unsigned i = 0; i < idx; ++i)
        {
            cnt += ind[offset[i]];
        }

        TData ass  = 0;
        auto start = offset[idx];
        auto nidx  = ind[start];
        auto nbnd  = ind[start + 1];
        start += 2;

        auto startbnd = start + nidx;

        // assemble bndptr components wtih local ids
        for (unsigned j = 0; j < nbnd; ++j)
        {
            ass += bndptr[ind[startbnd + j]];
        }

        // add in local point in rank ordered assembly
        ass += inoutptr[ind[start]] * sign[cnt];

        // copy rank ordered assembled values back to local values
        for (unsigned j = 0; j < nidx; ++j)
        {
            inoutptr[ind[start + j]] = ass * sign[cnt + j];
        }
    }
}

} // namespace Nektar::Operators::detail
#endif

#include "Operators/AssmbScatr/AssmbScatrDeviceOnHostKernelLaunchers.hpp"
#include "Operators/AssmbScatr/AssmbScatrHIPCUDAKernelLaunchers.hpp"
#include "Operators/AssmbScatr/AssmbScatrSYCLKernelLaunchers.hpp"
