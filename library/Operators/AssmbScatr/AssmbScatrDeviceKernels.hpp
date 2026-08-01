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

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AssembleScatrKernel(
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
NEK_DEVICE_KERNEL static void AssembleScatrBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    TData *inoutptr, TData *bndptr, const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass              = 0;
        const unsigned ioffset = offset[idx];
        const unsigned nidx    = nassemble[idx];
        const unsigned nbnd    = nbndvals[idx];

        // assemble values
        for (unsigned j = 0; j < nidx; ++j)
        {
            const unsigned ind = ioffset + j;
            ass += inoutptr[index[ind]] * sign[ind];
        }

        // copy one assembled values back to local values
        inoutptr[index[ioffset]] = ass * sign[ioffset];

        // put assembled values into boudnary array
        for (unsigned j = 0; j < nbnd; ++j)
        {
            const unsigned ind = ioffset + nidx + j;
            bndptr[index[ind]] = ass;
        }
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AssembleFromBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    const unsigned *norder, const TData *bndptr, TData *inoutptr,
    const TthreadBlock &threadBlock)
{
    const unsigned idx0   = getGlobalIdx(threadBlock);
    const unsigned stride = getGlobalRange(threadBlock);

    for (unsigned idx = idx0; idx < nvals; idx += stride)
    {
        TData ass              = 0;
        const unsigned ioffset = offset[idx];
        const unsigned nidx    = nassemble[idx];
        const unsigned nbnd    = nbndvals[idx];
        const unsigned nord    = norder[idx];

        // assemble bndptr components wtih local ids
        for (unsigned j = 0; j < nord; ++j)
        {
            const unsigned ind = ioffset + nidx + j;
            ass += bndptr[index[ind]];
        }

        // add in local point in rank ordered assembly
        ass += inoutptr[index[ioffset]] * sign[ioffset];

        // assemble rest of points from where we left off
        for (unsigned j = nord; j < nbnd; ++j)
        {
            const unsigned ind = ioffset + nidx + j;
            ass += bndptr[index[ind]];
        }

        // copy rank assembled values back to local values
        for (unsigned j = 0; j < nidx; ++j)
        {
            const unsigned ind   = ioffset + j;
            inoutptr[index[ind]] = ass * sign[ind];
        }
    }
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleScatrKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *index,
    const unsigned *offset, const int *sign, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(AssembleScatrKernel, gridSize,
                                          blockSize, 0, nvals, nassemble, index,
                                          offset, sign, inoutptr);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleScatrBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    TData *inoutptr, TData *bndptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        AssembleScatrBndKernel<>, gridSize, blockSize, 0, nvals, nassemble,
        nbndvals, index, offset, sign, inoutptr, bndptr);
}

template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AssembleFromBndKernel(
    const unsigned nvals, const unsigned *nassemble, const unsigned *nbndvals,
    const unsigned *index, const unsigned *offset, const int *sign,
    const unsigned *norder, const TData *bndptr, TData *inoutptr)
{
    const unsigned blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned gridSize  = (nvals + blockSize - 1) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        AssembleFromBndKernel<>, gridSize, blockSize, 0, nvals, nassemble,
        nbndvals, index, offset, sign, norder, bndptr, inoutptr);
}
#endif

} // namespace Nektar::Operators::detail
