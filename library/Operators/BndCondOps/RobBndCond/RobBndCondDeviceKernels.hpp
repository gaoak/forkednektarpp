///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondDeviceKernels.hpp
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
template <bool negflag, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void RobBndCond1DKernel(
    const size_t nsize, const size_t *__restrict__ offsetPtr,
    const TData *__restrict__ matPtr, const size_t *__restrict__ mapPtr,
    const TData *__restrict__ incoeffPtr, TData *__restrict__ coeffPtr,
    const TthreadBlock &threadBlock)
{
    size_t idx0   = getGlobalIdx(threadBlock);
    size_t stride = getGlobalRange(threadBlock);

    for (size_t i = idx0; i < nsize; i += stride)
    {
        const size_t offset = offsetPtr[i];
        const size_t map    = mapPtr[i];

        TData *const ptr = coeffPtr + offset + map;
        const TData val  = matPtr[i] * incoeffPtr[offset + map];
        if constexpr (negflag)
        {
            Nektar::atomic_sub<NektarSpaces::GlobalScope>(ptr, val);
        }
        else
        {
            Nektar::atomic_add<NektarSpaces::GlobalScope>(ptr, val);
        }
    }
}

template <bool negflag, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void RobBndCond2DKernel(
    const size_t nsize, const unsigned int *__restrict__ ncoeffPtr,
    const size_t *__restrict__ offsetPtr,
    const size_t *__restrict__ matOffsetPtr,
    const size_t *__restrict__ mapOffsetPtr, const TData *__restrict__ matPtr,
    const size_t *__restrict__ mapPtr, const int *__restrict__ signPtr,
    const TData *__restrict__ incoeffPtr, TData *__restrict__ coeffPtr,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    TData *vEdgeCoeffs = (TData *)shmemptr;

    unsigned int j = getBlockIdx(threadBlock);

    while (j < nsize)
    {
        const unsigned int ncoeff = ncoeffPtr[j];
        const size_t offset       = offsetPtr[j];
        const size_t matOffset    = matOffsetPtr[j];
        const size_t mapOffset    = mapOffsetPtr[j];

        unsigned int idx0   = getLocalIdx(threadBlock);
        unsigned int stride = getLocalRange(threadBlock);

        for (unsigned int i = idx0; i < ncoeff; i += stride)
        {
            const size_t index = mapOffset + i;
            vEdgeCoeffs[i] =
                incoeffPtr[offset + mapPtr[index]] * signPtr[index];
        }

        Nektar::localBarrier(threadBlock);

        for (unsigned int i = idx0; i < ncoeff; i += stride)
        {
            TData tmp = 0.0;
            for (unsigned int k = 0; k < ncoeff; k++)
            {
                tmp += matPtr[matOffset + ncoeff * k + i] * vEdgeCoeffs[k];
            }

            const size_t index = mapOffset + i;
            TData *const ptr   = coeffPtr + offset + mapPtr[index];
            const TData val    = tmp * signPtr[index];
            if constexpr (negflag)
            {
                Nektar::atomic_sub<NektarSpaces::GlobalScope>(ptr, val);
            }
            else
            {
                Nektar::atomic_add<NektarSpaces::GlobalScope>(ptr, val);
            }
        }

        Nektar::localBarrier(threadBlock);

        j += getBlockRange(threadBlock);
    }
}

template <typename ExecSpace, bool negflag, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    RobBndCond1DKernel(const size_t nsize, const size_t *offsetPtr,
                       const TData *matPtr, const size_t *mapPtr,
                       const TData *incoeffPtr, TData *coeffPtr)
{
    const unsigned int blockSize = NektarSpaces::Device::warpSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(RobBndCond1DKernel<negflag>, gridSize,
                                          blockSize, 0, nsize, offsetPtr,
                                          matPtr, mapPtr, incoeffPtr, coeffPtr);
}

template <typename ExecSpace, bool negflag, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                            void>::type
    RobBndCond2DKernel(const unsigned int nmaxcoeff, const size_t nsize,
                       const unsigned int *ncoeffPtr, const size_t *offsetPtr,
                       const size_t *matOffsetPtr, const size_t *mapOffsetPtr,
                       const TData *matPtr, const size_t *mapPtr,
                       const int *signPtr, const TData *incoeffPtr,
                       TData *coeffPtr)
{
    const unsigned int shmemsize = sizeof(TData) * nmaxcoeff;
    const unsigned int blockSize = NektarSpaces::Device::warpSize;
    const unsigned int gridSize  = nsize;

    DEVICE_1DGRID_KERNEL_LAUNCHER(RobBndCond2DKernel<negflag>, gridSize,
                                  blockSize, shmemsize, 0, nsize, ncoeffPtr,
                                  offsetPtr, matOffsetPtr, mapOffsetPtr, matPtr,
                                  mapPtr, signPtr, incoeffPtr, coeffPtr);
}
#endif

} // namespace Nektar::Operators::detail
