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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)) ||                      \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
template <bool negflag, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void RobBndCond1DKernel(
    const unsigned int nsize, const unsigned int *__restrict__ offsetPtr,
    const TData *__restrict__ matPtr, const unsigned int *__restrict__ mapPtr,
    const TData *__restrict__ incoeffPtr, TData *__restrict__ coeffPtr,
    const TthreadBlock &threadBlock)
{
    unsigned int idx0   = getGlobalIdx(threadBlock);
    unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int i = idx0; i < nsize; i += stride)
    {
        const unsigned int offset = offsetPtr[i];
        const unsigned int map    = mapPtr[i];

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
NEK_DEVICE_INLINE static void RobBndCond2DKernel(
    const unsigned int nsize, const unsigned int *__restrict__ ncoeffPtr,
    const unsigned int *__restrict__ offsetPtr,
    const unsigned int *__restrict__ matOffsetPtr,
    const unsigned int *__restrict__ mapOffsetPtr,
    const TData *__restrict__ matPtr, const unsigned int *__restrict__ mapPtr,
    const int *__restrict__ signPtr, const TData *__restrict__ incoeffPtr,
    TData *__restrict__ coeffPtr, TData *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    TData *vEdgeCoeffs = shmemptr;

    unsigned int j = getBlockIdx(threadBlock);

    while (j < nsize)
    {
        const unsigned int ncoeff    = ncoeffPtr[j];
        const unsigned int offset    = offsetPtr[j];
        const unsigned int matOffset = matOffsetPtr[j];
        const unsigned int mapOffset = mapOffsetPtr[j];

        unsigned int idx0   = getLocalIdx(threadBlock);
        unsigned int stride = getLocalRange(threadBlock);

        for (unsigned int i = idx0; i < ncoeff; i += stride)
        {
            const unsigned int index = mapOffset + i;
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

            const unsigned int index = mapOffset + i;
            TData *const ptr         = coeffPtr + offset + mapPtr[index];
            const TData val          = tmp * signPtr[index];
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
#endif

} // namespace Nektar::Operators::detail

#include "Operators/BndCondOps/RobBndCond/RobBndCondDeviceOnHostKernelLaunchers.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondHIPCUDAKernelLaunchers.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondSYCLKernelLaunchers.hpp"
