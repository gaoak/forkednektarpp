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
NEK_DEVICE_INLINE static void AssembleKernel(const unsigned int nsize,
                                             const int *__restrict__ assmbPtr,
                                             const TData *__restrict__ signPtr,
                                             const TData *__restrict__ inptr,
                                             TData *__restrict__ outptr,
                                             const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getGlobalIdx(threadBlock);
    const unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        Nektar::atomic_add<NektarSpaces::GlobalScope>(
            outptr + assmbPtr[idx], signPtr[idx] * inptr[idx]);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleKernel(const unsigned int nsize,
                                             const int *__restrict__ assmbPtr,
                                             const TData sign,
                                             const TData *__restrict__ inptr,
                                             TData *__restrict__ outptr,
                                             const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getGlobalIdx(threadBlock);
    const unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        Nektar::atomic_add<NektarSpaces::GlobalScope>(outptr + assmbPtr[idx],
                                                      sign * inptr[idx]);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AssembleKernel(const unsigned int nsize,
                                             const int *__restrict__ assmbPtr,
                                             const TData *__restrict__ inptr,
                                             TData *__restrict__ outptr,
                                             const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getGlobalIdx(threadBlock);
    const unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        Nektar::atomic_add<NektarSpaces::GlobalScope>(outptr + assmbPtr[idx],
                                                      inptr[idx]);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void GlobalToLocalKernel(
    const unsigned int nsize, const int *__restrict__ assmbPtr,
    const TData *__restrict__ signPtr, const TData *__restrict__ inptr,
    TData *__restrict__ outptr, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getGlobalIdx(threadBlock);
    const unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        outptr[idx] = signPtr[idx] * inptr[assmbPtr[idx]];
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void GlobalToLocalKernel(
    const unsigned int nsize, const int *__restrict__ assmbPtr,
    const TData sign, const TData *__restrict__ inptr,
    TData *__restrict__ outptr, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getGlobalIdx(threadBlock);
    const unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        outptr[idx] = sign * inptr[assmbPtr[idx]];
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void GlobalToLocalKernel(
    const unsigned int nsize, const int *__restrict__ assmbPtr,
    const TData *__restrict__ inptr, TData *__restrict__ outptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getGlobalIdx(threadBlock);
    const unsigned int stride = getGlobalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        outptr[idx] = inptr[assmbPtr[idx]];
    }
}

} // namespace Nektar::Operators::detail
#endif

#include "Operators/AssmbScatr/AssmbScatrDeviceOnHostKernelLaunchers.hpp"
#include "Operators/AssmbScatr/AssmbScatrHIPCUDAKernelLaunchers.hpp"
#include "Operators/AssmbScatr/AssmbScatrSYCLKernelLaunchers.hpp"
