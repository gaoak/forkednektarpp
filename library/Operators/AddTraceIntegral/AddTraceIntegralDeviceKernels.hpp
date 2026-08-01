///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralDeviceKernels.hpp
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
NEK_DEVICE_KERNEL static void AddTraceIntegralKernel(
    const size_t nsize, const size_t *NEK_RESTRICT traceCoeffsToElmtMapPtr,
    const int *NEK_RESTRICT traceCoeffsToElmtSignPtr,
    const size_t *NEK_RESTRICT traceCoeffsToElmtTracePtr,
    const TData *NEK_RESTRICT tracePtr, TData *NEK_RESTRICT outptr,
    const TthreadBlock &threadBlock)
{
    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        TData *const ptr = outptr + traceCoeffsToElmtMapPtr[idx];
        const TData val  = traceCoeffsToElmtSignPtr[idx] *
                          tracePtr[traceCoeffsToElmtTracePtr[idx]];
        Nektar::atomic_add<NektarSpaces::GlobalScope>(ptr, val);
    }
}

// Kernel Launchers.
template <
    typename ExecSpace, typename TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Device>, bool>
        Enable = true>
NEK_FORCE_INLINE static void AddTraceIntegralKernel(
    const size_t nsize, const size_t *traceCoeffsToElmtMapPtr,
    const int *traceCoeffsToElmtSignPtr,
    const size_t *traceCoeffsToElmtTracePtr, const TData *tracePtr,
    TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
        AddTraceIntegralKernel<>, gridSize, blockSize, 0, nsize,
        traceCoeffsToElmtMapPtr, traceCoeffsToElmtSignPtr,
        traceCoeffsToElmtTracePtr, tracePtr, outptr);
}
#endif

} // namespace Nektar::Operators::detail
