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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)) ||                      \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
namespace Nektar::Operators::detail
{

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddTraceIntegralKernel(
    const size_t nsize, const size_t *__restrict__ traceCoeffsToElmtMapPtr,
    const int *__restrict__ traceCoeffsToElmtSignPtr,
    const size_t *__restrict__ traceCoeffsToElmtTracePtr,
    const TData *__restrict__ tracePtr, TData *__restrict__ outptr,
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

} // namespace Nektar::Operators::detail
#endif

#include "Operators/AddTraceIntegral/AddTraceIntegralDeviceOnHostKernelLaunchers.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralHIPCUDAKernelLaunchers.hpp"
#include "Operators/AddTraceIntegral/AddTraceIntegralSYCLKernelLaunchers.hpp"
