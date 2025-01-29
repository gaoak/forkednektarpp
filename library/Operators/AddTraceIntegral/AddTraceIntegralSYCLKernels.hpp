///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralSYCLKernels.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
NEK_DEVICE_INLINE static void AddTraceIntegralKernel(
    const unsigned int nsize, const int *__restrict__ traceCoeffsToElmtMapPtr,
    const int *__restrict__ traceCoeffsToElmtSignPtr,
    const int *__restrict__ traceCoeffsToElmtTracePtr,
    const TData *__restrict__ tracePtr, TData *__restrict__ outptr,
    const sycl::nd_item<1> &item_ct1)
{
    const unsigned int idx0   = item_ct1.get_global_id(0);
    const unsigned int stride = item_ct1.get_global_range(0);

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        TData *const ptr = outptr + traceCoeffsToElmtMapPtr[idx];
        const TData val  = traceCoeffsToElmtSignPtr[idx] *
                          tracePtr[traceCoeffsToElmtTracePtr[idx]];
        Nektar::atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(ptr,
                                                                          val);
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
AddTraceIntegralKernel(const unsigned int nsize,
                       const int *traceCoeffsToElmtMapPtr,
                       const int *traceCoeffsToElmtSignPtr,
                       const int *traceCoeffsToElmtTracePtr,
                       const TData *tracePtr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<1>(gridSize * blockSize, blockSize),
             [=](sycl::nd_item<1> item_ct1) {
#pragma forceinline
                 AddTraceIntegralKernel<TData>(
                     nsize, traceCoeffsToElmtMapPtr, traceCoeffsToElmtSignPtr,
                     traceCoeffsToElmtTracePtr, tracePtr, outptr, item_ct1);
             });
     }).wait();
}

// Launchers
template <typename ExecSpace>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
ReOrderMapKernel([[maybe_unused]] const unsigned int nsize,
                 [[maybe_unused]] int *traceCoeffsToElmtMapPtr,
                 [[maybe_unused]] int *traceCoeffsToElmtSignPtr,
                 [[maybe_unused]] int *traceCoeffsToElmtTracePtr)
{
}

} // namespace Nektar::Operators::detail

#endif
