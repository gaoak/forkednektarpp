///////////////////////////////////////////////////////////////////////////////
//
// File: AddTraceIntegralCUDAKernels.cuh
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

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
__global__ void AddTraceIntegralKernel(
    const unsigned int nsize, const int *__restrict__ traceCoeffsToElmtMapPtr,
    const int *__restrict__ traceCoeffsToElmtSignPtr,
    const int *__restrict__ traceCoeffsToElmtTracePtr,
    const TData *__restrict__ tracePtr, TData *__restrict__ outptr)
{
    const unsigned int idx0   = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int idx = idx0; idx < nsize; idx += stride)
    {
        TData *const ptr = outptr + traceCoeffsToElmtMapPtr[idx];
        const TData val  = traceCoeffsToElmtSignPtr[idx] *
                          tracePtr[traceCoeffsToElmtTracePtr[idx]];
        Nektar::atomic_add<NektarSpaces::CUDA, NektarSpaces::GlobalScope>(ptr,
                                                                          val);
    }
}

// Launchers
template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
AddTraceIntegralKernel(const unsigned int nsize,
                       const int *traceCoeffsToElmtMapPtr,
                       const int *traceCoeffsToElmtSignPtr,
                       const int *traceCoeffsToElmtTracePtr,
                       const TData *tracePtr, TData *outptr)
{
    const unsigned int blockSize = NektarSpaces::CUDA::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    AddTraceIntegralKernel<TData><<<gridSize, blockSize>>>(
        nsize, traceCoeffsToElmtMapPtr, traceCoeffsToElmtSignPtr,
        traceCoeffsToElmtTracePtr, tracePtr, outptr);
}

// Launchers
template <typename ExecSpace>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::CUDA>,
                               void>::type
ReOrderMapKernel([[maybe_unused]] const unsigned int nsize,
                 [[maybe_unused]] int *traceCoeffsToElmtMapPtr,
                 [[maybe_unused]] int *traceCoeffsToElmtSignPtr,
                 [[maybe_unused]] int *traceCoeffsToElmtTracePtr)
{
}

} // namespace Nektar::Operators::detail

#endif
