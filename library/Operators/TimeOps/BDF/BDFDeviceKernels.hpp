///////////////////////////////////////////////////////////////////////////////
//
// File: BDFDeviceKernels.hpp
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
#include <deque>

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__)) ||                      \
    defined(NEKTAR_ENABLE_SYCL) || defined(NEKTAR_ENABLE_DEVICEONHOST)
namespace Nektar::Operators::detail
{

/*
 * Compute extrapolation of previous time steps.
 * extrapolation = \sum_q=0^{J-1} \alpha_q u^{n-q} where J = IntOrder
 */
template <typename TthreadBlock, typename TData, unsigned int IntOrder>
NEK_DEVICE_INLINE static void ExtrapolateKernel(
    const size_t nsize, const TData dt,
    const TData *__restrict__ const *__restrict__ solutions,
    TData *__restrict__ inoutPtr, const TthreadBlock &threadBlock)
{
    const size_t idx0   = getGlobalIdx(threadBlock);
    const size_t stride = getGlobalRange(threadBlock);

    // use tmp to avoid multiple read-write to memory
    TData tmp;

    for (size_t idx = idx0; idx < nsize; idx += stride)
    {
        if constexpr (IntOrder == 2)
        {
            tmp = 2.0 * solutions[0][idx];
            tmp += -1.0 / 2.0 * inoutPtr[idx];
            tmp /= dt;
        }
        if constexpr (IntOrder == 3)
        {
            tmp = 3.0 * solutions[0][idx];
            tmp += -3.0 / 2.0 * solutions[1][idx];
            tmp += 1.0 / 3.0 * inoutPtr[idx];
            tmp /= dt;
        }
        if constexpr (IntOrder == 4)
        {
            tmp = 4.0 * solutions[0][idx];
            tmp += -3.0 * solutions[1][idx];
            tmp += 4.0 / 3.0 * solutions[2][idx];
            tmp += -1.0 / 4.0 * inoutPtr[idx];
            tmp /= dt;
        }
        inoutPtr[idx] = tmp; // Write once to global memory
    }
}

} // namespace Nektar::Operators::detail
#endif

#include "Operators/TimeOps/BDF/BDFDeviceOnHostKernelLaunchers.hpp"
#include "Operators/TimeOps/BDF/BDFHIPCUDAKernelLaunchers.hpp"
#include "Operators/TimeOps/BDF/BDFSYCLKernelLaunchers.hpp"
