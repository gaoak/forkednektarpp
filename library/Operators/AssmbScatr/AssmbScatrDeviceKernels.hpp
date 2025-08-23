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
        TData ass = 0;
        for (unsigned j = offset[idx]; j < offset[idx + 1]; ++j)
        {
            ass += inoutptr[ind[j]] * sign[j];
        }
        for (unsigned j = offset[idx]; j < offset[idx + 1]; ++j)
        {
            inoutptr[ind[j]] = ass * sign[j];
        }
    }
}

} // namespace Nektar::Operators::detail
#endif

#include "Operators/AssmbScatr/AssmbScatrDeviceOnHostKernelLaunchers.hpp"
#include "Operators/AssmbScatr/AssmbScatrHIPCUDAKernelLaunchers.hpp"
#include "Operators/AssmbScatr/AssmbScatrSYCLKernelLaunchers.hpp"
