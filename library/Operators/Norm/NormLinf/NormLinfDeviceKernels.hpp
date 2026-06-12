///////////////////////////////////////////////////////////////////////////////
//
// File: NormLinfDeviceKernels.hpp
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

#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LinfKernel(const size_t nelmt,
                                         const unsigned int ndata,
                                         const unsigned int interleaveWidth,
                                         const TData *NEK_RESTRICT in,
                                         TData *NEK_RESTRICT norm,
                                         const TthreadBlock &threadBlock)
{
    TData acc = 0.0;
    size_t e  = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t lane   = e % interleaveWidth;
        const size_t group  = e / interleaveWidth;
        const size_t offset = group * interleaveWidth * ndata + lane;

        for (unsigned int q = 0; q < ndata; ++q)
        {
            const size_t index = offset + q * interleaveWidth;
            const TData value  = abs(in[index]);
            acc                = acc > value ? acc : value;
        }

        e += getGlobalRange(threadBlock);
    }

    blockReduceMax(acc, threadBlock, norm);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void LinfKernelLauncher(const size_t nelmt,
                                          const unsigned int ndata,
                                          const unsigned int interleaveWidth,
                                          const TData *NEK_RESTRICT in,
                                          TData *NEK_RESTRICT norm,
                                          const TthreadBlock &threadBlock)
{
    LinfKernel(nelmt, ndata, interleaveWidth, in, norm, threadBlock);
}

#endif

} // namespace Nektar::Operators::detail
