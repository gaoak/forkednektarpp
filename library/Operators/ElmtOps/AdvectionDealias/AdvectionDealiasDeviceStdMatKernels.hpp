///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasDeviceStdMatKernels.hpp
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
// Description: Fine-grid product for 3/2-rule dealiased advection, StdMat
// convention.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void AdvectionDealiasCombineStdMatKernel(
    const size_t nsize, const unsigned int coordDim, const TData *advVel,
    const size_t advVelOffset, const unsigned int nhomo, const TData *grad,
    const size_t gradOffset, TData *out, const TData scale,
    const TthreadBlock &threadBlock)
{
    size_t idx             = getGlobalIdx<0>(threadBlock);
    const unsigned int c   = getBlockIdx<1>(threadBlock);
    const TData *gradptr   = grad + nsize * c;
    TData *outptr          = out + nsize * c;
    const TData *advVelPtr = advVel + nsize * (c % nhomo);
    while (idx < nsize)
    {
        TData tmp = advVelPtr[idx] * gradptr[idx];
        for (unsigned int d = 1u; d < coordDim; ++d)
        {
            tmp += advVelPtr[d * advVelOffset + idx] *
                   gradptr[d * gradOffset + idx];
        }

        if constexpr (APPEND)
        {
            outptr[idx] += scale * tmp;
        }
        else
        {
            outptr[idx] = scale * tmp;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::Operators::detail
