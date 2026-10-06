///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceStdMatKernels.hpp
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

#include <LibUtilities/Backends/Backends_Device_API.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void MultiplyByDerivFactorKernel(
    const unsigned nqTot, const unsigned ncoord, const unsigned dimension,
    const unsigned outDim, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *dfptr, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx               = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const TData *inptr       = in + nsize * (outDim * nmode * c + m);
    TData *outptr            = out + nsize * (outDim * nmode * c + m);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3];
        if constexpr (DEFORMED)
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[(ndf - 1) * nqTot * e + nqTot * (k * dimension) +
                               idx] *
                         inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[(ndf - 1) * nqTot * e +
                                    nqTot * (k * dimension + d) + idx] *
                              inptr[idx + d * inoffset];
                }
            }
        }
        else
        {
            for (unsigned int k = 0; k < ncoord; k++)
            {
                tmp[k] = dfptr[(ndf * e + k * dimension)] * inptr[idx];
                for (unsigned int d = 1; d < dimension; d++)
                {
                    tmp[k] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + d * inoffset];
                }
            }
        }

        for (unsigned int k = 0; k < ncoord; k++)
        {
            outptr[k * outoffset + idx] = tmp[k];
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}

template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void MultiplyByDerivDirFactorKernel(
    const unsigned dir, const unsigned nqTot, const unsigned ncoord,
    const unsigned dimension, const size_t nelmt, const size_t inoffset,
    const TData *dfptr, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx           = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    const TData *inptr   = in + nsize * p;
    TData *outptr        = out + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp;
        if constexpr (DEFORMED)
        {
            tmp =
                dfptr[(ndf - 1) * nqTot * e + nqTot * (dir * dimension) + idx] *
                inptr[idx];
            for (unsigned int d = 1; d < dimension; d++)
            {
                tmp += dfptr[(ndf - 1) * nqTot * e +
                             nqTot * (dir * dimension + d) + idx] *
                       inptr[idx + d * inoffset];
            }
        }
        else
        {
            tmp = dfptr[(ndf * e + dir * dimension)] * inptr[idx];
            for (unsigned int d = 1; d < dimension; d++)
            {
                tmp += dfptr[(ndf * e + dir * dimension + d)] *
                       inptr[idx + d * inoffset];
            }
        }

        if constexpr (APPEND)
        {
            outptr[idx] += tmp;
        }
        else
        {
            outptr[idx] = tmp;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::MultiRegions::detail
