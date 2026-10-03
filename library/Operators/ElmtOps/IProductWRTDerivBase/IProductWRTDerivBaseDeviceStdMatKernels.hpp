///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceStdMatKernels.hpp
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

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// The planes and variables run on the second and third grid dimensions.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void JacobianDerivFactorKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *jacptr, const TData *dfptr,
    const TData *in, TData *out, const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx               = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
    const TData *inptr       = in + nsize * (inDim * nmode * c + m);
    TData *outptr            = out + nsize * (nmode * c + m);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3];
        if constexpr (DEFORMED)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] =
                    dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf - 1) * nqTot * e +
                                    nqTot * (k * dimension + d) + idx] *
                              inptr[idx + k * inoffset];
                }
            }
        }
        else
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] = dfptr[(ndf * e + d)] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + k * inoffset];
                }
            }
        }

        const auto wj = DEFORMED ? jacptr[idx] : jacptr[e];
        for (unsigned int d = 0; d < dimension; d++)
        {
            outptr[d * outoffset + idx] = tmp[d] * wj;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}

// The planes and variables run on the second and third grid dimensions.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void JacobianDerivFactorWeightsKernel(
    const unsigned int nqTot, const unsigned int ncoord,
    const unsigned int dimension, const size_t nelmt, const size_t inoffset,
    const size_t outoffset, const TData *jacptr, const TData *dfptr,
    const TData *weights, const TData *in, TData *out,
    const TthreadBlock &threadBlock)
{
    const auto ndf     = ncoord * dimension;
    const size_t nsize = nqTot * nelmt;

    size_t idx               = getGlobalIdx<0>(threadBlock);
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    const unsigned int inDim = (nmode > 1) ? 3u : ncoord;
    const TData *inptr       = in + nsize * (inDim * nmode * c + m);
    TData *outptr            = out + nsize * (nmode * c + m);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        TData tmp[3];
        if constexpr (DEFORMED)
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] =
                    dfptr[(ndf - 1) * nqTot * e + nqTot * d + idx] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf - 1) * nqTot * e +
                                    nqTot * (k * dimension + d) + idx] *
                              inptr[idx + k * inoffset];
                }
            }
        }
        else
        {
            for (unsigned int d = 0; d < dimension; d++)
            {
                tmp[d] = dfptr[(ndf * e + d)] * inptr[idx];
                for (unsigned int k = 1; k < ncoord; ++k)
                {
                    tmp[d] += dfptr[(ndf * e + k * dimension + d)] *
                              inptr[idx + k * inoffset];
                }
            }
        }

        const size_t i = idx - e * nqTot;
        const auto wj  = (DEFORMED ? jacptr[idx] : jacptr[e]) * weights[i];
        for (unsigned int d = 0; d < dimension; d++)
        {
            outptr[d * outoffset + idx] = tmp[d] * wj;
        }

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::Operators::detail
