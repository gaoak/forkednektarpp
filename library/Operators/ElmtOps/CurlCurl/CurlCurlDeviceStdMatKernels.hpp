///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceStdMatKernels.hpp
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
// Scalar curl of a two-dimensional vector field, omega = dv/dx - du/dy.
//
// @param deriv holds the standard (tensorial) derivatives produced by the
// standard matrix multiply, ordered as deriv[(d * 2 + c) * derivoffset + idx]
// for direction d and component c. The chain rule and the curl are applied in
// a single sweep. The planes run on the second grid dimension.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void Curl2DScalarStdMatKernel(
    const unsigned int nqTot, const size_t nelmt, const size_t derivoffset,
    const TData *dfptr, const TData *deriv, TData *omega,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 4u;
    const size_t nsize         = nqTot * nelmt;

    size_t idx            = getGlobalIdx<0>(threadBlock);
    const unsigned int p  = getBlockIdx<1>(threadBlock);
    const TData *derivptr = deriv + nsize * p;
    TData *omegaptr       = omega + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        const size_t dfbase =
            DEFORMED ? ndf * nqTot * e + (idx - e * nqTot) : ndf * e;
        const unsigned int dfstride = DEFORMED ? nqTot : 1u;

        const TData d0u = derivptr[0u * derivoffset + idx];
        const TData d0v = derivptr[1u * derivoffset + idx];
        const TData d1u = derivptr[2u * derivoffset + idx];
        const TData d1v = derivptr[3u * derivoffset + idx];

        const TData dvdx = d0v * dfptr[dfbase + 0u * dfstride] +
                           d1v * dfptr[dfbase + 1u * dfstride];
        const TData dudy = d0u * dfptr[dfbase + 2u * dfstride] +
                           d1u * dfptr[dfbase + 3u * dfstride];

        omegaptr[idx] = dvdx - dudy;

        idx += getGlobalRange<0>(threadBlock);
    }
}

// Vector curl of a two-dimensional scalar field,
// out = {d(omega)/dy, -d(omega)/dx}.
//
// @param deriv holds the standard derivatives of omega, ordered as
// deriv[d * derivoffset + idx] for direction d. The planes run on the second
// grid dimension.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void Curl2DVectorStdMatKernel(
    const unsigned int nqTot, const size_t nelmt, const size_t derivoffset,
    const size_t outoffset, const TData *dfptr, const TData *deriv, TData *out,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 4u;
    const size_t nsize         = nqTot * nelmt;

    size_t idx            = getGlobalIdx<0>(threadBlock);
    const unsigned int p  = getBlockIdx<1>(threadBlock);
    const TData *derivptr = deriv + nsize * p;
    TData *outptr         = out + nsize * p;
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        const size_t dfbase =
            DEFORMED ? ndf * nqTot * e + (idx - e * nqTot) : ndf * e;
        const unsigned int dfstride = DEFORMED ? nqTot : 1u;

        const TData d0 = derivptr[0u * derivoffset + idx];
        const TData d1 = derivptr[1u * derivoffset + idx];

        outptr[idx] = d0 * dfptr[dfbase + 2u * dfstride] +
                      d1 * dfptr[dfbase + 3u * dfstride];
        outptr[outoffset + idx] = -(d0 * dfptr[dfbase + 0u * dfstride] +
                                    d1 * dfptr[dfbase + 1u * dfstride]);

        idx += getGlobalRange<0>(threadBlock);
    }
}

// Curl of a three-dimensional vector field. Used for both passes of the
// curl-curl operator, i.e. omega = curl(u) and out = curl(omega).
//
// @param deriv holds the standard derivatives ordered as
// deriv[(d * 3 + c) * derivoffset + idx] for direction d and component c.
// Only the six off-diagonal physical derivatives are formed, the diagonal
// ones cancel out.
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL static void Curl3DStdMatKernel(
    const unsigned int nqTot, const size_t nelmt, const size_t derivoffset,
    const size_t outoffset, const TData *dfptr, const TData *deriv, TData *out,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const size_t nsize         = nqTot * nelmt;

    size_t idx = getGlobalIdx<0>(threadBlock);
    while (idx < nsize)
    {
        const size_t e = idx / nqTot;

        const size_t dfbase =
            DEFORMED ? ndf * nqTot * e + (idx - e * nqTot) : ndf * e;
        const unsigned int dfstride = DEFORMED ? nqTot : 1u;

        TData d0[3], d1[3], d2[3];
        for (unsigned int c = 0u; c < 3u; ++c)
        {
            d0[c] = deriv[(0u * 3u + c) * derivoffset + idx];
            d1[c] = deriv[(1u * 3u + c) * derivoffset + idx];
            d2[c] = deriv[(2u * 3u + c) * derivoffset + idx];
        }

        TData df[9];
        for (unsigned int m = 0u; m < 9u; ++m)
        {
            df[m] = dfptr[dfbase + m * dfstride];
        }

        // dw/dy - dv/dz
        out[idx] = (d0[2] * df[3] + d1[2] * df[4] + d2[2] * df[5]) -
                   (d0[1] * df[6] + d1[1] * df[7] + d2[1] * df[8]);
        // du/dz - dw/dx
        out[outoffset + idx] = (d0[0] * df[6] + d1[0] * df[7] + d2[0] * df[8]) -
                               (d0[2] * df[0] + d1[2] * df[1] + d2[2] * df[2]);
        // dv/dx - du/dy
        out[2u * outoffset + idx] =
            (d0[1] * df[0] + d1[1] * df[1] + d2[1] * df[2]) -
            (d0[0] * df[3] + d1[0] * df[4] + d2[0] * df[5]);

        idx += getGlobalRange<0>(threadBlock);
    }
}
#endif

} // namespace Nektar::Operators::detail
