///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFacKernels.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline constexpr unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nq0 + nq1;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
              * = nullptr>
inline constexpr unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1,
                                                        const unsigned int nq2)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return nq0 + 2u * nq1 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        return nq0 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        return nq0 + nq1 + nq2;
    }
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[warpsize * q + ilane];
        }

        // Multiply by derivative factors.
        for (unsigned int k = 0u; k < ncoord; k++)
        {
            out[k * outoffset + index] = d0 * df[k * warpsize + dfindex];
        }
    }
}

template <bool DEFORMED, unsigned int DIR, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[warpsize * q + ilane];
        }

        // Multiply by derivative factors.
        if constexpr (APPEND)
        {
            out[index] += d0 * df[DIR * warpsize + dfindex];
        }
        else
        {
            out[index] += d0 * df[DIR * warpsize + dfindex];
        }
    }
}

template <bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor1DKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT in0,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[warpsize * q + ilane];
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += d0;
        }
        else
        {
            out[warpsize * i + ilane] = d0;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[warpsize * (nq0 * q + i) + ilane];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors.
            out[index] = d0 * df[0u * warpsize + dfindex] +
                         d1 * df[1u * warpsize + dfindex];
            out[outoffset + index] = d0 * df[2u * warpsize + dfindex] +
                                     d1 * df[3u * warpsize + dfindex];
            if (ncoord == 3u)
            {
                out[2u * outoffset + index] = d0 * df[4u * warpsize + dfindex] +
                                              d1 * df[5u * warpsize + dfindex];
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, unsigned int DIR,
          bool APPEND, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[warpsize * (nq0 * q + i) + ilane];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors.
            if constexpr (APPEND)
            {
                out[index] += d0 * df[2u * DIR * warpsize + dfindex] +
                              d1 * df[(2u * DIR + 1u) * warpsize + dfindex];
            }
            else
            {
                out[index] = d0 * df[2u * DIR * warpsize + dfindex] +
                             d1 * df[(2u * DIR + 1u) * warpsize + dfindex];
            }
        }
    }
}

template <bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor2DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[i * nq0 + q] * in0[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[j * nq1 + q] * in1[warpsize * (nq0 * q + i) + ilane];
            }

            if constexpr (APPEND)
            {
                out[warpsize * cnt_ji + ilane] += d0 + d1;
            }
            else
            {
                out[warpsize * cnt_ji + ilane] = d0 + d1;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[q * nq1 + j] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[q * nq2 + k] *
                          in[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                              SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp0 = f1m[j] * f2[k] * d0;
                    TData tmp1 = f0[i] * tmp0;
                    TData tmp2 = f2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + f1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    d0 *= f2[k];
                    d2 += f0[i] * d0;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= f2[k];
                    d1 *= f2[k];
                    d2 += f0[i] * d0 + f1[j] * d1;
                }

                // Multiply by derivative factors.
                out[index] = d0 * df[0u * warpsize + dfindex] +
                             d1 * df[1u * warpsize + dfindex] +
                             d2 * df[2u * warpsize + dfindex];
                out[outoffset + index] = d0 * df[3u * warpsize + dfindex] +
                                         d1 * df[4u * warpsize + dfindex] +
                                         d2 * df[5u * warpsize + dfindex];
                out[2u * outoffset + index] = d0 * df[6u * warpsize + dfindex] +
                                              d1 * df[7u * warpsize + dfindex] +
                                              d2 * df[8u * warpsize + dfindex];
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, unsigned int DIR,
          bool APPEND, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[q * nq1 + j] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[q * nq2 + k] *
                          in[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                              SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp0 = f1m[j] * f2[k] * d0;
                    TData tmp1 = f0[i] * tmp0;
                    TData tmp2 = f2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + f1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    d0 *= f2[k];
                    d2 += f0[i] * d0;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= f2[k];
                    d1 *= f2[k];
                    d2 += f0[i] * d0 + f1[j] * d1;
                }

                // Multiply by derivative factors.
                if constexpr (APPEND)
                {
                    out[index] += d0 * df[(3u * DIR) * warpsize + dfindex] +
                                  d1 * df[(3u * DIR + 1) * warpsize + dfindex] +
                                  d2 * df[(3u * DIR + 2) * warpsize + dfindex];
                }
                else
                {
                    out[index] = d0 * df[(3u * DIR) * warpsize + dfindex] +
                                 d1 * df[(3u * DIR + 1) * warpsize + dfindex] +
                                 d2 * df[(3u * DIR + 2) * warpsize + dfindex];
                }
            }
        }
    }
}

template <bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor3DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    const TData *NEK_RESTRICT in2, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[i * nq0 + q] *
                          in0[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[j * nq1 + q] *
                          in1[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[k * nq2 + q] *
                          in2[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_kji + ilane] += d0 + d1 + d2;
                }
                else
                {
                    out[warpsize * cnt_kji + ilane] = d0 + d1 + d2;
                }
            }
        }
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t nelmt,
    const unsigned int outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * warpsize * iwarp;
        TData *outptr      = out + nq0 * warpsize * iwarp;
        PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, outoffset, D0,
                                          dfptr, inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <bool DEFORMED, typename TthreadBlock, unsigned int DIR, bool APPEND,
          typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacKernel(
    const unsigned int ncoord, const unsigned int nq0, const size_t nelmt,
    const unsigned int outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * warpsize * iwarp;
        TData *outptr      = out + nq0 * warpsize * iwarp;
        PhysDerivDir1DSumFacKernel<DEFORMED, DIR, APPEND>(
            ilane, ncoord, nq0, outoffset, D0, dfptr, inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const unsigned int outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = 2 * ncoord;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;
        PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, outoffset, D0, D1, s_f0, s_f1, dfptr,
            inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, unsigned int DIR,
          bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir2DSumFacKernel(
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int ndf    = 4u;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;

        PhysDerivDir2DSumFacKernel<SHAPE_TYPE, DEFORMED, DIR, APPEND>(
            ilane, nq0, nq1, D0, D1, s_f0, s_f1, dfptr, inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const unsigned int outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;
        PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, outoffset, D0, D1, D2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, unsigned int DIR,
          bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir3DSumFacKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *NEK_RESTRICT shmemptr,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * warpsize * iwarp;
        TData *outptr      = out + nqTot * warpsize * iwarp;
        PhysDerivDir3DSumFacKernel<SHAPE_TYPE, DEFORMED, DIR, APPEND>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            inptr, outptr);
        e += getGlobalRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDeriv1DKernelLauncher(NonTemplated1DPhysSizeParameters sizeParam1D,
                              const size_t nelmt, const size_t outoffset,
                              const TData *NEK_RESTRICT D0,
                              const TData *NEK_RESTRICT df,
                              const TData *NEK_RESTRICT in,
                              TData *NEK_RESTRICT out,
                              const TthreadBlock &threadBlock)
{
    PhysDeriv1DSumFacKernel<DEFORMED>(sizeParam1D.ncoord(), sizeParam1D.nq0(),
                                      nelmt, outoffset, D0, df, in, out,
                                      threadBlock);
}

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, typename TthreadBlock,
          typename TData /*,
unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv1DKernelLauncher(
        [[maybe_unused]] Templated1DPhysSizeParameters<ncoord, nq0>,
        const size_t nelmt, const size_t outoffset,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        const TthreadBlock &threadBlock)
{
    PhysDeriv1DSumFacKernel<DEFORMED>(ncoord, nq0, nelmt, outoffset, D0, df, in,
                                      out, threadBlock);
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, unsigned int DIR, bool APPEND,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDerivDir1DKernelLauncher(NonTemplated1DPhysSizeParameters sizeParam1D,
                                 const size_t nelmt,
                                 const TData *NEK_RESTRICT D0,
                                 const TData *NEK_RESTRICT df,
                                 const TData *NEK_RESTRICT in,
                                 TData *NEK_RESTRICT out,
                                 const TthreadBlock &threadBlock)
{
    PhysDerivDir1DSumFacKernel<DEFORMED, DIR, APPEND>(
        sizeParam1D.ncoord(), sizeParam1D.nq0(), nelmt, D0, df, in, out,
        threadBlock);
}

// Size based template version.
template <typename Implementation, bool DEFORMED, unsigned int DIR, bool APPEND,
          unsigned int ncoord, unsigned int nq0, typename TthreadBlock,
          typename TData /*,
unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv1DKernelLauncher(
        [[maybe_unused]] Templated1DPhysSizeParameters<ncoord, nq0>,
        const size_t nelmt, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, const TthreadBlock &threadBlock)
{
    PhysDerivDir1DSumFacKernel<DEFORMED, DIR, APPEND>(ncoord, nq0, nelmt, D0,
                                                      df, in, out, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDeriv2DKernelLauncher(
        NonTemplated2DPhysSizeParameters sizeParam2D, const size_t nelmt,
        const unsigned int outoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam2D.ncoord(), sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt,
        outoffset, D0, D1, f0, f1, df, in, out, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int ncoord, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData /*,
           unsigned int maxThreadPerBlock =
               GetDeviceBlockSize<Implementation>(nq0 *nq1)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv2DKernelLauncher(
        [[maybe_unused]] Templated2DPhysSizeParameters<ncoord, nq0, nq1>,
        const size_t nelmt, const unsigned int outoffset,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, nelmt, outoffset, D0, D1, f0, f1, df, in, out,
        shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int DIR, bool APPEND, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDerivDir2DKernelLauncher(
        NonTemplated2DPhysSizeParameters sizeParam2D, const size_t nelmt,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivDir2DSumFacKernel<SHAPE_TYPE, DEFORMED, DIR, APPEND>(
        sizeParam2D.ncoord(), sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, D0,
        D1, f0, f1, df, in, out, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int DIR, bool APPEND, unsigned int ncoord, unsigned int nq0,
          unsigned int nq1, typename TthreadBlock, typename TData/*,
                                                                   unsigned int maxThreadPerBlock =
                                                                   GetDeviceBlockSize<Implementation>(nq0 *nq1)*/>
NEK_DEVICE_KERNEL 
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
 /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
PhysDerivDir2DKernelLauncher(
        [[maybe_unused]] Templated2DPhysSizeParameters<ncoord, nq0, nq1>,
const size_t nelmt, const TData *NEK_RESTRICT D0,
                             const TData *NEK_RESTRICT D1,
                             const TData *NEK_RESTRICT f0,
                             const TData *NEK_RESTRICT f1,
                             const TData *NEK_RESTRICT df,
                             const TData *NEK_RESTRICT in,
                             TData *NEK_RESTRICT out, unsigned char* shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivDir2DSumFacKernel<SHAPE_TYPE, DEFORMED, DIR, APPEND>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out, shmemptr,
        threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDeriv3DKernelLauncher(
        NonTemplated3DPhysSizeParameters sizeParam3D, const size_t nelmt,
        const size_t outoffset, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nq0(), sizeParam3D.nq1(), sizeParam3D.nq2(), nelmt,
        outoffset, D0, D1, D2, f0, f1, f1m, f2, df, in, out, shmemptr,
        threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nq0, unsigned int nq1, unsigned int nq2,
          typename TthreadBlock,
          typename TData /*,
           unsigned int maxThreadPerBlock =
               GetDeviceBlockSize<Implementation>(nq0 *nq1 *nq2)*/
          >
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    /*__LAUNCH_BOUNDS__(maxThreadPerBlock)*/
    PhysDeriv3DKernelLauncher(
        [[maybe_unused]] Templated3DPhysSizeParameters<nq0, nq1, nq2>,
        const size_t nelmt, const unsigned int outoffset,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, nelmt, outoffset, D0, D1, D2, f0, f1, f1m, f2, df, in,
        out, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int DIR, bool APPEND, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDerivDir3DKernelLauncher(
        NonTemplated3DPhysSizeParameters sizeParam3D, const size_t nelmt,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivDir3DSumFacKernel<SHAPE_TYPE, DEFORMED, DIR, APPEND>(
        sizeParam3D.nq0(), sizeParam3D.nq1(), sizeParam3D.nq2(), nelmt, D0, D1,
        D2, f0, f1, f1m, f2, df, in, out, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int DIR, bool APPEND, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFac>>::type
    PhysDerivDir3DKernelLauncher(
        [[maybe_unused]] Templated3DPhysSizeParameters<nq0, nq1, nq2>,
        const size_t nelmt, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    PhysDerivDir3DSumFacKernel<SHAPE_TYPE, DEFORMED, DIR, APPEND>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out,
        shmemptr, threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
