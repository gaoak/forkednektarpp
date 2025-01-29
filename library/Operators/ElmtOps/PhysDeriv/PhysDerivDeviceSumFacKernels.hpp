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

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline constexpr unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return 0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            return nq0 + nq1;
        }
    }
    else
    {
        return 0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline constexpr unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                                        const unsigned int nq1,
                                                        const unsigned int nq2)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return 0;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return nq0 + 2u * nq1 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            return nq0 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            return nq0 + nq1 + nq2;
        }
    }
    else
    {
        return 0;
    }
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int outsize, const TData *__restrict__ D0,
    const TData *__restrict__ df, const TData *__restrict__ in,
    TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

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
        for (unsigned int d = 0u; d < ncoord; d++)
        {
            out[d * outsize * nq0 + index] = d0 * df[d * warpsize + dfindex];
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const unsigned int outsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

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
            if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * outsize * nqTot + index] =
                    d0 * df[(2u * d) * warpsize + dfindex] +
                    d1 * df[(2u * d + 1u) * warpsize + dfindex];
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int outsize,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, [[maybe_unused]] const TData *__restrict__ f0,
    [[maybe_unused]] const TData *__restrict__ f1,
    [[maybe_unused]] const TData *__restrict__ f1m,
    [[maybe_unused]] const TData *__restrict__ f2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

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
                if constexpr (SHAPE_TYPE == LibUtilities::Tet)
                {
                    TData tmp0 = f1m[j] * f2[k] * d0;
                    TData tmp1 = f0[i] * tmp0;
                    TData tmp2 = f2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + f1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
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
                for (unsigned int d = 0u; d < ncoord; d++)
                {
                    out[d * outsize * nqTot + index] =
                        d0 * df[(3u * d) * warpsize + dfindex] +
                        d1 * df[(3u * d + 1u) * warpsize + dfindex] +
                        d2 * df[(3u * d + 2u) * warpsize + dfindex];
                }
            }
        }
    }
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/PhysDeriv/PhysDerivCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSYCLSumFacKernels.hpp"
