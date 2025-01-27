///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivKokkosSumFacKernels.hpp
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

#if defined(NEKTAR_ENABLE_KOKKOS)

namespace Nektar::Operators::detail
{

using team_handle = Kokkos::TeamPolicy<>::member_type;
template <typename TData>
using ScratchMemoryView =
    Kokkos::View<TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int outsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        // Compute tensorial derivative.
        TData d0 = 0.0;
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

template <bool APPEND, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION void SumDerivTensor1DKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT in0,
    TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // Compute tensorial derivative.
        TData d0 = 0.0;
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

template <bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv1DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const unsigned int outsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const team_handle &team)
{
    unsigned int dfsize = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nq0;
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nq0), [&](const unsigned int &i) {
            const unsigned int dfindex = DEFORMED ? i : 0;

            // Compute tensorial derivative.
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[q];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * outsize * nq0 + i] = d0 * df[d * dfsize + dfindex];
            }
        });
}

template <bool APPEND, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION void SumDerivTensor1DQPKernel(
    const unsigned int nq0, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT in0, TData *KOKKOS_RESTRICT out,
    const team_handle &team)
{
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                         [&](const unsigned int &i) {
                             // Compute tensorial derivative.
                             // Direction 0
                             TData d0 = 0.0;
                             for (unsigned int q = 0u; q < nq0; ++q)
                             {
                                 d0 += D0[i * nq0 + q] * in0[q];
                             }

                             if constexpr (APPEND)
                             {
                                 out[i] += d0;
                             }
                             else
                             {
                                 out[i] = d0;
                             }
                         });

    team.team_barrier();
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const unsigned int outsize,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT f0,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT f1,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
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
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
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

template <bool APPEND, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void SumDerivTensor2DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT in0, const TData *KOKKOS_RESTRICT in1,
    TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[i * nq0 + q] * in0[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
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
KOKKOS_INLINE_FUNCTION static void PhysDeriv2DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int outsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT f0,
    const TData *KOKKOS_RESTRICT f1, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const team_handle &team)
{
    const unsigned int nqTot = nq0 * nq1;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                         [&](const unsigned int &idx) {
                             const unsigned int i       = idx % nq0;
                             const unsigned int j       = idx / nq0;
                             const unsigned int dfindex = DEFORMED ? idx : 0;

                             // Compute tensorial derivative.
                             // Direction 0
                             TData d0 = 0.0;
                             for (unsigned int q = 0u; q < nq0; ++q)
                             {
                                 d0 += D0[q * nq0 + i] * in[nq0 * j + q];
                             }

                             // Direction 1
                             TData d1 = 0.0;
                             for (unsigned int q = 0u; q < nq1; ++q)
                             {
                                 d1 += D1[q * nq1 + j] * in[nq0 * q + i];
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
                                 out[d * outsize * nqTot + idx] =
                                     d0 * df[(2u * d) * dfsize + dfindex] +
                                     d1 * df[(2u * d + 1u) * dfsize + dfindex];
                             }
                         });

    team.team_barrier();
}

template <bool APPEND, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void SumDerivTensor2DQPKernel(
    const unsigned int nq0, const unsigned int nq1,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT in0, const TData *KOKKOS_RESTRICT in1,
    TData *KOKKOS_RESTRICT out, const team_handle &team)
{
    const unsigned int nqTot = nq0 * nq1;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                         [&](const unsigned int &idx) {
                             const unsigned int i = idx % nq0;
                             const unsigned int j = idx / nq0;

                             // Compute tensorial derivative.
                             // Direction 0
                             TData d0 = 0.0;
                             for (unsigned int q = 0u; q < nq0; ++q)
                             {
                                 d0 += D0[i * nq0 + q] * in0[nq0 * j + q];
                             }

                             // Direction 1
                             TData d1 = 0.0;
                             for (unsigned int q = 0u; q < nq1; ++q)
                             {
                                 d1 += D1[j * nq1 + q] * in1[nq0 * q + i];
                             }

                             if constexpr (APPEND)
                             {
                                 out[idx] += d0 + d1;
                             }
                             else
                             {
                                 out[idx] = d0 + d1;
                             }
                         });

    team.team_barrier();
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int outsize,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT D2,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT f0,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT f1,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT f1m,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT f2,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
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
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[q * nq1 + j] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
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

template <bool APPEND, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void SumDerivTensor3DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT in0, const TData *KOKKOS_RESTRICT in1,
    const TData *KOKKOS_RESTRICT in2, TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[i * nq0 + q] *
                          in0[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[j * nq1 + q] *
                          in1[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
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

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv3DSumFacQPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int outsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT f0, const TData *KOKKOS_RESTRICT f1,
    const TData *KOKKOS_RESTRICT f1m, const TData *KOKKOS_RESTRICT f2,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const team_handle &team)
{
    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nqTot), [&](const unsigned int &idx) {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = (idx / nq0) % nq1;
            const unsigned int k       = idx / (nq0 * nq1);
            const unsigned int dfindex = DEFORMED ? idx : 0;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[nq0 * nq1 * k + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[nq0 * nq1 * k + nq0 * q + i];
            }

            // Direction 2
            TData d2 = 0.0;
            for (unsigned int q = 0u; q < nq2; ++q)
            {
                d2 += D2[q * nq2 + k] * in[nq0 * nq1 * q + nq0 * j + i];
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
                out[d * outsize * nqTot + idx] =
                    d0 * df[(3u * d) * dfsize + dfindex] +
                    d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                    d2 * df[(3u * d + 2u) * dfsize + dfindex];
            }
        });

    team.team_barrier();
}

template <bool APPEND, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void SumDerivTensor3DQPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT D2, const TData *KOKKOS_RESTRICT in0,
    const TData *KOKKOS_RESTRICT in1, const TData *KOKKOS_RESTRICT in2,
    TData *KOKKOS_RESTRICT out, const team_handle &team)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, nqTot), [&](const unsigned int &idx) {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq0 * nq1);

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[i * nq0 + q] * in0[nq0 * nq1 * k + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[j * nq1 + q] * in1[nq0 * nq1 * k + nq0 * q + i];
            }

            // Direction 2
            TData d2 = 0.0;
            for (unsigned int q = 0u; q < nq2; ++q)
            {
                d2 += D2[k * nq2 + q] * in2[nq0 * nq1 * q + nq0 * j + i];
            }

            if constexpr (APPEND)
            {
                out[idx] += d0 + d1 + d2;
            }
            else
            {
                out[idx] = d0 + d1 + d2;
            }
        });

    team.team_barrier();
}

// General Launcher
template <typename Implementation, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION void PhysDeriv1DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const team_handle &team)
{
    const unsigned int ndf = ncoord;
    unsigned int dfsize    = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nq0;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e =
            team.league_rank() * team.team_size() + team.team_rank();
        while (e < nelmt)
        {
            const unsigned int ilane    = e % warpsize;
            const unsigned int iwarp    = e / warpsize;
            const unsigned int dfoffset = ndf * dfsize * warpsize * iwarp;
            const unsigned int offset   = nq0 * warpsize * iwarp;

            const TData *dfptr = df + dfoffset;
            const TData *inptr = in + offset;
            TData *outptr      = out + offset;
            PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt, D0,
                                              dfptr, inptr, outptr);
            e += team.team_size() * team.league_size();
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        const unsigned int e = team.league_rank();

        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nq0 * e;

        const TData *dfptr = df + dfoffset;
        const TData *inptr = in + offset;
        TData *outptr      = out + offset;
        PhysDeriv1DSumFacQPKernel<DEFORMED>(ncoord, nq0, nelmt, D0, dfptr,
                                            inptr, outptr, team);
    }
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION void PhysDeriv2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT f0,
    const TData *KOKKOS_RESTRICT f1, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    TData *KOKKOS_RESTRICT shmemptr, const team_handle &team)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        TData *s_f0 = nullptr;
        TData *s_f1 = nullptr;

        // Precompute geometric factors.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            s_f0 = shmemptr;
            s_f1 = s_f0 + nq0;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0),
                [&](const unsigned int &idx) { s_f0[idx] = f0[idx]; });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq1),
                [&](const unsigned int &idx) { s_f1[idx] = f1[idx]; });

            team.team_barrier();
        }

        unsigned int e =
            team.league_rank() * team.team_size() + team.team_rank();
        while (e < nelmt)
        {
            const unsigned int ilane    = e % warpsize;
            const unsigned int iwarp    = e / warpsize;
            const unsigned int dfoffset = ndf * dfsize * warpsize * iwarp;
            const unsigned int offset   = nqTot * warpsize * iwarp;

            const TData *dfptr = df + dfoffset;
            const TData *inptr = in + offset;
            TData *outptr      = out + offset;
            PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt, D0, D1, s_f0, s_f1, dfptr,
                inptr, outptr);
            e += team.team_size() * team.league_size();
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        const unsigned int e = team.league_rank();

        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        const TData *dfptr = df + dfoffset;
        const TData *inptr = in + offset;
        TData *outptr      = out + offset;
        PhysDeriv2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(ncoord, nq0, nq1, nelmt,
                                                        D0, D1, f0, f1, dfptr,
                                                        inptr, outptr, team);
    }
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT f0, const TData *KOKKOS_RESTRICT f1,
    const TData *KOKKOS_RESTRICT f1m, const TData *KOKKOS_RESTRICT f2,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, TData *KOKKOS_RESTRICT shmemptr,
    const team_handle &team)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    unsigned int dfsize        = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        TData *s_f0  = nullptr;
        TData *s_f1  = nullptr;
        TData *s_f1m = nullptr;
        TData *s_f2  = nullptr;

        // Precompute geometric factors.
        if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            s_f0  = shmemptr;
            s_f1  = s_f0 + nq0;
            s_f1m = s_f1 + nq1;
            s_f2  = s_f1m + nq1;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0),
                [&](const unsigned int &idx) { s_f0[idx] = f0[idx]; });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                 [&](const unsigned int &idx) {
                                     s_f1[idx]  = f1[idx];
                                     s_f1m[idx] = f1m[idx];
                                 });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq2),
                [&](const unsigned int &idx) { s_f2[idx] = f2[idx]; });

            team.team_barrier();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            s_f0 = shmemptr;
            s_f2 = s_f0 + nq0;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0),
                [&](const unsigned int &idx) { s_f0[idx] = f0[idx]; });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq2),
                [&](const unsigned int &idx) { s_f2[idx] = f2[idx]; });

            team.team_barrier();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            s_f0 = shmemptr;
            s_f1 = s_f0 + nq0;
            s_f2 = s_f1 + nq1;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0),
                [&](const unsigned int &idx) { s_f0[idx] = f0[idx]; });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq1),
                [&](const unsigned int &idx) { s_f1[idx] = f1[idx]; });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq2),
                [&](const unsigned int &idx) { s_f2[idx] = f2[idx]; });

            team.team_barrier();
        }

        unsigned int e =
            team.league_rank() * team.team_size() + team.team_rank();
        while (e < nelmt)
        {
            const unsigned int ilane    = e % warpsize;
            const unsigned int iwarp    = e / warpsize;
            const unsigned int dfoffset = ndf * dfsize * warpsize * iwarp;
            const unsigned int offset   = nqTot * warpsize * iwarp;

            const TData *dfptr = df + dfoffset;
            const TData *inptr = in + offset;
            TData *outptr      = out + offset;
            PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt, D0, D1, D2, s_f0, s_f1, s_f1m,
                s_f2, dfptr, inptr, outptr);
            e += team.team_size() * team.league_size();
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        const unsigned int e = team.league_rank();

        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        const TData *dfptr = df + dfoffset;
        const TData *inptr = in + offset;
        TData *outptr      = out + offset;
        PhysDeriv3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, dfptr, inptr,
            outptr, team);
    }
}

// Launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int ncoord,
                                               const unsigned int nq0,
                                               const unsigned int nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    const unsigned int blocksize = GetKokkosBlockSize<Implementation>(nq0);
    const unsigned int gridsize  = GetKokkosGridSize<Implementation>(nelmt);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize),
        KOKKOS_LAMBDA(const team_handle &team) {
            PhysDeriv1DKernel<Implementation, DEFORMED>(ncoord, nq0, nelmt, D0,
                                                        df, in, out, team);
        });
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int ncoord, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    PhysDeriv1DKernel<ExecSpace, Implementation, DEFORMED>(ncoord, nq0, nelmt,
                                                           D0, df, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *f0,
    const TData *f1, const TData *df, const TData *in, TData *out)
{
    constexpr unsigned int slevel = 0u;
    const unsigned int nshared =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
    const unsigned int shmemsize =
        ScratchMemoryView<TData>::shmem_size(nshared);
    const unsigned int blocksize =
        GetKokkosBlockSize<Implementation>(nq0 * nq1);
    const unsigned int gridsize = GetKokkosGridSize<Implementation>(nelmt);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmemsize)),
        KOKKOS_LAMBDA(const team_handle &team) {
            ScratchMemoryView<TData> shmem(team.team_scratch(slevel), nshared);
            PhysDeriv2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out,
                shmem.data(), team);
        });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(const unsigned int nelmt,
                                               const TData *D0, const TData *D1,
                                               const TData *f0, const TData *f1,
                                               const TData *df, const TData *in,
                                               TData *out)
{
    PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *in, TData *out)
{
    constexpr unsigned int slevel = 0u;
    const unsigned int nshared =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2);
    const unsigned int shmemsize =
        ScratchMemoryView<TData>::shmem_size(nshared);
    const unsigned int blocksize =
        GetKokkosBlockSize<Implementation>(nq0 * nq1 * nq2);
    const unsigned int gridsize = GetKokkosGridSize<Implementation>(nelmt);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmemsize)),
        KOKKOS_LAMBDA(const team_handle &team) {
            ScratchMemoryView<TData> shmem(team.team_scratch(slevel), nshared);
            PhysDeriv3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
                nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out,
                shmem.data(), team);
        });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *in, TData *out)
{
    PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out);
}

} // namespace Nektar::Operators::detail

#endif
