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
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT D0,
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
            out[d * nelmt * nq0 + index] = d0 * df[d * warpsize + dfindex];
        }
    }
}

template <bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv1DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const team_handle &team)
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
                out[d * nelmt * nq0 + i] = d0 * df[d * dfsize + dfindex];
            }
        });
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT xfrm0,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT xfrm1,
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
                d0 *= xfrm0[j];
                d1 += d0 * xfrm1[i];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nqTot + index] =
                    d0 * df[(2u * d) * warpsize + dfindex] +
                    d1 * df[(2u * d + 1u) * warpsize + dfindex];
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv2DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT df,
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
                                 TData xfrm0 = 2.0 / (1.0 - Z1[j]);
                                 TData xfrm1 = 0.5 * (1.0 + Z0[i]);
                                 d0 *= xfrm0;
                                 d1 += d0 * xfrm1;
                             }

                             // Multiply by derivative factors.
                             for (unsigned int d = 0u; d < ncoord; d++)
                             {
                                 out[d * nelmt * nqTot + idx] =
                                     d0 * df[(2u * d) * dfsize + dfindex] +
                                     d1 * df[(2u * d + 1u) * dfsize + dfindex];
                             }
                         });

    team.team_barrier();
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT D2,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT xfrm_eta0,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT xfrm_eta1,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT xfrm_eta1m,
    [[maybe_unused]] const TData *KOKKOS_RESTRICT xfrm_eta2,
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
                    TData xfrm = xfrm_eta1m[j] * xfrm_eta2[k];
                    TData tmp0 = xfrm * d0;
                    TData tmp1 = xfrm_eta0[i] * tmp0;
                    TData tmp2 = xfrm_eta2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + xfrm_eta1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
                {
                    d0 *= xfrm_eta2[k];
                    d2 += xfrm_eta0[i] * d0;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= xfrm_eta2[k];
                    d1 *= xfrm_eta2[k];
                    d2 += xfrm_eta0[i] * d0 + xfrm_eta1[j] * d1;
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0u; d < ncoord; d++)
                {
                    out[d * nelmt * nqTot + index] =
                        d0 * df[(3u * d) * warpsize + dfindex] +
                        d1 * df[(3u * d + 1u) * warpsize + dfindex] +
                        d2 * df[(3u * d + 2u) * warpsize + dfindex];
                }
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
KOKKOS_INLINE_FUNCTION static void PhysDeriv3DSumFacQPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const team_handle &team)
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
                TData xfrm_eta0  = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                TData xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                TData xfrm_eta2  = 2.0 / (1.0 - Z2[k]);

                TData xfrm = xfrm_eta1m * xfrm_eta2;
                TData tmp0 = xfrm * d0;
                TData tmp1 = xfrm_eta0 * tmp0;
                TData tmp2 = xfrm_eta2 * d1;
                d0         = tmp0;
                d1         = tmp1 + tmp2;
                d2 += tmp1 + xfrm_eta1 * tmp2;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d1 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nqTot + idx] =
                    d0 * df[(3u * d) * dfsize + dfindex] +
                    d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                    d2 * df[(3u * d + 2u) * dfsize + dfindex];
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
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT df,
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

        TData *s_xfrm0 = nullptr;
        TData *s_xfrm1 = nullptr;

        // Precompute geometric factors.
        if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            s_xfrm0 = shmemptr;
            s_xfrm1 = s_xfrm0 + nq1;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                 [&](const unsigned int &idx) {
                                     s_xfrm0[idx] = 2.0 / (1.0 - Z1[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                 [&](const unsigned int &idx) {
                                     s_xfrm1[idx] = 0.5 * (1.0 + Z0[idx]);
                                 });

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
                ilane, ncoord, nq0, nq1, nelmt, D0, D1, s_xfrm0, s_xfrm1, dfptr,
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
                                                        D0, D1, Z0, Z1, dfptr,
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
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    TData *KOKKOS_RESTRICT shmemptr, const team_handle &team)
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

        TData *s_xfrm_eta0  = nullptr;
        TData *s_xfrm_eta1  = nullptr;
        TData *s_xfrm_eta1m = nullptr;
        TData *s_xfrm_eta2  = nullptr;

        // Precompute geometric factors.
        if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            s_xfrm_eta0  = shmemptr;
            s_xfrm_eta1  = s_xfrm_eta0 + nq0;
            s_xfrm_eta1m = s_xfrm_eta1 + nq1;
            s_xfrm_eta2  = s_xfrm_eta1m + nq1;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta1m[idx] = 2.0 / (1.0 - Z1[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
                                 });

            team.team_barrier();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            s_xfrm_eta0 = shmemptr;
            s_xfrm_eta2 = s_xfrm_eta0 + nq0;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
                                 });

            team.team_barrier();
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            s_xfrm_eta0 = shmemptr;
            s_xfrm_eta1 = s_xfrm_eta0 + nq0;
            s_xfrm_eta2 = s_xfrm_eta1 + nq1;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                 [&](const unsigned int &idx) {
                                     s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
                                 });

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
                ilane, nq0, nq1, nq2, nelmt, D0, D1, D2, s_xfrm_eta0,
                s_xfrm_eta1, s_xfrm_eta1m, s_xfrm_eta2, dfptr, inptr, outptr);
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
            nq0, nq1, nq2, nelmt, D0, D1, D2, Z0, Z1, Z2, dfptr, inptr, outptr,
            team);
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
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *Z0,
    const TData *Z1, const TData *df, const TData *in, TData *out)
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
                ncoord, nq0, nq1, nelmt, D0, D1, Z0, Z1, df, in, out,
                shmem.data(), team);
        });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(const unsigned int nelmt,
                                               const TData *D0, const TData *D1,
                                               const TData *Z0, const TData *Z1,
                                               const TData *df, const TData *in,
                                               TData *out)
{
    PhysDeriv2DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, Z0, Z1, df, in, out);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *Z0, const TData *Z1, const TData *Z2, const TData *df,
    const TData *in, TData *out)
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
                nq0, nq1, nq2, nelmt, D0, D1, D2, Z0, Z1, Z2, df, in, out,
                shmem.data(), team);
        });
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(const unsigned int nelmt,
                                               const TData *D0, const TData *D1,
                                               const TData *D2, const TData *Z0,
                                               const TData *Z1, const TData *Z2,
                                               const TData *df, const TData *in,
                                               TData *out)
{
    PhysDeriv3DKernel<SHAPE_TYPE, ExecSpace, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, Z0, Z1, Z2, df, in, out);
}

} // namespace Nektar::Operators::detail

#endif
