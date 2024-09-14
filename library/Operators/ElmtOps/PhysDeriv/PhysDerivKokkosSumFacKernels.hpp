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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED>
void PhysDeriv1DKernel(const unsigned int nq0, const unsigned int ncoord,
                       const unsigned int nelmt, const unsigned int nsize,
                       const unsigned int dfsize,
                       const TData *KOKKOS_RESTRICT D0,
                       const TData *KOKKOS_RESTRICT df,
                       const TData *KOKKOS_RESTRICT in,
                       TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, NektarSpaces::KOKKOS::defaultBlockSize),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int i = 0u; i < nq0; ++i)
            {
                const unsigned int index =
                    nq0 * warpsize * iwarp + warpsize * i + ilane;
                const unsigned int dfindex = DEFORMED ? index : e;

                // Compute tensorial derivative.
                TData d0 = 0.0;
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[nq0 * warpsize * iwarp + warpsize * q + ilane];
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0u; d < ncoord; d++)
                {
                    out[d * nsize + index] = d0 * df[d * dfsize + dfindex];
                }
            }
        });
}

template <typename TData, bool DEFORMED>
void PhysDeriv1DKernel_QP(const unsigned int nq0, const unsigned int ncoord,
                          const unsigned int nelmt, const unsigned int nsize,
                          const unsigned int dfsize,
                          const TData *KOKKOS_RESTRICT D0,
                          const TData *KOKKOS_RESTRICT df,
                          const TData *KOKKOS_RESTRICT in,
                          TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            const unsigned int e      = team.league_rank();
            const unsigned int offset = nq0 * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0), [&](const unsigned int &i) {
                    const unsigned int index   = offset + i;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    // Compute tensorial derivative.
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += D0[q * nq0 + i] * in[offset + q];
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] = d0 * df[d * dfsize + dfindex];
                    }
                });
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv2DKernel(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, NektarSpaces::KOKKOS::defaultBlockSize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_D0 = SHMEM ? &scratch[0] : (TData *)D0;
            TData *s_D1 = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
            TData *s_xfrm0, *s_xfrm1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq0),
                    [&](const unsigned int &idx) { s_D0[idx] = D0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1 * nq1),
                    [&](const unsigned int &idx) { s_D1[idx] = D1[idx]; });
            }

            // Precompute geometric factors.
            if (SHAPETYPE == LibUtilities::Tri)
            {
                s_xfrm0 = SHMEM ? s_D1 + nq1 * nq1 : &scratch[0];
                s_xfrm1 = s_xfrm0 + nq1;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_xfrm0[idx] = 2.0 / (1.0 - Z1[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_xfrm1[idx] = 0.5 * (1.0 + Z0[idx]);
                                     });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += D0[q * nq0 + i] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * j + q) + ilane];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += D1[q * nq1 + j] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * q + i) + ilane];
                    }

                    // Moving from standard to collapsed coordinates.
                    if (SHAPETYPE == LibUtilities::Tri)
                    {
                        d0 *= s_xfrm0[i];
                        d1 += d0 * s_xfrm1[i];
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(2u * d) * dfsize + dfindex] +
                            d1 * df[(2u * d + 1u) * dfsize + dfindex];
                    }
                }
            }
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv2DKernel_QP(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_wsp = &scratch[0];
            TData *s_D0  = SHMEM ? s_wsp + nqTot : (TData *)D0;
            TData *s_D1  = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq0),
                    [&](const unsigned int &idx) { s_D0[idx] = D0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1 * nq1),
                    [&](const unsigned int &idx) { s_D1[idx] = D1[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e      = team.league_rank();
            const unsigned int offset = nqTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     s_wsp[idx] = in[offset + idx];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                    team, nq0, nq1),
                [&](const unsigned int &i, const unsigned int &j) {
                    const unsigned int cnt_ji  = nq0 * j + i;
                    const unsigned int index   = offset + cnt_ji;
                    const unsigned int dfindex = DEFORMED ? index : e;
                    TData xfrm0, xfrm1;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] * s_wsp[nq0 * j + q];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] * s_wsp[nq0 * q + i];
                    }

                    // Moving from standard to collapsed coordinates.
                    if (SHAPETYPE == LibUtilities::Tri)
                    {
                        xfrm0 = 2.0 / (1.0 - Z1[j]);
                        xfrm1 = 0.5 * (1.0 + Z0[i]);
                        d0 *= xfrm0;
                        d1 += d0 * xfrm1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(2u * d) * dfsize + dfindex] +
                            d1 * df[(2u * d + 1u) * dfsize + dfindex];
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv2DKernel_QP_1D(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT D0, const TData *KOKKOS_RESTRICT D1,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_wsp = &scratch[0];
            TData *s_D0  = SHMEM ? s_wsp + nqTot : (TData *)D0;
            TData *s_D1  = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq0),
                    [&](const unsigned int &idx) { s_D0[idx] = D0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1 * nq1),
                    [&](const unsigned int &idx) { s_D1[idx] = D1[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e      = team.league_rank();
            const unsigned int offset = nqTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     s_wsp[idx] = in[offset + idx];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nqTot),
                [&](const unsigned int &idx) {
                    const unsigned int i       = idx % nq0;
                    const unsigned int j       = idx / nq0;
                    const unsigned int index   = offset + idx;
                    const unsigned int dfindex = DEFORMED ? index : e;
                    TData xfrm0, xfrm1;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] * s_wsp[nq0 * j + q];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] * s_wsp[nq0 * q + i];
                    }

                    // Moving from standard to collapsed coordinates.
                    if (SHAPETYPE == LibUtilities::Tri)
                    {
                        xfrm0 = 2.0 / (1.0 - Z1[j]);
                        xfrm1 = 0.5 * (1.0 + Z0[i]);
                        d0 *= xfrm0;
                        d1 += d0 * xfrm1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(2u * d) * dfsize + dfindex] +
                            d1 * df[(2u * d + 1u) * dfsize + dfindex];
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv3DKernel(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int ncoord   = 3u;
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, NektarSpaces::KOKKOS::defaultBlockSize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_D0 = SHMEM ? &scratch[0] : (TData *)D0;
            TData *s_D1 = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
            TData *s_D2 = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
            TData *s_xfrm_eta0, *s_xfrm_eta1, *s_xfrm_eta1m, *s_xfrm_eta2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq0),
                    [&](const unsigned int &idx) { s_D0[idx] = D0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1 * nq1),
                    [&](const unsigned int &idx) { s_D1[idx] = D1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2 * nq2),
                    [&](const unsigned int &idx) { s_D2[idx] = D2[idx]; });
            }

            // Precompute geometric factors.
            if (SHAPETYPE == LibUtilities::Tet)
            {
                s_xfrm_eta0  = SHMEM ? s_D2 + nq2 * nq2 : &scratch[0];
                s_xfrm_eta1  = s_xfrm_eta0 + nq0;
                s_xfrm_eta1m = s_xfrm_eta1 + nq1;
                s_xfrm_eta2  = s_xfrm_eta1m + nq1;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta0[idx] =
                                             0.5 * (1.0 + Z0[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta1[idx] =
                                             0.5 * (1.0 + Z1[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta1m[idx] =
                                             2.0 / (1.0 - Z1[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta2[idx] =
                                             2.0 / (1.0 - Z2[idx]);
                                     });
            }
            else if (SHAPETYPE == LibUtilities::Prism)
            {
                s_xfrm_eta0 = SHMEM ? s_D2 + nq2 * nq2 : &scratch[0];
                s_xfrm_eta2 = s_xfrm_eta0 + nq0;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta0[idx] =
                                             0.5 * (1.0 + Z0[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta2[idx] =
                                             2.0 / (1.0 - Z2[idx]);
                                     });
            }
            else if (SHAPETYPE == LibUtilities::Pyr)
            {
                s_xfrm_eta0 = SHMEM ? s_D2 + nq2 * nq2 : &scratch[0];
                s_xfrm_eta1 = s_xfrm_eta0 + nq0;
                s_xfrm_eta2 = s_xfrm_eta1 + nq1;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta0[idx] =
                                             0.5 * (1.0 + Z0[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta1[idx] =
                                             0.5 * (1.0 + Z1[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                     [&](const unsigned int &idx) {
                                         s_xfrm_eta2[idx] =
                                             2.0 / (1.0 - Z2[idx]);
                                     });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
            {
                for (unsigned int j = 0u; j < nq1; j++)
                {
                    for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        const unsigned int dfindex = DEFORMED ? index : e;

                        // Compute tensorial derivative.
                        // Direction 0
                        TData d0 = 0.0;
                        for (unsigned int q = 0u; q < nq0; ++q)
                        {
                            d0 += D0[q * nq0 + i] *
                                  in[nqTot * warpsize * iwarp +
                                     warpsize * (nq0 * nq1 * k + nq0 * j + q) +
                                     ilane];
                        }

                        // Direction 1
                        TData d1 = 0.0;
                        for (unsigned int q = 0u; q < nq1; ++q)
                        {
                            d1 += D1[q * nq1 + j] *
                                  in[nqTot * warpsize * iwarp +
                                     warpsize * (nq0 * nq1 * k + nq0 * q + i) +
                                     ilane];
                        }

                        // Direction 2
                        TData d2 = 0.0;
                        for (unsigned int q = 0u; q < nq2; ++q)
                        {
                            d2 += D2[q * nq2 + k] *
                                  in[nqTot * warpsize * iwarp +
                                     warpsize * (nq0 * nq1 * q + nq0 * j + i) +
                                     ilane];
                        }

                        // Moving from standard to collapsed coordinates.
                        if (SHAPETYPE == LibUtilities::Tet)
                        {
                            TData xfrm = s_xfrm_eta1m[j] * s_xfrm_eta2[k];
                            TData tmp0 = xfrm * d0;
                            TData tmp1 = s_xfrm_eta0[i] * tmp0;
                            TData tmp2 = s_xfrm_eta2[k] * d1;
                            d0         = tmp0;
                            d1         = tmp1 + tmp2;
                            d2 += tmp1 + s_xfrm_eta1[j] * tmp2;
                        }
                        else if (SHAPETYPE == LibUtilities::Prism)
                        {
                            d0 *= s_xfrm_eta2[k];
                            d2 += s_xfrm_eta0[i] * d0;
                        }
                        else if (SHAPETYPE == LibUtilities::Pyr)
                        {
                            d0 *= s_xfrm_eta2[k];
                            d1 *= s_xfrm_eta2[k];
                            d2 += s_xfrm_eta0[i] * d0 + s_xfrm_eta1[j] * d1;
                        }

                        // Multiply by derivative factors.
                        for (unsigned int d = 0u; d < ncoord; d++)
                        {
                            out[d * nsize + index] =
                                d0 * df[(3u * d) * dfsize + dfindex] +
                                d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                                d2 * df[(3u * d + 2u) * dfsize + dfindex];
                        }
                    }
                }
            }
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv3DKernel_QP(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_wsp = &scratch[0];
            TData *s_D0  = SHMEM ? s_wsp + nqTot : (TData *)D0;
            TData *s_D1  = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
            TData *s_D2  = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq0),
                    [&](const unsigned int &idx) { s_D0[idx] = D0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1 * nq1),
                    [&](const unsigned int &idx) { s_D1[idx] = D1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2 * nq2),
                    [&](const unsigned int &idx) { s_D2[idx] = D2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e      = team.league_rank();
            const unsigned int offset = nqTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     s_wsp[idx] = in[offset + idx];
                                 });

            team.team_barrier();

            // Compute tensorial derivative.
            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq0, nq1, nq2),
                [&](const unsigned int &i, const unsigned int &j,
                    const unsigned int &k) {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    const unsigned int index   = offset + cnt_kji;
                    const unsigned int dfindex = DEFORMED ? index : e;
                    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] *
                              s_wsp[nq0 * nq1 * k + nq0 * j + q];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] *
                              s_wsp[nq0 * nq1 * k + nq0 * q + i];
                    }

                    // Direction 2
                    TData d2 = 0.0;
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += s_D2[q * nq2 + k] *
                              s_wsp[nq0 * nq1 * q + nq0 * j + i];
                    }

                    // Moving from standard to collapsed coordinates.
                    if (SHAPETYPE == LibUtilities::Tet)
                    {
                        xfrm_eta0  = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                        xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                        xfrm_eta2  = 2.0 / (1.0 - Z2[k]);

                        TData xfrm = xfrm_eta1m * xfrm_eta2;
                        TData tmp0 = xfrm * d0;
                        TData tmp1 = xfrm_eta0 * tmp0;
                        TData tmp2 = xfrm_eta2 * d1;
                        d0         = tmp0;
                        d1         = tmp1 + tmp2;
                        d2 += tmp1 + xfrm_eta1 * tmp2;
                    }
                    else if (SHAPETYPE == LibUtilities::Prism)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0;
                    }
                    else if (SHAPETYPE == LibUtilities::Pyr)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d1 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(3u * d) * dfsize + dfindex] +
                            d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                            d2 * df[(3u * d + 2u) * dfsize + dfindex];
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv3DKernel_QP_1D(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *KOKKOS_RESTRICT D0,
    const TData *KOKKOS_RESTRICT D1, const TData *KOKKOS_RESTRICT D2,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_wsp = &scratch[0];
            TData *s_D0  = SHMEM ? s_wsp + nqTot : (TData *)D0;
            TData *s_D1  = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
            TData *s_D2  = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq0),
                    [&](const unsigned int &idx) { s_D0[idx] = D0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1 * nq1),
                    [&](const unsigned int &idx) { s_D1[idx] = D1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2 * nq2),
                    [&](const unsigned int &idx) { s_D2[idx] = D2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e      = team.league_rank();
            const unsigned int offset = nqTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     s_wsp[idx] = in[offset + idx];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nqTot),
                [&](const unsigned int &idx) {
                    const unsigned int i       = idx % nq0;
                    const unsigned int j       = (idx / nq0) % nq1;
                    const unsigned int k       = idx / (nq0 * nq1);
                    unsigned int index         = offset + idx;
                    const unsigned int dfindex = DEFORMED ? index : e;
                    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] *
                              s_wsp[nq0 * nq1 * k + nq0 * j + q];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] *
                              s_wsp[nq0 * nq1 * k + nq0 * q + i];
                    }

                    // Direction 2
                    TData d2 = 0.0;
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += s_D2[q * nq2 + k] *
                              s_wsp[nq0 * nq1 * q + nq0 * j + i];
                    }

                    // Moving from standard to collapsed coordinates.
                    if (SHAPETYPE == LibUtilities::Tet)
                    {
                        xfrm_eta0  = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                        xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                        xfrm_eta2  = 2.0 / (1.0 - Z2[k]);

                        TData xfrm = xfrm_eta1m * xfrm_eta2;
                        TData tmp0 = xfrm * d0;
                        TData tmp1 = xfrm_eta0 * tmp0;
                        TData tmp2 = xfrm_eta2 * d1;
                        d0         = tmp0;
                        d1         = tmp1 + tmp2;
                        d2 += tmp1 + xfrm_eta1 * tmp2;
                    }
                    else if (SHAPETYPE == LibUtilities::Prism)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0;
                    }
                    else if (SHAPETYPE == LibUtilities::Pyr)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d1 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nsize + index] =
                            d0 * df[(3u * d) * dfsize + dfindex] +
                            d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                            d2 * df[(3u * d + 2u) * dfsize + dfindex];
                    }
                });

            team.team_barrier();
        });
}

// Launchers
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
PhysDeriv1DKernel(const unsigned int nq0, const unsigned int ncoord,
                  const unsigned int nelmts, const unsigned int nsize,
                  const unsigned int dfsize, const TData *D0, const TData *df,
                  const TData *in, TData *out)
{
    if constexpr (MULTILEVEL)
    {
        PhysDeriv1DKernel_QP<TData, DEFORMED>(nq0, ncoord, nelmts, nsize,
                                              dfsize, D0, df, in, out);
    }
    else
    {
        PhysDeriv1DKernel<TData, DEFORMED>(nq0, ncoord, nelmts, nsize, dfsize,
                                           D0, df, in, out);
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true, bool SHMEM = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
PhysDeriv2DKernel(LibUtilities::ShapeType shapetype, const unsigned int nq0,
                  const unsigned int nq1, const unsigned int ncoord,
                  const unsigned int nelmts, const unsigned int nsize,
                  const unsigned int dfsize, const TData *D0, const TData *D1,
                  const TData *Z0, const TData *Z1, const TData *df,
                  const TData *in, TData *out)
{
    unsigned int nshared = SHMEM ? nq0 * nq0 + nq1 * nq1 : 0u;

    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1;
            PhysDeriv2DKernel_QP<TData, LibUtilities::Quad, DEFORMED, SHMEM>(
                nshared, nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1,
                nullptr, nullptr, df, in, out);
        }
        else
        {
            PhysDeriv2DKernel<TData, LibUtilities::Quad, DEFORMED, SHMEM>(
                nshared, nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1,
                nullptr, nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1;
            PhysDeriv2DKernel_QP<TData, LibUtilities::Tri, DEFORMED, SHMEM>(
                nshared, nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1, Z0,
                Z1, df, in, out);
        }
        else
        {
            nshared += nq0 * nq1;
            PhysDeriv2DKernel<TData, LibUtilities::Tri, DEFORMED, SHMEM>(
                nshared, nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1, Z0,
                Z1, df, in, out);
        }
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true, bool SHMEM = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
PhysDeriv3DKernel(LibUtilities::ShapeType shapetype, const unsigned int nq0,
                  const unsigned int nq1, const unsigned int nq2,
                  const unsigned int nelmts, const unsigned int nsize,
                  const unsigned int dfsize, const TData *D0, const TData *D1,
                  const TData *D2, const TData *Z0, const TData *Z1,
                  const TData *Z2, const TData *df, const TData *in, TData *out)
{
    unsigned int nshared = SHMEM ? nq0 * nq0 + nq1 * nq1 + nq2 * nq2 : 0u;

    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            PhysDeriv3DKernel_QP<TData, LibUtilities::Hex, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                nullptr, nullptr, nullptr, df, in, out);
        }
        else
        {
            PhysDeriv3DKernel<TData, LibUtilities::Hex, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                nullptr, nullptr, nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            PhysDeriv3DKernel_QP<TData, LibUtilities::Tet, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2, Z0,
                Z1, Z2, df, in, out);
        }
        else
        {
            nshared += nq0 + 2u * nq1 + nq2;
            PhysDeriv3DKernel<TData, LibUtilities::Tet, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2, Z0,
                Z1, Z2, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            PhysDeriv3DKernel_QP<TData, LibUtilities::Prism, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2, Z0,
                nullptr, Z2, df, in, out);
        }
        else
        {
            nshared += nq0 + nq2;
            PhysDeriv3DKernel<TData, LibUtilities::Prism, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2, Z0,
                nullptr, Z2, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            PhysDeriv3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2, Z0,
                Z1, Z2, df, in, out);
        }
        else
        {
            nshared += nq0 + nq1 + nq2;
            PhysDeriv3DKernel<TData, LibUtilities::Pyr, DEFORMED, SHMEM>(
                nshared, nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2, Z0,
                Z1, Z2, df, in, out);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
