///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseKokkosSumFacKernels.hpp
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

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseSegKernel(const unsigned int ssize, const unsigned int nm0,
                              const unsigned int nq0, const unsigned int nelmt,
                              const TData *KOKKOS_RESTRICT basis0,
                              const TData *KOKKOS_RESTRICT w0,
                              const TData *KOKKOS_RESTRICT jac,
                              const TData *KOKKOS_RESTRICT in,
                              TData *KOKKOS_RESTRICT out,
                              const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_w0     = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)w0;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int p = 0u; p < nm0; ++p)
            {
                TData sum = 0.0;
                for (unsigned int i = 0u; i < nq0; ++i)
                {
                    const unsigned int index =
                        nq0 * warpsize * iwarp + warpsize * i + ilane;
                    const unsigned int jacindex = DEFORMED ? index : e;
                    sum += in[index] * s_basis0[p * nq0 + i] * jac[jacindex] *
                           s_w0[i];
                }

                if (SCALE)
                {
                    sum *= scale;
                }

                const unsigned int index =
                    nm0 * warpsize * iwarp + warpsize * p + ilane;
                if (APPEND)
                {
                    out[index] += sum;
                }
                else
                {
                    out[index] = sum;
                }
            }
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseSegKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT jac,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

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
            TData *s_wsp0   = &scratch[0];
            TData *s_basis0 = SHMEM ? s_wsp0 + nq0 : (TData *)basis0;
            TData *s_w0     = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)w0;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nq0 * e;
            const unsigned int outoffset = nm0 * e;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0), [&](const unsigned int &i) {
                    const unsigned int index    = inoffset + i;
                    const unsigned int jacindex = DEFORMED ? index : e;
                    s_wsp0[i]                   = in[index] * jac[jacindex];
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0), [&](const unsigned int &p) {
                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i)
                    {
                        sum += s_wsp0[i] * s_basis0[p * nq0 + i] * s_w0[i];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    const unsigned int index = outoffset + p;
                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseQuadKernel(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT jac,
    TData *KOKKOS_RESTRICT wsp, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_w0     = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int p = 0u; p < nm0; ++p)
            {
                for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
                {
                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_ji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;
                        sum += in[index] * s_basis0[p * nq0 + i] *
                               jac[jacindex] * s_w0[i];
                    }
                    wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] = sum;
                }

                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    TData sum = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j)
                    {
                        sum +=
                            wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] *
                            s_basis1[q * nq1 + j] * s_w1[j];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    const unsigned int index = nmTot * warpsize * iwarp +
                                               warpsize * (nm0 * q + p) + ilane;
                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                }
            }
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseQuadKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT jac,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
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
            TData *s_wsp0   = &scratch[0];
            TData *s_wsp1   = s_wsp0 + nqTot;
            TData *s_basis0 = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_w0     = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                    team, nq1, nm0),
                [&](const unsigned int &j, const unsigned int &p) {
                    const unsigned int cnt_pj = nq1 * p + j;
                    unsigned int cnt_ji       = nq0 * j;

                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
                    }
                    s_wsp1[cnt_pj] = sum;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                    team, nm0, nm1),
                [&](const unsigned int &p, const unsigned int &q) {
                    const unsigned int cnt_pq = nm0 * q + p;
                    const unsigned int index  = outoffset + cnt_pq;
                    unsigned int cnt_pj       = nq1 * p;

                    TData sum = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
                    {
                        sum += s_wsp1[cnt_pj] * s_basis1[q * nq1 + j] * s_w1[j];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseQuadKernel_QP_1D(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT jac,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
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
            TData *s_wsp0   = &scratch[0];
            TData *s_wsp1   = s_wsp0 + nqTot;
            TData *s_basis0 = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_w0     = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nq1),
                [&](const unsigned int &idx) {
                    const unsigned int j = idx % nq1;
                    const unsigned int p = idx / nq1;
                    unsigned int cnt_ji  = nq0 * j;

                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
                    }
                    s_wsp1[idx] = sum;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nm1),
                [&](const unsigned int &idx) {
                    const unsigned int p     = idx % nm0;
                    const unsigned int q     = idx / nm0;
                    const unsigned int index = outoffset + idx;
                    unsigned int cnt_pj      = nq1 * p;

                    TData sum = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
                    {
                        sum += s_wsp1[cnt_pj] * s_basis1[q * nq1 + j] * s_w1[j];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseTriKernel(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool correct,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT jac, TData *KOKKOS_RESTRICT wsp,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_w0     = SHMEM ? s_basis1 + nmTot * nq1 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
                {
                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_ji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;
                        sum += in[index] * s_basis0[p * nq0 + i] *
                               jac[jacindex] * s_w0[i];
                    }
                    wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] = sum;
                }

                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    TData sum = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j)
                    {
                        sum +=
                            wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] *
                            s_basis1[mode_pq * nq1 + j] * s_w1[j];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * mode_pq + ilane;
                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                }
            }

            // Correction for singular vertex in collpased coordinates.
            // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
            // With contributions from every quadrature point
            if (correct)
            {
                TData iprod_01 = 0.0;
                for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
                {
                    const unsigned int index = nqTot * warpsize * iwarp + ilane;
                    const unsigned int jacindex = DEFORMED ? index : e;

                    TData tmp = s_w1[j] * s_basis1[nq1 + j];
                    if constexpr (!DEFORMED)
                    {
                        tmp *= jac[jacindex];
                    }

                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_ji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;

                        TData prod = in[index] * tmp * s_w0[i];
                        if (DEFORMED)
                        {
                            prod *= jac[jacindex];
                        }
                        iprod_01 += prod * s_basis0[nq0 + i];
                    }
                }

                const unsigned int index =
                    nmTot * warpsize * iwarp + warpsize + ilane;
                if (SCALE)
                {
                    out[index] += iprod_01 * scale;
                }
                else
                {
                    out[index] += iprod_01;
                }
            }
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseTriKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool correct,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
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
            TData *s_wsp0     = &scratch[0];
            TData *s_wsp1     = s_wsp0 + nqTot;
            TData *s_iprod_01 = s_wsp1 + nm0 * nq1;
            TData *s_basis0   = SHMEM ? s_iprod_01 + 1u : (TData *)basis0;
            TData *s_basis1   = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_w0       = SHMEM ? s_basis1 + nmTot * nq1 : (TData *)w0;
            TData *s_w1       = SHMEM ? s_w0 + nq0 : (TData *)w1;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                    team, nq1, nm0),
                [&](const unsigned int &j, const unsigned int &p) {
                    const unsigned int cnt_pj = nq1 * p + j;
                    unsigned int cnt_ji       = nq0 * j;

                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
                    }
                    s_wsp1[cnt_pj] = sum;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                    team, nm1, nm0),
                [&](const unsigned int &q, const unsigned int &p) {
                    if (q < nm1 - p)
                    {
                        const unsigned int mode_pq =
                            (2u * nm1 - p + 1u) * p / 2u + q;
                        const unsigned int index = outoffset + mode_pq;
                        unsigned int cnt_pj      = nq1 * p;

                        TData sum = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
                        {
                            sum += s_wsp1[cnt_pj] *
                                   s_basis1[mode_pq * nq1 + j] * s_w1[j];
                        }

                        if (SCALE)
                        {
                            sum *= scale;
                        }

                        if (APPEND)
                        {
                            out[index] += sum;
                        }
                        else
                        {
                            out[index] = sum;
                        }
                    }
                });

            // Correction for singular vertex in collpased coordinates.
            // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
            // With contributions from every quadrature point
            if (correct)
            {
                Kokkos::single(Kokkos::PerTeam(team),
                               [&]() { *s_iprod_01 = 0.0; });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                        team, nq0, nq1),
                    [&](const unsigned int &i, const unsigned int &j) {
                        TData tmp                 = s_w1[j] * s_basis1[nq1 + j];
                        const unsigned int cnt_ji = nq0 * j + i;
                        TData prod = s_wsp0[cnt_ji] * tmp * s_w0[i];
                        Kokkos::atomic_add(s_iprod_01,
                                           prod * s_basis0[nq0 + i]);
                    });

                team.team_barrier();

                const unsigned int index = outoffset + 1u;
                if (SCALE)
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[index] += (*s_iprod_01) * scale;
                    });
                }
                else
                {
                    Kokkos::single(Kokkos::PerTeam(team),
                                   [&]() { out[index] += (*s_iprod_01); });
                }
            }

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseTriKernel_QP_1D(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool correct,
    const unsigned int *KOKKOS_RESTRICT pindex,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
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
            TData *s_wsp0     = &scratch[0];
            TData *s_wsp1     = s_wsp0 + nqTot;
            TData *s_iprod_01 = s_wsp1 + nm0 * nq1;
            TData *s_basis0   = SHMEM ? s_iprod_01 + 1u : (TData *)basis0;
            TData *s_basis1   = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_w0       = SHMEM ? s_basis1 + nmTot * nq1 : (TData *)w0;
            TData *s_w1       = SHMEM ? s_w0 + nq0 : (TData *)w1;

            // Temporary solution, to be removed - TODO
            unsigned int *vpindex = (unsigned int *)pindex;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0),
                                 [&](const unsigned int &p) {
                                     for (unsigned int q = 0; q < nm1 - p; q++)
                                     {
                                         const unsigned int mode_pq =
                                             (2u * nm1 - p + 1u) * p / 2u + q;
                                         vpindex[mode_pq] = p;
                                     }
                                 });

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nq1),
                [&](const unsigned int &idx) {
                    const unsigned int j = idx % nq1;
                    const unsigned int p = idx / nq1;
                    unsigned int cnt_ji  = nq0 * j;

                    TData sum = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
                    }
                    s_wsp1[idx] = sum;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nmTot),
                [&](const unsigned int &idx) {
                    const unsigned int p     = pindex[idx];
                    const unsigned int index = outoffset + idx;
                    unsigned int cnt_pj      = nq1 * p;

                    TData sum = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
                    {
                        sum +=
                            s_wsp1[cnt_pj] * s_basis1[idx * nq1 + j] * s_w1[j];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                });

            // Correction for singular vertex in collpased coordinates.
            // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
            // With contributions from every quadrature point
            if (correct)
            {
                *s_iprod_01 = 0.0;

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq1),
                    [&](const unsigned int &idx) {
                        const unsigned int i = idx % nq0;
                        const unsigned int j = idx / nq0;
                        TData tmp            = s_w1[j] * s_basis1[nq1 + j];
                        TData prod           = s_wsp0[idx] * tmp * s_w0[i];
                        Kokkos::atomic_add(s_iprod_01,
                                           prod * s_basis0[nq0 + i]);
                    });

                team.team_barrier();

                const unsigned int index = outoffset + 1u;
                if (SCALE)
                {
                    out[index] += (*s_iprod_01) * scale;
                }
                else
                {
                    out[index] += (*s_iprod_01);
                }
            }

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseHexKernel(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT basis2, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT w2,
    const TData *KOKKOS_RESTRICT jac, TData *KOKKOS_RESTRICT wsp,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
{
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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm2 * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;
            TData *wsp0              = wsp;
            TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;

            for (unsigned int p = 0u; p < nm0; ++p)
            {
                for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2;
                     ++k)
                {
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                    {
                        TData sum_kj = 0.0;
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;
                            const unsigned int jacindex = DEFORMED ? index : e;
                            sum_kj += in[index] * s_basis0[i + nq0 * p] *
                                      jac[jacindex] * s_w0[i];
                        }
                        wsp0[nq1 * nq2 * warpsize * iwarp + warpsize * cnt_kj +
                             ilane] = sum_kj;
                    }
                }

                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                        {
                            sum_k += wsp0[nq1 * nq2 * warpsize * iwarp +
                                          warpsize * cnt_kj + ilane] *
                                     s_basis1[q * nq1 + j] * s_w1[j];
                        }
                        wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] =
                            sum_k;
                    }

                    for (unsigned int r = 0u; r < nm2; ++r)
                    {
                        const unsigned int cnt_rqp =
                            nm0 * nm1 * r + nm0 * q + p;
                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * cnt_rqp + ilane;

                        TData sum = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k)
                        {
                            sum += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                        ilane] *
                                   s_basis2[r * nq2 + k] * s_w2[k];
                        }

                        if (SCALE)
                        {
                            sum *= scale;
                        }

                        if (APPEND)
                        {
                            out[index] += sum;
                        }
                        else
                        {
                            out[index] = sum;
                        }
                    }
                }
            }
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseHexKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT basis2, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT w2,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

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
            TData *s_wsp0 = &scratch[0];
            TData *s_wsp1 = s_wsp0 + nqTot;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm2 * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq1, nq2, nm0),
                [&](const unsigned int &j, const unsigned int &k,
                    const unsigned int &p) {
                    const unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k + j;
                    unsigned int cnt_kji       = nq0 * nq1 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq2, nm1, nm0),
                [&](const unsigned int &k, const unsigned int &q,
                    const unsigned int &p) {
                    const unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;
                    unsigned int cnt_pkj       = nq2 * nq1 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_wsp1[cnt_pkj] * s_basis1[q * nq1 + j] * s_w1[j];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nm2, nm1, nm0),
                [&](const unsigned int &r, const unsigned int &q,
                    const unsigned int &p) {
                    const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;
                    const unsigned int index   = outoffset + cnt_rqp;
                    unsigned int cnt_pqk       = nm1 * nq2 * p + nq2 * q;

                    TData sum = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum +=
                            s_wsp2[cnt_pqk] * s_basis2[r * nq2 + k] * s_w2[k];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                });

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseHexKernel_QP_1D(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT basis2, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT w2,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

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
            TData *s_wsp0 = &scratch[0];
            TData *s_wsp1 = s_wsp0 + nqTot;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm2 * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nq1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int j = idx % nq1;
                    const unsigned int k = (idx / nq1) % nq2;
                    const unsigned int p = idx / (nq1 * nq2);
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
                    }
                    s_wsp1[idx] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nm1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int k = idx % nq2;
                    const unsigned int q = (idx / nq2) % nm1;
                    const unsigned int p = idx / (nq2 * nm1);
                    unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_wsp1[cnt_pkj] * s_basis1[q * nq1 + j] * s_w1[j];
                    }
                    s_wsp2[idx] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nm1 * nm2),
                [&](const unsigned int &idx) {
                    const unsigned int p     = idx % nm0;
                    const unsigned int q     = (idx / nm0) % nm1;
                    const unsigned int r     = idx / (nm0 * nm1);
                    const unsigned int index = outoffset + idx;
                    unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

                    TData sum = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum +=
                            s_wsp2[cnt_pqk] * s_basis2[r * nq2 + k] * s_w2[k];
                    }

                    if (SCALE)
                    {
                        sum *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum;
                    }
                    else
                    {
                        out[index] = sum;
                    }
                });

            team.team_barrier();
        });
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseTetKernel(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT basis2,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT w2, const TData *KOKKOS_RESTRICT jac,
    TData *KOKKOS_RESTRICT wsp, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nmTot * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm01 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;
            TData *wsp0              = wsp;
            TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;
            TData *prod              = wsp1 + nq2 * nelmt;

            for (unsigned int p = 0u, mode_pq = 0u, mode_pqr = 0u; p < nm0; ++p)
            {
                for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2;
                     ++k)
                {
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                    {
                        TData sum_kj = 0.0;
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;
                            const unsigned int jacindex = DEFORMED ? index : e;
                            sum_kj += in[index] * s_basis0[i + nq0 * p] *
                                      jac[jacindex] * s_w0[i];
                        }
                        wsp0[nq1 * nq2 * warpsize * iwarp + warpsize * cnt_kj +
                             ilane] = sum_kj;
                    }
                }

                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                        {
                            sum_k += wsp0[nq1 * nq2 * warpsize * iwarp +
                                          warpsize * cnt_kj + ilane] *
                                     s_basis1[mode_pq * nq1 + j] * s_w1[j];
                        }
                        wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] =
                            sum_k;
                    }

                    for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode_pqr)
                    {
                        TData tmp = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k)
                        {
                            tmp += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                        ilane] *
                                   s_basis2[mode_pqr * nq2 + k] * s_w2[k];
                        }

                        if (SCALE)
                        {
                            tmp *= scale;
                        }

                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * mode_pqr + ilane;
                        if (APPEND)
                        {
                            out[index] += tmp;
                        }
                        else
                        {
                            out[index] = tmp;
                        }
                    }
                }
            }

            // Add correction for collapsed coordinate.
            if (correct)
            {
                for (unsigned int r = 0u; r < nm2; ++r)
                {
                    prod[nm2 * warpsize * iwarp + warpsize * r + ilane] = 0.0;
                }

                for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
                {
                    TData tmpQ2 = s_w2[k];
                    if constexpr (!DEFORMED)
                    {
                        tmpQ2 *= jac[e];
                    }

                    for (unsigned int j = 0u; j < nq1; ++j)
                    {
                        TData tmpQ1 = tmpQ2 * s_w1[j];
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;

                            // Store jac * quadrature weight
                            TData tmpQ = tmpQ1 * s_w0[i];
                            if constexpr (DEFORMED)
                            {
                                tmpQ *= jac[index];
                            }

                            // top vertex
                            TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                            tmp += s_basis0[nq0 + i] * s_basis1[j];
                            tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                            tmp *= s_basis2[nq2 + k];
                            tmp *= in[index] * tmpQ;
                            prod[nm2 * warpsize * iwarp + warpsize * (nm2 - 1) +
                                 ilane] += tmp;

                            // bottom vertex
                            tmp = s_basis0[nq0 + i] * s_basis1[nq1 + j] *
                                  s_basis2[k] * in[index] * tmpQ;
                            prod[nm2 * warpsize * iwarp + ilane] += tmp;

                            // singular edge
                            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                            {
                                tmp = s_basis2[(r + 1) * nq2 + k] *
                                      s_basis1[nq1 + j] * s_basis0[nq0 + i] *
                                      in[index] * tmpQ;
                                prod[nm2 * warpsize * iwarp + warpsize * r +
                                     ilane] += tmp;
                            }
                        }
                    }
                }

                if (SCALE)
                {
                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize + ilane;
                    out[index] += prod[nm2 * warpsize * iwarp +
                                       warpsize * (nm2 - 1) + ilane] *
                                  scale;
                    for (unsigned int r = 0u; r < nm2 - 1u; ++r)
                    {
                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * (nm2 + r) + ilane;
                        out[index] += prod[nm2 * warpsize * iwarp +
                                           warpsize * r + ilane] *
                                      scale;
                    }
                }
                else
                {
                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize + ilane;
                    out[index] += prod[nm2 * warpsize * iwarp +
                                       warpsize * (nm2 - 1) + ilane];
                    for (unsigned int r = 0u; r < nm2 - 1u; ++r)
                    {
                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * (nm2 + r) + ilane;
                        out[index] +=
                            prod[nm2 * warpsize * iwarp + warpsize * r + ilane];
                    }
                }
            }
        });
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseTetKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT basis2,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT w2, const TData *KOKKOS_RESTRICT jac,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

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
            TData *s_prod   = &scratch[0];
            TData *s_wsp0   = s_prod + nm2;
            TData *s_wsp1   = s_wsp0 + nqTot;
            TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 = SHMEM ? s_wsp2 + nm01 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nmTot * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm01 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq1, nq2, nm0),
                [&](const unsigned int &j, const unsigned int &k,
                    const unsigned int &p) {
                    const unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;
                    unsigned int cnt_kji       = nq0 * nq1 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq2, nm1, nm0),
                [&](const unsigned int &k, const unsigned int &q,
                    const unsigned int &p) {
                    if (q < nm1 - p)
                    {
                        const unsigned int mode_pq =
                            (2u * nm1 - p + 1u) * p / 2u + q;
                        unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

                        TData sum_k = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                        {
                            sum_k += s_basis1[mode_pq * nq1 + j] *
                                     s_wsp1[cnt_pkj] * s_w1[j];
                        }
                        s_wsp2[mode_pq * nq2 + k] = sum_k;
                    }
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nm2, nm1, nm0),
                [&](const unsigned int &r, const unsigned int &q,
                    const unsigned int &p) {
                    if (q < nm1 - p)
                    {
                        if (r < nm2 - p - q)
                        {
                            const unsigned int mode_pq =
                                (2u * nm1 - p + 1u) * p / 2u + q;
                            unsigned int mode_pqr =
                                (2u * (nm2 - p) - q + 1u) * q;
                            mode_pqr += nm2 * (nm2 + 1u) * p;
                            mode_pqr -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
                            mode_pqr += (p - 1u) * p * (2u * p - 1u) / 6u;
                            mode_pqr /= 2u;
                            const unsigned int index = outoffset + mode_pqr + r;

                            TData tmp = 0.0;
                            for (unsigned int k = 0u; k < nq2; ++k)
                            {
                                tmp += s_wsp2[mode_pq * nq2 + k] *
                                       s_basis2[(mode_pqr + r) * nq2 + k] *
                                       s_w2[k];
                            }

                            if (SCALE)
                            {
                                tmp *= scale;
                            }

                            if (APPEND)
                            {
                                out[index] += tmp;
                            }
                            else
                            {
                                out[index] = tmp;
                            }
                        }
                    }
                });

            // Add correction for collapsed coordinate.
            if (correct)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nm2),
                    [&](const unsigned int &r) { s_prod[r] = 0.0; });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                        team, nq0, nq1, nq2),
                    [&](const unsigned int &i, const unsigned int &j,
                        const unsigned int &k) {
                        TData tmpQ2 = s_w2[k];
                        TData tmpQ1 = tmpQ2 * s_w1[j];
                        const unsigned int cnt_kji =
                            nq1 * nq0 * k + nq0 * j + i;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * s_w0[i];

                        // top vertex
                        TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                        tmp += s_basis0[nq0 + i] * s_basis1[j];
                        tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp *= s_basis2[nq2 + k];
                        tmp *= s_wsp0[cnt_kji] * tmpQ;
                        Kokkos::atomic_add(s_prod + nm2 - 1, tmp);

                        // bottom vertex
                        tmp = s_basis0[nq0 + i] * s_basis1[nq1 + j] *
                              s_basis2[k] * s_wsp0[cnt_kji] * tmpQ;
                        Kokkos::atomic_add(s_prod, tmp);

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp = s_basis2[(r + 1) * nq2 + k] *
                                  s_basis1[nq1 + j] * s_basis0[nq0 + i] *
                                  s_wsp0[cnt_kji] * tmpQ;
                            Kokkos::atomic_add(s_prod + r, tmp);
                        }
                    });

                team.team_barrier();

                if (SCALE)
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[outoffset + 1] += s_prod[nm2 - 1] * scale;
                    });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm2 - 1u),
                        [&](const unsigned int &r) {
                            out[outoffset + nm2 + r] += s_prod[r] * scale;
                        });
                }
                else
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[outoffset + 1] += s_prod[nm2 - 1];
                    });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm2 - 1u),
                        [&](const unsigned int &r) {
                            out[outoffset + nm2 + r] += s_prod[r];
                        });
                }
            }

            team.team_barrier();
        });
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBaseTetKernel_QP_1D(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const unsigned int *KOKKOS_RESTRICT pindex1,
    const unsigned int *KOKKOS_RESTRICT qindex1,
    const unsigned int *KOKKOS_RESTRICT pindex2,
    const unsigned int *KOKKOS_RESTRICT qindex2,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT basis2, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT w2,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

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
            TData *s_prod = &scratch[0];
            TData *s_wsp0 = s_prod + nm2;
            TData *s_wsp1 = s_wsp0 + nqTot;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nmTot * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Temporary solution, to be removed - TODO
            unsigned int *vpindex1 = (unsigned int *)pindex1;
            unsigned int *vqindex1 = (unsigned int *)qindex1;
            unsigned int *vpindex2 = (unsigned int *)pindex2;
            unsigned int *vqindex2 = (unsigned int *)qindex2;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0), [&](const unsigned int &p) {
                    for (unsigned int q = 0; q < nm1 - p; q++)
                    {
                        const unsigned int mode_pq =
                            (2u * nm1 - p + 1u) * p / 2u + q;
                        vpindex1[mode_pq]     = p;
                        vqindex1[mode_pq]     = q;
                        unsigned int mode_pqr = (2u * (nm2 - p) - q + 1u) * q;
                        mode_pqr += nm2 * (nm2 + 1u) * p;
                        mode_pqr -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
                        mode_pqr += (p - 1u) * p * (2u * p - 1u) / 6u;
                        mode_pqr /= 2u;
                        for (unsigned int r = 0; r < nm2 - p - q;
                             r++, mode_pqr++)
                        {
                            vpindex2[mode_pqr] = p;
                            vqindex2[mode_pqr] = q;
                        }
                    }
                });

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm01 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nq1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int j = idx % nq1;
                    const unsigned int k = (idx / nq1) % nq2;
                    const unsigned int p = idx / (nq1 * nq2);
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
                    }
                    s_wsp1[idx] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm01 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int mode_pq = idx / nq2;
                    const unsigned int p       = pindex1[mode_pq];
                    // const unsigned int q       = qindex1[mode_pq];
                    const unsigned int k = idx % nq2;
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k += s_basis1[mode_pq * nq1 + j] * s_wsp1[cnt_pkj] *
                                 s_w1[j];
                    }
                    s_wsp2[idx] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int p     = pindex2[idx];
                                     const unsigned int q     = qindex2[idx];
                                     const unsigned int index = outoffset + idx;
                                     const unsigned int mode_pq =
                                         (2u * nm1 - p + 1u) * p / 2u + q;

                                     TData tmp = 0.0;
                                     for (unsigned int k = 0u; k < nq2; ++k)
                                     {
                                         tmp += s_wsp2[mode_pq * nq2 + k] *
                                                s_basis2[idx * nq2 + k] *
                                                s_w2[k];
                                     }

                                     if (SCALE)
                                     {
                                         tmp *= scale;
                                     }

                                     if (APPEND)
                                     {
                                         out[index] += tmp;
                                     }
                                     else
                                     {
                                         out[index] = tmp;
                                     }
                                 });

            // Add correction for collapsed coordinate.
            if (correct)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nm2),
                    [&](const unsigned int &idx) { s_prod[idx] = 0.0; });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq1 * nq2),
                    [&](const unsigned int &idx) {
                        const unsigned int i = idx % nq0;
                        const unsigned int j = (idx / nq0) % nq1;
                        const unsigned int k = idx / (nq0 * nq1);
                        TData tmpQ2          = s_w2[k];
                        TData tmpQ1          = tmpQ2 * s_w1[j];

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * s_w0[i];

                        // top vertex
                        TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                        tmp += s_basis0[nq0 + i] * s_basis1[j];
                        tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp *= s_basis2[nq2 + k];
                        tmp *= s_wsp0[idx] * tmpQ;
                        Kokkos::atomic_add(s_prod + nm2 - 1, tmp);

                        // bottom vertex
                        tmp = s_basis0[nq0 + i] * s_basis1[nq1 + j] *
                              s_basis2[k] * s_wsp0[idx] * tmpQ;
                        Kokkos::atomic_add(s_prod, tmp);

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp = s_basis2[(r + 1) * nq2 + k] *
                                  s_basis1[nq1 + j] * s_basis0[nq0 + i] *
                                  s_wsp0[idx] * tmpQ;
                            Kokkos::atomic_add(s_prod + r, tmp);
                        }
                    });

                team.team_barrier();

                if (SCALE)
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[outoffset + 1] += s_prod[nm2 - 1] * scale;
                    });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm2 - 1u),
                        [&](const unsigned int &r) {
                            out[outoffset + nm2 + r] += s_prod[r] * scale;
                        });
                }
                else
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[outoffset + 1] += s_prod[nm2 - 1];
                    });
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm2 - 1u),
                        [&](const unsigned int &r) {
                            out[outoffset + nm2 + r] += s_prod[r];
                        });
                }
            }

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBasePrismKernel(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT basis2,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT w2, const TData *KOKKOS_RESTRICT jac,
    TData *KOKKOS_RESTRICT wsp, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;

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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm02 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm02 * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;
            TData *wsp0              = wsp;
            TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;
            TData *wsp2              = wsp1 + nq2 * nelmt;

            for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
            {
                for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2;
                     ++k)
                {
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                    {
                        TData sum_kj = 0.0;
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;
                            const unsigned int jacindex = DEFORMED ? index : e;
                            sum_kj += in[index] * s_basis0[nq0 * p + i] *
                                      jac[jacindex] * s_w0[i];
                        }
                        wsp0[nq1 * nq2 * warpsize * iwarp + warpsize * cnt_kj +
                             ilane] = sum_kj;
                    }
                }

                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                        {
                            sum_k += wsp0[nq1 * nq2 * warpsize * iwarp +
                                          warpsize * cnt_kj + ilane] *
                                     s_basis1[q * nq1 + j] * s_w1[j];
                        }
                        wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] =
                            sum_k;
                    }

                    for (int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * mode_pqr + ilane;
                        unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k)
                        {
                            sum_k += wsp1[nq2 * warpsize * iwarp +
                                          warpsize * k + ilane] *
                                     s_basis2[(mode_pr + r) * nq2 + k] *
                                     s_w2[k];
                        }

                        if (SCALE)
                        {
                            sum_k *= scale;
                        }

                        if (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                }
            }

            // Add correction for collapsed coordinate.
            if (correct)
            {
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    wsp2[nm1 * warpsize * iwarp + warpsize * q + ilane] = 0.0;
                }

                for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
                {
                    TData k_weight = s_w2[k];
                    if constexpr (!DEFORMED)
                    {
                        k_weight *= jac[e];
                    }

                    for (unsigned int j = 0u; j < nq1; ++j)
                    {
                        TData kj_weight = k_weight * s_w1[j];
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;
                            TData prod = kj_weight * s_w0[i] * in[index];
                            if constexpr (DEFORMED)
                            {
                                prod *= jac[index];
                            }

                            for (unsigned int q = 0u; q < nm1; ++q)
                            {
                                wsp2[nm1 * warpsize * iwarp + warpsize * q +
                                     ilane] += prod * s_basis2[nq2 + k] *
                                               s_basis1[q * nq1 + j] *
                                               s_basis0[nq0 + i];
                            }
                        }
                    }
                }

                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    const unsigned int index = nmTot * warpsize * iwarp +
                                               warpsize * (nm2 * q + 1u) +
                                               ilane;
                    if (SCALE)
                    {
                        out[index] += wsp2[nm1 * warpsize * iwarp +
                                           warpsize * q + ilane] *
                                      scale;
                    }
                    else
                    {
                        out[index] +=
                            wsp2[nm1 * warpsize * iwarp + warpsize * q + ilane];
                    }
                }
            }
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBasePrismKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT basis2,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT w2, const TData *KOKKOS_RESTRICT jac,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;

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
            TData *s_wsp0 = &scratch[0];
            TData *s_wsp1 = s_wsp0 + nqTot;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm02 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm02 * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq1, nq2, nm0),
                [&](const unsigned int &j, const unsigned int &k,
                    const unsigned int &p) {
                    const unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;
                    unsigned int cnt_kji       = nq1 * nq0 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq2, nm1, nm0),
                [&](const unsigned int &k, const unsigned int &q,
                    const unsigned int &p) {
                    const unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;
                    unsigned int cnt_pkj       = nq1 * nq2 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nm2, nm1, nm0),
                [&](const unsigned int &r, const unsigned int &q,
                    const unsigned int &p) {
                    if (r < nm2 - p)
                    {
                        unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;
                        unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;
                        const unsigned int mode_pqr =
                            mode_pr * nm1 + (nm2 - p) * q + r;
                        const unsigned int index = outoffset + mode_pqr;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += s_basis2[(mode_pr + r) * nq2 + k] *
                                     s_w2[k] * s_wsp2[cnt_pqk];
                        }

                        if (SCALE)
                        {
                            sum_k *= scale;
                        }

                        if (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                });

            team.team_barrier();

            // Add correction for collapsed coordinate.
            if (correct)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nm1),
                    [&](const unsigned int &q) { s_wsp2[q] = 0.0; });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                        team, nq0, nq1, nq2),
                    [&](const unsigned int &i, const unsigned int &j,
                        const unsigned int &k) {
                        TData k_weight  = s_w2[k];
                        TData kj_weight = k_weight * s_w1[j];
                        const unsigned int cnt_kji =
                            nq1 * nq0 * k + nq0 * j + i;
                        TData prod = kj_weight * s_w0[i] * s_wsp0[cnt_kji];
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            Kokkos::atomic_add(s_wsp2 + q,
                                               prod * s_basis2[nq2 + k] *
                                                   s_basis1[q * nq1 + j] *
                                                   s_basis0[nq0 + i]);
                        }
                    });

                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1),
                                     [&](const unsigned int &q) {
                                         const unsigned int index =
                                             outoffset + nm2 * q + 1u;
                                         if constexpr (SCALE)
                                         {
                                             out[index] += s_wsp2[q] * scale;
                                         }
                                         else
                                         {
                                             out[index] += s_wsp2[q];
                                         }
                                     });
            }

            team.team_barrier();
        });
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBasePrismKernel_QP_1D(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const unsigned int *KOKKOS_RESTRICT pindex,
    const unsigned int *KOKKOS_RESTRICT qindex,
    const unsigned int *KOKKOS_RESTRICT rindex,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT basis2, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT w2,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;

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
            TData *s_wsp0 = &scratch[0];
            TData *s_wsp1 = s_wsp0 + nqTot;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm02 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Temporary solution, to be removed - TODO
            unsigned int *vpindex = (unsigned int *)pindex;
            unsigned int *vqindex = (unsigned int *)qindex;
            unsigned int *vrindex = (unsigned int *)rindex;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0), [&](const unsigned int &p) {
                    for (unsigned int q = 0u; q < nm1; q++)
                    {
                        for (unsigned int r = 0u; r < nm2 - p; r++)
                        {
                            unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;
                            unsigned int mode_pqr =
                                mode_pr * nm1 + (nm2 - p) * q + r;
                            vpindex[mode_pqr] = p;
                            vqindex[mode_pqr] = q;
                            vrindex[mode_pqr] = r;
                        }
                    }
                });

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm02 * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nq1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int j = idx % nq1;
                    const unsigned int k = (idx / nq1) % nq2;
                    const unsigned int p = idx / (nq1 * nq2);
                    unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
                    }
                    s_wsp1[idx] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nm1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int k = idx % nq2;
                    const unsigned int q = (idx / nq2) % nm1;
                    const unsigned int p = idx / (nq2 * nm1);
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[idx] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nmTot),
                [&](const unsigned int &idx) {
                    const unsigned int p = pindex[idx];
                    const unsigned int q = qindex[idx];
                    const unsigned int r = rindex[idx];
                    const unsigned int mode_pr =
                        (2u * nm2 - p + 1u) * p / 2u + r;
                    const unsigned int index = outoffset + idx;
                    unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum_k += s_basis2[mode_pr * nq2 + k] * s_w2[k] *
                                 s_wsp2[cnt_pqk];
                    }

                    if (SCALE)
                    {
                        sum_k *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                });

            team.team_barrier();

            // Add correction for collapsed coordinate.
            if (correct)
            {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nm1),
                    [&](const unsigned int &idx) { s_wsp2[idx] = 0.0; });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nqTot),
                    [&](const unsigned int &idx) {
                        const unsigned int i = idx % nq0;
                        const unsigned int j = (idx / nq0) % nq1;
                        const unsigned int k = idx / (nq0 * nq1);
                        TData k_weight       = s_w2[k];
                        TData kj_weight      = k_weight * s_w1[j];
                        TData prod = kj_weight * s_w0[i] * s_wsp0[idx];
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            Kokkos::atomic_add(s_wsp2 + q,
                                               prod * s_basis2[nq2 + k] *
                                                   s_basis1[q * nq1 + j] *
                                                   s_basis0[nq0 + i]);
                        }
                    });

                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1),
                                     [&](const unsigned int &idx) {
                                         const unsigned int index =
                                             outoffset + nm2 * idx + 1u;
                                         if constexpr (SCALE)
                                         {
                                             out[index] += s_wsp2[idx] * scale;
                                         }
                                         else
                                         {
                                             out[index] += s_wsp2[idx];
                                         }
                                     });
            }

            team.team_barrier();
        });
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBasePyrKernel(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT basis2,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT w2, const TData *KOKKOS_RESTRICT jac,
    TData *KOKKOS_RESTRICT wsp, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
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
            TData *s_basis0 = SHMEM ? &scratch[0] : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;
            TData *wsp0              = wsp;
            TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;

            for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
            {
                for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2;
                     ++k)
                {
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                    {
                        TData sum_kj = 0.0;
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;
                            const unsigned int jacindex = DEFORMED ? index : e;
                            sum_kj += in[index] * s_basis0[nq0 * p + i] *
                                      jac[jacindex] * s_w0[i];
                        }
                        wsp0[nq1 * nq2 * warpsize * iwarp + warpsize * cnt_kj +
                             ilane] = sum_kj;
                    }
                }

                for (unsigned int q = 0u; q < p; ++q)
                {
                    for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                        {
                            sum_k += wsp0[nq1 * nq2 * warpsize * iwarp +
                                          warpsize * cnt_kj + ilane] *
                                     s_basis1[q * nq1 + j] * s_w1[j];
                        }
                        wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] =
                            sum_k;
                    }

                    for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k)
                        {
                            sum_k += wsp1[nq2 * warpsize * iwarp +
                                          warpsize * k + ilane] *
                                     s_basis2[mode_pqr * nq2 + k] * s_w2[k];
                        }

                        if (SCALE)
                        {
                            sum_k *= scale;
                        }

                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * mode_pqr + ilane;
                        if (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                }

                for (unsigned int q = p; q < nm1; ++q)
                {
                    for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                        {
                            sum_k += wsp0[nq1 * nq2 * warpsize * iwarp +
                                          warpsize * cnt_kj + ilane] *
                                     s_basis1[q * nq1 + j] * s_w1[j];
                        }
                        wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] =
                            sum_k;
                    }

                    for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode_pqr)
                    {
                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k)
                        {
                            sum_k += wsp1[nq2 * warpsize * iwarp +
                                          warpsize * k + ilane] *
                                     s_basis2[mode_pqr * nq2 + k] * s_w2[k];
                        }

                        if (SCALE)
                        {
                            sum_k *= scale;
                        }

                        const unsigned int index = nmTot * warpsize * iwarp +
                                                   warpsize * mode_pqr + ilane;
                        if (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                }
            }

            // Add correction for collapsed coordinate.
            if (correct)
            {
                TData prod = 0.0;
                for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
                {
                    TData tmpQ2 = s_w2[k];
                    if constexpr (!DEFORMED)
                    {
                        tmpQ2 *= jac[e];
                    }

                    for (unsigned int j = 0u; j < nq1; ++j)
                    {
                        TData tmpQ1 = tmpQ2 * s_w1[j];
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;

                            // Store jac * quadrature weight
                            TData tmpQ = tmpQ1 * s_w0[i];
                            if constexpr (DEFORMED)
                            {
                                tmpQ *= jac[index];
                            }

                            // top vertex
                            TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                            tmp += s_basis0[nq0 + i] * s_basis1[j];
                            tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                            tmp *= s_basis2[nq2 + k];
                            tmp *= in[index] * tmpQ;
                            prod += tmp;
                        }
                    }
                }

                // add to existing entry
                const unsigned int index =
                    nmTot * warpsize * iwarp + warpsize + ilane;
                if (SCALE)
                {
                    out[index] += prod * scale;
                }
                else
                {
                    out[index] += prod;
                }
            }
        });
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBasePyrKernel_QP(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const TData *KOKKOS_RESTRICT basis0,
    const TData *KOKKOS_RESTRICT basis1, const TData *KOKKOS_RESTRICT basis2,
    const TData *KOKKOS_RESTRICT w0, const TData *KOKKOS_RESTRICT w1,
    const TData *KOKKOS_RESTRICT w2, const TData *KOKKOS_RESTRICT jac,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out,
    const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

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
            TData *s_prod = &scratch[0];
            TData *s_wsp0 = s_prod + 1u;
            TData *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nmTot * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq1, nq2, nm0),
                [&](const unsigned int &j, const unsigned int &k,
                    const unsigned int &p) {
                    const unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;
                    unsigned int cnt_kji       = k * nq1 * nq0 + j * nq0;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq2, nm1, nm0),
                [&](const unsigned int &k, const unsigned int &q,
                    const unsigned int &p) {
                    const unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;
                    unsigned int cnt_pkj       = nq1 * nq2 * p + k * nq1;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nm2, nm1, nm0),
                [&](const unsigned int &r, const unsigned int &q,
                    const unsigned int &p) {
                    unsigned int mode_pq = nm1 * (2u * nm2 + 1u - nm1) * p;
                    mode_pq -= (p - 1u) * p / 2u;
                    mode_pq -= (p - 1u) * p * (2u * p - 1u) / 6u;
                    mode_pq /= 2u;

                    if (q < p && r < nm2 - p)
                    {
                        const unsigned int mode_pqr =
                            mode_pq + q * (nm2 - p) + r;
                        const unsigned int index = outoffset + mode_pqr;
                        unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += s_basis2[mode_pqr * nq2 + k] * s_w2[k] *
                                     s_wsp2[cnt_pqk];
                        }

                        if (SCALE)
                        {
                            sum_k *= scale;
                        }

                        if (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                    else if (q >= p && r < nm2 - q)
                    {
                        unsigned int cnt_pqk  = nm1 * nq2 * p + nq2 * q;
                        unsigned int mode_pqr = mode_pq + p * (nm2 - p);
                        mode_pqr +=
                            ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u +
                            r;
                        const unsigned int index = outoffset + mode_pqr;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += s_basis2[mode_pqr * nq2 + k] * s_w2[k] *
                                     s_wsp2[cnt_pqk];
                        }
                        if (SCALE)
                        {
                            sum_k *= scale;
                        }

                        if (APPEND)
                        {
                            out[index] += sum_k;
                        }
                        else
                        {
                            out[index] = sum_k;
                        }
                    }
                });

            // Add correction for collapsed coordinate.
            if (correct)
            {
                Kokkos::single(Kokkos::PerTeam(team),
                               [&]() { (*s_prod) = 0.0; });

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                        team, nq0, nq1, nq2),
                    [&](const unsigned int &i, const unsigned int &j,
                        const unsigned int &k) {
                        TData tmpQ2 = s_w2[k];
                        TData tmpQ1 = tmpQ2 * s_w1[j];
                        const unsigned int cnt_kji =
                            nq0 * nq1 * k + nq0 * j + i;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * s_w0[i];

                        // top vertex
                        TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                        tmp += s_basis0[nq0 + i] * s_basis1[j];
                        tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp *= s_basis2[nq2 + k];
                        tmp *= s_wsp0[cnt_kji] * tmpQ;
                        Kokkos::atomic_add(s_prod, tmp);
                    });

                team.team_barrier();

                // add to existing entry
                if (SCALE)
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[outoffset + 1] += (*s_prod) * scale;
                    });
                }
                else
                {
                    Kokkos::single(Kokkos::PerTeam(team),
                                   [&]() { out[outoffset + 1] += (*s_prod); });
                }
            }

            team.team_barrier();
        });
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED,
          bool SHMEM = true>
void IProductWRTBasePyrKernel_QP_1D(
    const unsigned int ssize, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nmTot, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const bool correct, const unsigned int *KOKKOS_RESTRICT pindex,
    const unsigned int *KOKKOS_RESTRICT qindex,
    const TData *KOKKOS_RESTRICT basis0, const TData *KOKKOS_RESTRICT basis1,
    const TData *KOKKOS_RESTRICT basis2, const TData *KOKKOS_RESTRICT w0,
    const TData *KOKKOS_RESTRICT w1, const TData *KOKKOS_RESTRICT w2,
    const TData *KOKKOS_RESTRICT jac, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out, const TData scale = 1.0)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

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
            TData *s_prod = &scratch[0];
            TData *s_wsp0 = s_prod + 1u;
            TData *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
            TData *s_wsp2 = s_wsp1 + nm0 * nq1 * nq2;
            TData *s_basis0 =
                SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
            TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
            TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
            TData *s_w0     = SHMEM ? s_basis2 + nmTot * nq2 : (TData *)w0;
            TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
            TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

            // Temporary solution, to be removed - TODO
            unsigned int *vpindex = (unsigned int *)pindex;
            unsigned int *vqindex = (unsigned int *)qindex;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0), [&](const unsigned int &p) {
                    for (unsigned int q = 0u; q < nm1; q++)
                    {
                        unsigned int mode_pq = nm1 * (2u * nm2 + 1u - nm1) * p;
                        mode_pq -= (p - 1u) * p / 2u;
                        mode_pq -= (p - 1u) * p * (2u * p - 1u) / 6u;
                        mode_pq /= 2u;

                        if (q < p)
                        {
                            for (unsigned int r = 0; r < nm2 - p; r++)
                            {
                                const unsigned int mode_pqr =
                                    mode_pq + q * (nm2 - p) + r;
                                vpindex[mode_pqr] = p;
                                vqindex[mode_pqr] = q;
                            }
                        }
                        else
                        {
                            for (unsigned int r = 0; r < nm2 - q; r++)
                            {
                                unsigned int mode_pqr = mode_pq + p * (nm2 - p);
                                mode_pqr += ((2u * (nm2 - p) - (q - p) + 1u) *
                                             (q - p)) /
                                                2u +
                                            r;
                                vpindex[mode_pqr] = p;
                                vqindex[mode_pqr] = q;
                            }
                        }
                    }
                });

            // Copy to shared memory.
            if (SHMEM)
            {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm0 * nq0),
                                     [&](const unsigned int &idx) {
                                         s_basis0[idx] = basis0[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nm1 * nq1),
                                     [&](const unsigned int &idx) {
                                         s_basis1[idx] = basis1[idx];
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nmTot * nq2),
                                     [&](const unsigned int &idx) {
                                         s_basis2[idx] = basis2[idx];
                                     });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0),
                    [&](const unsigned int &idx) { s_w0[idx] = w0[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq1),
                    [&](const unsigned int &idx) { s_w1[idx] = w1[idx]; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq2),
                    [&](const unsigned int &idx) { s_w2[idx] = w2[idx]; });
            }

            // Copy to shared memory.
            const unsigned int e         = team.league_rank();
            const unsigned int inoffset  = nqTot * e;
            const unsigned int outoffset = nmTot * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nqTot),
                                 [&](const unsigned int &idx) {
                                     const unsigned int index = inoffset + idx;
                                     const unsigned int jacindex =
                                         DEFORMED ? index : e;
                                     s_wsp0[idx] = in[index] * jac[jacindex];
                                 });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nq1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int j = idx % nq1;
                    const unsigned int k = (idx / nq1) % nq2;
                    const unsigned int p = idx / (nq1 * nq2);
                    unsigned int cnt_kji = k * nq1 * nq0 + j * nq0;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
                    }
                    s_wsp1[idx] = sum_kj;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nm0 * nm1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int k = idx % nq2;
                    const unsigned int q = (idx / nq2) % nm1;
                    const unsigned int p = idx / (nq2 * nm1);
                    unsigned int cnt_pkj = nq1 * nq2 * p + k * nq1;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[idx] = sum_k;
                });

            team.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nmTot),
                [&](const unsigned int &idx) {
                    const unsigned int p     = pindex[idx];
                    const unsigned int q     = qindex[idx];
                    const unsigned int index = outoffset + idx;
                    unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum_k +=
                            s_basis2[idx * nq2 + k] * s_w2[k] * s_wsp2[cnt_pqk];
                    }
                    if (SCALE)
                    {
                        sum_k *= scale;
                    }

                    if (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                });

            // Add correction for collapsed coordinate.
            if (correct)
            {
                (*s_prod) = 0.0;

                team.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, nq0 * nq1 * nq2),
                    [&](const unsigned int &idx) {
                        const unsigned int i = idx % nq0;
                        const unsigned int j = (idx / nq0) % nq1;
                        const unsigned int k = idx / (nq0 * nq1);
                        TData tmpQ2          = s_w2[k];
                        TData tmpQ1          = tmpQ2 * s_w1[j];

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * s_w0[i];

                        // top vertex
                        TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                        tmp += s_basis0[nq0 + i] * s_basis1[j];
                        tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp *= s_basis2[nq2 + k];
                        tmp *= s_wsp0[idx] * tmpQ;
                        Kokkos::atomic_add(s_prod, tmp);
                    });

                team.team_barrier();

                // add to existing entry
                if (SCALE)
                {
                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                        out[outoffset + 1] += (*s_prod) * scale;
                    });
                }
                else
                {
                    Kokkos::single(Kokkos::PerTeam(team),
                                   [&]() { out[outoffset + 1] += (*s_prod); });
                }
            };

            team.team_barrier();
        });
}

template <typename ExecSpace, typename TData, bool SCALE, bool APPEND,
          bool DEFORMED, bool MULTILEVEL = true, bool SHMEM = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
IProductWRTBase1DKernel(const unsigned int nm0, const unsigned int nq0,
                        const unsigned int nelmts, const TData *basis0,
                        const TData *w0, const TData *jac, const TData *in,
                        TData *out, const TData scale = 1.0)
{
    unsigned int nshared = SHMEM ? nm0 * nq0 + nq0 : 0u;

    if constexpr (MULTILEVEL)
    {
        nshared += nq0;
        IProductWRTBaseSegKernel_QP<TData, SCALE, APPEND, DEFORMED, SHMEM>(
            nshared, nm0, nq0, nelmts, basis0, w0, jac, in, out, scale);
    }
    else
    {
        IProductWRTBaseSegKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
            nshared, nm0, nq0, nelmts, basis0, w0, jac, in, out, scale);
    }
}

template <typename ExecSpace, typename TData, bool SCALE, bool APPEND,
          bool DEFORMED, bool MULTILEVEL = true, bool SHMEM = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
IProductWRTBase2DKernel(LibUtilities::ShapeType shapetype,
                        const unsigned int nm0, const unsigned int nm1,
                        const unsigned int nq0, const unsigned int nq1,
                        const unsigned int nelmts, const bool correct,
                        const TData *basis0, const TData *basis1,
                        const TData *w0, const TData *w1, const TData *jac,
                        [[maybe_unused]] TData *wsp, const TData *in,
                        TData *out, const TData scale = 1.0)
{
    if (shapetype == LibUtilities::Quad)
    {
        const unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = nm0 * nq0 + nm1 * nq1 + nq0 + nq1;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 + nm0 * nq1;
            IProductWRTBaseQuadKernel_QP<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nmTot, nq0, nq1, nelmts, basis0, basis1, w0,
                w1, jac, in, out, scale);
            /* IProductWRTBaseQuadKernel_QP_1D<TData, SCALE, APPEND, DEFORMED,
                                            SHMEM>(
                nshared, nm0, nm1, nmTot, nq0, nq1, nelmts, basis0, basis1, w0,
                w1, jac, in, out, scale);*/
        }
        else
        {
            IProductWRTBaseQuadKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nmTot, nq0, nq1, nelmts, basis0, basis1, w0,
                w1, jac, wsp, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = nm0 * nq0 + nmTot * nq1 + nq0 + nq1;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 + nm0 * nq1 + 1u;
            IProductWRTBaseTriKernel_QP<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nmTot, nq0, nq1, nelmts, correct, basis0,
                basis1, w0, w1, jac, in, out, scale);
            /* unsigned int *pindex = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "pindex", nmTot * sizeof(unsigned int));
            IProductWRTBaseTriKernel_QP_1D<TData, SCALE, APPEND, DEFORMED,
                                           SHMEM>(nshared,
                nm0, nm1, nmTot, nq0, nq1, nelmts, correct, pindex, basis0,
                basis1, w0, w1, jac, in, out, scale);*/
        }
        else
        {
            IProductWRTBaseTriKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nmTot, nq0, nq1, nelmts, correct, basis0,
                basis1, w0, w1, jac, wsp, in, out, scale);
        }
    }
}

template <typename ExecSpace, typename TData, bool SCALE, bool APPEND,
          bool DEFORMED, bool MULTILEVEL = true, bool SHMEM = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
IProductWRTBase3DKernel(LibUtilities::ShapeType shapetype,
                        const unsigned int nm0, const unsigned int nm1,
                        const unsigned int nm2, const unsigned int nq0,
                        const unsigned int nq1, const unsigned int nq2,
                        const unsigned int nelmts, const bool correct,
                        const TData *basis0, const TData *basis1,
                        const TData *basis2, const TData *w0, const TData *w1,
                        const TData *w2, const TData *jac,
                        [[maybe_unused]] TData *wsp, const TData *in,
                        TData *out, const TData scale = 1.0)
{
    if (shapetype == LibUtilities::Hex)
    {
        const unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared =
            SHMEM ? nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 + nq1 + nq2 : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
            IProductWRTBaseHexKernel_QP<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, basis0,
                basis1, basis2, w0, w1, w2, jac, in, out, scale);
            /* IProductWRTBaseHexKernel_QP_1D<TData, SCALE, APPEND, DEFORMED,
                                           SHMEM>(nshared,
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, basis0, basis1,
                basis2, w0, w1, w2, jac, in, out, scale);*/
        }
        else
        {
            IProductWRTBaseHexKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, basis0,
                basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        const unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        unsigned int nshared =
            SHMEM ? nm0 * nq0 + nm01 * nq1 + nmTot * nq2 + nq0 + nq1 + nq2 : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm01 * nq2 + nm2;
            IProductWRTBaseTetKernel_QP<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
            /* unsigned int *pindex1 = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "pindex1", nm01 * sizeof(unsigned int));
            unsigned int *qindex1 = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "qindex1", nm01 * sizeof(unsigned int));
            unsigned int *pindex2 = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "pindex2", nmTot * sizeof(unsigned int));
            unsigned int *qindex2 = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "qindex2", nmTot * sizeof(unsigned int));
            IProductWRTBaseTetKernel_QP_1D<TData, SCALE, APPEND, DEFORMED,
                                           SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                pindex1, qindex1, pindex2, qindex2, basis0, basis1, basis2, w0,
                w1, w2, jac, in, out, scale);*/
        }
        else
        {
            IProductWRTBaseTetKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        const unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        unsigned int nshared =
            SHMEM ? nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nq0 + nq1 + nq2 : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared +=
                nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nm1;
            IProductWRTBasePrismKernel_QP<TData, SCALE, APPEND, DEFORMED,
                                          SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
            /* unsigned int *pindex = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "pindex", nmTot * sizeof(unsigned int));
            unsigned int *qindex = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "qindex", nmTot * sizeof(unsigned int));
            unsigned int *rindex = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "rindex", nmTot * sizeof(unsigned int));
            IProductWRTBasePrismKernel_QP_1D<TData, SCALE, APPEND, DEFORMED,
                                             SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                pindex, qindex, rindex, basis0, basis1, basis2, w0, w1, w2, jac,
                in, out, scale);*/
        }
        else
        {
            IProductWRTBasePrismKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared =
            SHMEM ? nm0 * nq0 + nm1 * nq1 + nmTot * nq2 + nq0 + nq1 + nq2 : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 1u;
            IProductWRTBasePyrKernel_QP<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale);
            /*unsigned int *pindex = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "pindex", nmTot * sizeof(unsigned int));
            unsigned int *qindex = (unsigned int *)Kokkos::kokkos_malloc<
                Kokkos::DefaultExecutionSpace::memory_space>(
                "qindex", nmTot * sizeof(unsigned int));
            IProductWRTBasePyrKernel_QP_1D<TData, SCALE, APPEND, DEFORMED,
                                           SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                pindex, qindex, basis0, basis1, basis2, w0, w1, w2, jac, in,
                out, scale);*/
        }
        else
        {
            IProductWRTBasePyrKernel<TData, SCALE, APPEND, DEFORMED, SHMEM>(
                nshared, nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts, correct,
                basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
