///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseKokkosSumFacKernels.hpp
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
void IProductWRTDerivBase1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int blocksize = NektarSpaces::KOKKOS::defaultBlockSize;
    const unsigned int gridsize =
        std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();

            while (e < nelmt)
            {
                const unsigned int iwarp = e / warpsize;
                const unsigned int ilane = e % warpsize;

                for (unsigned int i = 0u; i < nq0; ++i)
                {
                    const unsigned int index =
                        nq0 * warpsize * iwarp + warpsize * i + ilane;
                    const unsigned int dfindex =
                        DEFORMED ? nq0 * ncoord * warpsize * iwarp +
                                       warpsize * i * ncoord + ilane
                                 : ncoord * warpsize * iwarp + ilane;

                    TData sum = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        sum +=
                            df[d * warpsize + dfindex] * in[d * nsize + index];
                    }
                    out[index] = sum;
                }

                e += team.team_size() * team.league_size();
            }
        });
}

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nq0 * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0), [&](const unsigned int &i) {
                    const unsigned int index = offset + i;
                    const unsigned int dfindex =
                        DEFORMED ? ncoord * index : ncoord * e;

                    TData sum = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        sum += df[d + dfindex] * in[d * nsize + index];
                    }
                    out[index] = sum;
                });
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase2DKernel(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;
    const auto ndf           = 2 * ncoord;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    const unsigned int blocksize = NektarSpaces::KOKKOS::defaultBlockSize;
    const unsigned int gridsize =
        std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_f0, *s_f1;

            // Pre-compute factor.
            if (SHAPETYPE == LibUtilities::Tri)
            {
                s_f0 = &scratch[0];
                s_f1 = s_f0 + nq1;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_f0[idx] = 2.0 / (1.0 - Z1[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
                                     });

                team.team_barrier();
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();

            while (e < nelmt)
            {
                const unsigned int iwarp = e / warpsize;
                const unsigned int ilane = e % warpsize;

                for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
                {
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_ji + ilane;
                        const unsigned int dfindex =
                            DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                           warpsize * cnt_ji * ndf + ilane
                                     : ndf * warpsize * iwarp + ilane;

                        TData sum1 = 0.0, sum2 = 0.0;
                        for (unsigned int d = 0; d < ncoord; ++d)
                        {
                            TData tmp = in[d * nsize + index];
                            sum1 += df[(2u * d) * warpsize + dfindex] * tmp;
                            sum2 +=
                                df[(2u * d + 1u) * warpsize + dfindex] * tmp;
                        }

                        // Moving from standard to collapsed coordinates.
                        if (SHAPETYPE == LibUtilities::Quad)
                        {
                            out[index]         = sum1;
                            out[nsize + index] = sum2;
                        }
                        else if (SHAPETYPE == LibUtilities::Tri)
                        {
                            out[index] = (sum1 + sum2 * s_f1[i]) * s_f0[j];
                            out[nsize + index] = sum2;
                        }
                    }
                }

                e += team.team_size() * team.league_size();
            }
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;
    const auto ndf           = 2 * ncoord;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nqTot * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                    team, nq0, nq1),
                [&](const unsigned int &i, const unsigned int &j) {
                    TData f0, f1;

                    if (SHAPETYPE == LibUtilities::Tri)
                    {
                        f0 = 2.0 / (1.0 - Z1[j]);
                        f1 = 0.5 * (1.0 + Z0[i]);
                    }

                    const unsigned int cnt_ji = nq0 * j + i;
                    const unsigned int index  = offset + cnt_ji;
                    const unsigned int dfindex =
                        DEFORMED ? ndf * index : ndf * e;

                    TData sum1 = 0.0, sum2 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(2u * d) + dfindex] * tmp;
                        sum2 += df[(2u * d + 1u) + dfindex] * tmp;
                    }

                    // Moving from standard to collapsed coordinates.
                    if (SHAPETYPE == LibUtilities::Quad)
                    {
                        out[index]         = sum1;
                        out[nsize + index] = sum2;
                    }
                    else if (SHAPETYPE == LibUtilities::Tri)
                    {
                        out[index]         = (sum1 + sum2 * f1) * f0;
                        out[nsize + index] = sum2;
                    }
                });
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase2DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;
    const auto ndf           = 2 * ncoord;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nqTot * e;

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0 * nq1),
                                 [&](const unsigned int &idx) {
                                     const unsigned int i     = idx % nq0;
                                     const unsigned int j     = idx / nq0;
                                     const unsigned int index = offset + idx;
                                     const unsigned int dfindex =
                                         DEFORMED ? ndf * index : ndf * e;
                                     TData f0, f1;

                                     if (SHAPETYPE == LibUtilities::Tri)
                                     {
                                         f0 = 2.0 / (1.0 - Z1[j]);
                                         f1 = 0.5 * (1.0 + Z0[i]);
                                     }

                                     TData sum1 = 0.0, sum2 = 0.0;
                                     for (unsigned int d = 0u; d < ncoord; ++d)
                                     {
                                         TData tmp = in[d * nsize + index];
                                         sum1 += df[(2u * d) + dfindex] * tmp;
                                         sum2 +=
                                             df[(2u * d + 1u) + dfindex] * tmp;
                                     }

                                     // Moving from standard to collapsed
                                     // coordinates.
                                     if (SHAPETYPE == LibUtilities::Quad)
                                     {
                                         out[index]         = sum1;
                                         out[nsize + index] = sum2;
                                     }
                                     else if (SHAPETYPE == LibUtilities::Tri)
                                     {
                                         out[index] = (sum1 + sum2 * f1) * f0;
                                         out[nsize + index] = sum2;
                                     }
                                 });
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase3DKernel(
    const unsigned int ssize, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT Z2,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    const auto ndf                  = 9u;
    constexpr unsigned int warpsize = NektarSpaces::KOKKOS::width;

    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int shmem_size = Kokkos::View<
        TData *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
    const unsigned int slevel = 0u;

    const unsigned int blocksize = NektarSpaces::KOKKOS::defaultBlockSize;
    const unsigned int gridsize =
        std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(gridsize, blocksize)
            .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const team_handle &team) {
            // Set shared memory.
            Kokkos::View<TData *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                scratch(team.team_scratch(slevel), ssize);
            TData *s_f0, *s_f1, *s_f2, *s_f3;

            // Pre-compute factor.
            if (SHAPETYPE == LibUtilities::Tet)
            {
                s_f0 = &scratch[0];
                s_f1 = s_f0 + nq1;
                s_f2 = s_f1 + nq0;
                s_f3 = s_f2 + nq2;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_f0[idx] = 2.0 / (1.0 - Z1[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                     [&](const unsigned int &idx) {
                                         s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_f3[idx] = 0.5 * (1.0 + Z1[idx]);
                                     });

                team.team_barrier();
            }
            else if (SHAPETYPE == LibUtilities::Prism)
            {
                s_f1 = &scratch[0];
                s_f2 = s_f1 + nq0;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                     [&](const unsigned int &idx) {
                                         s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
                                     });

                team.team_barrier();
            }
            else if (SHAPETYPE == LibUtilities::Pyr)
            {
                s_f1 = &scratch[0];
                s_f2 = s_f1 + nq0;
                s_f3 = s_f2 + nq2;

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq0),
                                     [&](const unsigned int &idx) {
                                         s_f1[idx] = 0.5 * (1.0 + Z0[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq2),
                                     [&](const unsigned int &idx) {
                                         s_f2[idx] = 2.0 / (1.0 - Z2[idx]);
                                     });

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nq1),
                                     [&](const unsigned int &idx) {
                                         s_f3[idx] = 0.5 * (1.0 + Z1[idx]);
                                     });

                team.team_barrier();
            }

            unsigned int e =
                team.league_rank() * team.team_size() + team.team_rank();

            while (e < nelmt)
            {
                const unsigned int iwarp = e / warpsize;
                const unsigned int ilane = e % warpsize;

                for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
                {
                    for (unsigned int j = 0u; j < nq1; ++j)
                    {
                        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                        {
                            const unsigned int index =
                                nqTot * warpsize * iwarp + warpsize * cnt_kji +
                                ilane;
                            const unsigned int dfindex =
                                DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                               warpsize * cnt_kji * ndf + ilane
                                         : ndf * warpsize * iwarp + ilane;

                            TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                            for (unsigned int d = 0u; d < ncoord; ++d)
                            {
                                TData tmp = in[d * nsize + index];
                                sum1 += df[(3u * d) * warpsize + dfindex] * tmp;
                                sum2 += df[(3u * d + 1u) * warpsize + dfindex] *
                                        tmp;
                                sum3 += df[(3u * d + 2u) * warpsize + dfindex] *
                                        tmp;
                            }

                            if (SHAPETYPE == LibUtilities::Hex)
                            {
                                out[index]              = sum1;
                                out[nsize + index]      = sum2;
                                out[2u * nsize + index] = sum3;
                            }
                            else if (SHAPETYPE == LibUtilities::Tet)
                            {
                                out[index] = (sum1 + (sum2 + sum3) * s_f1[i]) *
                                             s_f0[j] * s_f2[k];
                                out[nsize + index] =
                                    (sum2 + sum3 * s_f3[j]) * s_f2[k];
                                out[2u * nsize + index] = sum3;
                            }
                            else if (SHAPETYPE == LibUtilities::Prism)
                            {
                                out[index] = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                                out[nsize + index]      = sum2;
                                out[2u * nsize + index] = sum3;
                            }
                            else if (SHAPETYPE == LibUtilities::Pyr)
                            {
                                out[index] = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                                out[nsize + index] =
                                    (sum2 + sum3 * s_f3[j]) * s_f2[k];
                                out[2u * nsize + index] = sum3;
                            }
                        }
                    }
                }

                e += team.team_size() * team.league_size();
            }
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT Z2,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const auto ndf           = 9u;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nqTot * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                    team, nq0, nq1, nq2),
                [&](const unsigned int &i, const unsigned int &j,
                    const unsigned int &k) {
                    TData f0, f1, f2, f3;

                    if (SHAPETYPE == LibUtilities::Tet ||
                        SHAPETYPE == LibUtilities::Prism ||
                        SHAPETYPE == LibUtilities::Pyr)
                    {
                        f1 = 0.5 * (1.0 + Z0[i]);
                        f2 = 2.0 / (1.0 - Z2[k]);
                    }

                    if (SHAPETYPE == LibUtilities::Tet ||
                        SHAPETYPE == LibUtilities::Pyr)
                    {
                        f3 = 0.5 * (1.0 + Z1[j]);
                    }

                    if (SHAPETYPE == LibUtilities::Tet)
                    {
                        f0 = 2.0 / (1.0 - Z1[j]);
                    }

                    const unsigned int index =
                        offset + nq0 * nq1 * k + nq0 * j + i;
                    const unsigned int dfindex =
                        DEFORMED ? ndf * index : ndf * e;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(3u * d) + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) + dfindex] * tmp;
                    }

                    if (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]              = sum1;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index] = (sum1 + (sum2 + sum3) * f1) * f0 * f2;
                        out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Prism)
                    {
                        out[index]              = (sum1 + sum3 * f1) * f2;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Pyr)
                    {
                        out[index]              = (sum1 + sum3 * f1) * f2;
                        out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                        out[2u * nsize + index] = sum3;
                    }
                });
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase3DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT Z2,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const auto ndf           = 9;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nqTot * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0 * nq1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int i     = idx % nq0;
                    const unsigned int j     = (idx / nq0) % nq1;
                    const unsigned int k     = idx / (nq0 * nq1);
                    const unsigned int index = offset + idx;
                    const unsigned int dfindex =
                        DEFORMED ? ndf * index : ndf * e;
                    TData f0, f1, f2, f3;

                    if (SHAPETYPE == LibUtilities::Tet ||
                        SHAPETYPE == LibUtilities::Prism ||
                        SHAPETYPE == LibUtilities::Pyr)
                    {
                        f1 = 0.5 * (1.0 + Z0[i]);
                        f2 = 2.0 / (1.0 - Z2[k]);
                    }

                    if (SHAPETYPE == LibUtilities::Tet ||
                        SHAPETYPE == LibUtilities::Pyr)
                    {
                        f3 = 0.5 * (1.0 + Z1[j]);
                    }

                    if (SHAPETYPE == LibUtilities::Tet)
                    {
                        f0 = 2.0 / (1.0 - Z1[j]);
                    }

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(3u * d) + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) + dfindex] * tmp;
                    }

                    if (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]              = sum1;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index] = (sum1 + (sum2 + sum3) * f1) * f0 * f2;
                        out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Prism)
                    {
                        out[index]              = (sum1 + sum3 * f1) * f2;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Pyr)
                    {
                        out[index]              = (sum1 + sum3 * f1) * f2;
                        out[nsize + index]      = (sum2 + sum3 * f3) * f2;
                        out[2u * nsize + index] = sum3;
                    }
                });
        });
}

// Launchers
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
IProductWRTDerivBase1DKernel(const unsigned int nq0, const unsigned int ncoord,
                             const unsigned int nelmts,
                             const unsigned int nsize, const TData *df,
                             const TData *in, TData *out)
{
    if constexpr (MULTILEVEL)
    {
        IProductWRTDerivBase1DKernel_QP<TData, DEFORMED>(nq0, ncoord, nelmts,
                                                         nsize, df, in, out);
    }
    else
    {
        IProductWRTDerivBase1DKernel<TData, DEFORMED>(nq0, ncoord, nelmts,
                                                      nsize, df, in, out);
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
IProductWRTDerivBase2DKernel(LibUtilities::ShapeType shapetype,
                             const unsigned int nq0, const unsigned int nq1,
                             const unsigned int ncoord,
                             const unsigned int nelmts,
                             const unsigned int nsize, const TData *Z0,
                             const TData *Z1, const TData *df, const TData *in,
                             TData *out)
{
    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Quad,
                                            DEFORMED>(
                nq0, nq1, ncoord, nelmts, nsize, nullptr, nullptr, df, in, out);
        }
        else
        {
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Quad, DEFORMED>(
                0u, nq0, nq1, ncoord, nelmts, nsize, nullptr, nullptr, df, in,
                out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Tri, DEFORMED>(
                nq0, nq1, ncoord, nelmts, nsize, Z0, Z1, df, in, out);
        }
        else
        {
            unsigned int nshared = nq0 + nq1;
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Tri, DEFORMED>(
                nshared, nq0, nq1, ncoord, nelmts, nsize, Z0, Z1, df, in, out);
        }
    }
}

template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
IProductWRTDerivBase3DKernel(LibUtilities::ShapeType shapetype,
                             const unsigned int nq0, const unsigned int nq1,
                             const unsigned int nq2, const unsigned int ncoord,
                             const unsigned int nelmts,
                             const unsigned int nsize, const TData *Z0,
                             const TData *Z1, const TData *Z2, const TData *df,
                             const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Hex, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, nullptr, nullptr, nullptr,
                df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Hex, DEFORMED>(
                0u, nq0, nq1, nq2, ncoord, nelmts, nsize, nullptr, nullptr,
                nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Tet, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, Z0, Z1, Z2, df, in, out);
        }
        else
        {
            unsigned int nshared = nq0 + 2 * nq1 + nq2;
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Tet, DEFORMED>(
                nshared, nq0, nq1, nq2, ncoord, nelmts, nsize, Z0, Z1, Z2, df,
                in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Prism,
                                            DEFORMED>(nq0, nq1, nq2, ncoord,
                                                      nelmts, nsize, Z0,
                                                      nullptr, Z2, df, in, out);
        }
        else
        {
            unsigned int nshared = nq0 + nq2;
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Prism, DEFORMED>(
                nshared, nq0, nq1, nq2, ncoord, nelmts, nsize, Z0, nullptr, Z2,
                df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, Z0, Z1, Z2, df, in, out);
        }
        else
        {
            unsigned int nshared = nq0 + nq1 + nq2;
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Pyr, DEFORMED>(
                nshared, nq0, nq1, nq2, ncoord, nelmts, nsize, Z0, Z1, Z2, df,
                in, out);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
