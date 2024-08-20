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
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = 32u;

    Kokkos::parallel_for(
        nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int i = 0u; i < nq0; ++i)
            {
                const unsigned int index =
                    nq0 * warpsize * iwarp + warpsize * i + ilane;
                const unsigned int dfindex = DEFORMED ? index : e;

                TData sum = 0.0;
                for (unsigned int d = 0u; d < ncoord; ++d)
                {
                    sum += df[d * dfsize + dfindex] * in[d * nsize + index];
                }
                out[index] = sum;
            }
        });
}

template <typename TData, bool DEFORMED>
void IProductWRTDerivBase1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT df, const TData *KOKKOS_RESTRICT in,
    TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nq0 * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0), [&](const unsigned int &i) {
                    const unsigned int index   = offset + i;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    TData sum = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        sum += df[d * dfsize + dfindex] * in[d * nsize + index];
                    }
                    out[index] = sum;
                });
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1;

    Kokkos::parallel_for(
        nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    TData f0, f1;

                    if (SHAPETYPE == LibUtilities::Tri)
                    {
                        f0 = 2.0 / (1.0 - Z1[j]);
                        f1 = 0.5 * (1.0 + Z0[i]);
                    }

                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0;
                    for (unsigned int d = 0; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
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
                }
            }
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;

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

                    const unsigned int cnt_ji  = nq0 * j + i;
                    const unsigned int index   = offset + cnt_ji;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
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
    const unsigned int dfsize, const TData *KOKKOS_RESTRICT Z0,
    const TData *KOKKOS_RESTRICT Z1, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nqTot * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0 * nq1),
                [&](const unsigned int &idx) {
                    const unsigned int i       = idx % nq0;
                    const unsigned int j       = idx / nq0;
                    const unsigned int index   = offset + idx;
                    const unsigned int dfindex = DEFORMED ? index : e;
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
                        sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
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
void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    constexpr unsigned int warpsize = 32u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    Kokkos::parallel_for(
        nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
            const unsigned int iwarp = e / warpsize;
            const unsigned int ilane = e % warpsize;

            for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
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
                            f0 = 2.0 * f2 / (1.0 - Z1[j]);
                        }

                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        const unsigned int dfindex = DEFORMED ? index : e;

                        TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                        for (unsigned int d = 0u; d < ncoord; ++d)
                        {
                            TData tmp = in[d * nsize + index];
                            sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                            sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                            sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
                        }

                        if (SHAPETYPE == LibUtilities::Hex)
                        {
                            out[index]              = sum1;
                            out[nsize + index]      = sum2;
                            out[2u * nsize + index] = sum3;
                        }
                        else if (SHAPETYPE == LibUtilities::Tet)
                        {
                            out[index] = (sum1 + (sum2 + sum3) * f1) * f0;
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
                    }
                }
            }
        });
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
void IProductWRTDerivBase3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;

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
                        f0 = 2.0 * f2 / (1.0 - Z1[j]);
                    }

                    const unsigned int index =
                        offset + nq0 * nq1 * k + nq0 * j + i;
                    const unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
                    }

                    if (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]              = sum1;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index]         = (sum1 + (sum2 + sum3) * f1) * f0;
                        out[nsize + index] = (sum2 + sum3 * f3) * f2;
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
    const unsigned int nsize, const unsigned int dfsize,
    const TData *KOKKOS_RESTRICT Z0, const TData *KOKKOS_RESTRICT Z1,
    const TData *KOKKOS_RESTRICT Z2, const TData *KOKKOS_RESTRICT df,
    const TData *KOKKOS_RESTRICT in, TData *KOKKOS_RESTRICT out)
{
    typedef Kokkos::TeamPolicy<>::member_type team_handle;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_handle &team) {
            unsigned int e = team.league_rank();

            const unsigned int offset = nqTot * e;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nq0 * nq1 * nq2),
                [&](const unsigned int &idx) {
                    const unsigned int i       = idx % nq0;
                    const unsigned int j       = (idx / nq0) % nq1;
                    const unsigned int k       = idx / (nq0 * nq1);
                    const unsigned int index   = offset + idx;
                    const unsigned int dfindex = DEFORMED ? index : e;
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
                        f0 = 2.0 * f2 / (1.0 - Z1[j]);
                    }

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0u; d < ncoord; ++d)
                    {
                        TData tmp = in[d * nsize + index];
                        sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
                        sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
                        sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
                    }

                    if (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]              = sum1;
                        out[nsize + index]      = sum2;
                        out[2u * nsize + index] = sum3;
                    }
                    else if (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index]         = (sum1 + (sum2 + sum3) * f1) * f0;
                        out[nsize + index] = (sum2 + sum3 * f3) * f2;
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
                             const unsigned int nsize,
                             const unsigned int dfsize, const TData *df,
                             const TData *in, TData *out)
{
    if constexpr (MULTILEVEL)
    {
        IProductWRTDerivBase1DKernel_QP<TData, DEFORMED>(
            nq0, ncoord, nelmts, nsize, dfsize, df, in, out);
    }
    else
    {
        IProductWRTDerivBase1DKernel<TData, DEFORMED>(
            nq0, ncoord, nelmts, nsize, dfsize, df, in, out);
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
                             const unsigned int nsize,
                             const unsigned int dfsize, const TData *Z0,
                             const TData *Z1, const TData *df, const TData *in,
                             TData *out)
{
    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Quad,
                                            DEFORMED>(nq0, nq1, ncoord, nelmts,
                                                      nsize, dfsize, nullptr,
                                                      nullptr, df, in, out);
        }
        else
        {
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Quad, DEFORMED>(
                nq0, nq1, ncoord, nelmts, nsize, dfsize, nullptr, nullptr, df,
                in, out);
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase2DKernel_QP<TData, LibUtilities::Tri, DEFORMED>(
                nq0, nq1, ncoord, nelmts, nsize, dfsize, Z0, Z1, df, in, out);
        }
        else
        {
            IProductWRTDerivBase2DKernel<TData, LibUtilities::Tri, DEFORMED>(
                nq0, nq1, ncoord, nelmts, nsize, dfsize, Z0, Z1, df, in, out);
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
                             const unsigned int nsize,
                             const unsigned int dfsize, const TData *Z0,
                             const TData *Z1, const TData *Z2, const TData *df,
                             const TData *in, TData *out)
{
    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Hex, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, nullptr, nullptr,
                nullptr, df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Hex, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, nullptr, nullptr,
                nullptr, df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Tet, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, Z0, Z1, Z2, df,
                in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Tet, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, Z0, Z1, Z2, df,
                in, out);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Prism,
                                            DEFORMED>(nq0, nq1, nq2, ncoord,
                                                      nelmts, nsize, dfsize, Z0,
                                                      nullptr, Z2, df, in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Prism, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, Z0, nullptr, Z2,
                df, in, out);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (MULTILEVEL)
        {
            IProductWRTDerivBase3DKernel_QP<TData, LibUtilities::Pyr, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, Z0, Z1, Z2, df,
                in, out);
        }
        else
        {
            IProductWRTDerivBase3DKernel<TData, LibUtilities::Pyr, DEFORMED>(
                nq0, nq1, nq2, ncoord, nelmts, nsize, dfsize, Z0, Z1, Z2, df,
                in, out);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
