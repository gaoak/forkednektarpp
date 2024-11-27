///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSYCLSumFacKernels.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

#include "Operators/Utils/SYCLQueue.hpp"

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict D0, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    unsigned int e = item_ct1.get_global_id(2);

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
                out[d * nelmt * nq0 + index] = d0 * df[d * warpsize + dfindex];
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict D0, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out,
    const sycl::nd_item<3> &item_ct1)
{
    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nq0 : 1;
        const unsigned int dfoffset = ncoord * dfsize * e;
        const unsigned int offset   = nq0 * e;

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            const unsigned int index   = offset + i;
            const unsigned int dfindex = DEFORMED ? dfoffset + i : dfoffset;

            // Compute tensorial derivative.
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[offset + q];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nq0 + index] = d0 * df[d * dfsize + dfindex];
            }
        }

        e += item_ct1.get_group_range(2);
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, bool SHMEM,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict Z0,
    const TData *__restrict Z1, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf = 2 * ncoord;

    const unsigned int nqTot = nq0 * nq1;

    TData *s_D0 = SHMEM ? shared : (TData *)D0;
    TData *s_D1 = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_xfrm0, *s_xfrm1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_D1[idx] = D1[idx];
        }

        // Precompute geometric factors.
        if constexpr (SHAPETYPE == LibUtilities::Tri)
        {
            s_xfrm0 = s_D1 + nq1 * nq1;
            s_xfrm1 = s_xfrm0 + nq1;

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm0[idx] = 2.0 / (1.0 - Z1[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm1[idx] = 0.5 * (1.0 + Z0[idx]);
            }
        }
    }

    if constexpr (SHAPETYPE == LibUtilities::Tri || SHMEM)
    {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index =
                    nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                   warpsize * cnt_ji * ndf + ilane
                             : ndf * warpsize * iwarp + ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += s_D0[q * nq0 + i] *
                          in[nqTot * warpsize * iwarp +
                             warpsize * (nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += s_D1[q * nq1 + j] *
                          in[nqTot * warpsize * iwarp +
                             warpsize * (nq0 * q + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    if constexpr (SHMEM)
                    {
                        d0 *= s_xfrm0[j];
                        d1 += d0 * s_xfrm1[i];
                    }
                    else
                    {
                        d0 *= 2.0 / (1.0 - Z1[j]);
                        d1 += d0 * 0.5 * (1.0 + Z0[i]);
                    }
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

        e += item_ct1.get_global_range(2);
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict Z0,
    const TData *__restrict Z1, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[offset + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[offset + nq0 * q + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                TData xfrm0 = 2.0 / (1.0 - Z1[j]);
                TData xfrm1 = 0.5 * (1.0 + Z0[i]);
                d0 *= xfrm0;
                d1 += d0 * xfrm1;
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nqTot + index] =
                    d0 * df[(2u * d) * dfsize + dfindex] +
                    d1 * df[(2u * d + 1u) * dfsize + dfindex];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, bool SHMEM,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict D2,
    const TData *__restrict Z0, const TData *__restrict Z1,
    const TData *__restrict Z2, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_D0              = SHMEM ? shared : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_D2              = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
    TData *s_xfrm_eta0, *s_xfrm_eta1, *s_xfrm_eta1m, *s_xfrm_eta2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_D1[idx] = D1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_D2[idx] = D2[idx];
        }

        // Precompute geometric factors.
        if constexpr (SHAPETYPE == LibUtilities::Tet)
        {
            s_xfrm_eta0  = s_D2 + nq2 * nq2;
            s_xfrm_eta1  = s_xfrm_eta0 + nq0;
            s_xfrm_eta1m = s_xfrm_eta1 + nq1;
            s_xfrm_eta2  = s_xfrm_eta1m + nq1;

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta1m[idx] = 2.0 / (1.0 - Z1[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
            }
        }
        else if constexpr (SHAPETYPE == LibUtilities::Prism)
        {
            s_xfrm_eta0 = s_D2 + nq2 * nq2;
            s_xfrm_eta2 = s_xfrm_eta0 + nq0;

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
            }
        }
        else if constexpr (SHAPETYPE == LibUtilities::Pyr)
        {
            s_xfrm_eta0 = s_D2 + nq2 * nq2;
            s_xfrm_eta1 = s_xfrm_eta0 + nq0;
            s_xfrm_eta2 = s_xfrm_eta1 + nq1;

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
            }

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
                 idx += item_ct1.get_local_range(2))
            {
                s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
            }
        }
    }

    if constexpr (SHAPETYPE != LibUtilities::Hex || SHMEM)
    {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
        {
            for (unsigned int j = 0u; j < nq1; j++)
            {
                for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane;
                    const unsigned int dfindex =
                        DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                       warpsize * cnt_kji * ndf + ilane
                                 : ndf * warpsize * iwarp + ilane;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += s_D0[q * nq0 + i] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * k + nq0 * j + q) +
                                 ilane];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += s_D1[q * nq1 + j] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * k + nq0 * q + i) +
                                 ilane];
                    }

                    // Direction 2
                    TData d2 = 0.0;
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += s_D2[q * nq2 + k] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * q + nq0 * j + i) +
                                 ilane];
                    }

                    // Moving from standard to collapsed coordinates.
                    if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        if constexpr (SHMEM)
                        {
                            TData xfrm = s_xfrm_eta1m[j] * s_xfrm_eta2[k];
                            TData tmp0 = xfrm * d0;
                            TData tmp1 = s_xfrm_eta0[i] * tmp0;
                            TData tmp2 = s_xfrm_eta2[k] * d1;
                            d0         = tmp0;
                            d1         = tmp1 + tmp2;
                            d2 += tmp1 + s_xfrm_eta1[j] * tmp2;
                        }
                        else
                        {
                            TData xfrm =
                                2.0 / (1.0 - Z1[j]) * 2.0 / (1.0 - Z2[k]);
                            TData tmp0 = xfrm * d0;
                            TData tmp1 = 0.5 * (1.0 + Z0[i]) * tmp0;
                            TData tmp2 = 2.0 / (1.0 - Z2[k]) * d1;
                            d0         = tmp0;
                            d1         = tmp1 + tmp2;
                            d2 += tmp1 + 0.5 * (1.0 + Z1[j]) * tmp2;
                        }
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        if constexpr (SHMEM)
                        {
                            d0 *= s_xfrm_eta2[k];
                            d2 += s_xfrm_eta0[i] * d0;
                        }
                        else
                        {
                            d0 *= 2.0 / (1.0 - Z2[k]);
                            d2 += 0.5 * (1.0 + Z0[i]) * d0;
                        }
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        if constexpr (SHMEM)
                        {
                            d0 *= s_xfrm_eta2[k];
                            d1 *= s_xfrm_eta2[k];
                            d2 += s_xfrm_eta0[i] * d0 + s_xfrm_eta1[j] * d1;
                        }
                        else
                        {
                            d0 *= 2.0 / (1.0 - Z2[k]);
                            d1 *= 2.0 / (1.0 - Z2[k]);
                            d2 += 0.5 * (1.0 + Z0[i]) * d0 +
                                  0.5 * (1.0 + Z1[j]) * d1;
                        }
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

        e += item_ct1.get_global_range(2);
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict D2,
    const TData *__restrict Z0, const TData *__restrict Z1,
    const TData *__restrict Z2, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = (idx / nq0) % nq1;
            const unsigned int k       = idx / (nq0 * nq1);
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 +=
                    D0[q * nq0 + i] * in[offset + nq0 * nq1 * k + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 +=
                    D1[q * nq1 + j] * in[offset + nq0 * nq1 * k + nq0 * q + i];
            }

            // Direction 2
            TData d2 = 0.0;
            for (unsigned int q = 0u; q < nq2; ++q)
            {
                d2 +=
                    D2[q * nq2 + k] * in[offset + nq0 * nq1 * q + nq0 * j + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tet)
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
            else if constexpr (SHAPETYPE == LibUtilities::Prism)
            {
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Pyr)
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
                out[d * nelmt * nqTot + index] =
                    d0 * df[(3u * d) * dfsize + dfindex] +
                    d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                    d2 * df[(3u * d + 2u) * dfsize + dfindex];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

// Launchers

// PhysDeriv1DKernel
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int nq0,
                                               const unsigned int ncoord,
                                               const unsigned int nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (MULTILEVEL)
    {
        const unsigned int SYCLBlockSize =
            std::min(nq0, NektarSpaces::SYCL::defaultBlockSize);
        const unsigned int SYCLGridSize = std::min(nelmt, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     PhysDeriv1DKernel_QP<DEFORMED>(nq0, ncoord, nelmt, D0, df,
                                                    in, out, item);
                 });
         }).wait();
    }
    else
    {
        const unsigned int SYCLBlockSize = NektarSpaces::SYCL::defaultBlockSize;
        const unsigned int SYCLGridSize =
            std::min((nelmt + SYCLBlockSize - 1u) / SYCLBlockSize, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     PhysDeriv1DKernel<DEFORMED>(nq0, ncoord, nelmt, D0, df, in,
                                                 out, item);
                 });
         }).wait();
    }
}

//
// PhysDeriv2DKernel
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *Z0,
    const TData *Z1, const TData *df, const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (MULTILEVEL)
    {
        const unsigned int SYCLBlockSize =
            std::min(nq0 * nq1, NektarSpaces::SYCL::defaultBlockSize);
        const unsigned int SYCLGridSize = std::min(nelmt, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     PhysDeriv2DKernel_QP<SHAPE_TYPE, DEFORMED>(
                         nq0, nq1, ncoord, nelmt, D0, D1, Z0, Z1, df, in, out,
                         item);
                 });
         }).wait();
    }
    else
    {
        const unsigned int nshared =
            PhysDerivSharedMemorySize<SHAPE_TYPE>(nq0, nq1);
        const unsigned int SYCLBlockSize = NektarSpaces::SYCL::defaultBlockSize;
        const unsigned int SYCLGridSize =
            std::min((nelmt + SYCLBlockSize - 1u) / SYCLBlockSize, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                   cgh);
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     TData *shmptr = shared
                                         .template get_multi_ptr<
                                             sycl::access::decorated::no>()
                                         .get();
                     PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED, true>(
                         nq0, nq1, ncoord, nelmt, D0, D1, Z0, Z1, df, in, out,
                         shmptr, item);
                 });
         }).wait();
    }
}

// PhysDeriv3DKernel
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *Z0, const TData *Z1, const TData *Z2, const TData *df,
    const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (MULTILEVEL)
    {
        const unsigned int SYCLBlockSize =
            std::min(nq0 * nq1 * nq2, NektarSpaces::SYCL::defaultBlockSize);
        const unsigned int SYCLGridSize = std::min(nelmt, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     PhysDeriv3DKernel_QP<SHAPE_TYPE, DEFORMED>(
                         nq0, nq1, nq2, nelmt, D0, D1, D2, Z0, Z1, Z2, df, in,
                         out, item);
                 });
         }).wait();
    }
    else
    {
        const unsigned int nshared =
            PhysDerivSharedMemorySize<SHAPE_TYPE>(nq0, nq1, nq2);
        const unsigned int SYCLBlockSize = NektarSpaces::SYCL::defaultBlockSize;
        const unsigned int SYCLGridSize =
            std::min((nelmt + SYCLBlockSize - 1u) / SYCLBlockSize, 2147483647u);

        Q.submit([=](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                   cgh);
             const sycl::range<3> blocksize(1, 1, SYCLBlockSize);
             const sycl::range<3> gridsize(1, 1, SYCLGridSize);
             cgh.parallel_for(
                 sycl::nd_range<3>(gridsize * blocksize, blocksize),
                 [=](sycl::nd_item<3> item) {
                     TData *shmptr = shared
                                         .template get_multi_ptr<
                                             sycl::access::decorated::no>()
                                         .get();
                     PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED, true>(
                         nq0, nq1, nq2, nelmt, D0, D1, D2, Z0, Z1, Z2, df, in,
                         out, shmptr, item);
                 });
         }).wait();
    }
}

} // namespace Nektar::Operators::detail

#endif
