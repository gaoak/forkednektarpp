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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/Spaces.hpp"
#include "Operators/SYCLQueue.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED>
void PhysDeriv1DKernel(const unsigned int nq0, const unsigned int ncoord,
                       const unsigned int nelmt, const unsigned int nsize,
                       const unsigned int dfsize, const TData *__restrict D0,
                       const TData *__restrict df, const TData *__restrict in,
                       TData *__restrict out, const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::SYCL::width;

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData, bool DEFORMED>
void PhysDeriv1DKernel_QP(const unsigned int nq0, const unsigned int ncoord,
                          const unsigned int nelmt, const unsigned int nsize,
                          const unsigned int dfsize, const TData *__restrict D0,
                          const TData *__restrict df,
                          const TData *__restrict in, TData *__restrict out,
                          const sycl::nd_item<3> &item_ct1)
{
    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int offset = nq0 * e;

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
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
        }

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv2DKernel(const unsigned int nq0, const unsigned int nq1,
                       const unsigned int ncoord, const unsigned int nelmt,
                       const unsigned int nsize, const unsigned int dfsize,
                       const TData *__restrict D0, const TData *__restrict D1,
                       const TData *__restrict Z0, const TData *__restrict Z1,
                       const TData *__restrict df, const TData *__restrict in,
                       TData *__restrict out, TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::SYCL::width;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_D0              = SHMEM ? shared : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
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
    }

    // Precompute geometric factors.
    if constexpr (SHAPETYPE == LibUtilities::Tri)
    {
        s_xfrm0 = SHMEM ? s_D1 + nq1 * nq1 : shared;
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

    if constexpr (SHAPETYPE == LibUtilities::Tri || SHMEM)
    {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                const unsigned int dfindex = DEFORMED ? index : e;

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
                    d0 *= s_xfrm0[j];
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict Z0,
    const TData *__restrict Z1, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp             = shared;
    TData *s_D0              = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData xfrm0, xfrm1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
        {
            s_D1[idx] = D1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp[idx] = in[offset + idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
             j += item_ct1.get_local_range(1))
        {
            for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                 i += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_ji  = nq0 * j + i;
                const unsigned int index   = offset + cnt_ji;
                const unsigned int dfindex = DEFORMED ? index : e;

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
                if constexpr (SHAPETYPE == LibUtilities::Tri)
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
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv2DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict Z0,
    const TData *__restrict Z1, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp             = shared;
    TData *s_D0              = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData xfrm0, xfrm1;

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
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp[idx] = in[offset + idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? index : e;

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
            if constexpr (SHAPETYPE == LibUtilities::Tri)
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
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv3DKernel(const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nelmt,
                       const unsigned int nsize, const unsigned int dfsize,
                       const TData *__restrict D0, const TData *__restrict D1,
                       const TData *__restrict D2, const TData *__restrict Z0,
                       const TData *__restrict Z1, const TData *__restrict Z2,
                       const TData *__restrict df, const TData *__restrict in,
                       TData *__restrict out, TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ncoord   = 3u;
    constexpr unsigned int warpsize = NektarSpaces::SYCL::width;

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
    }

    // Precompute geometric factors.
    if constexpr (SHAPETYPE == LibUtilities::Tet)
    {
        s_xfrm_eta0  = SHMEM ? s_D2 + nq2 * nq2 : shared;
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
        s_xfrm_eta0 = SHMEM ? s_D2 + nq2 * nq2 : shared;
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
        s_xfrm_eta0 = SHMEM ? s_D2 + nq2 * nq2 : shared;
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

    if constexpr (SHAPETYPE != LibUtilities::Hex || SHMEM)
    {
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                    const unsigned int dfindex = DEFORMED ? index : e;

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
                        TData xfrm = s_xfrm_eta1m[j] * s_xfrm_eta2[k];
                        TData tmp0 = xfrm * d0;
                        TData tmp1 = s_xfrm_eta0[i] * tmp0;
                        TData tmp2 = s_xfrm_eta2[k] * d1;
                        d0         = tmp0;
                        d1         = tmp1 + tmp2;
                        d2 += tmp1 + s_xfrm_eta1[j] * tmp2;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        d0 *= s_xfrm_eta2[k];
                        d2 += s_xfrm_eta0[i] * d0;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict D2,
    const TData *__restrict Z0, const TData *__restrict Z1,
    const TData *__restrict Z2, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp             = shared;
    TData *s_D0              = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_D2              = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nq0 * nq0; idx += stride)
        {
            s_D0[idx] = D0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1 * nq1; idx += stride)
        {
            s_D1[idx] = D1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2 * nq2; idx += stride)
        {
            s_D2[idx] = D2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            s_wsp[idx] = in[offset + idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Compute tensorial derivative.
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                     i += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;
                    const unsigned int index   = offset + cnt_kji;
                    const unsigned int dfindex = DEFORMED ? index : e;

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
                    if constexpr (SHAPETYPE == LibUtilities::Tet)
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
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                        xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                        d0 *= xfrm_eta2;
                        d2 += xfrm_eta0 * d0;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
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
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED,
          bool SHMEM = true>
void PhysDeriv3DKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const unsigned int nsize,
    const unsigned int dfsize, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict D2,
    const TData *__restrict Z0, const TData *__restrict Z1,
    const TData *__restrict Z2, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp             = shared;
    TData *s_D0              = SHMEM ? s_wsp + nqTot : (TData *)D0;
    TData *s_D1              = SHMEM ? s_D0 + nq0 * nq0 : (TData *)D1;
    TData *s_D2              = SHMEM ? s_D1 + nq1 * nq1 : (TData *)D2;
    TData xfrm_eta0, xfrm_eta1, xfrm_eta1m, xfrm_eta2;

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
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int offset = nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp[idx] = in[offset + idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = (idx / nq0) % nq1;
            const unsigned int k       = idx / (nq0 * nq1);
            unsigned int index         = offset + idx;
            const unsigned int dfindex = DEFORMED ? index : e;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += s_D0[q * nq0 + i] * s_wsp[nq0 * nq1 * k + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += s_D1[q * nq1 + j] * s_wsp[nq0 * nq1 * k + nq0 * q + i];
            }

            // Direction 2
            TData d2 = 0.0;
            for (unsigned int q = 0u; q < nq2; ++q)
            {
                d2 += s_D2[q * nq2 + k] * s_wsp[nq0 * nq1 * q + nq0 * j + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tet)
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
            else if constexpr (SHAPETYPE == LibUtilities::Prism)
            {
                xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Pyr)
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
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

// Launchers

// PhysDeriv1DKernel
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    PhysDeriv1DKernel(const unsigned int nq0, const unsigned int ncoord,
                      const unsigned int nelmts, const unsigned int nsize,
                      const unsigned int dfsize, const TData *D0,
                      const TData *df, const TData *in, TData *out)
{
    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0, NektarSpaces::SYCL::defaultBlockSize)
                   : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize = std::min(
        MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize, 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (MULTILEVEL)
    {
        Q.submit([=](sycl::handler &cgh) {
             const sycl::range<3> localSize(1, 1,  NektarSpaces::SYCL::width);
             const sycl::range<3> globalSize(1, 1, gridsize *  NektarSpaces::SYCL::width);

             cgh.parallel_for(sycl::nd_range<3>(globalSize, localSize),
                              [=](sycl::nd_item<3> item) {
                                  PhysDeriv1DKernel_QP<TData, DEFORMED>(
                                      nq0, ncoord, nelmts, nsize, dfsize, D0,
                                      df, in, out, item);
                              });
         }).wait();
    }
    else
    {
        Q.submit([=](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shared(sycl::range<1>(nq0 * nq0),
                                                   cgh);

             const sycl::range<3> localSize(1, 1, blocksize);
             const sycl::range<3> globalSize(1, 1, gridsize * blocksize);

             cgh.parallel_for(sycl::nd_range<3>(globalSize, localSize),
                              [=](sycl::nd_item<3> item) {
                                  TData *shmPtr = shared.get_pointer();
                                  PhysDeriv1DKernel<TData, DEFORMED>(
                                      nq0, ncoord, nelmts, nsize, dfsize, D0,
                                      df, in, out, shmPtr, item);
                              });
         }).wait();
    }
}

//
// PhysDeriv2DKernel
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true, bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    PhysDeriv2DKernel(LibUtilities::ShapeType shapetype, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int ncoord,
                      const unsigned int nelmts, const unsigned int nsize,
                      const unsigned int dfsize, const TData *D0,
                      const TData *D1, const TData *Z0, const TData *Z1,
                      const TData *df, const TData *in, TData *out)
{
    const sycl::range<3> blocksize2d(1, std::min(nq0, 16u), std::min(nq1, 16u));
    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0 * nq1, NektarSpaces::SYCL::defaultBlockSize)
                   : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize = std::min(
        MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize, 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    unsigned int nshared = SHMEM ? nq0 * nq0 + nq1 * nq1 : 0u;

    if (shapetype == LibUtilities::Quad)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1;

            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 // Set WorkGroup dimensions
                 sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * blocksize2d;

                 cgh.parallel_for(
                     sycl::nd_range<3>(globalSize, blocksize2d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv2DKernel_QP<TData, LibUtilities::Quad,
                                              DEFORMED, SHMEM>(
                             nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1,
                             nullptr, nullptr, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
        else
        {
            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 // Set WorkGroup dimensions
                 const sycl::range<3> wgSize(1, 1, blocksize);
                 const sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * wgSize;

                 cgh.parallel_for(
                     sycl::nd_range<3>(globalSize, wgSize),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv2DKernel<TData, LibUtilities::Quad, DEFORMED,
                                           SHMEM>(
                             nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1,
                             nullptr, nullptr, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1;

            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);
                 // Set WorkGroup dimensions
                 sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * blocksize2d;

                 cgh.parallel_for(
                     sycl::nd_range<3>(globalSize, blocksize2d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv2DKernel_QP<TData, LibUtilities::Tri,
                                              DEFORMED, SHMEM>(
                             nq0, nq1, ncoord, nelmts, nsize, dfsize, D0, D1,
                             Z0, Z1, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
        else
        {
            nshared += nq0 * nq1;

            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 // Set WorkGroup dimensions
                 sycl::range<3> wgSize(1, 1, blocksize);
                 sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * wgSize;

                 cgh.parallel_for(
                     sycl::nd_range<3>(globalSize, wgSize),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv2DKernel<TData, LibUtilities::Tri, DEFORMED,
                                           SHMEM>(nq0, nq1, ncoord, nelmts,
                                                  nsize, dfsize, D0, D1, Z0, Z1,
                                                  df, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
}

// PhysDeriv3DKernel
template <typename ExecSpace, typename TData, bool DEFORMED,
          bool MULTILEVEL = true, bool SHMEM = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    PhysDeriv3DKernel(LibUtilities::ShapeType shapetype, const unsigned int nq0,
                      const unsigned int nq1, const unsigned int nq2,
                      const unsigned int nelmts, const unsigned int nsize,
                      const unsigned int dfsize, const TData *D0,
                      const TData *D1, const TData *D2, const TData *Z0,
                      const TData *Z1, const TData *Z2, const TData *df,
                      const TData *in, TData *out)
{
    const sycl::range<3> blocksize3d(std::min(nq0, 8u), std::min(nq1, 8u),
                                     std::min(nq2, 8u));
    const unsigned int blocksize =
        MULTILEVEL
            ? std::min(nq0 * nq1 * nq2, NektarSpaces::SYCL::defaultBlockSize)
            : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize = std::min(
        MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize, 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    unsigned int nshared = SHMEM ? (nq0 * nq0 + nq1 * nq1 + nq2 * nq2) : 0u;

    if (shapetype == LibUtilities::Hex)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel_QP<TData, LibUtilities::Hex,
                                              DEFORMED, SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             nullptr, nullptr, nullptr, df, in, out, shmPtr,
                             item);
                     });
             }).wait();
        }
        else
        {
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel<TData, LibUtilities::Hex, DEFORMED,
                                           SHMEM>(nq0, nq1, nq2, nelmts, nsize,
                                                  dfsize, D0, D1, D2, nullptr,
                                                  nullptr, nullptr, df, in, out,
                                                  shmPtr, item);
                     });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel_QP<TData, LibUtilities::Tet,
                                              DEFORMED, SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             Z0, Z1, Z2, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
        else
        {
            nshared += nq0 + 2u * nq1 + nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel<TData, LibUtilities::Tet, DEFORMED,
                                           SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             Z0, Z1, Z2, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel_QP<TData, LibUtilities::Prism,
                                              DEFORMED, SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             Z0, nullptr, Z2, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
        else
        {
            nshared += nq0 + nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel<TData, LibUtilities::Prism, DEFORMED,
                                           SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             Z0, nullptr, Z2, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel_QP<TData, LibUtilities::Pyr,
                                              DEFORMED, SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             Z0, Z1, Z2, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
        else
        {
            nshared += nq0 + nq1 + nq2;
            Q.submit([=](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         PhysDeriv3DKernel<TData, LibUtilities::Pyr, DEFORMED,
                                           SHMEM>(
                             nq0, nq1, nq2, nelmts, nsize, dfsize, D0, D1, D2,
                             Z0, Z1, Z2, df, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
