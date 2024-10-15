///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSYCLSumFacKernels.hpp
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

template <bool SHMEM, typename TData>
void BwdTransSegKernel(const unsigned int nm0, const unsigned int nq0,
                       const unsigned int nelmt, const TData *__restrict basis0,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    TData *s_basis0 = SHMEM ? shared : (TData *)basis0;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p)
            {
                tmp += in[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                       s_basis0[p * nq0 + i];
            }
            out[nq0 * warpsize * iwarp + warpsize * i + ilane] = tmp;
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransSegKernel_QP(const unsigned int nm0, const unsigned int nq0,
                          const unsigned int nelmt,
                          const TData *__restrict basis0,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict shared,
                          const sycl::nd_item<3> &item_ct1)
{
    TData *s_wsp0   = shared;
    TData *s_basis0 = SHMEM ? s_wsp0 + nm0 : (TData *)basis0;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nm0 * e;
        TData *outptr      = out + nq0 * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; p++)
            {
                tmp += s_wsp0[p] * s_basis0[p * nq0 + i];
            }
            outptr[i] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransQuadKernel(const unsigned int nm0, const unsigned int nm1,
                        const unsigned int nmTot, const unsigned int nq0,
                        const unsigned int nq1, const unsigned int nelmt,
                        const TData *__restrict basis0,
                        const TData *__restrict basis1, TData *__restrict wsp,
                        const TData *__restrict in, TData *__restrict out,
                        TData *__restrict shared,
                        const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            // direction 0
            for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += in[nmTot * warpsize * iwarp + warpsize * cnt_qp +
                              ilane] *
                           s_basis0[p * nq0 + i];
                }
                wsp[nm1 * warpsize * iwarp + warpsize * q + ilane] = tmp;
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += wsp[nm1 * warpsize * iwarp + warpsize * q + ilane] *
                           s_basis1[q * nq1 + j];
                }
                out[nqTot * warpsize * iwarp + warpsize * (nq0 * j + i) +
                    ilane] = tmp;
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransQuadKernel_QP(const unsigned int nm0, const unsigned int nm1,
                           const unsigned int nmTot, const unsigned int nq0,
                           const unsigned int nq1, const unsigned int nelmt,
                           const TData *__restrict basis0,
                           const TData *__restrict basis1,
                           const TData *__restrict in, TData *__restrict out,
                           TData *__restrict shared,
                           const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm1 * nq0 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int i = item_ct1.get_local_id(1); i < nq0;
             i += item_ct1.get_local_range(1))
        {
            for (unsigned int q = item_ct1.get_local_id(2); q < nm1;
                 q += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_iq = nm1 * i + q;
                unsigned int cnt_qp       = nm0 * q;

                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += s_wsp0[cnt_qp] * s_basis0[p * nq0 + i];
                }
                s_wsp1[cnt_iq] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
             j += item_ct1.get_local_range(1))
        {
            for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                 i += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_ji = nq0 * j + i;
                unsigned int cnt_iq       = nm1 * i;

                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
                {
                    tmp += s_wsp1[cnt_iq] * s_basis1[q * nq1 + j];
                }
                outptr[cnt_ji] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransQuadKernel_QP_1D(const unsigned int nm0, const unsigned int nm1,
                              const unsigned int nmTot, const unsigned int nq0,
                              const unsigned int nq1, const unsigned int nelmt,
                              const TData *__restrict basis0,
                              const TData *__restrict basis1,
                              const TData *__restrict in, TData *__restrict out,
                              TData *__restrict shared,
                              const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm1 * nq0 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nm1;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int q = idx % nm1;
            const unsigned int i = idx / nm1;
            unsigned int cnt_qp  = nm0 * q;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
            {
                tmp += s_wsp0[cnt_qp] * s_basis0[p * nq0 + i];
            }
            s_wsp1[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            unsigned int cnt_iq  = nm1 * i;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
            {
                tmp += s_wsp1[cnt_iq] * s_basis1[q * nq1 + j];
            }
            outptr[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransTriKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nmTot, const unsigned int nq0,
                       const unsigned int nq1, const unsigned int nelmt,
                       const bool correct, const TData *__restrict basis0,
                       const TData *__restrict basis1, TData *__restrict wsp,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;

        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            // direction 1
            for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
            {
                TData tmp = 0.0;
                for (unsigned int q = 0u; q < (nm1 - p); ++q, ++mode_pq)
                {
                    tmp += in[nmTot * warpsize * iwarp + warpsize * mode_pq +
                              ilane] *
                           s_basis1[mode_pq * nq1 + j];
                }
                wsp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
            }

            // direction 0
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p)
                {
                    tmp += wsp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                           s_basis0[p * nq0 + i];
                }

                if (correct)
                {
                    tmp += in[nmTot * warpsize * iwarp + warpsize + ilane] *
                           s_basis0[nq0 + i] * s_basis1[nq1 + j];
                }

                out[nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane] = tmp;
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransTriKernel_QP(const unsigned int nm0, const unsigned int nm1,
                          const unsigned int nmTot, const unsigned int nq0,
                          const unsigned int nq1, const unsigned int nelmt,
                          const bool correct, const TData *__restrict basis0,
                          const TData *__restrict basis1,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict shared,
                          const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nmTot * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
             j += item_ct1.get_local_range(1))
        {
            for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                 p += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_jp = nm0 * j + p;
                unsigned int mode_pq      = (2u * nm1 - p + 1u) * p / 2u;

                TData tmp = 0.0;
                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    tmp += s_basis1[mode_pq * nq1 + j] * s_wsp0[mode_pq];
                }
                s_wsp1[cnt_jp] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
             j += item_ct1.get_local_range(1))
        {
            for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                 i += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_ij = nq0 * j + i;
                unsigned int cnt_jp       = nm0 * j;

                TData tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
                {
                    tmp += s_wsp1[cnt_jp] * s_basis0[p * nq0 + i];
                }

                if (correct)
                {
                    tmp += s_wsp0[1] * s_basis0[nq0 + i] * s_basis1[nq1 + j];
                }

                outptr[cnt_ij] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransTriKernel_QP_1D(const unsigned int nm0, const unsigned int nm1,
                             const unsigned int nmTot, const unsigned int nq0,
                             const unsigned int nq1, const unsigned int nelmt,
                             const bool correct, const TData *__restrict basis0,
                             const TData *__restrict basis1,
                             const TData *__restrict in, TData *__restrict out,
                             TData *__restrict shared,
                             const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p = idx % nm0;
            const unsigned int j = idx / nm0;
            unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                tmp += s_basis1[mode_pq * nq1 + j] * s_wsp0[mode_pq];
            }
            s_wsp1[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = idx / nq0;
            unsigned int cnt_jp  = nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
            {
                tmp += s_wsp1[cnt_jp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                tmp += s_wsp0[1] * s_basis0[nq0 + i] * s_basis1[nq1 + j];
            }

            outptr[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransHexKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nm2, const unsigned int nmTot,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nelmt,
                       const TData *__restrict basis0,
                       const TData *__restrict basis1,
                       const TData *__restrict basis2, TData *__restrict wsp,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *wsp1        = wsp;
        TData *wsp2        = wsp1 + (nm1 * nm2) * nelmt;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            // direction 0
            for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u; r < nm2; ++r)
            {
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * cnt_rqp + ilane] *
                               s_basis0[p * nq0 + i];
                    }
                    wsp1[nm1 * nm2 * warpsize * iwarp + warpsize * cnt_rq +
                         ilane] = tmp;
                }
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                    {
                        tmp += wsp1[nm1 * nm2 * warpsize * iwarp +
                                    warpsize * cnt_rq + ilane] *
                               s_basis1[q * nq1 + j];
                    }
                    wsp2[nm2 * warpsize * iwarp + warpsize * r + ilane] = tmp;
                }

                // direction 2
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2; ++r)
                    {
                        tmp += wsp2[nm2 * warpsize * iwarp + warpsize * r +
                                    ilane] *
                               s_basis2[r * nq2 + k];
                    }
                    out[nqTot * warpsize * iwarp +
                        warpsize * (k * nq1 * nq0 + j * nq0 + i) + ilane] = tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransHexKernel_QP(const unsigned int nm0, const unsigned int nm1,
                          const unsigned int nm2, const unsigned int nmTot,
                          const unsigned int nq0, const unsigned int nq1,
                          const unsigned int nq2, const unsigned int nelmt,
                          const TData *__restrict basis0,
                          const TData *__restrict basis1,
                          const TData *__restrict basis2,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict shared,
                          const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nq0 * nm1 * nm2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq1 * nq0 * nm2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

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
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nm2 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int i = item_ct1.get_local_id(0); i < nq0;
             i += item_ct1.get_local_range(0))
        {
            for (unsigned int r = item_ct1.get_local_id(1); r < nm2;
                 r += item_ct1.get_local_range(1))
            {
                for (unsigned int q = item_ct1.get_local_id(2); q < nm1;
                     q += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r + q;
                    unsigned int cnt_rqp       = nm1 * nm0 * r + nm0 * q;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += s_wsp0[cnt_rqp] * s_basis0[p * nq0 + i];
                    }
                    s_wsp1[cnt_irq] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int j = item_ct1.get_local_id(0); j < nq1;
             j += item_ct1.get_local_range(0))
        {
            for (unsigned int i = item_ct1.get_local_id(1); i < nq0;
                 i += item_ct1.get_local_range(1))
            {
                for (unsigned int r = item_ct1.get_local_id(2); r < nm2;
                     r += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i + r;
                    unsigned int cnt_irq       = nm1 * nm2 * i + nm1 * r;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
                    {
                        tmp += s_wsp1[cnt_irq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[cnt_jir] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
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
                    unsigned int cnt_jir       = nq0 * nm2 * j + nm2 * i;

                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
                    {
                        tmp += s_wsp2[cnt_jir] * s_basis2[r * nq2 + k];
                    }
                    outptr[cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransHexKernel_QP_1D(const unsigned int nm0, const unsigned int nm1,
                             const unsigned int nm2, const unsigned int nmTot,
                             const unsigned int nq0, const unsigned int nq1,
                             const unsigned int nq2, const unsigned int nelmt,
                             const TData *__restrict basis0,
                             const TData *__restrict basis1,
                             const TData *__restrict basis2,
                             const TData *__restrict in, TData *__restrict out,
                             TData *__restrict shared,
                             const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nq0 * nm1 * nm2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq1 * nq0 * nm2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nm1 * nm2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int q = idx % nm1;
            const unsigned int r = (idx / nm1) % nm2;
            const unsigned int i = idx / (nm1 * nm2);
            unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
            {
                tmp += s_wsp0[cnt_rqp] * s_basis0[p * nq0 + i];
            }
            s_wsp1[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq1 * nm2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int r = idx % nm2;
            const unsigned int i = (idx / nm2) % nq0;
            const unsigned int j = idx / (nm2 * nq0);
            unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
            {
                tmp += s_wsp1[cnt_irq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i = idx % nq0;
            const unsigned int j = (idx / nq0) % nq1;
            const unsigned int k = idx / (nq1 * nq0);
            unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
            {
                tmp += s_wsp2[cnt_jir] * s_basis2[r * nq2 + k];
            }
            outptr[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransTetKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nm2, const unsigned int nmTot,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nelmt,
                       const bool correct, const TData *__restrict basis0,
                       const TData *__restrict basis1,
                       const TData *__restrict basis2, TData *__restrict wsp,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_basis0 = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm01 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *fpq         = wsp;
        TData *fp          = fpq + nm01 * nelmt;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            // direction 2
            for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
                 p < nm0; ++p)
            {
                for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p - q;
                         ++r, ++mode2, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[nq2 * mode2 + k];
                    }
                    fpq[nm01 * warpsize * iwarp + warpsize * mode_pq + ilane] =
                        tmp;
                }

                // increment mode in case order1!=order2
                for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
                {
                    mode2 += nm2 - p - q;
                }
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
                    {
                        tmp += fpq[nm01 * warpsize * iwarp +
                                   warpsize * mode_pq + ilane] *
                               s_basis1[mode_pq * nq1 + j];
                    }
                    fp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
                }

                // direction 0
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp +=
                            fp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                            s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 *
                               in[nmTot * warpsize * iwarp + warpsize + ilane];

                        // bottom vertex
                        tmp1 = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[k];
                        tmp += tmp1 * in[nmTot * warpsize * iwarp +
                                         warpsize * nm2 + ilane];

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp1 = s_basis1[nq1 + j] * s_basis0[nq0 + i];
                            tmp1 *= s_basis2[(r + 1u) * nq2 + k];
                            tmp += tmp1 * in[nmTot * warpsize * iwarp +
                                             warpsize * (nm2 + r) + ilane];
                        }
                    }

                    out[nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane] =
                        tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransTetKernel_QP(const unsigned int nm0, const unsigned int nm1,
                          const unsigned int nm2, const unsigned int nmTot,
                          const unsigned int nq0, const unsigned int nq1,
                          const unsigned int nq2, const unsigned int nelmt,
                          const bool correct, const TData *__restrict basis0,
                          const TData *__restrict basis1,
                          const TData *__restrict basis2,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict shared,
                          const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_wsp0   = shared;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_wsp2   = s_wsp1 + nm01 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;

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
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm01 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int p = item_ct1.get_local_id(1); p < nm0;
                 p += item_ct1.get_local_range(1))
            {
                for (unsigned int q = item_ct1.get_local_id(2); q < nm1 - p;
                     q += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_kpq =
                        nm01 * k + (2u * nm1 - p + 1u) * p / 2u + q;
                    unsigned int mode2 = (2u * (nm2 - p) - q + 1u) * q;
                    mode2 += nm2 * (nm2 + 1u) * p;
                    mode2 -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
                    mode2 += (p - 1u) * p * (2u * p - 1u) / 6u;
                    mode2 /= 2u;
                    unsigned int mode_pqr =
                        mode2 - ((nm2 > nm1)
                                     ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u
                                     : 0u);

                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p - q;
                         ++r, ++mode2, ++mode_pqr)
                    {
                        tmp += s_wsp0[mode_pqr] * s_basis2[k + nq2 * mode2];
                    }
                    s_wsp1[cnt_kpq] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_pq        = (2u * nm1 - p + 1u) * p / 2u;
                    unsigned int cnt_kpq        = nm01 * k + mode_pq;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1 - p;
                         ++q, ++cnt_kpq, ++mode_pq)
                    {
                        tmp += s_wsp1[cnt_kpq] * s_basis1[mode_pq * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
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
                    unsigned int mode_kjp      = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 * s_wsp0[1];

                        // bottom vertex
                        tmp1 = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[k];
                        tmp += tmp1 * s_wsp0[nm2];

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp1 = s_basis1[nq1 + j] * s_basis0[nq0 + i];
                            tmp1 *= s_basis2[(r + 1u) * nq2 + k];
                            tmp += tmp1 * s_wsp0[nm2 + r];
                        }
                    }

                    outptr[cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransTetKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const unsigned int *__restrict pindex,
    const unsigned int *__restrict qindex, const TData *__restrict basis0,
    const TData *__restrict basis1, const TData *__restrict basis2,
    const TData *__restrict in, TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm01  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_wsp0   = shared;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_wsp2   = s_wsp1 + nm01 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;

    // Temporary solution, to be removed - TODO
    unsigned int *vpindex = (unsigned int *)pindex;
    unsigned int *vqindex = (unsigned int *)qindex;
    for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
         p += item_ct1.get_local_range(2))
    {
        for (unsigned int q = 0; q < nm1 - p; q++)
        {
            unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
            vpindex[mode_pq]     = p;
            vqindex[mode_pq]     = q;
        }
    }

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm01 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nm1 * nm2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int k = idx / nm01;
            const unsigned int p = pindex[idx % nm01];
            const unsigned int q = qindex[idx % nm01];
            unsigned int mode2   = (2u * (nm2 - p) - q + 1u) * q;
            mode2 += nm2 * (nm2 + 1u) * p;
            mode2 -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
            mode2 += (p - 1u) * p * (2u * p - 1u) / 6u;
            mode2 /= 2u;
            unsigned int mode_pqr =
                mode2 -
                ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
            {
                tmp += s_wsp0[mode_pqr] * s_basis2[k + nq2 * mode2];
            }
            s_wsp1[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p = idx % nm0;
            const unsigned int j = (idx / nm0) % nq1;
            const unsigned int k = idx / (nm0 * nq1);
            unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;
            unsigned int cnt_kpq = nm01 * k + mode_pq;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1 - p; ++q, ++cnt_kpq, ++mode_pq)
            {
                tmp += s_wsp1[cnt_kpq] * s_basis1[mode_pq * nq1 + j];
            }
            s_wsp2[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i  = idx % nq0;
            const unsigned int j  = (idx / nq0) % nq1;
            const unsigned int k  = idx / (nq0 * nq1);
            unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
            {
                tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                // top vertex
                TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp1 *= s_basis2[nq2 + k];
                tmp += tmp1 * s_wsp0[1];

                // bottom vertex
                tmp1 = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp1 *= s_basis2[k];
                tmp += tmp1 * s_wsp0[nm2];

                // singular edge
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    tmp1 = s_basis1[nq1 + j] * s_basis0[nq0 + i];
                    tmp1 *= s_basis2[(r + 1u) * nq2 + k];
                    tmp += tmp1 * s_wsp0[nm2 + r];
                }
            }

            outptr[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransPrismKernel(const unsigned int nm0, const unsigned int nm1,
                         const unsigned int nm2, const unsigned int nmTot,
                         const unsigned int nq0, const unsigned int nq1,
                         const unsigned int nq2, const unsigned int nelmt,
                         const bool correct, const TData *__restrict basis0,
                         const TData *__restrict basis1,
                         const TData *__restrict basis2, TData *__restrict wsp,
                         const TData *__restrict in, TData *__restrict out,
                         TData *__restrict shared,
                         const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm12 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *fpq         = wsp;
        TData *fp          = fpq + nm0 * nm1 * nelmt;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            // direction 2
            for (unsigned int p = 0u, mode_pr = 0u, mode_pq = 0u, mode_pqr = 0u;
                 p < nm0; ++p)
            {
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[(mode_pr + r) * nq2 + k];
                    }
                    fpq[nm0 * nm1 * warpsize * iwarp + warpsize * mode_pq +
                        ilane] = tmp;
                }
                mode_pr += nm2 - p;
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                    {
                        tmp += fpq[nm0 * nm1 * warpsize * iwarp +
                                   warpsize * mode_pq + ilane] *
                               s_basis1[q * nq1 + j];
                    }
                    fp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
                }

                // direction 0
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp +=
                            fp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                            s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            tmp += s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                                   s_basis0[nq0 + i] *
                                   in[nmTot * warpsize * iwarp +
                                      warpsize * (nm2 * q + 1u) + ilane];
                        }
                    }

                    out[nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane] =
                        tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransPrismKernel_QP(const unsigned int nm0, const unsigned int nm1,
                            const unsigned int nm2, const unsigned int nmTot,
                            const unsigned int nq0, const unsigned int nq1,
                            const unsigned int nq2, const unsigned int nelmt,
                            const bool correct, const TData *__restrict basis0,
                            const TData *__restrict basis1,
                            const TData *__restrict basis2,
                            const TData *__restrict in, TData *__restrict out,
                            TData *__restrict shared,
                            const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nm12 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int p = item_ct1.get_local_id(1); p < nm0;
                 p += item_ct1.get_local_range(1))
            {
                for (unsigned int q = item_ct1.get_local_id(2); q < nm1;
                     q += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p + q;
                    unsigned int mode_pr        = (2u * nm2 - p + 1u) * p / 2u;
                    unsigned int mode_pqr       = mode_pr * nm1 + (nm2 - p) * q;

                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - p;
                         ++r, ++mode_pqr, ++mode_pr)
                    {
                        tmp += s_wsp0[mode_pqr] * s_basis2[mode_pr * nq2 + k];
                    }
                    s_wsp1[mode_kpq] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_kpq       = nm0 * nm1 * k + nm1 * p;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
                    {
                        tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
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
                    unsigned int mode_kjp      = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            tmp += s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                                   s_basis0[nq0 + i] * s_wsp0[q * nm2 + 1u];
                        }
                    }

                    outptr[cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransPrismKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict basis0, const TData *__restrict basis1,
    const TData *__restrict basis2, const TData *__restrict in,
    TData *__restrict out, TData *__restrict shared,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nmTot;
    TData *s_wsp2            = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm12 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int q  = idx % nm1;
            const unsigned int p  = (idx / nm1) % nm0;
            const unsigned int k  = idx / (nm1 * nm0);
            unsigned int mode_pr  = (2u * nm2 - p + 1u) * p / 2u;
            unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;

            TData tmp = 0.0;
            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr, ++mode_pr)
            {
                tmp += s_wsp0[mode_pqr] * s_basis2[mode_pr * nq2 + k];
            }
            s_wsp1[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p  = idx % nm0;
            const unsigned int j  = (idx / nm0) % nq1;
            const unsigned int k  = idx / (nm0 * nq1);
            unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
            {
                tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i  = idx % nq0;
            const unsigned int j  = (idx / nq0) % nq1;
            const unsigned int k  = idx / (nq0 * nq1);
            unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
            {
                tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                           s_basis0[nq0 + i] * s_wsp0[q * nm2 + 1u];
                }
            }

            outptr[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransPyrKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nm2, const unsigned int nmTot,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nelmt,
                       const bool correct, const TData *__restrict basis0,
                       const TData *__restrict basis1,
                       const TData *__restrict basis2, TData *__restrict wsp,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict shared,
                       const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_basis0 = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        TData *fpq         = wsp;
        TData *fp          = fpq + nm0 * nm1 * nelmt;

        for (int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            // direction 2
            for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u, mode_pq = 0u;
                 p < nm0; ++p)
            {
                for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2 - std::max(p, q);
                         ++r, ++mode2, ++mode_pqr)
                    {
                        tmp += in[nmTot * warpsize * iwarp +
                                  warpsize * mode_pqr + ilane] *
                               s_basis2[mode2 * nq2 + k];
                    }
                    fpq[nm0 * nm1 * warpsize * iwarp + warpsize * mode_pq +
                        ilane] = tmp;
                }

                // increment mode in case nm2>nm1
                for (unsigned int q = nm1; q < nm2; ++q)
                {
                    mode2 += nm2 - q;
                }
            }

            // direction 1
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_pq)
                    {
                        tmp += fpq[nm0 * nm1 * warpsize * iwarp +
                                   warpsize * mode_pq + ilane] *
                               s_basis1[q * nq1 + j];
                    }
                    fp[nm0 * warpsize * iwarp + warpsize * p + ilane] = tmp;
                }

                // direction 0
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p)
                    {
                        tmp +=
                            fp[nm0 * warpsize * iwarp + warpsize * p + ilane] *
                            s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 *
                               in[nmTot * warpsize * iwarp + warpsize + ilane];
                    }

                    out[nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane] =
                        tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransPyrKernel_QP(const unsigned int nm0, const unsigned int nm1,
                          const unsigned int nm2, const unsigned int nmTot,
                          const unsigned int nq0, const unsigned int nq1,
                          const unsigned int nq2, const unsigned int nelmt,
                          const bool correct, const TData *__restrict basis0,
                          const TData *__restrict basis1,
                          const TData *__restrict basis2,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict shared,
                          const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_wsp0   = shared;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_wsp2   = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

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
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1) *
                item_ct1.get_local_id(0) +
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2) *
                                    item_ct1.get_local_range(1) *
                                    item_ct1.get_local_range(0);
        for (unsigned int idx = idx0; idx < nmTot; idx += stride)
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int p = item_ct1.get_local_id(1); p < nm0;
                 p += item_ct1.get_local_range(1))
            {
                for (unsigned int q = item_ct1.get_local_id(2); q < nm1;
                     q += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p + q;
                    unsigned int mode2 =
                        (nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u
                                    : 0u;
                    unsigned int mode_pqr = nm1 * (2u * nm2 + 1u - nm1) * p;
                    mode_pqr -= (p - 1u) * p / 2u;
                    mode_pqr -= (p - 1u) * p * (2u * p - 1u) / 6u;
                    mode_pqr /= 2u;

                    if (q < p)
                    {
                        mode_pqr += q * (nm2 - p);
                        mode2 += mode_pqr;
                        TData tmp = 0.0;
                        for (unsigned int r = 0u; r < nm2 - p;
                             ++r, ++mode2, ++mode_pqr)
                        {
                            tmp += s_wsp0[mode_pqr] * s_basis2[mode2 * nq2 + k];
                        }
                        s_wsp1[mode_kpq] = tmp;
                    }
                    else
                    {
                        mode_pqr += p * (nm2 - p);
                        mode_pqr +=
                            ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;
                        mode2 += mode_pqr;

                        TData tmp = 0.0;
                        for (unsigned int r = 0u; r < nm2 - q;
                             ++r, ++mode2, ++mode_pqr)
                        {
                            tmp += s_wsp0[mode_pqr] * s_basis2[mode2 * nq2 + k];
                        }
                        s_wsp1[mode_kpq] = tmp;
                    }
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_kpq       = nm0 * nm1 * k + nm1 * p;

                    TData tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
                    {
                        tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
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
                    unsigned int mode_kjp      = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                        tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                        tmp1 *= s_basis2[nq2 + k];
                        tmp += tmp1 * s_wsp0[1];
                    }

                    outptr[cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SHMEM, typename TData>
void BwdTransPyrKernel_QP_1D(const unsigned int nm0, const unsigned int nm1,
                             const unsigned int nm2, const unsigned int nmTot,
                             const unsigned int nq0, const unsigned int nq1,
                             const unsigned int nq2, const unsigned int nelmt,
                             const bool correct, const TData *__restrict basis0,
                             const TData *__restrict basis1,
                             const TData *__restrict basis2,
                             const TData *__restrict in, TData *__restrict out,
                             TData *__restrict shared,
                             const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_wsp0   = shared;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_wsp2   = s_wsp1 + (nm0 * nm1 * nq2);
    TData *s_basis0 = SHMEM ? s_wsp2 + nq2 * nq1 * nm0 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            s_wsp0[idx] = inptr[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 2
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int q = idx % nm1;
            const unsigned int p = (idx / nm1) % nm0;
            const unsigned int k = idx / (nm1 * nm0);
            unsigned int mode2 =
                (nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u;
            unsigned int mode_pqr = nm1 * (2u * nm2 + 1u - nm1) * p;
            mode_pqr -= (p - 1u) * p / 2u;
            mode_pqr -= (p - 1u) * p * (2u * p - 1u) / 6u;
            mode_pqr /= 2u;

            if (q < p)
            {
                mode_pqr += q * (nm2 - p);
                mode2 += mode_pqr;
                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
                {
                    tmp += s_wsp0[mode_pqr] * s_basis2[mode2 * nq2 + k];
                }
                s_wsp1[idx] = tmp;
            }
            else
            {
                mode_pqr += p * (nm2 - p);
                mode_pqr += ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;
                mode2 += mode_pqr;

                TData tmp = 0.0;
                for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
                {
                    tmp += s_wsp0[mode_pqr] * s_basis2[mode2 * nq2 + k];
                }
                s_wsp1[idx] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 1
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p  = idx % nm0;
            const unsigned int j  = (idx / nm0) % nq1;
            const unsigned int k  = idx / (nm0 * nq1);
            unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

            TData tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
            {
                tmp += s_wsp1[mode_kpq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // direction 0
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int i  = idx % nq0;
            const unsigned int j  = (idx / nq0) % nq1;
            const unsigned int k  = idx / (nq0 * nq1);
            unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

            TData tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
            {
                tmp += s_wsp2[mode_kjp] * s_basis0[p * nq0 + i];
            }

            if (correct)
            {
                // top vertex
                TData tmp1 = s_basis0[i] * s_basis1[nq1 + j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[j];
                tmp1 += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp1 *= s_basis2[nq2 + k];
                tmp += tmp1 * s_wsp0[1];
            }

            outptr[idx] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

// Kernel launchers
template <typename ExecSpace, typename Implementation, bool SHMEM,
          typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    BwdTrans1DKernel(const unsigned int nm0, const unsigned int nq0,
                     const unsigned int nelmt, const TData *basis0,
                     const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same<Implementation, Operators::SumFacQP>::value;

    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0, NektarSpaces::SYCL::defaultBlockSize)
                   : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize = std::min(
        MULTILEVEL ? nelmt : (nelmt + blocksize - 1u) / blocksize, 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (MULTILEVEL)
    {
        unsigned int nshared = nm0 + (SHMEM ? nm0 * nq0 : 0u);

        Q.submit([=](sycl::handler &cgh) {
             // Create local shared memory
             sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                   cgh);
             // Set WorkGroup dimensions
             const sycl::range<3> wgSize(
                 1, 1, NektarSpaces::vector_width<TData>::value);
             const sycl::range<3> globalSize =
                 sycl::range<3>(1, 1, gridsize) * wgSize;

             cgh.parallel_for(
                 sycl::nd_range<3>(globalSize, wgSize),
                 [=](sycl::nd_item<3> item) {
                     TData *shmPtr = shared
                                         .template get_multi_ptr<
                                             sycl::access::decorated::no>()
                                         .get();
                     BwdTransSegKernel_QP<SHMEM>(nm0, nq0, nelmt, basis0, in,
                                                 out, shmPtr, item);
                 });
         }).wait();
    }
    else
    {
        unsigned int nshared = nm0 * nq0;

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
                     BwdTransSegKernel<SHMEM>(nm0, nq0, nelmt, basis0, in, out,
                                              shmPtr, item);
                 });
         }).wait();
    }
}

template <typename ExecSpace, typename Implementation, bool SHMEM,
          typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    BwdTrans2DKernel(LibUtilities::ShapeType shapetype, const unsigned int nm0,
                     const unsigned int nm1, const unsigned int nq0,
                     const unsigned int nq1, const unsigned int nelmt,
                     const bool correct, const TData *basis0,
                     const TData *basis1, TData *wsp, const TData *in,
                     TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same<Implementation, Operators::SumFacQP>::value;

    const sycl::range<3> blocksize2d(1, std::min(nq1, 16u), std::min(nq0, 16u));
    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0 * nq1, NektarSpaces::SYCL::defaultBlockSize)
                   : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize = std::min(
        MULTILEVEL ? nelmt : (nelmt + blocksize - 1u) / blocksize, 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    if (shapetype == LibUtilities::Quad)
    {
        const unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = SHMEM ? (nq0 * nm0 + nq1 * nm1) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + nq0 * nm1;
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize2d,
                                       blocksize2d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         BwdTransQuadKernel_QP<SHMEM>(nm0, nm1, nmTot, nq0, nq1,
                                                      nelmt, basis0, basis1, in,
                                                      out, shmPtr, item);
                     });
             }).wait();
            // BwdTransQuadKernel_QP_1D<SHMEM>
            //    <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nmTot, nq0, nq1, nelmt, basis0, basis1, in,
            //         out);
        }
        else
        {
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
                         BwdTransQuadKernel<SHMEM>(nm0, nm1, nmTot, nq0, nq1,
                                                   nelmt, basis0, basis1, wsp,
                                                   in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = SHMEM ? (nm0 * nq0 + nmTot * nq1) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + nm0 * nq1;
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize2d,
                                       blocksize2d),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         BwdTransTriKernel_QP<SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmt, correct, basis0,
                             basis1, in, out, shmPtr, item);
                     });
             }).wait();
            // BwdTransTriKernel_QP_1D<SHMEM>
            //     <<<nelmt, blocksize, nshared>>>(nm0, nm1, nmTot, nq0, nq1,
            //                                        nelmt, correct, basis0,
            //                                        basis1, in, out);
        }
        else
        {
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
                         BwdTransTriKernel<SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmt, correct, basis0,
                             basis1, wsp, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
}

template <typename ExecSpace, typename Implementation, bool SHMEM,
          typename TData>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    BwdTrans3DKernel(LibUtilities::ShapeType shapetype, const unsigned int nm0,
                     const unsigned int nm1, const unsigned int nm2,
                     const unsigned int nq0, const unsigned int nq1,
                     const unsigned int nq2, const unsigned int nelmt,
                     const bool correct, const TData *basis0,
                     const TData *basis1, const TData *basis2,
                     [[maybe_unused]] TData *wsp, const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same<Implementation, Operators::SumFacQP>::value;

    const sycl::range<3> blocksize3d(std::min(nq0, 8u), std::min(nq1, 8u),
                                     std::min(nq2, 8u));
    const unsigned int blocksize =
        MULTILEVEL
            ? std::min(nq0 * nq1 * nq2, NektarSpaces::SYCL::defaultBlockSize)
            : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize = std::min(
        MULTILEVEL ? nelmt : (nelmt + blocksize - 1u) / blocksize, 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    if (shapetype == LibUtilities::Hex)
    {
        const unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared = SHMEM ? (nq0 * nm0 + nq1 * nm1 + nq2 * nm2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += (nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nmTot +
                        (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2));

            Q.submit([&](sycl::handler &cgh) {
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
                         BwdTransHexKernel_QP<SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, basis0,
                             basis1, basis2, in, out, shmPtr, item);
                     });
             }).wait();
            // BwdTransHexKernel_QP_1D<SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(nm0, nm1, nm2, nmTot, nq0,
            //                                          nq1, nq2, nelmt,
            //                                          basis0, basis1, basis2,
            //                                          in, out);
        }
        else
        {
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
                         BwdTransHexKernel<SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, basis0,
                             basis1, basis2, wsp, in, out, shmPtr, item);
                     });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        const unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        unsigned int nshared =
            SHMEM ? (nm0 * nq0 + nm01 * nq1 + nmode2 * nq2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += (nmTot + ((2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2) +
                        (nm0 * nq1 * nq2));

            Q.submit([&](sycl::handler &cgh) {
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
                         BwdTransTetKernel_QP<SHMEM>(nm0, nm1, nm2, nmTot, nq0,
                                                     nq1, nq2, nelmt, correct,
                                                     basis0, basis1, basis2, in,
                                                     out, shmPtr, item);
                     });
             }).wait();
            // unsigned int *pindex;
            // unsigned int *qindex;
            // cudaMalloc((void **)&pindex, sizeof(unsigned int)*nm01);
            // cudaMalloc((void **)&qindex, sizeof(unsigned int)*nm01);
            // BwdTransTetKernel_QP_1D<SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct,
            //         pindex, qindex, basis0, basis1, basis2, in, out);
        }
        else
        {
            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 // Set WorkGroup dimensions
                 sycl::range<3> wgSize(1, 1, blocksize);
                 sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * wgSize;

                 cgh.parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                                  [=](sycl::nd_item<3> item) {
                                      TData *shmPtr =
                                          shared
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get();
                                      BwdTransTetKernel<SHMEM>(
                                          nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
                                          nelmt, correct, basis0, basis1,
                                          basis2, wsp, in, out, shmPtr, item);
                                  });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        const unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
        unsigned int nshared =
            SHMEM ? (nm0 * nq0 + nm1 * nq1 + nm12 * nq2) : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
            Q.submit([&](sycl::handler &cgh) {
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
                         BwdTransPrismKernel_QP<SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                             correct, basis0, basis1, basis2, in, out, shmPtr,
                             item);
                     });
             }).wait();
            // BwdTransPrismKernel_QP_1D<SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct,
            //         basis0, basis1, basis2, in, out);
        }
        else
        {
            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 // Set WorkGroup dimensions
                 sycl::range<3> wgSize(1, 1, blocksize);
                 sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * wgSize;

                 cgh.parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                                  [=](sycl::nd_item<3> item) {
                                      TData *shmPtr =
                                          shared
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get();
                                      BwdTransPrismKernel<SHMEM>(
                                          nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
                                          nelmt, correct, basis0, basis1,
                                          basis2, wsp, in, out, shmPtr, item);
                                  });
             }).wait();
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        unsigned int nshared =
            SHMEM ? nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 : 0u;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
            Q.submit([&](sycl::handler &cgh) {
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
                         BwdTransPyrKernel_QP<SHMEM>(nm0, nm1, nm2, nmTot, nq0,
                                                     nq1, nq2, nelmt, correct,
                                                     basis0, basis1, basis2, in,
                                                     out, shmPtr, item);
                     });
             }).wait();
            // BwdTransPyrKernel_QP_1D<SHMEM>
            //     <<<gridsize, blocksize, nshared>>>(
            //         nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, correct,
            //         basis0, basis1, basis2, in, out);
        }
        else
        {
            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 // Set WorkGroup dimensions
                 sycl::range<3> wgSize(1, 1, blocksize);
                 sycl::range<3> globalSize =
                     sycl::range<3>(1, 1, gridsize) * wgSize;

                 cgh.parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                                  [=](sycl::nd_item<3> item) {
                                      TData *shmPtr =
                                          shared
                                              .template get_multi_ptr<
                                                  sycl::access::decorated::no>()
                                              .get();
                                      BwdTransPyrKernel<SHMEM>(
                                          nm0, nm1, nm2, nmTot, nq0, nq1, nq2,
                                          nelmt, correct, basis0, basis1,
                                          basis2, wsp, in, out, shmPtr, item);
                                  });
             }).wait();
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
