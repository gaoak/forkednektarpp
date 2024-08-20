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

#include "Operators/Common/Spaces.hpp"
#include "Operators/SYCLQueue.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
void BwdTransSegKernel(const unsigned int nm0, const unsigned int nq0,
                       const unsigned int nelmt, const TData *__restrict basis0,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict s_basis0,
                       const sycl::nd_item<3> &item_ct1)
{

    // Copy to shared memory.
    unsigned int sIndex = item_ct1.get_local_id(2);
    while (sIndex < nm0 * nq0)
    {
        s_basis0[sIndex] = basis0[sIndex];
        sIndex += item_ct1.get_local_range(2);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nm0 * e;
        unsigned int outoffset = nq0 * e;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            TData tmp = 0.0;
            for (unsigned int p = 0; p < nm0; ++p)
            {
                tmp += in[inoffset + p] * s_basis0[p * nq0 + i];
            }
            out[outoffset + i] = tmp;
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransSegKernel_QP(const unsigned int nm0, const unsigned int nq0,
                          const unsigned int nelmt,
                          const TData *__restrict basis0,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict s_wsp0,
                          const sycl::nd_item<3> &item_ct1)
{

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nm0 * e;
        unsigned int outoffset = nq0 * e;

        // Copy to shared memory.
        for (unsigned int p = 0; p < nm0; p++)
        {
            s_wsp0[p] = in[inoffset + p];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            TData tmp = 0.0;
            for (unsigned int p = 0; p < nm0; p++)
            {
                tmp += s_wsp0[p] * basis0[p * nq0 + i];
            }
            out[outoffset + i] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransQuadKernel(const unsigned int nm0, const unsigned int nm1,
                        const unsigned int nmTot, const unsigned int nq0,
                        const unsigned int nq1, const unsigned int nelmt,
                        const TData *__restrict basis0,
                        const TData *__restrict basis1,
                        const TData *__restrict in, TData *__restrict out,
                        TData *__restrict shared, TData *__restrict g_wsp,
                        const sycl::nd_item<3> &item_ct1)
{

    TData *s_basis0 = shared;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    // Copy to shared memory.
    unsigned int sIndex = item_ct1.get_local_id(2);
    while (sIndex < nm0 * nq0)
    {
        s_basis0[sIndex] = basis0[sIndex];
        sIndex += item_ct1.get_local_range(2);
    }

    sIndex = item_ct1.get_local_id(2);
    while (sIndex < nm1 * nq1)
    {
        s_basis1[sIndex] = basis1[sIndex];
        sIndex += item_ct1.get_local_range(2);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    // Assign correct area of global memory to thread
    TData *wsp = g_wsp + (e * nm1);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * e;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int q = 0, cnt_qp = 0; q < nm1; ++q)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += in[inoffset + cnt_qp] * s_basis0[p * nq0 + i];
                }
                wsp[q] = tmp;
            }

            for (unsigned int j = 0; j < nq1; ++j)
            {
                TData tmp = 0.0;
                for (unsigned int q = 0; q < nm1; ++q)
                {
                    tmp += wsp[q] * s_basis1[q * nq1 + j];
                }
                out[outoffset + nq0 * j + i] = tmp;
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransQuadKernel_QP(const unsigned int nm0, const unsigned int nm1,
                           const unsigned int nmTot, const unsigned int nq0,
                           const unsigned int nq1, const unsigned int nelmt,
                           const TData *__restrict basis0,
                           const TData *__restrict basis1,
                           const TData *__restrict in, TData *__restrict out,
                           TData *__restrict shared,
                           const sycl::nd_item<3> &item_ct1)
{
    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nmTot;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * e;

        // Copy to shared memory.
        for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
             q += item_ct1.get_local_range(1))
        {
            unsigned int cnt_qp = nm0 * q;

            for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                 p += item_ct1.get_local_range(2))
            {
                s_wsp0[cnt_qp + p] = in[inoffset + cnt_qp + p];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                unsigned int cnt_iq = nm1 * i + q;
                unsigned int cnt_qp = nm0 * q;

                TData tmp = 0.0;
                for (unsigned int p = 0; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += s_wsp0[cnt_qp] * basis0[p * nq0 + i];
                }
                s_wsp1[cnt_iq] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                unsigned int cnt_iq = nm1 * i;
                unsigned int cnt_ji = nq0 * j + i;

                TData tmp = 0.0;
                for (unsigned int q = 0; q < nm1; ++q, ++cnt_iq)
                {
                    tmp += s_wsp1[cnt_iq] * basis1[q * nq1 + j];
                }
                out[outoffset + cnt_ji] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransTriKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nmTot, const unsigned int nq0,
                       const unsigned int nq1, const unsigned int nelmt,
                       const bool correct, const TData *__restrict basis0,
                       const TData *__restrict basis1,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict g_wsp,
                       const sycl::nd_item<3> &item_ct1)
{

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    // Assign correct area of global memory to thread
    TData *wsp = g_wsp + (e * nm0);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * e;

        for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
        {
            for (unsigned int p = 0, mode_pq = 0; p < nm0; ++p)
            {
                TData tmp = 0.0;
                for (unsigned int q = 0; q < (nm1 - p); ++q, ++mode_pq)
                {
                    tmp += basis1[mode_pq * nq1 + j] * in[inoffset + mode_pq];
                }
                wsp[p] = tmp;
            }

            for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
            {
                TData tmp = 0.0;
                for (unsigned int p = 0; p < nm0; ++p)
                {
                    tmp += wsp[p] * basis0[p * nq0 + i];
                }

                if (correct)
                {
                    tmp += in[inoffset + 1] * basis0[nq0 + i] * basis1[nq1 + j];
                }

                out[outoffset + cnt_ji] = tmp;
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransTriKernel_QP(const unsigned int nm0, const unsigned int nm1,
                          const unsigned int nmTot, const unsigned int nq0,
                          const unsigned int nq1, const unsigned int nelmt,
                          const bool correct, const TData *__restrict basis0,
                          const TData *__restrict basis1,
                          const TData *__restrict in, TData *__restrict out,
                          TData *__restrict shared,
                          const sycl::nd_item<3> &item_ct1)
{

    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nmTot;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * e;

        // Copy to shared memory.
        for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
             p += item_ct1.get_local_range(2))
        {
            unsigned int mode_pq = (2 * nm1 - p + 1) * p / 2;

            for (unsigned int q = item_ct1.get_local_id(1); q < nm1 - p;
                 q += item_ct1.get_local_range(1))
            {
                s_wsp0[mode_pq + q] = in[inoffset + mode_pq + q];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
             p += item_ct1.get_local_range(2))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                unsigned int mode_pq = (2 * nm1 - p + 1) * p / 2;
                unsigned int cnt_jp  = nm0 * j + p;

                TData tmp = 0.0;
                for (unsigned int q = 0; q < (nm1 - p); ++q, ++mode_pq)
                {
                    tmp += basis1[mode_pq * nq1 + j] * s_wsp0[mode_pq];
                }
                s_wsp1[cnt_jp] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                unsigned int cnt_jp = nm0 * j;
                unsigned int cnt_ij = nq0 * j + i;

                TData tmp = 0.0;
                for (unsigned int p = 0; p < nm0; ++p, ++cnt_jp)
                {
                    tmp += s_wsp1[cnt_jp] * basis0[p * nq0 + i];
                }

                if (correct)
                {
                    tmp += s_wsp0[1] * basis0[nq0 + i] * basis1[nq1 + j];
                }

                out[outoffset + cnt_ij] = tmp;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransHexKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict basis0, const TData *__restrict basis1,
    const TData *__restrict basis2, const TData *__restrict in,
    TData *__restrict out, TData *__restrict shared, TData *__restrict g_wsp1,
    TData *__restrict g_wsp2, const sycl::nd_item<3> &item_ct1)
{

    TData *s_basis0 = shared;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_basis2 = s_basis1 + nm1 * nq1;

    // Copy to shared memory.
    unsigned int sIndex = item_ct1.get_local_id(2);
    while (sIndex < nm0 * nq0)
    {
        s_basis0[sIndex] = basis0[sIndex];
        sIndex += item_ct1.get_local_range(2);
    }

    sIndex = item_ct1.get_local_id(2);
    while (sIndex < nm1 * nq1)
    {
        s_basis1[sIndex] = basis1[sIndex];
        sIndex += item_ct1.get_local_range(2);
    }

    sIndex = item_ct1.get_local_id(2);
    while (sIndex < nm2 * nq2)
    {
        s_basis2[sIndex] = basis2[sIndex];
        sIndex += item_ct1.get_local_range(2);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    TData *wsp1 = g_wsp1 + (e * nm1 * nm2);
    TData *wsp2 = g_wsp2 + (e * nm2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int r = 0, cnt_rqp = 0, cnt_rq = 0; r < nm2; ++r)
            {
                for (unsigned int q = 0; q < nm1; ++q, ++cnt_rq)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += in[inoffset + cnt_rqp] * s_basis0[p * nq0 + i];
                    }
                    wsp1[cnt_rq] = tmp;
                }
            }

            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int r = 0, cnt_rq = 0; r < nm2; ++r)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1; ++q, ++cnt_rq)
                    {
                        tmp += wsp1[cnt_rq] * s_basis1[q * nq1 + j];
                    }
                    wsp2[r] = tmp;
                }

                for (unsigned int k = 0; k < nq2; ++k)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2; ++r)
                    {
                        tmp += wsp2[r] * s_basis2[r * nq2 + k];
                    }
                    out[outoffset + k * nq1 * nq0 + j * nq0 + i] = tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
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

    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nmTot;
    TData *s_wsp2 = s_wsp1 + (nq0 * nm1 * nm2);

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        // Copy to shared memory.
        for (unsigned int r = item_ct1.get_local_id(0); r < nm2;
             r += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    s_wsp0[cnt_rqp + p] = in[inoffset + cnt_rqp + p];
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            for (unsigned int r = item_ct1.get_local_id(0); r < nm2;
                 r += item_ct1.get_local_range(0))
            {
                for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                     q += item_ct1.get_local_range(1))
                {
                    unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;
                    unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r + q;

                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += s_wsp0[cnt_rqp] * basis0[p * nq0 + i];
                    }
                    s_wsp1[cnt_irq] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
             j += item_ct1.get_local_range(1))
        {
            for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                 i += item_ct1.get_local_range(2))
            {
                for (unsigned int r = item_ct1.get_local_id(0); r < nm2;
                     r += item_ct1.get_local_range(0))
                {
                    unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;
                    unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i + r;

                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1; ++q, ++cnt_irq)
                    {
                        tmp += s_wsp1[cnt_irq] * basis1[q * nq1 + j];
                    }
                    s_wsp2[cnt_jir] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                     i += item_ct1.get_local_range(2))
                {
                    unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;

                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2; ++r, ++cnt_jir)
                    {
                        tmp += s_wsp2[cnt_jir] * basis2[r * nq2 + k];
                    }
                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData> // not working for nm2 > nm1
void BwdTransTetKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nm2, const unsigned int nmTot,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nelmt,
                       const bool correct, const TData *__restrict basis0,
                       const TData *__restrict basis1,
                       const TData *__restrict basis2,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict g_fpq, TData *__restrict g_fp,
                       sycl::nd_item<3> &item_ct1)
{
    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    // This is the largest cnt_pq will get.
    // Allows for smaller memory footprint than nm0 * nm1
    unsigned int nm01 = (2 * nm1 - nm0 + 1) * nm0 / 2;

    TData *fpq = g_fpq + (e * nm01);
    TData *fp  = g_fp + (e * nm0);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
        {
            for (unsigned int p = 0, cnt_pq = 0, mode_pqr = 0; p < nm0; ++p)
            {
                for (unsigned int q = 0; q < nm1 - p; ++q, ++cnt_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2 - p - q; ++r, ++mode_pqr)
                    {
                        tmp += basis2[k + nq2 * mode_pqr] *
                               in[inoffset + mode_pqr];
                    }
                    fpq[cnt_pq] = tmp;
                }

                // increment mode in case order1!=order2
                for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
                {
                    mode_pqr += nm2 - p - q;
                }
            }

            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int p = 0, mode_pq = 0; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1 - p; ++q, ++mode_pq)
                    {
                        tmp += fpq[mode_pq] * basis1[mode_pq * nq1 + j];
                    }
                    fp[p] = tmp;
                }

                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p)
                    {
                        tmp += basis0[p * nq0 + i] * fp[p];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = basis0[i] * basis1[nq1 + j];
                        tmp1 += basis0[nq0 + i] * basis1[j];
                        tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp1 *= basis2[nq2 + k];
                        tmp += tmp1 * in[inoffset + 1];

                        // bottom vertex
                        tmp1 = basis0[nq0 + i] * basis1[nq1 + j];
                        tmp1 *= basis2[k];
                        tmp += tmp1 * in[inoffset + nm2];

                        // singular edge
                        for (unsigned int r = 1; r < nm2 - 1; ++r)
                        {
                            tmp1 = basis1[nq1 + j] * basis0[nq0 + i];
                            tmp1 *= basis2[(r + 1) * nq2 + k];
                            tmp += tmp1 * in[inoffset + nm2 + r];
                        }
                    }

                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData> // not working for nm2 > nm1
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

    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nmTot;
    TData *s_wsp2 = s_wsp1 + ((2 * nm1 - nm0 + 1) * nm0 / 2 * nq2);

    unsigned int nm01 = (2 * nm1 - nm0 + 1) * nm0 / 2;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        // Copy to shared memory.
        for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
             p += item_ct1.get_local_range(2))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1 - p;
                 q += item_ct1.get_local_range(1))
            {
                unsigned int mode_pqr = (2 * (nm2 - p) - q + 1) * q;
                mode_pqr += nm2 * (nm2 + 1) * p;
                mode_pqr -= (2 * nm2 + 1) * (p - 1) * p / 2;
                mode_pqr += (p - 1) * p * (2 * p - 1) / 6;
                mode_pqr /= 2;

                for (unsigned int r = item_ct1.get_local_id(0); r < nm2 - p - q;
                     r += item_ct1.get_local_range(0))
                {
                    s_wsp0[mode_pqr + r] = in[inoffset + mode_pqr + r];
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                 p += item_ct1.get_local_range(2))
            {
                for (unsigned int q = item_ct1.get_local_id(1); q < nm1 - p;
                     q += item_ct1.get_local_range(1))
                {
                    unsigned int cnt_kpq =
                        nm01 * k + (2 * nm1 - p + 1) * p / 2 + q;
                    unsigned int mode_pqr = (2 * (nm2 - p) - q + 1) * q;
                    mode_pqr += nm2 * (nm2 + 1) * p;
                    mode_pqr -= (2 * nm2 + 1) * (p - 1) * p / 2;
                    mode_pqr += (p - 1) * p * (2 * p - 1) / 6;
                    mode_pqr /= 2;

                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2 - p - q; ++r, ++mode_pqr)
                    {
                        tmp += basis2[k + nq2 * mode_pqr] * s_wsp0[mode_pqr];
                    }
                    s_wsp1[cnt_kpq] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    unsigned int mode_pq  = (2 * nm1 - p + 1) * p / 2;
                    unsigned int cnt_kpq  = nm01 * k + mode_pq;
                    unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;

                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1 - p; ++q, ++cnt_kpq)
                    {
                        tmp +=
                            s_wsp1[cnt_kpq] * basis1[(mode_pq + q) * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                     i += item_ct1.get_local_range(2))
                {
                    unsigned int cnt_kji  = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += basis0[p * nq0 + i] * s_wsp2[mode_kjp];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = basis0[i] * basis1[nq1 + j];
                        tmp1 += basis0[nq0 + i] * basis1[j];
                        tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp1 *= basis2[nq2 + k];
                        tmp += tmp1 * s_wsp0[1];

                        // bottom vertex
                        tmp1 = basis0[nq0 + i] * basis1[nq1 + j];
                        tmp1 *= basis2[k];
                        tmp += tmp1 * s_wsp0[nm2];

                        // singular edge
                        for (unsigned int r = 1; r < nm2 - 1; ++r)
                        {
                            tmp1 = basis1[nq1 + j] * basis0[nq0 + i];
                            tmp1 *= basis2[(r + 1) * nq2 + k];
                            tmp += tmp1 * s_wsp0[nm2 + r];
                        }
                    }

                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData>
void BwdTransPrismKernel(const unsigned int nm0, const unsigned int nm1,
                         const unsigned int nm2, const unsigned int nmTot,
                         const unsigned int nq0, const unsigned int nq1,
                         const unsigned int nq2, const unsigned int nelmt,
                         const bool correct, const TData *__restrict basis0,
                         const TData *__restrict basis1,
                         const TData *__restrict basis2,
                         const TData *__restrict in, TData *__restrict out,
                         TData *__restrict g_fpq, TData *__restrict g_fp,
                         const sycl::nd_item<3> &item_ct1)
{
    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    // Assign correct area of global memory to thread
    TData *fpq = g_fpq + (e * nm0 * nm1);
    TData *fp  = g_fp + (e * nm0);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
        {
            for (unsigned int p = 0, mode_pr = 0, mode_pq = 0, mode_pqr = 0;
                 p < nm0; ++p)
            {
                for (unsigned int q = 0; q < nm1; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        tmp += in[inoffset + mode_pqr] *
                               basis2[(mode_pr + r) * nq2 + k];
                    }
                    fpq[mode_pq] = tmp;
                }
                mode_pr += nm2 - p;
            }

            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int p = 0, mode_pq = 0; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1; ++q, ++mode_pq)
                    {
                        tmp += fpq[mode_pq] * basis1[q * nq1 + j];
                    }
                    fp[p] = tmp;
                }

                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p)
                    {
                        tmp += fp[p] * basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        for (unsigned int q = 0; q < nm1; ++q)
                        {
                            tmp += basis2[nq2 + k] * basis1[q * nq1 + j] *
                                   basis0[nq0 + i] * in[inoffset + nm2 * q + 1];
                        }
                    }

                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
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

    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nmTot;
    TData *s_wsp2 = s_wsp1 + (nm0 * nm1 * nq2);

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        // Copy to shared memory.
        for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
             p += item_ct1.get_local_range(2))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                unsigned int mode_pr  = (2 * nm2 - p + 1) * p / 2;
                unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;

                for (unsigned int r = item_ct1.get_local_id(0); r < nm2 - p;
                     r += item_ct1.get_local_range(0))
                {
                    s_wsp0[mode_pqr + r] = in[inoffset + mode_pqr + r];
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                 p += item_ct1.get_local_range(2))
            {
                for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                     q += item_ct1.get_local_range(1))
                {
                    unsigned int mode_pr  = (2 * nm2 - p + 1) * p / 2;
                    unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;
                    unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p + q;

                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        tmp +=
                            s_wsp0[mode_pqr] * basis2[(mode_pr + r) * nq2 + k];
                    }
                    s_wsp1[mode_kpq] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;
                    unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;

                    TData tmp = 0.0;
                    for (int q = 0; q < nm1; ++q, ++mode_kpq)
                    {
                        tmp += s_wsp1[mode_kpq] * basis1[q * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                     i += item_ct1.get_local_range(2))
                {
                    unsigned int cnt_kji  = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (int p = 0; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        for (int q = 0; q < nm1; ++q)
                        {
                            tmp += basis2[nq2 + k] * basis1[q * nq1 + j] *
                                   basis0[nq0 + i] * s_wsp0[q * nm2 + 1];
                        }
                    }

                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename TData> // not working for nm2 > nm1
void BwdTransPyrKernel(const unsigned int nm0, const unsigned int nm1,
                       const unsigned int nm2, const unsigned int nmTot,
                       const unsigned int nq0, const unsigned int nq1,
                       const unsigned int nq2, const unsigned int nelmt,
                       const bool correct, const TData *__restrict basis0,
                       const TData *__restrict basis1,
                       const TData *__restrict basis2,
                       const TData *__restrict in, TData *__restrict out,
                       TData *__restrict g_fpq, TData *__restrict g_fp,
                       const sycl::nd_item<3> &item_ct1)
{
    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    // Assign correct area of global memory to thread
    TData *fpq = g_fpq + (e * nm0 * nm1);
    TData *fp  = g_fp + (e * nm0);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        for (int k = 0, cnt_kji = 0; k < nq2; ++k)
        {
            for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0; p < nm0; ++p)
            {
                for (unsigned int q = 0; q < p; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2 - p; ++r, ++mode_pqr)
                    {
                        tmp += basis2[mode_pqr * nq2 + k] *
                               in[inoffset + mode_pqr];
                    }
                    fpq[mode_pq] = tmp;
                }

                for (unsigned int q = p; q < nm1; ++q, ++mode_pq)
                {
                    TData tmp = 0.0;
                    for (unsigned int r = 0; r < nm2 - q; ++r, ++mode_pqr)
                    {
                        tmp += basis2[mode_pqr * nq2 + k] *
                               in[inoffset + mode_pqr];
                    }
                    fpq[mode_pq] = tmp;
                }

                // increment mode in case nm2>nm1
                for (unsigned int q = nm1; q < nm2 - p; ++q)
                {
                    mode_pqr += nm2 - q;
                }
            }

            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int p = 0, mode_pq = 0; p < nm0; ++p)
                {
                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1; ++q, ++mode_pq)
                    {
                        tmp += fpq[mode_pq] * basis1[q * nq1 + j];
                    }
                    fp[p] = tmp;
                }

                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p)
                    {
                        tmp += fp[p] * basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = basis0[i] * basis1[nq1 + j];
                        tmp1 += basis0[nq0 + i] * basis1[j];
                        tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp1 *= basis2[nq2 + k];
                        tmp += tmp1 * in[inoffset + 1];
                    }

                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData> // not working for nm2 > nm1
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

    TData *s_wsp0 = shared;
    TData *s_wsp1 = s_wsp0 + nmTot;
    TData *s_wsp2 = s_wsp1 + (nm0 * nm1 * nq2);

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        unsigned int inoffset  = nmTot * e;
        unsigned int outoffset = nq0 * nq1 * nq2 * e;

        // Copy to shared memory.
        for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
             p += item_ct1.get_local_range(2))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                unsigned int mode_pqr = nm1 * (2 * nm2 + 1 - nm1) * p;
                mode_pqr -= (p - 1) * p / 2;
                mode_pqr -= (p - 1) * p * (2 * p - 1) / 6;
                mode_pqr /= 2;

                if (q < p)
                {
                    mode_pqr += q * (nm2 - p);
                    for (unsigned int r = item_ct1.get_local_id(0); r < nm2 - p;
                         r += item_ct1.get_local_range(0))
                    {
                        s_wsp0[mode_pqr + r] = in[inoffset + mode_pqr + r];
                    }
                }
                else
                {
                    mode_pqr += p * (nm2 - p);
                    mode_pqr += ((2 * (nm2 - p) - (q - p) + 1) * (q - p)) / 2;
                    for (unsigned int r = item_ct1.get_local_id(0); r < nm2 - q;
                         r += item_ct1.get_local_range(0))
                    {
                        s_wsp0[mode_pqr + r] = in[inoffset + mode_pqr + r];
                    }
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                 p += item_ct1.get_local_range(2))
            {
                for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                     q += item_ct1.get_local_range(1))
                {
                    unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p + q;
                    unsigned int mode_pqr = nm1 * (2 * nm2 + 1 - nm1) * p;
                    mode_pqr -= (p - 1) * p / 2;
                    mode_pqr -= (p - 1) * p * (2 * p - 1) / 6;
                    mode_pqr /= 2;

                    if (q < p)
                    {
                        mode_pqr += q * (nm2 - p);
                        TData tmp = 0.0;
                        for (unsigned int r = 0; r < nm2 - p; ++r, ++mode_pqr)
                        {
                            tmp +=
                                basis2[mode_pqr * nq2 + k] * s_wsp0[mode_pqr];
                        }
                        s_wsp1[mode_kpq] = tmp;
                    }
                    else
                    {
                        mode_pqr += p * (nm2 - p);
                        mode_pqr +=
                            ((2 * (nm2 - p) - (q - p) + 1) * (q - p)) / 2;

                        TData tmp = 0.0;
                        for (unsigned int r = 0; r < nm2 - q; ++r, ++mode_pqr)
                        {
                            tmp +=
                                basis2[mode_pqr * nq2 + k] * s_wsp0[mode_pqr];
                        }
                        s_wsp1[mode_kpq] = tmp;
                    }
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j + p;
                    unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

                    TData tmp = 0.0;
                    for (unsigned int q = 0; q < nm1; ++q, ++mode_kpq)
                    {
                        tmp += s_wsp1[mode_kpq] * basis1[q * nq1 + j];
                    }
                    s_wsp2[mode_kjp] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
             k += item_ct1.get_local_range(0))
        {
            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                     i += item_ct1.get_local_range(2))
                {
                    unsigned int cnt_kji  = nq0 * nq1 * k + nq0 * j + i;
                    unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

                    TData tmp = 0.0;
                    for (unsigned int p = 0; p < nm0; ++p, ++mode_kjp)
                    {
                        tmp += s_wsp2[mode_kjp] * basis0[p * nq0 + i];
                    }

                    if (correct)
                    {
                        // top vertex
                        TData tmp1 = basis0[i] * basis1[nq1 + j];
                        tmp1 += basis0[nq0 + i] * basis1[j];
                        tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp1 *= basis2[nq2 + k];
                        tmp += tmp1 * s_wsp0[1];
                    }

                    out[outoffset + cnt_kji] = tmp;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

// Kernel launchers
template <typename ExecSpace, typename TData, bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    BwdTrans1DKernel(const unsigned int nm0, const unsigned int nq0,
                     const unsigned int nelmt, const TData *basis0,
                     const TData *in, TData *out)
{
    const unsigned int gridsize  = MULTILEVEL ? std::min(nq0, 256u) : 256u;
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmt : (nelmt + gridsize - 1u) / gridsize, 2147483647u);

    if constexpr (MULTILEVEL)
    {
        SYCLQueue::GetInstance()
            .submit([=](sycl::handler &cgh) {
                unsigned int nshared = nm0;
                // Create local shared memory
                sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                      cgh);
                // Set WorkGroup dimensions
                const sycl::range<3> wgSize(1, 1, 32);
                const sycl::range<3> globalSize =
                    sycl::range<3>(1, 1, gridsize) * wgSize;

                cgh.parallel_for(
                    sycl::nd_range<3>(globalSize, wgSize),
                    [=](sycl::nd_item<3> item) {
                        TData *shmPtr = shared
                                            .template get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get();
                        BwdTransSegKernel_QP<TData>(nm0, nq0, nelmt, basis0, in,
                                                    out, shmPtr, item);
                    });
            })
            .wait();
    }
    else
    {
        SYCLQueue::GetInstance()
            .submit([=](sycl::handler &cgh) {
                unsigned int nshared = nm0 * nq0;
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
                        BwdTransSegKernel<TData>(nm0, nq0, nelmt, basis0, in,
                                                 out, shmPtr, item);
                    });
            })
            .wait();
    }
}

template <typename ExecSpace, typename TData, bool MULTILEVEL = true>
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
    const sycl::range<3> blocksize2d(1, std::min(nq1, 16u), std::min(nq0, 16u));
    const unsigned int gridsize = MULTILEVEL ? std::min(nq0 * nq1, 256u) : 256u;
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmt : (nelmt + gridsize - 1u) / gridsize, 2147483647u);

    if (shapetype == LibUtilities::Quad)
    {
        const unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = nq0 * nm0 + nq1 * nm1;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + nq0 * nm1;

            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    // Create local shared memory
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);

                    sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * blocksize2d;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, blocksize2d),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransQuadKernel_QP<TData>(
                                nm0, nm1, nmTot, nq0, nq1, nelmt, basis0,
                                basis1, in, out, shmPtr, item);
                        });
                })
                .wait();
        }
        else
        {
            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);
                    const sycl::range<3> wgSize(1, 1, blocksize);
                    const sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * wgSize;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, wgSize),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransQuadKernel<TData>(nm0, nm1, nmTot, nq0, nq1,
                                                      nelmt, basis0, basis1, in,
                                                      out, shmPtr, wsp, item);
                        });
                })
                .wait();
        }
    }
    else if (shapetype == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = nm0 * nq0 + nmTot * nq1;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + nm0 * nq1;

            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);

                    sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * blocksize2d;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, blocksize2d),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransTriKernel_QP<TData>(
                                nm0, nm1, nmTot, nq0, nq1, nelmt, correct,
                                basis0, basis1, in, out, shmPtr, item);
                        });
                })
                .wait();
        }
        else
        {
            sycl::range<3> wgSize(1, 1, blocksize);
            sycl::range<3> globalSize = sycl::range<3>(1, 1, gridsize) * wgSize;

            SYCLQueue::GetInstance()
                .parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                              [=](sycl::nd_item<3> item) {
                                  BwdTransTriKernel<TData>(
                                      nm0, nm1, nmTot, nq0, nq1, nelmt, correct,
                                      basis0, basis1, in, out, wsp, item);
                              })
                .wait();
        }
    }
}

template <typename ExecSpace, typename TData, bool MULTILEVEL = true>
inline
    typename std::enable_if<std::is_same<ExecSpace, NektarSpaces::SYCL>::value,
                            void>::type
    BwdTrans3DKernel(LibUtilities::ShapeType shapetype, const unsigned int nm0,
                     const unsigned int nm1, const unsigned int nm2,
                     const unsigned int nq0, const unsigned int nq1,
                     const unsigned int nq2, const unsigned int nelmt,
                     const bool correct, const TData *basis0,
                     const TData *basis1, const TData *basis2, TData *wsp,
                     const TData *in, TData *out)
{
    const sycl::range<3> blocksize3d(std::min(nq0, 8u), std::min(nq1, 8u),
                                     std::min(nq2, 8u));
    const unsigned int gridsize =
        MULTILEVEL ? std::min(nq0 * nq1 * nq2, 256u) : 256u;
    const unsigned int blocksize = std::min(
        MULTILEVEL ? nelmt : (nelmt + gridsize - 1u) / gridsize, 2147483647u);

    if (shapetype == LibUtilities::Hex)
    {
        const unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared = nq0 * nm0 + nq1 * nm1 + nq2 * nm2;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nmTot +
                       (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2);

            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);
                    sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * blocksize3d;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, blocksize3d),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransHexKernel_QP<TData>(
                                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                basis0, basis1, basis2, in, out, shmPtr, item);
                        });
                })
                .wait();
        }
        else
        {
            /**
             * NOTE: THIS HAS BEEN LEFT IN FOR COMPLETENESS
             * The orginal kernel was not designed to use `wsp` parameter and
             * since the kernels will need to be redone to match the CUDA
             * versions it will be updated then.
             */
            sycl::queue &Q = SYCLQueue::GetInstance();
            sycl::range<3> wgSize(1, 1, blocksize);
            sycl::range<3> globalSize = sycl::range<3>(1, 1, gridsize) * wgSize;
            TData *device_wsp1 =
                sycl::malloc_device<TData>(globalSize.size() * nm1 * nm2, Q);
            TData *device_wsp2 =
                sycl::malloc_device<TData>(globalSize.size() * nm2, Q);

            Q.submit([=](sycl::handler &cgh) {
                 // Create local shared memory accessor
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(globalSize, wgSize),
                     [=](sycl::nd_item<3> item) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         BwdTransHexKernel<TData>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, basis0,
                             basis1, basis2, in, out, shmPtr, device_wsp1,
                             device_wsp2, item);
                     });
             }).wait();

            sycl::free(device_wsp1, Q);
            sycl::free(device_wsp2, Q);
        }
    }
    else if (shapetype == LibUtilities::Tet)
    {
        const unsigned int nmTot =
            LibUtilities::StdTetData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        unsigned int nshared    = nm0 * nq0 + nm01 * nq1 + nmTot * nq2;

        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + ((2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2) +
                       (nm0 * nq1 * nq2);

            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);
                    sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * blocksize3d;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, blocksize3d),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransTetKernel_QP<TData>(
                                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                correct, basis0, basis1, basis2, in, out,
                                shmPtr, item);
                        });
                })
                .wait();
        }
        else
        {
            /**
             * NOTE: THIS HAS BEEN LEFT IN FOR COMPLETENESS
             * The orginal kernel was not designed to use `wsp` parameter and
             * since the kernels will need to be redone to match the CUDA
             * versions it will be updated then.
             */
            sycl::queue &Q = SYCLQueue::GetInstance();
            sycl::range<3> wgSize(1, 1, blocksize);
            sycl::range<3> globalSize = sycl::range<3>(1, 1, gridsize) * wgSize;
            TData *device_fpq         = sycl::malloc_device<TData>(
                globalSize.size() * (2 * nm1 - nm0 + 1) * nm0 / 2, Q);
            TData *device_fp =
                sycl::malloc_device<TData>(globalSize.size() * nm0, Q);

            Q.parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                           [=](sycl::nd_item<3> item) {
                               BwdTransTetKernel<TData>(
                                   nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                   correct, basis0, basis1, basis2, in, out,
                                   device_fpq, device_fp, item);
                           })
                .wait();

            sycl::free(device_fpq, Q);
            sycl::free(device_fp, Q);
        }
    }
    else if (shapetype == LibUtilities::Prism)
    {
        const unsigned int nmTot =
            LibUtilities::StdPrismData::getNumberOfCoefficients(nm0, nm1, nm2);
        const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
        unsigned int nshared    = nm0 * nq0 + nm1 * nq1 + nm12 * nq2;
        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);
                    sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * blocksize3d;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, blocksize3d),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransPrismKernel_QP<TData>(
                                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                correct, basis0, basis1, basis2, in, out,
                                shmPtr, item);
                        });
                })
                .wait();
        }
        else
        {
            /**
             * NOTE: THIS HAS BEEN LEFT IN FOR COMPLETENESS
             * The orginal kernel was not designed to use `wsp` parameter and
             * since the kernels will need to be redone to match the CUDA
             * versions it will be updated then.
             */
            sycl::queue &Q = SYCLQueue::GetInstance();
            sycl::range<3> wgSize(1, 1, blocksize);
            sycl::range<3> globalSize = sycl::range<3>(1, 1, gridsize) * wgSize;
            TData *device_fpq =
                sycl::malloc_device<TData>(globalSize.size() * nm0 * nm1, Q);
            TData *device_fp =
                sycl::malloc_device<TData>(globalSize.size() * nm0, Q);

            Q.parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                           [=](sycl::nd_item<3> item) {
                               BwdTransPrismKernel<TData>(
                                   nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                   correct, basis0, basis1, basis2, in, out,
                                   device_fpq, device_fp, item);
                           })
                .wait();

            sycl::free(device_fpq, Q);
            sycl::free(device_fp, Q);
        }
    }
    else if (shapetype == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::StdPyrData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared = nm0 * nq0 + nm1 * nq1 + nmTot * nq2;

        if constexpr (MULTILEVEL)
        {
            nshared += nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
            SYCLQueue::GetInstance()
                .submit([=](sycl::handler &cgh) {
                    unsigned int nshared =
                        nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
                    sycl::local_accessor<TData, 1> shared(
                        sycl::range<1>(nshared), cgh);
                    sycl::range<3> globalSize =
                        sycl::range<3>(1, 1, gridsize) * blocksize3d;

                    cgh.parallel_for(
                        sycl::nd_range<3>(globalSize, blocksize3d),
                        [=](sycl::nd_item<3> item) {
                            TData *shmPtr =
                                shared
                                    .template get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get();
                            BwdTransPyrKernel_QP<TData>(
                                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                correct, basis0, basis1, basis2, in, out,
                                shmPtr, item);
                        });
                })
                .wait();
        }
        else
        {
            /**
             * NOTE: THIS HAS BEEN LEFT IN FOR COMPLETENESS
             * The orginal kernel was not designed to use `wsp` parameter and
             * since the kernels will need to be redone to match the CUDA
             * versions it will be updated then.
             */
            sycl::queue &Q = SYCLQueue::GetInstance();
            sycl::range<3> wgSize(1, 1, blocksize);
            sycl::range<3> globalSize = sycl::range<3>(1, 1, gridsize) * wgSize;
            TData *device_fpq =
                sycl::malloc_device<TData>(globalSize.size() * nm0 * nm1, Q);
            TData *device_fp =
                sycl::malloc_device<TData>(globalSize.size() * nm0, Q);

            Q.parallel_for(sycl::nd_range<3>(globalSize, wgSize),
                           [=](sycl::nd_item<3> item) {
                               BwdTransPyrKernel<TData>(
                                   nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt,
                                   correct, basis0, basis1, basis2, in, out,
                                   device_fpq, device_fp, item);
                           })
                .wait();

            sycl::free(device_fpq, Q);
            sycl::free(device_fp, Q);
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
