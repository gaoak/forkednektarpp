///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSYCLSumFacKernels.hpp
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

namespace Nektar::Operators::detail
{

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseSegKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
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
                sum += in[index] * basis0[p * nq0 + i] * jac[jacindex] * w0[i];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            const unsigned int index =
                nm0 * warpsize * iwarp + warpsize * p + ilane;
            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseSegKernel_QP(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale, TData *__restrict__ shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    TData *s_wsp0 = shmemptr;

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nq0 * e;
        const unsigned int outoffset = nm0 * e;

        // Copy to shared memory.
        for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
             i += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + i;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[i]                   = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
             p += item_ct1.get_local_range(2))
        {
            TData sum = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i)
            {
                sum += s_wsp0[i] * basis0[p * nq0 + i] * w0[i];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            const unsigned int index = outoffset + p;
            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int p = 0u; p < nm0; ++p)
        {
            for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
            {
                TData sum = 0.0;
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                    const unsigned int jacindex = DEFORMED ? index : e;
                    sum +=
                        in[index] * basis0[p * nq0 + i] * jac[jacindex] * w0[i];
                }
                wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] = sum;
            }

            for (unsigned int q = 0u; q < nm1; ++q)
            {
                TData sum = 0.0;
                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    sum += wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] *
                           basis1[q * nq1 + j] * w1[j];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                const unsigned int index =
                    nmTot * warpsize * iwarp + warpsize * (nm0 * q + p) + ilane;
                if constexpr (APPEND)
                {
                    out[index] += sum;
                }
                else
                {
                    out[index] = sum;
                }
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseQuadKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale, TData *__restrict__ shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;

    TData *s_wsp0   = shmemptr;
    TData *s_wsp1   = s_wsp0 + nqTot;
    TData *s_basis0 = s_wsp1 + nm0 * nq1;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_w0     = s_basis1 + nm1 * nq1;
    TData *s_w1     = s_w0 + nq0;

    // Copy to shared memory.
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

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
         idx += item_ct1.get_local_range(2))
    {
        s_w0[idx] = w0[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
         idx += item_ct1.get_local_range(2))
    {
        s_w1[idx] = w1[idx];
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int j = idx % nq1;
            const unsigned int p = idx / nq1;
            unsigned int cnt_ji  = nq0 * j;

            TData sum = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
            }
            s_wsp1[idx] = sum;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p     = idx % nm0;
            const unsigned int q     = idx / nm0;
            const unsigned int index = outoffset + idx;
            unsigned int cnt_pj      = nq1 * p;

            TData sum = 0.0;
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
            {
                sum += s_wsp1[cnt_pj] * s_basis1[q * nq1 + j] * s_w1[j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseTriKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    TData *__restrict__ wsp, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
        {
            for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
            {
                TData sum = 0.0;
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                    const unsigned int jacindex = DEFORMED ? index : e;
                    sum +=
                        in[index] * basis0[p * nq0 + i] * jac[jacindex] * w0[i];
                }
                wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] = sum;
            }

            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                TData sum = 0.0;
                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    sum += wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] *
                           basis1[mode_pq * nq1 + j] * w1[j];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                const unsigned int index =
                    nmTot * warpsize * iwarp + warpsize * mode_pq + ilane;
                if constexpr (APPEND)
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
        if (isModified)
        {
            TData iprod_01 = 0.0;
            for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
            {
                const unsigned int index    = nqTot * warpsize * iwarp + ilane;
                const unsigned int jacindex = DEFORMED ? index : e;

                TData tmp = w1[j] * basis1[nq1 + j];
                if constexpr (!DEFORMED)
                {
                    tmp *= jac[jacindex];
                }

                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                    const unsigned int jacindex = DEFORMED ? index : e;

                    TData prod = in[index] * tmp * w0[i];
                    if constexpr (DEFORMED)
                    {
                        prod *= jac[jacindex];
                    }
                    iprod_01 += prod * basis0[nq0 + i];
                }
            }

            const unsigned int index =
                nmTot * warpsize * iwarp + warpsize + ilane;
            if constexpr (SCALE)
            {
                out[index] += iprod_01 * scale;
            }
            else
            {
                out[index] += iprod_01;
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseTriKernel_QP(
    const unsigned int nm0, [[maybe_unused]] const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ pindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    TData *__restrict__ shmemptr, const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;

    TData *s_wsp0   = shmemptr;
    TData *s_wsp1   = s_wsp0 + nqTot;
    TData *s_prod   = s_wsp1 + nm0 * nq1;
    TData *s_basis0 = s_prod + 1u;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_w0     = s_basis1 + nmTot * nq1;
    TData *s_w1     = s_w0 + nq0;

    // Copy to shared memory.
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

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
         idx += item_ct1.get_local_range(2))
    {
        s_w0[idx] = w0[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
         idx += item_ct1.get_local_range(2))
    {
        s_w1[idx] = w1[idx];
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int j = idx % nq1;
            const unsigned int p = idx / nq1;
            unsigned int cnt_ji  = nq0 * j;

            TData sum = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
            }
            s_wsp1[idx] = sum;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p     = pindex[idx];
            const unsigned int index = outoffset + idx;
            unsigned int cnt_pj      = nq1 * p;

            TData sum = 0.0;
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
            {
                sum += s_wsp1[cnt_pj] * s_basis1[idx * nq1 + j] * s_w1[j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        // Correction for singular vertex in collpased coordinates.
        // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
        // With contributions from every quadrature point
        if (isModified)
        {
            item_ct1.barrier(sycl::access::fence_space::local_space);

            TData prod = 0.0;

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                 idx += item_ct1.get_local_range(2))
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = idx / nq0;
                TData tmp            = s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp *= s_wsp0[idx] * s_w0[i] * s_w1[j];

                prod += tmp;
            }

            prod = sycl::reduce_over_group(item_ct1.get_sub_group(), prod,
                                           sycl::plus<>());

            if (item_ct1.get_sub_group().get_local_id() == 0)
            {
                if constexpr (SCALE)
                {
                    prod *= scale;
                }

                atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
                    out + outoffset + 1u, prod);
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseHexKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;
        TData *wsp0              = wsp;
        TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;

        for (unsigned int p = 0u; p < nm0; ++p)
        {
            for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[i + nq0 * p] *
                                  jac[jacindex] * w0[i];
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
                                 basis1[q * nq1 + j] * w1[j];
                    }
                    wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] = sum_k;
                }

                for (unsigned int r = 0u; r < nm2; ++r)
                {
                    const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;
                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * cnt_rqp + ilane;

                    TData sum = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        sum += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                    ilane] *
                               basis2[r * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum *= scale;
                    }

                    if constexpr (APPEND)
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

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseHexKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const TData scale, TData *__restrict__ shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData *s_wsp0   = shmemptr;
    TData *s_wsp1   = s_wsp0 + nqTot;
    TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = s_wsp2 + nm0 * nm1 * nq2;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_basis2 = s_basis1 + nm1 * nq1;
    TData *s_w0     = s_basis2 + nm2 * nq2;
    TData *s_w1     = s_w0 + nq0;
    TData *s_w2     = s_w1 + nq1;

    // Copy to shared memory.
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

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
         idx += item_ct1.get_local_range(2))
    {
        s_w0[idx] = w0[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
         idx += item_ct1.get_local_range(2))
    {
        s_w1[idx] = w1[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
         idx += item_ct1.get_local_range(2))
    {
        s_w2[idx] = w2[idx];
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int j = idx % nq1;
            const unsigned int k = (idx / nq1) % nq2;
            const unsigned int p = idx / (nq1 * nq2);
            unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

            TData sum_kj = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                sum_kj += s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
            }
            s_wsp1[idx] = sum_kj;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int k = idx % nq2;
            const unsigned int q = (idx / nq2) % nm1;
            const unsigned int p = idx / (nq2 * nm1);
            unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k;

            TData sum_k = 0.0;
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
            {
                sum_k += s_wsp1[cnt_pkj] * s_basis1[q * nq1 + j] * s_w1[j];
            }
            s_wsp2[idx] = sum_k;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1 * nm2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p     = idx % nm0;
            const unsigned int q     = (idx / nm0) % nm1;
            const unsigned int r     = idx / (nm0 * nm1);
            const unsigned int index = outoffset + idx;
            unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

            TData sum = 0.0;
            for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
            {
                sum += s_wsp2[cnt_pqk] * s_basis2[r * nq2 + k] * s_w2[k];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[index] += sum;
            }
            else
            {
                out[index] = sum;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseTetKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;
        TData *wsp0              = wsp;
        TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;
        TData *prod              = wsp1 + nq2 * nelmt;

        for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u;
             p < nm0; ++p)
        {
            for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[i + nq0 * p] *
                                  jac[jacindex] * w0[i];
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
                                 basis1[mode_pq * nq1 + j] * w1[j];
                    }
                    wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] = sum_k;
                }

                for (unsigned int r = 0u; r < nm2 - p - q;
                     ++r, ++mode2, ++mode_pqr)
                {
                    TData tmp = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        tmp += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                    ilane] *
                               basis2[mode2 * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        tmp *= scale;
                    }

                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * mode_pqr + ilane;
                    if constexpr (APPEND)
                    {
                        out[index] += tmp;
                    }
                    else
                    {
                        out[index] = tmp;
                    }
                }
            }

            // increment mode in case order1!=order2
            for (int q = nm1 - p; q < nm2 - p; ++q)
            {
                mode2 += nm2 - p - q;
            }
        }

        // Add correction for collapsed coordinate.
        if (isModified)
        {
            for (unsigned int r = 0u; r < nm2; ++r)
            {
                prod[nm2 * warpsize * iwarp + warpsize * r + ilane] = 0.0;
            }

            for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                TData tmpQ2 = w2[k];
                if constexpr (!DEFORMED)
                {
                    tmpQ2 *= jac[e];
                }

                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    TData tmpQ1 = tmpQ2 * w1[j];
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * w0[i];
                        if constexpr (DEFORMED)
                        {
                            tmpQ *= jac[index];
                        }

                        // top vertex
                        TData tmp = basis0[i] * basis1[nq1 + j];
                        tmp += basis0[nq0 + i] * basis1[j];
                        tmp += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp *= basis2[nq2 + k];
                        tmp *= in[index] * tmpQ;
                        prod[nm2 * warpsize * iwarp + warpsize * (nm2 - 1) +
                             ilane] += tmp;

                        // bottom vertex
                        tmp = basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] *
                              in[index] * tmpQ;
                        prod[nm2 * warpsize * iwarp + ilane] += tmp;

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp = basis2[(r + 1) * nq2 + k] * basis1[nq1 + j] *
                                  basis0[nq0 + i] * in[index] * tmpQ;
                            prod[nm2 * warpsize * iwarp + warpsize * r +
                                 ilane] += tmp;
                        }
                    }
                }
            }

            if constexpr (SCALE)
            {
                const unsigned int index =
                    nmTot * warpsize * iwarp + warpsize + ilane;
                out[index] += prod[nm2 * warpsize * iwarp +
                                   warpsize * (nm2 - 1) + ilane] *
                              scale;
                for (unsigned int r = 0u; r < nm2 - 1u; ++r)
                {
                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * (nm2 + r) + ilane;
                    out[index] +=
                        prod[nm2 * warpsize * iwarp + warpsize * r + ilane] *
                        scale;
                }
            }
            else
            {
                const unsigned int index =
                    nmTot * warpsize * iwarp + warpsize + ilane;
                out[index] +=
                    prod[nm2 * warpsize * iwarp + warpsize * (nm2 - 1) + ilane];
                for (unsigned int r = 0u; r < nm2 - 1u; ++r)
                {
                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * (nm2 + r) + ilane;
                    out[index] +=
                        prod[nm2 * warpsize * iwarp + warpsize * r + ilane];
                }
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBaseTetKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ pindex1,
    const unsigned int *__restrict__ pindex2,
    const unsigned int *__restrict__ qindex2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    TData *__restrict__ shmemptr, const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    TData *s_prod   = shmemptr;
    TData *s_wsp0   = s_prod + nm2;
    TData *s_wsp1   = s_wsp0 + nqTot;
    TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = s_wsp2 + nm01 * nq2;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_basis2 = s_basis1 + nm01 * nq1;
    TData *s_w0     = s_basis2 + nmode2 * nq2;
    TData *s_w1     = s_w0 + nq0;
    TData *s_w2     = s_w1 + nq1;

    // Copy to shared memory.
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

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
         idx += item_ct1.get_local_range(2))
    {
        s_w0[idx] = w0[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
         idx += item_ct1.get_local_range(2))
    {
        s_w1[idx] = w1[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
         idx += item_ct1.get_local_range(2))
    {
        s_w2[idx] = w2[idx];
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int j = idx % nq1;
            const unsigned int k = (idx / nq1) % nq2;
            const unsigned int p = idx / (nq1 * nq2);
            unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j;

            TData sum_kj = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                sum_kj += s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
            }
            s_wsp1[idx] = sum_kj;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm01 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int mode_pq = idx / nq2;
            const unsigned int p       = pindex1[mode_pq];
            const unsigned int k       = idx % nq2;
            unsigned int cnt_pkj       = nq1 * nq2 * p + nq1 * k;

            TData sum_k = 0.0;
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
            {
                sum_k +=
                    s_basis1[mode_pq * nq1 + j] * s_wsp1[cnt_pkj] * s_w1[j];
            }
            s_wsp2[idx] = sum_k;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p       = pindex2[idx];
            const unsigned int q       = qindex2[idx];
            const unsigned int index   = outoffset + idx;
            const unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
            const unsigned int mode2 =
                idx +
                ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);

            TData tmp = 0.0;
            for (unsigned int k = 0u; k < nq2; ++k)
            {
                tmp += s_wsp2[mode_pq * nq2 + k] * s_basis2[mode2 * nq2 + k] *
                       s_w2[k];
            }

            if constexpr (SCALE)
            {
                tmp *= scale;
            }

            if constexpr (APPEND)
            {
                out[index] += tmp;
            }
            else
            {
                out[index] = tmp;
            }
        }

        // Add correction for collapsed coordinate.
        if (isModified)
        {
            item_ct1.barrier(sycl::access::fence_space::local_space);

            constexpr unsigned int NM2_MAX = 8;
            if (nm2 <= NM2_MAX)
            {
                TData prod[NM2_MAX] = {0.0};
                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    // Store jac * quadrature weight
                    TData tmpQ = s_w2[k] * s_w1[j] * s_w0[i];

                    // top vertex
                    TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                    tmp += s_basis0[nq0 + i] * s_basis1[j];
                    tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                    tmp *= s_basis2[nq2 + k];
                    tmp *= s_wsp0[idx] * tmpQ;
                    prod[nm2 - 1u] += tmp;

                    // singular edge
                    tmpQ *= s_basis1[nq1 + j] * s_basis0[nq0 + i] * s_wsp0[idx];
                    for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                    {
                        prod[r] += s_basis2[(r + 1u) * nq2 + k] * tmpQ;
                    }

                    // bottom vertex
                    prod[0] += s_basis2[k] * tmpQ;
                }

                for (unsigned int r = 0u; r < nm2; ++r)
                {
                    prod[r] = sycl::reduce_over_group(item_ct1.get_sub_group(),
                                                      prod[r], sycl::plus<>());

                    if (item_ct1.get_sub_group().get_local_id() == 0)
                    {
                        if constexpr (SCALE)
                        {
                            prod[r] *= scale;
                        }

                        if (r == nm2 - 1u)
                        {
                            atomic_add<NektarSpaces::SYCL,
                                       NektarSpaces::GlobalScope>(
                                out + outoffset + 1u, prod[nm2 - 1u]);
                        }
                        else
                        {
                            atomic_add<NektarSpaces::SYCL,
                                       NektarSpaces::GlobalScope>(
                                out + outoffset + nm2 + r, prod[r]);
                        }
                    }
                }
            }
            else
            {
                TData prod0 = 0.0;
                TData prod1 = 0.0;

                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    // Store jac * quadrature weight
                    TData tmpQ = s_w2[k] * s_w1[j] * s_w0[i];

                    // top vertex
                    TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                    tmp += s_basis0[nq0 + i] * s_basis1[j];
                    tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                    tmp *= s_basis2[nq2 + k];
                    tmp *= s_wsp0[idx] * tmpQ;
                    prod0 += tmp;

                    // bottom vertex
                    prod1 += s_basis0[nq0 + i] * s_basis1[nq1 + j] *
                             s_basis2[k] * s_wsp0[idx] * tmpQ;
                }

                prod0 = sycl::reduce_over_group(item_ct1.get_sub_group(), prod0,
                                                sycl::plus<>());
                prod1 = sycl::reduce_over_group(item_ct1.get_sub_group(), prod1,
                                                sycl::plus<>());

                if (item_ct1.get_sub_group().get_local_id() == 0)
                {
                    if constexpr (SCALE)
                    {
                        prod0 *= scale;
                        prod1 *= scale;
                    }

                    atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
                        out + outoffset + 1, prod0);
                    atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
                        out + outoffset + nm2, prod1);
                }

                // singular edge
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    TData prod = 0.0;

                    for (unsigned int idx = item_ct1.get_local_id(2);
                         idx < nqTot; idx += item_ct1.get_local_range(2))
                    {
                        const unsigned int i = idx % nq0;
                        const unsigned int j = (idx / nq0) % nq1;
                        const unsigned int k = idx / (nq0 * nq1);

                        // Store jac * quadrature weight
                        TData tmpQ = s_w2[k] * s_w1[j] * s_w0[i];

                        prod += s_basis2[(r + 1) * nq2 + k] *
                                s_basis1[nq1 + j] * s_basis0[nq0 + i] *
                                s_wsp0[idx] * tmpQ;
                    }

                    prod = sycl::reduce_over_group(item_ct1.get_sub_group(),
                                                   prod, sycl::plus<>());

                    if (item_ct1.get_sub_group().get_local_id() == 0)
                    {
                        if constexpr (SCALE)
                        {
                            prod *= scale;
                        }

                        atomic_add<NektarSpaces::SYCL,
                                   NektarSpaces::GlobalScope>(
                            out + outoffset + nm2 + r, prod);
                    }
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

/**
NOTE: The total declared local variable size exceeds 128 bytes. This may cause
high register pressure with a sub-group size of 32 depending on hardware.
Leaving it for now but will have to profile later and make appropriate changes
if necessary.
*/
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBasePrismKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;
        TData *wsp0              = wsp;
        TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;
        TData *wsp2              = wsp1 + nq2 * nelmt;

        for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
        {
            for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[nq0 * p + i] *
                                  jac[jacindex] * w0[i];
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
                                 basis1[q * nq1 + j] * w1[j];
                    }
                    wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] = sum_k;
                }

                for (int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
                {
                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * mode_pqr + ilane;
                    unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;

                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        sum_k += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                      ilane] *
                                 basis2[(mode_pr + r) * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum_k *= scale;
                    }

                    if constexpr (APPEND)
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
        if (isModified)
        {
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                wsp2[nm1 * warpsize * iwarp + warpsize * q + ilane] = 0.0;
            }

            for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                TData k_weight = w2[k];
                if constexpr (!DEFORMED)
                {
                    k_weight *= jac[e];
                }

                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    TData kj_weight = k_weight * w1[j];
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        TData prod = kj_weight * basis2[nq2 + k] *
                                     basis0[nq0 + i] * w0[i] * in[index];
                        if constexpr (DEFORMED)
                        {
                            prod *= jac[index];
                        }

                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            wsp2[nm1 * warpsize * iwarp + warpsize * q +
                                 ilane] += prod * basis1[q * nq1 + j];
                        }
                    }
                }
            }

            for (unsigned int q = 0u; q < nm1; ++q)
            {
                const unsigned int index = nmTot * warpsize * iwarp +
                                           warpsize * (nm2 * q + 1u) + ilane;
                if constexpr (SCALE)
                {
                    out[index] +=
                        wsp2[nm1 * warpsize * iwarp + warpsize * q + ilane] *
                        scale;
                }
                else
                {
                    out[index] +=
                        wsp2[nm1 * warpsize * iwarp + warpsize * q + ilane];
                }
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBasePrismKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex,
    const unsigned int *__restrict__ rindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    TData *__restrict__ shmemptr, const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;

    TData *s_wsp0   = shmemptr;
    TData *s_wsp1   = s_wsp0 + nqTot;
    TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = s_wsp2 + nm0 * nm1 * nq2;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_basis2 = s_basis1 + nm1 * nq1;
    TData *s_w0     = s_basis2 + nm02 * nq2;
    TData *s_w1     = s_w0 + nq0;
    TData *s_w2     = s_w1 + nq1;

    // Copy to shared memory.
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

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nm02 * nq2;
         idx += item_ct1.get_local_range(2))
    {
        s_basis2[idx] = basis2[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
         idx += item_ct1.get_local_range(2))
    {
        s_w0[idx] = w0[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
         idx += item_ct1.get_local_range(2))
    {
        s_w1[idx] = w1[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
         idx += item_ct1.get_local_range(2))
    {
        s_w2[idx] = w2[idx];
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int j = idx % nq1;
            const unsigned int k = (idx / nq1) % nq2;
            const unsigned int p = idx / (nq1 * nq2);
            unsigned int cnt_kji = nq1 * nq0 * k + nq0 * j;

            TData sum_kj = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                sum_kj += s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
            }
            s_wsp1[idx] = sum_kj;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int k = idx % nq2;
            const unsigned int q = (idx / nq2) % nm1;
            const unsigned int p = idx / (nq2 * nm1);
            unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

            TData sum_k = 0.0;
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
            {
                sum_k += s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
            }
            s_wsp2[idx] = sum_k;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p       = pindex[idx];
            const unsigned int q       = qindex[idx];
            const unsigned int r       = rindex[idx];
            const unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u + r;
            const unsigned int index   = outoffset + idx;
            unsigned int cnt_pqk       = nm1 * nq2 * p + nq2 * q;

            TData sum_k = 0.0;
            for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
            {
                sum_k +=
                    s_basis2[mode_pr * nq2 + k] * s_w2[k] * s_wsp2[cnt_pqk];
            }

            if constexpr (SCALE)
            {
                sum_k *= scale;
            }

            if constexpr (APPEND)
            {
                out[index] += sum_k;
            }
            else
            {
                out[index] = sum_k;
            }
        }

        // Add correction for collapsed coordinate.
        if (isModified)
        {
            item_ct1.barrier(sycl::access::fence_space::local_space);

            constexpr unsigned int NM1_MAX = 8;
            if (nm1 <= NM1_MAX)
            {
                TData prod[NM1_MAX] = {0.0};

                for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                     idx += item_ct1.get_local_range(2))
                {
                    const unsigned int i = idx % nq0;
                    const unsigned int j = (idx / nq0) % nq1;
                    const unsigned int k = idx / (nq0 * nq1);

                    // Store jac * quadrature weight
                    TData tmpQ = s_w2[k] * s_w1[j] * s_w0[i];

                    TData tmp = tmpQ * s_wsp0[idx];
                    tmp *= s_basis2[nq2 + k] * s_basis0[nq0 + i];
                    for (unsigned int q = 0u; q < nm1; ++q)
                    {
                        prod[q] += tmp * s_basis1[q * nq1 + j];
                    }
                }

                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    prod[q] = sycl::reduce_over_group(item_ct1.get_sub_group(),
                                                      prod[q], sycl::plus<>());

                    if (item_ct1.get_sub_group().get_local_id() == 0)
                    {
                        if constexpr (SCALE)
                        {
                            prod[q] *= scale;
                        }

                        atomic_add<NektarSpaces::SYCL,
                                   NektarSpaces::GlobalScope>(
                            out + outoffset + nm2 * q + 1u, prod[q]);
                    }
                }
            }
            else
            {
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    TData prod = 0.0;

                    for (unsigned int idx = item_ct1.get_local_id(2);
                         idx < nqTot; idx += item_ct1.get_local_range(2))
                    {
                        const unsigned int i = idx % nq0;
                        const unsigned int j = (idx / nq0) % nq1;
                        const unsigned int k = idx / (nq0 * nq1);

                        // Store jac * quadrature weight
                        TData tmpQ = s_w2[k] * s_w1[j] * s_w0[i];

                        TData tmp = tmpQ * s_wsp0[idx];
                        tmp *= s_basis2[nq2 + k] * s_basis1[q * nq1 + j] *
                               s_basis0[nq0 + i];
                        prod += tmp;
                    }

                    prod = sycl::reduce_over_group(item_ct1.get_sub_group(),
                                                   prod, sycl::plus<>());

                    if (item_ct1.get_sub_group().get_local_id() == 0)
                    {
                        if constexpr (SCALE)
                        {
                            prod *= scale;
                        }

                        atomic_add<NektarSpaces::SYCL,
                                   NektarSpaces::GlobalScope>(
                            out + outoffset + nm2 * q + 1u, prod);
                    }
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBasePyrKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int e = item_ct1.get_global_id(2);

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;
        TData *wsp0              = wsp;
        TData *wsp1              = wsp0 + nq2 * nq1 * nelmt;

        for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0; ++p)
        {
            for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
                        const unsigned int jacindex = DEFORMED ? index : e;
                        sum_kj += in[index] * basis0[nq0 * p + i] *
                                  jac[jacindex] * w0[i];
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
                                 basis1[q * nq1 + j] * w1[j];
                    }
                    wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] = sum_k;
                }

                for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
                {
                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        sum_k += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                      ilane] *
                                 basis2[mode2 * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum_k *= scale;
                    }

                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * mode_pqr + ilane;
                    if constexpr (APPEND)
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
                                 basis1[q * nq1 + j] * w1[j];
                    }
                    wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] = sum_k;
                }

                for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
                {
                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        sum_k += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                      ilane] *
                                 basis2[mode2 * nq2 + k] * w2[k];
                    }

                    if constexpr (SCALE)
                    {
                        sum_k *= scale;
                    }

                    const unsigned int index =
                        nmTot * warpsize * iwarp + warpsize * mode_pqr + ilane;
                    if constexpr (APPEND)
                    {
                        out[index] += sum_k;
                    }
                    else
                    {
                        out[index] = sum_k;
                    }
                }
            }

            // increment mode in case order1!=order2
            for (int q = nm1; q < nm2; ++q)
            {
                mode2 += nm2 - q;
            }
        }

        // Add correction for collapsed coordinate.
        if (isModified)
        {
            TData prod = 0.0;
            for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
            {
                TData tmpQ2 = w2[k];
                if constexpr (!DEFORMED)
                {
                    tmpQ2 *= jac[e];
                }

                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    TData tmpQ1 = tmpQ2 * w1[j];
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;

                        // Store jac * quadrature weight
                        TData tmpQ = tmpQ1 * w0[i];
                        if constexpr (DEFORMED)
                        {
                            tmpQ *= jac[index];
                        }

                        // top vertex
                        TData tmp = basis0[i] * basis1[nq1 + j];
                        tmp += basis0[nq0 + i] * basis1[j];
                        tmp += basis0[nq0 + i] * basis1[nq1 + j];
                        tmp *= basis2[nq2 + k];
                        tmp *= in[index] * tmpQ;
                        prod += tmp;
                    }
                }
            }

            // add to existing entry
            const unsigned int index =
                nmTot * warpsize * iwarp + warpsize + ilane;
            if constexpr (SCALE)
            {
                out[index] += prod * scale;
            }
            else
            {
                out[index] += prod;
            }
        }

        e += item_ct1.get_global_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBasePyrKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out, const TData scale,
    TData *__restrict__ shmemptr, const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

    TData *s_prod   = shmemptr;
    TData *s_wsp0   = s_prod + 1u;
    TData *s_wsp1   = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = s_wsp2 + nm0 * nm1 * nq2;
    TData *s_basis1 = s_basis0 + nm0 * nq0;
    TData *s_basis2 = s_basis1 + nm1 * nq1;
    TData *s_w0     = s_basis2 + nmode2 * nq2;
    TData *s_w1     = s_w0 + nq0;
    TData *s_w2     = s_w1 + nq1;

    // Copy to shared memory.
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

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0;
         idx += item_ct1.get_local_range(2))
    {
        s_w0[idx] = w0[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq1;
         idx += item_ct1.get_local_range(2))
    {
        s_w1[idx] = w1[idx];
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq2;
         idx += item_ct1.get_local_range(2))
    {
        s_w2[idx] = w2[idx];
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nq1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int j = idx % nq1;
            const unsigned int k = (idx / nq1) % nq2;
            const unsigned int p = idx / (nq1 * nq2);
            unsigned int cnt_kji = k * nq1 * nq0 + j * nq0;

            TData sum_kj = 0.0;
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                sum_kj += s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
            }
            s_wsp1[idx] = sum_kj;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nm0 * nm1 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int k = idx % nq2;
            const unsigned int q = (idx / nq2) % nm1;
            const unsigned int p = idx / (nq2 * nm1);
            unsigned int cnt_pkj = nq1 * nq2 * p + k * nq1;

            TData sum_k = 0.0;
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
            {
                sum_k += s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
            }
            s_wsp2[idx] = sum_k;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmTot;
             idx += item_ct1.get_local_range(2))
        {
            const unsigned int p = pindex[idx];
            const unsigned int q = qindex[idx];
            const unsigned int mode2 =
                idx +
                ((nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u);
            const unsigned int index = outoffset + idx;
            unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

            TData sum_k = 0.0;
            for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
            {
                sum_k += s_basis2[mode2 * nq2 + k] * s_w2[k] * s_wsp2[cnt_pqk];
            }
            if constexpr (SCALE)
            {
                sum_k *= scale;
            }

            if constexpr (APPEND)
            {
                out[index] += sum_k;
            }
            else
            {
                out[index] = sum_k;
            }
        }

        // Add correction for collapsed coordinate.
        if (isModified)
        {
            item_ct1.barrier(sycl::access::fence_space::local_space);

            TData prod = 0.0;

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                 idx += item_ct1.get_local_range(2))
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);

                // Store jac * quadrature weight
                TData tmpQ = s_w2[k] * s_w1[j] * s_w0[i];

                // top vertex
                TData tmp = s_basis0[i] * s_basis1[nq1 + j];
                tmp += s_basis0[nq0 + i] * s_basis1[j];
                tmp += s_basis0[nq0 + i] * s_basis1[nq1 + j];
                tmp *= s_basis2[nq2 + k];
                tmp *= s_wsp0[idx] * tmpQ;
                prod += tmp;
            }

            prod = sycl::reduce_over_group(item_ct1.get_sub_group(), prod,
                                           sycl::plus<>());

            if (item_ct1.get_sub_group().get_local_id() == 0)
            {
                if constexpr (SCALE)
                {
                    prod *= scale;
                }

                atomic_add<NektarSpaces::SYCL, NektarSpaces::GlobalScope>(
                    out + outoffset + 1, prod);
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict out, const TData scale,
    [[maybe_unused]] TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        IProductWRTBaseSegKernel<SCALE, APPEND, DEFORMED>(
            nm0, nq0, nelmt, basis0, w0, jac, in, out, scale, item_ct1);
    }
    else
    {
        IProductWRTBaseSegKernel_QP<SCALE, APPEND, DEFORMED>(
            nm0, nq0, nelmt, basis0, w0, jac, in, out, scale, shmemptr,
            item_ct1);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, [[maybe_unused]] const unsigned int *index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, [[maybe_unused]] TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict out, const TData scale,
    [[maybe_unused]] TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            IProductWRTBaseQuadKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, basis0, basis1, w0, w1, jac,
                wsp, in, out, scale, item_ct1);
        }
        else
        {
            IProductWRTBaseQuadKernel_QP<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, basis0, basis1, w0, w1, jac,
                in, out, scale, shmemptr, item_ct1);
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            IProductWRTBaseTriKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, basis0, basis1,
                w0, w1, jac, wsp, in, out, scale, item_ct1);
        }
        else
        {
            IProductWRTBaseTriKernel_QP<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
                basis1, w0, w1, jac, in, out, scale, shmemptr, item_ct1);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0,
    [[maybe_unused]] const unsigned int *index1,
    [[maybe_unused]] const unsigned int *index2,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, [[maybe_unused]] TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict out, const TData scale,
    [[maybe_unused]] TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            IProductWRTBaseHexKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, basis0, basis1,
                basis2, w0, w1, w2, jac, wsp, in, out, scale, item_ct1);
        }
        else
        {
            IProductWRTBaseHexKernel_QP<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, basis0, basis1,
                basis2, w0, w1, w2, jac, in, out, scale, shmemptr, item_ct1);
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            IProductWRTBaseTetKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, basis0,
                basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale, item_ct1);
        }
        else
        {
            IProductWRTBaseTetKernel_QP<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
                index1, index2, basis0, basis1, basis2, w0, w1, w2, jac, in,
                out, scale, shmemptr, item_ct1);
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            IProductWRTBasePrismKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, basis0,
                basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale, item_ct1);
        }
        else
        {
            IProductWRTBasePrismKernel_QP<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
                index1, index2, basis0, basis1, basis2, w0, w1, w2, jac, in,
                out, scale, shmemptr, item_ct1);
        }
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
        {
            IProductWRTBasePyrKernel<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, basis0,
                basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale, item_ct1);
        }
        else
        {
            IProductWRTBasePyrKernel_QP<SCALE, APPEND, DEFORMED>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0,
                index1, basis0, basis1, basis2, w0, w1, w2, jac, in, out, scale,
                shmemptr, item_ct1);
        }
    }
}

// Kernel launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *basis0, const TData *w0, const TData *jac, const TData *in,
    TData *out, const TData scale = 1.0)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        IProductWRTBaseSharedMemorySize<Implementation>(nq0, nm0);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nq0);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
                 IProductWRTBase1DKernel<Implementation, SCALE, APPEND,
                                         DEFORMED>(nm0, nq0, nelmt, basis0, w0,
                                                   jac, in, out, scale,
                                                   shmemptr, item_ct1);
             });
     }).wait();
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase1DKernel(
    const unsigned int nelmt, const TData *basis0, const TData *w0,
    const TData *jac, const TData *in, TData *out, const TData scale = 1.0)
{
    IProductWRTBase1DKernel<ExecSpace, Implementation, SCALE, APPEND, DEFORMED>(
        nm0, nq0, nelmt, basis0, w0, jac, in, out, scale);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *w0, const TData *w1, const TData *jac,
    [[maybe_unused]] TData *wsp, const TData *in, TData *out,
    const TData scale = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1,
                                                                    nm0, nm1);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nq0 * nq1);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
                 IProductWRTBase2DKernel<SHAPE_TYPE, Implementation, SCALE,
                                         APPEND, DEFORMED>(
                     nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0,
                     basis0, basis1, w0, w1, jac, wsp, in, out, scale, shmemptr,
                     item_ct1);
             });
     }).wait();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nm1, unsigned int nq0,
          unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase2DKernel(
    const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *w0, const TData *w1, const TData *jac,
    [[maybe_unused]] TData *wsp, const TData *in, TData *out,
    const TData scale = 1.0)
{
    IProductWRTBase2DKernel<SHAPE_TYPE, ExecSpace, Implementation, SCALE,
                            APPEND, DEFORMED>(
        nm0, nm1, nq0, nq1, nelmt, isModified, index0, basis0, basis1, w0, w1,
        jac, wsp, in, out, scale);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0,
    [[maybe_unused]] const unsigned int *index1,
    [[maybe_unused]] const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, [[maybe_unused]] TData *wsp,
    const TData *in, TData *out, const TData scale = 1.0)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);

    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        IProductWRTBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2, nm0, nm1, nm2);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nm0 * nm1 * nm2);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
                 IProductWRTBase3DKernel<SHAPE_TYPE, Implementation, SCALE,
                                         APPEND, DEFORMED>(
                     nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified,
                     index0, index1, index2, basis0, basis1, basis2, w0, w1, w2,
                     jac, wsp, in, out, scale, shmemptr, item_ct1);
             });
     }).wait();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool SCALE, bool APPEND, bool DEFORMED,
          unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nq0, unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void IProductWRTBase3DKernel(
    const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *index0,
    [[maybe_unused]] const unsigned int *index1,
    [[maybe_unused]] const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, [[maybe_unused]] TData *wsp,
    const TData *in, TData *out, const TData scale = 1.0)
{
    IProductWRTBase3DKernel<SHAPE_TYPE, ExecSpace, Implementation, SCALE,
                            APPEND, DEFORMED>(
        nm0, nm1, nm2, nq0, nq1, nq2, nelmt, isModified, index0, index1, index2,
        basis0, basis1, basis2, w0, w1, w2, jac, wsp, in, out, scale);
}

} // namespace Nektar::Operators::detail

#endif
