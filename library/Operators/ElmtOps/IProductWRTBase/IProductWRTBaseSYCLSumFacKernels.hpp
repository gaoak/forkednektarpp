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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseSegKernel(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    TData *s_basis0 = SHMEM ? shared : (TData *)basis0;
    TData *s_w0     = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)w0;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                sum +=
                    in[index] * s_basis0[p * nq0 + i] * jac[jacindex] * s_w0[i];
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseSegKernel_QP(
    const unsigned int nm0, const unsigned int nq0, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ w0,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    TData *s_wsp0   = shared;
    TData *s_basis0 = SHMEM ? s_wsp0 + nq0 : (TData *)basis0;
    TData *s_w0     = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)w0;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }
    }

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
                sum += s_wsp0[i] * s_basis0[p * nq0 + i] * s_w0[i];
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

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_w0              = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                    sum += in[index] * s_basis0[p * nq0 + i] * jac[jacindex] *
                           s_w0[i];
                }
                wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] = sum;
            }

            for (unsigned int q = 0u; q < nm1; ++q)
            {
                TData sum = 0.0;
                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    sum += wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] *
                           s_basis1[q * nq1 + j] * s_w1[j];
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseQuadKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_w0              = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;

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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(1); p < nm0;
             p += item_ct1.get_local_range(1))
        {
            for (unsigned int j = item_ct1.get_local_id(2); j < nq1;
                 j += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_pj = nq1 * p + j;
                unsigned int cnt_ji       = nq0 * j;

                TData sum = 0.0;
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
                }
                s_wsp1[cnt_pj] = sum;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
             q += item_ct1.get_local_range(1))
        {
            for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                 p += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_pq = nm0 * q + p;
                const unsigned int index  = outoffset + cnt_pq;
                unsigned int cnt_pj       = nq1 * p;

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
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseQuadKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_basis0          = SHMEM ? s_wsp1 + nm0 * nq1 : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_w0              = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;

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

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseTriKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    TData *__restrict__ wsp, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_w0              = SHMEM ? s_basis1 + nmTot * nq1 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nmTot * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                    sum += in[index] * s_basis0[p * nq0 + i] * jac[jacindex] *
                           s_w0[i];
                }
                wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] = sum;
            }

            for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
            {
                TData sum = 0.0;
                for (unsigned int j = 0u; j < nq1; ++j)
                {
                    sum += wsp[nq1 * warpsize * iwarp + warpsize * j + ilane] *
                           s_basis1[mode_pq * nq1 + j] * s_w1[j];
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
        if (correct)
        {
            TData iprod_01 = 0.0;
            for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
            {
                const unsigned int index    = nqTot * warpsize * iwarp + ilane;
                const unsigned int jacindex = DEFORMED ? index : e;

                TData tmp = s_w1[j] * s_basis1[nq1 + j];
                if constexpr (!DEFORMED)
                {
                    tmp *= jac[jacindex];
                }

                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                    const unsigned int jacindex = DEFORMED ? index : e;

                    TData prod = in[index] * tmp * s_w0[i];
                    if constexpr (DEFORMED)
                    {
                        prod *= jac[jacindex];
                    }
                    iprod_01 += prod * s_basis0[nq0 + i];
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseTriKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{

    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_iprod_01        = s_wsp1 + nm0 * nq1;
    TData *s_basis0          = SHMEM ? s_iprod_01 + 1u : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_w0              = SHMEM ? s_basis1 + nmTot * nq1 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;

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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

        // Copy to shared memory.
        const unsigned int idx0 =
            item_ct1.get_local_range(2) * item_ct1.get_local_id(1) +
            item_ct1.get_local_id(2);
        const unsigned int stride =
            item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
        for (unsigned int idx = idx0; idx < nqTot; idx += stride)
        {
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(1); p < nm0;
             p += item_ct1.get_local_range(1))
        {
            for (unsigned int j = item_ct1.get_local_id(2); j < nq1;
                 j += item_ct1.get_local_range(2))
            {
                const unsigned int cnt_pj = nq1 * p + j;
                unsigned int cnt_ji       = nq0 * j;

                TData sum = 0.0;
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
                {
                    sum += s_wsp0[cnt_ji] * s_basis0[p * nq0 + i] * s_w0[i];
                }
                s_wsp1[cnt_pj] = sum;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(1); p < nm0;
             p += item_ct1.get_local_range(1))
        {
            for (unsigned int q = item_ct1.get_local_id(2); q < nm1 - p;
                 q += item_ct1.get_local_range(2))
            {
                const unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u + q;
                const unsigned int index   = outoffset + mode_pq;
                unsigned int cnt_pj        = nq1 * p;

                TData sum = 0.0;
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pj)
                {
                    sum +=
                        s_wsp1[cnt_pj] * s_basis1[mode_pq * nq1 + j] * s_w1[j];
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

        // Correction for singular vertex in collpased coordinates.
        // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
        // With contributions from every quadrature point
        if (correct)
        {
            if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0)
            {
                *s_iprod_01 = 0.0;
            }

            // s_iprod_01 is not an array so can get away with allocating
            // atomic_ref once
            sycl::atomic_ref<TData, sycl::memory_order_relaxed,
                             sycl::memory_scope_work_group,
                             sycl::access::address_space::local_space>
                aref(s_iprod_01[0]);

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                 j += item_ct1.get_local_range(1))
            {
                TData tmp = s_w1[j] * s_basis1[nq1 + j];
                for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                     i += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_ji = nq0 * j + i;
                    TData prod                = s_wsp0[cnt_ji] * tmp * s_w0[i];
                    aref.fetch_add(prod * s_basis0[nq0 + i]);
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0)
            {
                const unsigned int index = outoffset + 1u;
                if constexpr (SCALE)
                {
                    out[index] += (*s_iprod_01) * scale;
                }
                else
                {
                    out[index] += (*s_iprod_01);
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseTriKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool correct, const unsigned int *__restrict__ pindex,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_iprod_01        = s_wsp1 + nm0 * nq1;
    TData *s_basis0          = SHMEM ? s_iprod_01 + 1u : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_w0              = SHMEM ? s_basis1 + nmTot * nq1 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;

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
        if (correct)
        {
            if (item_ct1.get_local_id(2) == 0)
            {
                *s_iprod_01 = 0.0;
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq1;
                 idx += item_ct1.get_local_range(2))
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = idx / nq0;
                TData tmp            = s_w1[j] * s_basis1[nq1 + j];
                TData prod           = s_wsp0[idx] * tmp * s_w0[i];
                atomic_add<NektarSpaces::SYCL, NektarSpaces::LocalScope>(
                    s_iprod_01, prod * s_basis0[nq0 + i]);
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (item_ct1.get_local_id(2) == 0)
            {
                const unsigned int index = outoffset + 1u;
                if constexpr (SCALE)
                {
                    out[index] += (*s_iprod_01) * scale;
                }
                else
                {
                    out[index] += (*s_iprod_01);
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseHexKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0              = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2              = SHMEM ? s_w1 + nq1 : (TData *)w2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                               s_basis2[r * nq2 + k] * s_w2[k];
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

/**
NOTE: The total declared local variable size exceeds 128 bytes. This may cause
high register pressure with a sub-group size of 32 depending on hardware.
Leaving it for now but will have to profile later and make appropriate changes
if necessary.
*/
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseHexKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_wsp2            = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

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
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int k = item_ct1.get_local_id(1); k < nq2;
                 k += item_ct1.get_local_range(1))
            {
                for (unsigned int j = item_ct1.get_local_id(2); j < nq1;
                     j += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pkj = nq2 * nq1 * p + nq1 * k + j;
                    unsigned int cnt_kji       = nq0 * nq1 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int k = item_ct1.get_local_id(2); k < nq2;
                     k += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;
                    unsigned int cnt_pkj       = nq2 * nq1 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_wsp1[cnt_pkj] * s_basis1[q * nq1 + j] * s_w1[j];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int r = item_ct1.get_local_id(0); r < nm2;
             r += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int p = item_ct1.get_local_id(2); p < nm0;
                     p += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;
                    const unsigned int index   = outoffset + cnt_rqp;
                    unsigned int cnt_pqk       = nm1 * nq2 * p + nq2 * q;

                    TData sum = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum +=
                            s_wsp2[cnt_pqk] * s_basis2[r * nq2 + k] * s_w2[k];
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

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseHexKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_wsp2            = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nm2 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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

/**
NOTE: The total declared local variable size exceeds 128 bytes. This may cause
high register pressure with a sub-group size of 32 depending on hardware.
Leaving it for now but will have to profile later and make appropriate changes
if necessary.
*/
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseTetKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    TData *s_basis0         = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1         = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2         = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;
    TData *s_w0             = SHMEM ? s_basis2 + nmode2 * nq2 : (TData *)w0;
    TData *s_w1             = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2             = SHMEM ? s_w1 + nq1 : (TData *)w2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                               s_basis2[mode2 * nq2 + k] * s_w2[k];
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
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;

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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

/**
NOTE: The total declared local variable size exceeds 128 bytes. This may cause
high register pressure with a sub-group size of 32 depending on hardware.
Leaving it for now but will have to profile later and make appropriate changes
if necessary.
*/
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseTetKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    TData *s_prod           = shared;
    TData *s_wsp0           = s_prod + nm2;
    TData *s_wsp1           = s_wsp0 + nqTot;
    TData *s_wsp2           = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0         = SHMEM ? s_wsp2 + nm01 * nq2 : (TData *)basis0;
    TData *s_basis1         = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2         = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;
    TData *s_w0             = SHMEM ? s_basis2 + nmode2 * nq2 : (TData *)w0;
    TData *s_w1             = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2             = SHMEM ? s_w1 + nq1 : (TData *)w2;

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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

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
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int k = item_ct1.get_local_id(1); k < nq2;
                 k += item_ct1.get_local_range(1))
            {
                for (unsigned int j = item_ct1.get_local_id(2); j < nq1;
                     j += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;
                    unsigned int cnt_kji       = nq0 * nq1 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[i + nq0 * p] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1 - p;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int k = item_ct1.get_local_id(2); k < nq2;
                     k += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_pq =
                        (2u * nm1 - p + 1u) * p / 2u + q;
                    unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k += s_basis1[mode_pq * nq1 + j] * s_wsp1[cnt_pkj] *
                                 s_w1[j];
                    }
                    s_wsp2[mode_pq * nq2 + k] = sum_k;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1 - p;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int r = item_ct1.get_local_id(2); r < nm2 - p - q;
                     r += item_ct1.get_local_range(2))
                {
                    const unsigned int mode_pq =
                        (2u * nm1 - p + 1u) * p / 2u + q;
                    unsigned int mode2 = (2u * (nm2 - p) - q + 1u) * q;
                    mode2 += nm2 * (nm2 + 1u) * p;
                    mode2 -= (2u * nm2 + 1u) * (p - 1u) * p / 2u;
                    mode2 += (p - 1u) * p * (2u * p - 1u) / 6u;
                    mode2 /= 2u;
                    const unsigned int mode_pqr =
                        mode2 - ((nm2 > nm1)
                                     ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u
                                     : 0u);
                    const unsigned int index = outoffset + mode_pqr + r;

                    TData tmp = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        tmp += s_wsp2[mode_pq * nq2 + k] *
                               s_basis2[(mode2 + r) * nq2 + k] * s_w2[k];
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
            }
        }

        // Add correction for collapsed coordinate.
        if (correct)
        {
            if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0)
            {
                for (unsigned int r = item_ct1.get_local_id(0); r < nm2;
                     r += item_ct1.get_local_range(0))
                {
                    s_prod[r] = 0.0;
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
                 k += item_ct1.get_local_range(0))
            {
                TData tmpQ2 = s_w2[k];
                for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                     j += item_ct1.get_local_range(1))
                {
                    TData tmpQ1 = tmpQ2 * s_w1[j];
                    for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                         i += item_ct1.get_local_range(2))
                    {
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
                        Nektar::atomic_add<NektarSpaces::SYCL,
                                           NektarSpaces::LocalScope>(
                            s_prod + nm2 - 1, tmp);

                        // bottom vertex
                        tmp = s_basis0[nq0 + i] * s_basis1[nq1 + j] *
                              s_basis2[k] * s_wsp0[cnt_kji] * tmpQ;
                        Nektar::atomic_add<NektarSpaces::SYCL,
                                           NektarSpaces::LocalScope>(s_prod,
                                                                     tmp);

                        // singular edge
                        for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                        {
                            tmp = s_basis2[(r + 1) * nq2 + k] *
                                  s_basis1[nq1 + j] * s_basis0[nq0 + i] *
                                  s_wsp0[cnt_kji] * tmpQ;
                            Nektar::atomic_add<NektarSpaces::SYCL,
                                               NektarSpaces::LocalScope>(
                                s_prod + r, tmp);
                        }
                    }
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            if constexpr (SCALE)
            {
                if (item_ct1.get_local_id(2) == 0 &&
                    item_ct1.get_local_id(1) == 0 &&
                    item_ct1.get_local_id(0) == 0)
                {
                    out[outoffset + 1] += s_prod[nm2 - 1] * scale;
                }
                if (item_ct1.get_local_id(0) == 0 &&
                    item_ct1.get_local_id(1) == 0)
                {
                    for (unsigned int r = item_ct1.get_local_id(2);
                         r < nm2 - 1u; r += item_ct1.get_local_range(2))
                    {
                        out[outoffset + nm2 + r] += s_prod[r] * scale;
                    }
                }
            }
            else
            {
                if (item_ct1.get_local_id(2) == 0 &&
                    item_ct1.get_local_id(1) == 0 &&
                    item_ct1.get_local_id(0) == 0)
                {
                    out[outoffset + 1] += s_prod[nm2 - 1];
                }
                if (item_ct1.get_local_id(0) == 0 &&
                    item_ct1.get_local_id(1) == 0)
                {
                    for (unsigned int r = item_ct1.get_local_id(2);
                         r < nm2 - 1u; r += item_ct1.get_local_range(2))
                    {
                        out[outoffset + nm2 + r] += s_prod[r];
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
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBaseTetKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const unsigned int *__restrict__ pindex1,
    const unsigned int *__restrict__ pindex2,
    const unsigned int *__restrict__ qindex2, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    TData *s_prod           = shared;
    TData *s_wsp0           = s_prod + nm2;
    TData *s_wsp1           = s_wsp0 + nqTot;
    TData *s_wsp2           = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm01 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nmode2 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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
        if (correct)
        {
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm2;
                 idx += item_ct1.get_local_range(2))
            {
                s_prod[idx] = 0.0;
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int idx = item_ct1.get_local_id(2);
                 idx < nq0 * nq1 * nq2; idx += item_ct1.get_local_range(2))
            {
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
                atomic_add<NektarSpaces::SYCL, NektarSpaces::LocalScope>(
                    s_prod + nm2 - 1, tmp);

                // bottom vertex
                tmp = s_basis0[nq0 + i] * s_basis1[nq1 + j] * s_basis2[k] *
                      s_wsp0[idx] * tmpQ;
                atomic_add<NektarSpaces::SYCL, NektarSpaces::LocalScope>(s_prod,
                                                                         tmp);

                // singular edge
                for (unsigned int r = 1u; r < nm2 - 1u; ++r)
                {
                    tmp = s_basis2[(r + 1) * nq2 + k] * s_basis1[nq1 + j] *
                          s_basis0[nq0 + i] * s_wsp0[idx] * tmpQ;
                    atomic_add<NektarSpaces::SYCL, NektarSpaces::LocalScope>(
                        s_prod + r, tmp);
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            if constexpr (SCALE)
            {
                if (item_ct1.get_local_id(2) == 0)
                {
                    out[outoffset + 1] += s_prod[nm2 - 1] * scale;
                }
                for (unsigned int idx = item_ct1.get_local_id(2);
                     idx < nm2 - 1u; idx += item_ct1.get_local_range(2))
                {
                    out[outoffset + nm2 + idx] += s_prod[idx] * scale;
                }
            }
            else
            {
                if (item_ct1.get_local_id(2) == 0)
                {
                    out[outoffset + 1] += s_prod[nm2 - 1];
                }
                for (unsigned int idx = item_ct1.get_local_id(2);
                     idx < nm2 - 1u; idx += item_ct1.get_local_range(2))
                {
                    out[outoffset + nm2 + idx] += s_prod[idx];
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
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBasePrismKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    TData *s_basis0          = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1          = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2          = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0              = SHMEM ? s_basis2 + nm02 * nq2 : (TData *)w0;
    TData *s_w1              = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2              = SHMEM ? s_w1 + nq1 : (TData *)w2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        for (unsigned int idx = idx0; idx < nm0 * nq0; idx += stride)
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = idx0; idx < nm1 * nq1; idx += stride)
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = idx0; idx < nm02 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                                 s_basis2[(mode_pr + r) * nq2 + k] * s_w2[k];
                    }

                    if (SCALE)
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
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;
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

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

/**
NOTE: The total declared local variable size exceeds 128 bytes. This may cause
high register pressure with a sub-group size of 32 depending on hardware.
Leaving it for now but will have to profile later and make appropriate changes
if necessary.
*/
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBasePrismKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_wsp2            = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nm02 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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

        for (unsigned int idx = idx0; idx < nm02 * nq2; idx += stride)
        {
            s_basis2[idx] = basis2[idx];
        }

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

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
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int k = item_ct1.get_local_id(1); k < nq2;
                 k += item_ct1.get_local_range(1))
            {
                for (unsigned int j = item_ct1.get_local_id(2); j < nq1;
                     j += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;
                    unsigned int cnt_kji       = nq1 * nq0 * k + nq0 * j;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int k = item_ct1.get_local_id(2); k < nq2;
                     k += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;
                    unsigned int cnt_pkj       = nq1 * nq2 * p + nq1 * k;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int r = item_ct1.get_local_id(2); r < nm2 - p;
                     r += item_ct1.get_local_range(2))
                {
                    unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q;
                    unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;
                    const unsigned int mode_pqr =
                        mode_pr * nm1 + (nm2 - p) * q + r;
                    const unsigned int index = outoffset + mode_pqr;

                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                    {
                        sum_k += s_basis2[(mode_pr + r) * nq2 + k] * s_w2[k] *
                                 s_wsp2[cnt_pqk];
                    }

                    if (SCALE)
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

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Add correction for collapsed coordinate.
        if (correct)
        {
            if (item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(0) == 0)
            {
                for (unsigned int q = item_ct1.get_local_id(2); q < nm1;
                     q += item_ct1.get_local_range(2))
                {
                    s_wsp2[q] = 0.0;
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
                 k += item_ct1.get_local_range(0))
            {
                TData k_weight = s_w2[k];
                for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                     j += item_ct1.get_local_range(1))
                {
                    TData kj_weight = k_weight * s_w1[j];
                    for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                         i += item_ct1.get_local_range(2))
                    {
                        const unsigned int cnt_kji =
                            nq1 * nq0 * k + nq0 * j + i;
                        TData prod = kj_weight * s_w0[i] * s_wsp0[cnt_kji];
                        for (unsigned int q = 0u; q < nm1; ++q)
                        {
                            Nektar::atomic_add<NektarSpaces::SYCL,
                                               NektarSpaces::LocalScope>(
                                s_wsp2 + q, prod * s_basis2[nq2 + k] *
                                                s_basis1[q * nq1 + j] *
                                                s_basis0[nq0 + i]);
                        }
                    }
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            if (item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(0) == 0)
            {
                for (unsigned int q = item_ct1.get_local_id(2); q < nm1;
                     q += item_ct1.get_local_range(2))
                {
                    const unsigned int index = outoffset + nm2 * q + 1u;
                    if constexpr (SCALE)
                    {
                        out[index] += s_wsp2[q] * scale;
                    }
                    else
                    {
                        out[index] += s_wsp2[q];
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
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBasePrismKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex,
    const unsigned int *__restrict__ rindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nm02  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    TData *s_wsp0            = shared;
    TData *s_wsp1            = s_wsp0 + nqTot;
    TData *s_wsp2            = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nm02 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Add correction for collapsed coordinate.
        if (correct)
        {
            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1;
                 idx += item_ct1.get_local_range(2))
            {
                s_wsp2[idx] = 0.0;
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nqTot;
                 idx += item_ct1.get_local_range(2))
            {
                const unsigned int i = idx % nq0;
                const unsigned int j = (idx / nq0) % nq1;
                const unsigned int k = idx / (nq0 * nq1);
                TData k_weight       = s_w2[k];
                TData kj_weight      = k_weight * s_w1[j];
                TData prod           = kj_weight * s_w0[i] * s_wsp0[idx];
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    atomic_add<NektarSpaces::SYCL, NektarSpaces::LocalScope>(
                        s_wsp2 + q, prod * s_basis2[nq2 + k] *
                                        s_basis1[q * nq1 + j] *
                                        s_basis0[nq0 + i]);
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int idx = item_ct1.get_local_id(2); idx < nm1;
                 idx += item_ct1.get_local_range(2))
            {
                const unsigned int index = outoffset + nm2 * idx + 1u;
                if constexpr (SCALE)
                {
                    out[index] += s_wsp2[idx] * scale;
                }
                else
                {
                    out[index] += s_wsp2[idx];
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
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBasePyrKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, TData *__restrict__ wsp,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_basis0 = SHMEM ? shared : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nmode2 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

    // Copy to shared memory.
    if constexpr (SHMEM)
    {
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    unsigned int e = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

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
                    wsp1[nq2 * warpsize * iwarp + warpsize * k + ilane] = sum_k;
                }

                for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
                {
                    TData sum_k = 0.0;
                    for (unsigned int k = 0u; k < nq2; ++k)
                    {
                        sum_k += wsp1[nq2 * warpsize * iwarp + warpsize * k +
                                      ilane] *
                                 s_basis2[mode2 * nq2 + k] * s_w2[k];
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
                                 s_basis1[q * nq1 + j] * s_w1[j];
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
                                 s_basis2[mode2 * nq2 + k] * s_w2[k];
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
                        const unsigned int index = nqTot * warpsize * iwarp +
                                                   warpsize * cnt_kji + ilane;

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
            if constexpr (SCALE)
            {
                out[index] += prod * scale;
            }
            else
            {
                out[index] += prod;
            }
        }

        e += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

/**
NOTE: The total declared local variable size exceeds 128 bytes. This may cause
high register pressure with a sub-group size of 32 depending on hardware.
Leaving it for now but will have to profile later and make appropriate changes
if necessary.
*/
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBasePyrKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ jac, const TData *__restrict__ in,
    TData *__restrict__ out, const sycl::nd_item<3> &item_ct1,
    TData *__restrict__ shared, const TData scale = 1.0)
{

    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_prod   = shared;
    TData *s_wsp0   = s_prod + 1u;
    TData *s_wsp1   = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nmode2 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_w0[idx] = w0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_w1[idx] = w1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_w2[idx] = w2[idx];
        }
    }

    unsigned int e = item_ct1.get_group(2);

    while (e < nelmt)
    {
        const unsigned int inoffset  = nqTot * e;
        const unsigned int outoffset = nmTot * e;

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
            const unsigned int index    = inoffset + idx;
            const unsigned int jacindex = DEFORMED ? index : e;
            s_wsp0[idx]                 = in[index] * jac[jacindex];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int k = item_ct1.get_local_id(1); k < nq2;
                 k += item_ct1.get_local_range(1))
            {
                for (unsigned int j = item_ct1.get_local_id(2); j < nq1;
                     j += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pkj = nq1 * nq2 * p + nq1 * k + j;
                    unsigned int cnt_kji       = k * nq1 * nq0 + j * nq0;

                    TData sum_kj = 0.0;
                    for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                    {
                        sum_kj +=
                            s_wsp0[cnt_kji] * s_basis0[nq0 * p + i] * s_w0[i];
                    }
                    s_wsp1[cnt_pkj] = sum_kj;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                for (unsigned int k = item_ct1.get_local_id(2); k < nq2;
                     k += item_ct1.get_local_range(2))
                {
                    const unsigned int cnt_pqk = nm1 * nq2 * p + nq2 * q + k;
                    unsigned int cnt_pkj       = nq1 * nq2 * p + k * nq1;

                    TData sum_k = 0.0;
                    for (unsigned int j = 0u; j < nq1; ++j, ++cnt_pkj)
                    {
                        sum_k +=
                            s_basis1[q * nq1 + j] * s_w1[j] * s_wsp1[cnt_pkj];
                    }
                    s_wsp2[cnt_pqk] = sum_k;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (unsigned int p = item_ct1.get_local_id(0); p < nm0;
             p += item_ct1.get_local_range(0))
        {
            for (unsigned int q = item_ct1.get_local_id(1); q < nm1;
                 q += item_ct1.get_local_range(1))
            {
                unsigned int mode2 =
                    (nm2 > nm1) ? p * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u : 0u;
                unsigned int mode_pq = nm1 * (2u * nm2 + 1u - nm1) * p;
                mode_pq -= (p - 1u) * p / 2u;
                mode_pq -= (p - 1u) * p * (2u * p - 1u) / 6u;
                mode_pq /= 2u;

                if (q < p)
                {
                    for (unsigned int r = item_ct1.get_local_id(2); r < nm2 - p;
                         r += item_ct1.get_local_range(2))
                    {
                        const unsigned int mode_pqr =
                            mode_pq + q * (nm2 - p) + r;
                        mode2 += mode_pqr;
                        const unsigned int index = outoffset + mode_pqr;
                        unsigned int cnt_pqk     = nm1 * nq2 * p + nq2 * q;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += s_basis2[mode2 * nq2 + k] * s_w2[k] *
                                     s_wsp2[cnt_pqk];
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
                else
                {
                    for (unsigned int r = item_ct1.get_local_id(2); r < nm2 - q;
                         r += item_ct1.get_local_range(2))
                    {
                        unsigned int cnt_pqk  = nm1 * nq2 * p + nq2 * q;
                        unsigned int mode_pqr = mode_pq + p * (nm2 - p);
                        mode_pqr +=
                            ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u +
                            r;
                        mode2 += mode_pqr;
                        const unsigned int index = outoffset + mode_pqr;

                        TData sum_k = 0.0;
                        for (unsigned int k = 0u; k < nq2; ++k, ++cnt_pqk)
                        {
                            sum_k += s_basis2[mode2 * nq2 + k] * s_w2[k] *
                                     s_wsp2[cnt_pqk];
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
        }

        // Add correction for collapsed coordinate.
        if (correct)
        {
            if (item_ct1.get_local_id(2) == 0 &&
                item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(0) == 0)
            {
                (*s_prod) = 0.0;
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int k = item_ct1.get_local_id(0); k < nq2;
                 k += item_ct1.get_local_range(0))
            {
                TData tmpQ2 = s_w2[k];
                for (unsigned int j = item_ct1.get_local_id(1); j < nq1;
                     j += item_ct1.get_local_range(1))
                {
                    TData tmpQ1 = tmpQ2 * s_w1[j];
                    for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
                         i += item_ct1.get_local_range(2))
                    {
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
                        Nektar::atomic_add<NektarSpaces::SYCL,
                                           NektarSpaces::LocalScope>(s_prod,
                                                                     tmp);
                    }
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            // add to existing entry
            if (item_ct1.get_local_id(2) == 0 &&
                item_ct1.get_local_id(1) == 0 && item_ct1.get_local_id(0) == 0)
            {
                if constexpr (SCALE)
                {
                    out[outoffset + 1] += (*s_prod) * scale;
                }
                else
                {
                    out[outoffset + 1] += (*s_prod);
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
template <bool SCALE, bool APPEND, bool DEFORMED, bool SHMEM, typename TData>
void IProductWRTBasePyrKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool correct,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ jac,
    const TData *__restrict__ in, TData *__restrict__ out,
    const sycl::nd_item<3> &item_ct1, TData *__restrict__ shared,
    const TData scale = 1.0)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    TData *s_prod   = shared;
    TData *s_wsp0   = s_prod + 1u;
    TData *s_wsp1   = s_wsp0 + nq0 * nq1 * nq2;
    TData *s_wsp2   = s_wsp1 + nm0 * nq1 * nq2;
    TData *s_basis0 = SHMEM ? s_wsp2 + nm0 * nm1 * nq2 : (TData *)basis0;
    TData *s_basis1 = SHMEM ? s_basis0 + nm0 * nq0 : (TData *)basis1;
    TData *s_basis2 = SHMEM ? s_basis1 + nm1 * nq1 : (TData *)basis2;
    TData *s_w0     = SHMEM ? s_basis2 + nmode2 * nq2 : (TData *)w0;
    TData *s_w1     = SHMEM ? s_w0 + nq0 : (TData *)w1;
    TData *s_w2     = SHMEM ? s_w1 + nq1 : (TData *)w2;

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
        if (correct)
        {
            if (item_ct1.get_local_id(2) == 0)
            {
                (*s_prod) = 0.0;
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (unsigned int idx = item_ct1.get_local_id(2);
                 idx < nq0 * nq1 * nq2; idx += item_ct1.get_local_range(2))
            {
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
                atomic_add<NektarSpaces::SYCL, NektarSpaces::LocalScope>(s_prod,
                                                                         tmp);
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            // add to existing entry
            if (item_ct1.get_local_id(2) == 0)
            {
                if constexpr (SCALE)
                {
                    out[outoffset + 1] += (*s_prod) * scale;
                }
                else
                {
                    out[outoffset + 1] += (*s_prod);
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        e += item_ct1.get_group_range(2);
    }
}

// Launchers
template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, bool SHMEM, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
IProductWRTBase1DKernel(const unsigned int nm0, const unsigned int nq0,
                        const unsigned int nelmts, const TData *basis0,
                        const TData *w0, const TData *jac, const TData *in,
                        TData *out, const TData scale = 1.0)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0, NektarSpaces::SYCL::defaultBlockSize)
                   : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize =
        std::min(MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize,
                 2147483647u);

    unsigned int nshared = SHMEM ? nm0 * nq0 + nq0 : 0u;

    sycl::queue &Q = SYCLQueue::GetInstance();

    if constexpr (MULTILEVEL)
    {
        nshared += nq0;

        Q.submit([&](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                   cgh);

             cgh.parallel_for(
                 sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                       sycl::range<3>(1, 1, blocksize),
                                   sycl::range<3>(1, 1, blocksize)),
                 [=](sycl::nd_item<3> item_ct1) {
                     TData *shmPtr = shared
                                         .template get_multi_ptr<
                                             sycl::access::decorated::no>()
                                         .get();
                     IProductWRTBaseSegKernel_QP<SCALE, APPEND, DEFORMED,
                                                 SHMEM>(
                         nm0, nq0, nelmts, basis0, w0, jac, in, out, item_ct1,
                         shmPtr, scale);
                 });
         }).wait();
    }
    else
    {
        Q.submit([&](sycl::handler &cgh) {
             sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                   cgh);

             cgh.parallel_for(
                 sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                       sycl::range<3>(1, 1, blocksize),
                                   sycl::range<3>(1, 1, blocksize)),
                 [=](sycl::nd_item<3> item_ct1) {
                     TData *shmPtr = shared
                                         .template get_multi_ptr<
                                             sycl::access::decorated::no>()
                                         .get();
                     IProductWRTBaseSegKernel<SCALE, APPEND, DEFORMED, SHMEM>(
                         nm0, nq0, nelmts, basis0, w0, jac, in, out, item_ct1,
                         shmPtr, scale);
                 });
         }).wait();
    }
}

template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, bool SHMEM, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
IProductWRTBase2DKernel(LibUtilities::ShapeType shapetype,
                        const unsigned int nm0, const unsigned int nm1,
                        const unsigned int nq0, const unsigned int nq1,
                        const unsigned int nelmts, const bool correct,
                        [[maybe_unused]] const unsigned int *index0,
                        const TData *basis0, const TData *basis1,
                        const TData *w0, const TData *w1, const TData *jac,
                        TData *wsp, const TData *in, TData *out,
                        const TData scale = 1.0)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    const sycl::range<3> blocksize2d =
        sycl::range<3>(1, std::min(nq0, 16u), std::min(nq1, 16u));
    const unsigned int blocksize =
        MULTILEVEL ? std::min(nq0 * nq1, NektarSpaces::SYCL::defaultBlockSize)
                   : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize =
        std::min(MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize,
                 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    if (shapetype == LibUtilities::Quad)
    {
        const unsigned int nmTot =
            LibUtilities::StdQuadData::getNumberOfCoefficients(nm0, nm1);
        unsigned int nshared = nm0 * nq0 + nm1 * nq1 + nq0 + nq1;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 + nm0 * nq1;

            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);
#if !defined(NEKTAR_USE_QP_1D_KERNEL)
                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize2d,
                                       blocksize2d),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseQuadKernel_QP<SCALE, APPEND, DEFORMED,
                                                      SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmts, basis0, basis1,
                             w0, w1, jac, in, out, item_ct1, shmPtr, scale);
#else
                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseQuadKernel_QP_1D<SCALE, APPEND,
                                                         DEFORMED, SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmts, basis0, basis1,
                             w0, w1, jac, in, out, item_ct1, shmPtr, scale);
#endif
                     });
             }).wait();
        }
        else
        {
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseQuadKernel<SCALE, APPEND, DEFORMED,
                                                   SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmts, basis0, basis1,
                             w0, w1, jac, wsp, in, out, item_ct1, shmPtr,
                             scale);
                     });
             }).wait();
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
#if !defined(NEKTAR_USE_QP_1D_KERNEL)
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize2d,
                                       blocksize2d),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseTriKernel_QP<SCALE, APPEND, DEFORMED,
                                                     SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmts, correct, basis0,
                             basis1, w0, w1, jac, in, out, item_ct1, shmPtr,
                             scale);
                     });
             }).wait();
#else
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseTriKernel_QP_1D<SCALE, APPEND, DEFORMED,
                                                        SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmts, correct, index0,
                             basis0, basis1, w0, w1, jac, in, out, item_ct1,
                             shmPtr, scale);
                     });
             }).wait();
#endif
        }
        else
        {
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseTriKernel<SCALE, APPEND, DEFORMED,
                                                  SHMEM>(
                             nm0, nm1, nmTot, nq0, nq1, nelmts, correct, basis0,
                             basis1, w0, w1, jac, wsp, in, out, item_ct1,
                             shmPtr, scale);
                     });
             }).wait();
        }
    }
}

template <typename ExecSpace, typename Implementation, bool SCALE, bool APPEND,
          bool DEFORMED, bool SHMEM, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
IProductWRTBase3DKernel(LibUtilities::ShapeType shapetype,
                        const unsigned int nm0, const unsigned int nm1,
                        const unsigned int nm2, const unsigned int nq0,
                        const unsigned int nq1, const unsigned int nq2,
                        const unsigned int nelmts, const bool correct,
                        [[maybe_unused]] const unsigned int *index0,
                        [[maybe_unused]] const unsigned int *index1,
                        [[maybe_unused]] const unsigned int *index2,
                        const TData *basis0, const TData *basis1,
                        const TData *basis2, const TData *w0, const TData *w1,
                        const TData *w2, const TData *jac, TData *wsp,
                        const TData *in, TData *out, const TData scale = 1.0)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    const sycl::range<3> blocksize3d =
        sycl::range<3>(std::min(nq0, 8u), std::min(nq1, 8u), std::min(nq2, 8u));
    const unsigned int blocksize =
        MULTILEVEL
            ? std::min(nq0 * nq1 * nq2, NektarSpaces::SYCL::defaultBlockSize)
            : NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridsize =
        std::min(MULTILEVEL ? nelmts : (nelmts + blocksize - 1u) / blocksize,
                 2147483647u);

    sycl::queue &Q = SYCLQueue::GetInstance();

    if (shapetype == LibUtilities::Hex)
    {
        const unsigned int nmTot =
            LibUtilities::StdHexData::getNumberOfCoefficients(nm0, nm1, nm2);
        unsigned int nshared =
            SHMEM ? nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 + nq1 + nq2 : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2;

            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);
#if !defined(NEKTAR_USE_QP_1D_KERNEL)
                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();

                         IProductWRTBaseHexKernel_QP<SCALE, APPEND, DEFORMED,
                                                     SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             basis0, basis1, basis2, w0, w1, w2, jac, in, out,
                             item_ct1, shmPtr, scale);
#else
                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseHexKernel_QP_1D<SCALE, APPEND, DEFORMED,
                                                        SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             basis0, basis1, basis2, w0, w1, w2, jac, in, out,
                             item_ct1, shmPtr, scale);
#endif
                     });
             }).wait();
        }
        else
        {
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseHexKernel<SCALE, APPEND, DEFORMED,
                                                  SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             basis0, basis1, basis2, w0, w1, w2, jac, wsp, in,
                             out, item_ct1, shmPtr, scale);
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
            SHMEM ? nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nq0 + nq1 + nq2
                  : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm01 * nq2 + nm2;

#if !defined(NEKTAR_USE_QP_1D_KERNEL)
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();

                         IProductWRTBaseTetKernel_QP<SCALE, APPEND, DEFORMED,
                                                     SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, basis0, basis1, basis2, w0, w1, w2, jac,
                             in, out, item_ct1, shmPtr, scale);
                     });
             }).wait();
#else
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();

                         IProductWRTBaseTetKernel_QP_1D<SCALE, APPEND, DEFORMED,
                                                        SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, index0, index1, index2, basis0, basis1,
                             basis2, w0, w1, w2, jac, in, out, item_ct1, shmPtr,
                             scale);
                     });
             }).wait();
#endif
        }
        else
        {
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBaseTetKernel<SCALE, APPEND, DEFORMED,
                                                  SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, basis0, basis1, basis2, w0, w1, w2, jac,
                             wsp, in, out, item_ct1, shmPtr, scale);
                     });
             }).wait();
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

#if !defined(NEKTAR_USE_QP_1D_KERNEL)
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);
                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();

                         IProductWRTBasePrismKernel_QP<SCALE, APPEND, DEFORMED,
                                                       SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, basis0, basis1, basis2, w0, w1, w2, jac,
                             in, out, item_ct1, shmPtr, scale);
                     });
             }).wait();
#else
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);
                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();

                         IProductWRTBasePrismKernel_QP_1D<SCALE, APPEND,
                                                          DEFORMED, SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, index0, index1, index2, basis0, basis1,
                             basis2, w0, w1, w2, jac, in, out, item_ct1, shmPtr,
                             scale);
                     });
             }).wait();
#endif
        }
        else
        {
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBasePrismKernel<SCALE, APPEND, DEFORMED,
                                                    SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, basis0, basis1, basis2, w0, w1, w2, jac,
                             wsp, in, out, item_ct1, shmPtr, scale);
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
            SHMEM ? nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nq0 + nq1 + nq2 : 0u;

        if constexpr (MULTILEVEL)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 1u;

#if !defined(NEKTAR_USE_QP_1D_KERNEL)
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           blocksize3d,
                                       blocksize3d),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBasePyrKernel_QP<SCALE, APPEND, DEFORMED,
                                                     SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, basis0, basis1, basis2, w0, w1, w2, jac,
                             in, out, item_ct1, shmPtr, scale);
                     });
             }).wait();
#else
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBasePyrKernel_QP_1D<SCALE, APPEND, DEFORMED,
                                                        SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, index0, index1, basis0, basis1, basis2,
                             w0, w1, w2, jac, in, out, item_ct1, shmPtr, scale);
                     });
             }).wait();
#endif
        }
        else
        {
            Q.submit([&](sycl::handler &cgh) {
                 sycl::local_accessor<TData, 1> shared(sycl::range<1>(nshared),
                                                       cgh);

                 cgh.parallel_for(
                     sycl::nd_range<3>(sycl::range<3>(1, 1, gridsize) *
                                           sycl::range<3>(1, 1, blocksize),
                                       sycl::range<3>(1, 1, blocksize)),
                     [=](sycl::nd_item<3> item_ct1) {
                         TData *shmPtr = shared
                                             .template get_multi_ptr<
                                                 sycl::access::decorated::no>()
                                             .get();
                         IProductWRTBasePyrKernel<SCALE, APPEND, DEFORMED,
                                                  SHMEM>(
                             nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmts,
                             correct, basis0, basis1, basis2, w0, w1, w2, jac,
                             wsp, in, out, item_ct1, shmPtr, scale);
                     });
             }).wait();
        }
    }
}

} // namespace Nektar::Operators::detail

#endif
