///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFacTOPKernels.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nm0)
{
    return nm0 + nm0 * nq0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nq1,
                                             const unsigned int nm0,
                                             const unsigned int nm1)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nq0 * nm0 + nq1 * nm1 + nmTot + nq0 * nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return nm0 * nq0 + nmTot * nq1 + nmTot + nm0 * nq1;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int BwdTransSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nmTot + (nq0 * nm1 * nm2) +
               (nq0 * nq1 * nm2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nmTot + (nm01 * nq2) +
               (nm0 * nq1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        return nm0 * nq0 + nm1 * nq1 + nm12 * nq2 + nmTot + (nm0 * nm1 * nq2) +
               (nm0 * nq1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nmTot +
               (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; p++)
        {
            tmp += in[p] * basis0[p * nq0 + i];
        }
        out[i] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nq0 * nm1; idx += stride)
    {
        const unsigned int q = idx % nm1;
        const unsigned int i = idx / nm1;
        unsigned int cnt_qp  = nm0 * q;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
        {
            tmp += in[cnt_qp] * basis0[p * nq0 + i];
        }
        wsp[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        unsigned int cnt_iq  = nm1 * i;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
        {
            tmp += wsp[cnt_iq] * basis1[q * nq1 + j];
        }
        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransTriSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int j = idx / nm0;
        unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            tmp += in[mode_pq] * basis1[mode_pq * nq1 + j];
        }
        wsp[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;
        unsigned int cnt_jp  = nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            tmp += in[1] * basis0[nq0 + i] * basis1[nq1 + j];
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_jp)
        {
            tmp += wsp[cnt_jp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransHexSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nq0 * nm1 * nm2; idx += stride)
    {
        const unsigned int q = idx % nm1;
        const unsigned int r = (idx / nm1) % nm2;
        const unsigned int i = idx / (nm1 * nm2);
        unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
        {
            tmp += in[cnt_rqp] * basis0[p * nq0 + i];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nq0 * nq1 * nm2; idx += stride)
    {
        const unsigned int r = idx % nm2;
        const unsigned int i = (idx / nm2) % nq0;
        const unsigned int j = idx / (nm2 * nq0);
        unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
        {
            tmp += wsp0[cnt_irq] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq1 * nq0);
        unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
        {
            tmp += wsp1[cnt_jir] * basis2[r * nq2 + k];
        }
        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransTetSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const unsigned int *__restrict__ pindex,
    const unsigned int *__restrict__ qindex, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp0, TData *__restrict__ wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nm01 * nq2; idx += stride)
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
#pragma unroll
        for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
        {
            tmp += in[mode_pqr] * basis2[k + nq2 * mode2];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int p = idx % nm0;
        const unsigned int j = (idx / nm0) % nq1;
        const unsigned int k = idx / (nm0 * nq1);
        unsigned int mode_pq = (2u * nm1 - p + 1u) * p / 2u;
        unsigned int cnt_kpq = nm01 * k + mode_pq;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1 - p; ++q, ++cnt_kpq, ++mode_pq)
        {
            tmp += wsp0[cnt_kpq] * basis1[mode_pq * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            // top vertex
            tmp += basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k] * in[1];

            // bottom vertex
            TData tmp1 = basis2[k] * in[nm2];

            // singular edge
#pragma unroll
            for (unsigned int r = 1u; r < nm2 - 1u; ++r)
            {
                tmp1 += basis2[(r + 1u) * nq2 + k] * in[nm2 + r];
            }
            tmp += basis1[nq1 + j] * basis0[nq0 + i] * tmp1;
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp1[mode_kjp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransPrismSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
    {
        const unsigned int q  = idx % nm1;
        const unsigned int p  = (idx / nm1) % nm0;
        const unsigned int k  = idx / (nm1 * nm0);
        unsigned int mode_pr  = (2u * nm2 - p + 1u) * p / 2u;
        unsigned int mode_pqr = mode_pr * nm1 + (nm2 - p) * q;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr, ++mode_pr)
        {
            tmp += in[mode_pqr] * basis2[mode_pr * nq2 + k];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int p  = idx % nm0;
        const unsigned int j  = (idx / nm0) % nq1;
        const unsigned int k  = idx / (nm0 * nq1);
        unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
        {
            tmp += wsp0[mode_kpq] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
#pragma unroll
            for (unsigned int q = 0u; q < nm1; ++q)
            {
                tmp += in[q * nm2 + 1u] * basis1[q * nq1 + j];
            }
            tmp *= basis2[nq2 + k] * basis0[nq0 + i];
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp1[mode_kjp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransPyrSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const bool isModified,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp0, TData *__restrict__ wsp1,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    // direction 2
    for (unsigned int idx = idx0; idx < nm0 * nm1 * nq2; idx += stride)
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
        mode_pqr += (q < p)
                        ? q * (nm2 - p)
                        : p * (nm2 - p) +
                              ((2u * (nm2 - p) - (q - p) + 1u) * (q - p)) / 2u;
        mode2 += mode_pqr;

        TData tmp             = 0.0;
        const unsigned ulimit = (q < p) ? nm2 - p : nm2 - q;
#pragma unroll
        for (unsigned int r = 0u; r < ulimit; ++r, ++mode2, ++mode_pqr)
        {
            tmp += in[mode_pqr] * basis2[mode2 * nq2 + k];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 1
    for (unsigned int idx = idx0; idx < nm0 * nq1 * nq2; idx += stride)
    {
        const unsigned int p  = idx % nm0;
        const unsigned int j  = (idx / nm0) % nq1;
        const unsigned int k  = idx / (nm0 * nq1);
        unsigned int mode_kpq = nm0 * nm1 * k + nm1 * p;

        TData tmp = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++mode_kpq)
        {
            tmp += wsp0[mode_kpq] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    // direction 0
    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i  = idx % nq0;
        const unsigned int j  = (idx / nq0) % nq1;
        const unsigned int k  = idx / (nq0 * nq1);
        unsigned int mode_kjp = nm0 * nq1 * k + nm0 * j;

        TData tmp = 0.0;

        if (isModified)
        {
            // top vertex
            tmp += basis0[i] * basis1[nq1 + j];
            tmp += basis0[nq0 + i] * basis1[j];
            tmp += basis0[nq0 + i] * basis1[nq1 + j];
            tmp *= basis2[nq2 + k] * in[1];
        }

#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++mode_kjp)
        {
            tmp += wsp1[mode_kjp] * basis0[p * nq0 + i];
        }

        out[idx] = tmp;
    }

    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransSumFacTOP1DKernel(
    const unsigned int nm0, const unsigned int nq0, const size_t nelmt,
    const TData *__restrict__ basis0, const TData *__restrict__ in,
    TData *__restrict__ out,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nm0 * e;
        TData *outptr      = out + nq0 * e;
        BwdTransSegSumFacTOPKernel(nm0, nq0, basis0, inptr, outptr,
                                   threadBlock);
        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void BwdTransSumFacTOP2DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const size_t nelmt,
    const bool isModified, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ nodToMod,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1;

    unsigned int offset{0}, nmode0{0}, nmode1{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = nm1 * nq0;
        nmode0 = nm0;
        nmode1 = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nmTot;
    }

    TData *s_wsp0   = (TData *)shmemptr;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_basis0 = s_wsp1 + offset;
    TData *s_basis1 = s_basis0 + nm0 * nq0;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            // Nodal to Modal in shapred memory
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, s_wsp0, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                s_wsp0[idx] = inptr[idx];
            }
            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                        s_basis1, s_wsp0, outptr, s_wsp1,
                                        threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                       s_basis0, s_basis1, s_wsp0, outptr,
                                       s_wsp1, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void BwdTransSumFacTOP3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ nodToMod,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp,
    [[maybe_unused]] unsigned char *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int offset0{0}, offset1{0}, nmode0{0}, nmode1{0}, nmode2{0};
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = nq0 * nm1 * nm2;
        offset1 = nq0 * nq1 * nm2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *s_wsp0   = (TData *)shmemptr;
    TData *s_wsp1   = s_wsp0 + nmTot;
    TData *s_wsp2   = s_wsp1 + offset0;
    TData *s_basis0 = s_wsp2 + offset1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
    {
        s_basis2[idx] = basis2[idx];
    }

    size_t e = getBlockIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const TData *inptr = in + nmTot * e;
        TData *outptr      = out + nqTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism ||
                      SHAPE_TYPE == LibUtilities::NodalTet)
        {
            // Nodal to Modal in shapred memory
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, s_wsp0, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                s_wsp0[idx] = inptr[idx];
            }
            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       s_basis0, s_basis1, s_basis2, s_wsp0,
                                       outptr, s_wsp1, s_wsp2, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       isModified, index0, index1, s_basis0,
                                       s_basis1, s_basis2, s_wsp0, outptr,
                                       s_wsp1, s_wsp2, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                         isModified, s_basis0, s_basis1,
                                         s_basis2, s_wsp0, outptr, s_wsp1,
                                         s_wsp2, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       isModified, s_basis0, s_basis1, s_basis2,
                                       s_wsp0, outptr, s_wsp1, s_wsp2,
                                       threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    BwdTrans1DKernelLauncher(const unsigned int nm0, const unsigned int nq0,
                             const size_t nelmt,
                             const TData *__restrict__ basis0,
                             const TData *__restrict__ in,
                             TData *__restrict__ out, unsigned char *shmemptr,
                             const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    BwdTransSumFacTOP1DKernel(nm0, nq0, nelmt, basis0, in, out, shmemptr,
                              threadBlock);
}

// Size based template version.
template <
    typename Implementation, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock)
        BwdTrans1DKernelLauncher(const size_t nelmt,
                                 const TData *__restrict__ basis0,
                                 const TData *__restrict__ in,
                                 TData *__restrict__ out,
                                 unsigned char *shmemptr,
                                 const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    BwdTransSumFacTOP1DKernel(nm0, nq0, nelmt, basis0, in, out, shmemptr,
                              threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    BwdTrans2DKernelLauncher(
        const unsigned int nm0, const unsigned int nm1,
        const unsigned int nmTot, const unsigned int nq0,
        const unsigned int nq1, const size_t nelmt, const bool isModified,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    BwdTransSumFacTOP2DKernel<SHAPE_TYPE>(nm0, nm1, nmTot, nq0, nq1, nelmt,
                                          isModified, basis0, basis1, nodToMod,
                                          in, out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TthreadBlock,
          typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) BwdTrans2DKernelLauncher(
        const size_t nelmt, const bool isModified,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    BwdTransSumFacTOP2DKernel<SHAPE_TYPE>(nm0, nm1, nmTot, nq0, nq1, nelmt,
                                          isModified, basis0, basis1, nodToMod,
                                          in, out, wsp, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    BwdTrans3DKernelLauncher(
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nmTot, const unsigned int nq0,
        const unsigned int nq1, const unsigned int nq2, const size_t nelmt,
        const bool isModified, const unsigned int *index0,
        const unsigned int *index1, const TData *__restrict__ basis0,
        const TData *__restrict__ basis1, const TData *__restrict__ basis2,
        const TData *__restrict__ nodToMod, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    BwdTransSumFacTOP3DKernel<SHAPE_TYPE>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        basis0, basis1, basis2, nodToMod, in, out, wsp, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) BwdTrans3DKernelLauncher(
        const size_t nelmt, const bool isModified, const unsigned int *index0,
        const unsigned int *index1, const TData *__restrict__ basis0,
        const TData *__restrict__ basis1, const TData *__restrict__ basis2,
        const TData *__restrict__ nodToMod, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    BwdTransSumFacTOP3DKernel<SHAPE_TYPE>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        basis0, basis1, basis2, nodToMod, in, out, wsp, shmemptr, threadBlock);
}

#endif

} // namespace Nektar::Operators::detail
