///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSYCLSumFacKernels.hpp
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

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void StdAlignDerivBase1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int insize, const TData *__restrict w0,
    const TData *__restrict df, const TData *__restrict jac,
    const TData *__restrict in, TData *__restrict out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        TData sum = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            sum += df[d * warpsize + dfindex] * in[d * insize * nq0 + index];
        }

        if constexpr (DEFORMED)
        {
            out[index] = sum * jac[index] * w0[i];
        }
        else
        {
            out[index] = sum * jac[0] * w0[i];
        }
    }
}

template <bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void StdAlignDerivBase1DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const unsigned int insize, const TData *__restrict w0,
    const TData *__restrict df, const TData *__restrict jac,
    const TData *__restrict in, TData *__restrict out,
    const sycl::nd_item<3> &item_ct1)
{
    unsigned int dfsize = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nq0;
    }

    for (unsigned int i = item_ct1.get_local_id(2); i < nq0;
         i += item_ct1.get_local_range(2))
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        TData sum = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            sum += df[d * dfsize + dfindex] * in[d * insize * nq0 + i];
        }

        if constexpr (DEFORMED)
        {
            out[i] = sum * jac[i] * w0[i];
        }
        else
        {
            out[i] = sum * jac[0] * w0[i];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void StdAlignDerivBase2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const unsigned int insize,
    const TData *__restrict w0, const TData *__restrict w1,
    [[maybe_unused]] const TData *__restrict f0,
    [[maybe_unused]] const TData *__restrict f1, const TData *__restrict df,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out0, TData *__restrict out1)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            TData sum1 = 0.0, sum2 = 0.0;
            for (unsigned int d = 0; d < ncoord; ++d)
            {
                TData tmp = in[d * insize * nqTot + index];
                sum1 += df[(2u * d) * warpsize + dfindex] * tmp;
                sum2 += df[(2u * d + 1u) * warpsize + dfindex] * tmp;
            }

            TData tmpQ = w0[i] * w1[j];
            if constexpr (DEFORMED)
            {
                tmpQ *= jac[warpsize * cnt_ji + ilane];
            }
            else
            {
                tmpQ *= jac[0];
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                out0[index] = sum1 * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                out0[index] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
                out1[index] = sum2 * tmpQ;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void StdAlignDerivBase2DSumFacQPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int insize, const TData *__restrict w0,
    const TData *__restrict w1, const TData *__restrict f0,
    const TData *__restrict f1, const TData *__restrict df,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out0, TData *__restrict out1,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int nqTot = nq0 * nq1;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq1;
         idx += item_ct1.get_local_range(2))
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        TData sum1 = 0.0, sum2 = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            TData tmp = in[d * insize * nqTot + idx];
            sum1 += df[(2u * d) * dfsize + dfindex] * tmp;
            sum2 += df[(2u * d + 1u) * dfsize + dfindex] * tmp;
        }

        TData tmpQ = w0[i] * w1[j];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            out0[idx] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void StdAlignDerivBase3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int insize,
    const TData *__restrict w0, const TData *__restrict w1,
    const TData *__restrict w2, [[maybe_unused]] const TData *__restrict f0,
    [[maybe_unused]] const TData *__restrict f1,
    [[maybe_unused]] const TData *__restrict f1m,
    [[maybe_unused]] const TData *__restrict f2, const TData *__restrict df,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out0, TData *__restrict out1, TData *__restrict out2)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
    {
        for (unsigned int j = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (unsigned int d = 0u; d < ncoord; ++d)
                {
                    TData tmp = in[d * insize * nqTot + index];
                    sum1 += df[(3u * d) * warpsize + dfindex] * tmp;
                    sum2 += df[(3u * d + 1u) * warpsize + dfindex] * tmp;
                    sum3 += df[(3u * d + 2u) * warpsize + dfindex] * tmp;
                }

                TData tmpQ = w0[i] * w1[j] * w2[k];
                if constexpr (DEFORMED)
                {
                    tmpQ *= jac[warpsize * cnt_kji + ilane];
                }
                else
                {
                    tmpQ *= jac[0];
                }

                if constexpr (SHAPE_TYPE == LibUtilities::Hex)
                {
                    out0[index] = sum1 * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
                {
                    out0[index] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
                    out1[index] = sum2 * tmpQ;
                    out2[index] = sum3 * tmpQ;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    TData tmp   = f2[k] * tmpQ;
                    out0[index] = (sum1 + sum3 * f0[i]) * tmp;
                    out1[index] = (sum2 + sum3 * f1[j]) * tmp;
                    out2[index] = sum3 * tmpQ;
                }
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void StdAlignDerivBase3DSumFacQPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int insize, const TData *__restrict w0,
    const TData *__restrict w1, const TData *__restrict w2,
    const TData *__restrict f0, const TData *__restrict f1,
    const TData *__restrict f1m, const TData *__restrict f2,
    const TData *__restrict df, const TData *__restrict jac,
    const TData *__restrict in, TData *__restrict out0, TData *__restrict out1,
    TData *__restrict out2, const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ncoord = 3u;

    const unsigned int nqTot = nq0 * nq1 * nq2;
    unsigned int dfsize      = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
    }

    for (unsigned int idx = item_ct1.get_local_id(2); idx < nq0 * nq1 * nq2;
         idx += item_ct1.get_local_range(2))
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        for (unsigned int d = 0u; d < ncoord; ++d)
        {
            TData tmp = in[d * insize * nqTot + idx];
            sum1 += df[(3u * d) * dfsize + dfindex] * tmp;
            sum2 += df[(3u * d + 1u) * dfsize + dfindex] * tmp;
            sum3 += df[(3u * d + 2u) * dfsize + dfindex] * tmp;
        }

        TData tmpQ = w0[i] * w1[j] * w2[k];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            out0[idx] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + sum3 * f0[i]) * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
}

// General Launcher
template <typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *__restrict dbasis0,
    const TData *__restrict w0, const TData *__restrict df,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out, TData *__restrict wsp, TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int ndf = ncoord;
    unsigned int dfsize    = 1u;
    unsigned int jacsize   = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nq0;
        jacsize *= nq0;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = item_ct1.get_global_id(2);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nq0 * warpsize * iwarp;
            TData *deriv       = wsp + nq0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            StdAlignDerivBase1DSumFacKernel<DEFORMED>(
                ilane, ncoord, nq0, nelmt, w0, dfptr, jacptr, inptr, deriv);
            IProductWRTBaseSegSumFacKernel<false, true, DEFORMED>(
                ilane, nm0, nq0, dbasis0, deriv, outptr, (TData)1.0);
            e += item_ct1.get_global_range(2);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        TData *deriv = shmemptr;

        unsigned int e = item_ct1.get_group(2);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nq0 * e;
            TData *outptr       = out + nm0 * e;

            StdAlignDerivBase1DSumFacQPKernel<DEFORMED>(
                ncoord, nq0, nelmt, w0, dfptr, jacptr, inptr, deriv, item_ct1);
            IProductWRTBaseSegSumFacQPKernel<false, true, DEFORMED>(
                nm0, nq0, dbasis0, deriv, outptr, (TData)1.0, item_ct1);

            e += item_ct1.get_group_range(2);
        }
    }
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict index0,
    const TData *__restrict basis0, const TData *__restrict basis1,
    const TData *__restrict D0, const TData *__restrict D1,
    const TData *__restrict w0, const TData *__restrict w1,
    const TData *__restrict f0, const TData *__restrict f1,
    const TData *__restrict df, const TData *__restrict jac,
    const TData *__restrict in, TData *__restrict out,
    [[maybe_unused]] TData *__restrict wsp, TData *__restrict shmemptr,
    const sycl::nd_item<3> &item_ct1)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;
    unsigned int dfsize      = 1u;
    unsigned int jacsize     = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        TData *s_f0 = nullptr;
        TData *s_f1 = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            s_f0 = shmemptr;
            s_f1 = s_f0 + nq0;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq1; idx += stride)
            {
                s_f1[idx] = f1[idx];
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

        unsigned int e = item_ct1.get_global_id(2);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nqTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * warpsize * iwarp;
            TData *deriv1      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;

            StdAlignDerivBase2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt, w0, w1, s_f0, s_f1, dfptr,
                jacptr, inptr, deriv0, deriv1);
            SumDerivTensor2DKernel<false, DEFORMED>(ilane, nq0, nq1, D0, D1,
                                                    deriv0, deriv1, deriv);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 = wsp + 3 * nqTot * nelmt + nq1 * warpsize * iwarp;
                IProductWRTBaseQuadSumFacKernel<false, true, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, deriv, outptr,
                    wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + 3 * nqTot * nelmt + nq1 * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<false, true, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1,
                    deriv, outptr, wsp0, (TData)1.0);
            }
            e += item_ct1.get_global_range(2);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset, nmode0, nmode1;
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            offset = nm0 * nq1;
            nmode0 = nm0;
            nmode1 = nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            offset = nm0 * nq1;
            nmode0 = nm0;
            nmode1 = nmTot;
        }

        TData *deriv0   = shmemptr;
        TData *deriv1   = deriv0 + nqTot;
        TData *deriv    = deriv1 + nqTot;
        TData *s_wsp0   = deriv + nqTot;
        TData *s_basis0 = s_wsp0 + offset;
        TData *s_basis1 = s_basis0 + nm0 * nq0;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        unsigned int e = item_ct1.get_group(2);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nqTot * e;
            TData *outptr       = out + nmTot * e;

            StdAlignDerivBase2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nelmt, w0, w1, f0, f1, dfptr, jacptr, inptr,
                deriv0, deriv1, item_ct1);
            SumDerivTensor2DQPKernel<false, DEFORMED>(nq0, nq1, D0, D1, deriv0,
                                                      deriv1, deriv, item_ct1);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                IProductWRTBaseQuadSumFacQPKernel<false, true, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, deriv,
                    outptr, s_wsp0, (TData)1.0, item_ct1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                IProductWRTBaseTriSumFacQPKernel<false, true, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, deriv, outptr, s_wsp0, (TData)1.0,
                    item_ct1);
            }

            e += item_ct1.get_group_range(2);
        }
    }
}

// General Launcher
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TData>
NEK_FORCE_INLINE void IProductWRTDerivBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict index0,
    [[maybe_unused]] const unsigned int *__restrict index1,
    [[maybe_unused]] const unsigned int *__restrict index2,
    const TData *__restrict basis0, const TData *__restrict basis1,
    const TData *__restrict basis2, const TData *__restrict D0,
    const TData *__restrict D1, const TData *__restrict D2,
    const TData *__restrict w0, const TData *__restrict w1,
    const TData *__restrict w2, const TData *__restrict f0,
    const TData *__restrict f1, const TData *__restrict f1m,
    const TData *__restrict f2, const TData *__restrict df,
    const TData *__restrict jac, const TData *__restrict in,
    TData *__restrict out, [[maybe_unused]] TData *__restrict wsp,
    TData *__restrict shmemptr, const sycl::nd_item<3> &item_ct1)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    unsigned int dfsize        = 1u;
    unsigned int jacsize       = 1u;
    if constexpr (DEFORMED)
    {
        dfsize *= nqTot;
        jacsize *= nqTot;
    }

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        TData *s_f0  = nullptr;
        TData *s_f1  = nullptr;
        TData *s_f1m = nullptr;
        TData *s_f2  = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = item_ct1.get_local_id(2);
        const unsigned int stride = item_ct1.get_local_range(2);
        if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            s_f0  = shmemptr;
            s_f1  = s_f0 + nq0;
            s_f1m = s_f1 + nq1;
            s_f2  = s_f1m + nq1;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq1; idx += stride)
            {
                s_f1[idx]  = f1[idx];
                s_f1m[idx] = f1m[idx];
            }

            for (unsigned int idx = idx0; idx < nq2; idx += stride)
            {
                s_f2[idx] = f2[idx];
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            s_f0 = shmemptr;
            s_f2 = s_f0 + nq0;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq2; idx += stride)
            {
                s_f2[idx] = f2[idx];
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            s_f0 = shmemptr;
            s_f1 = s_f0 + nq0;
            s_f2 = s_f1 + nq1;

            for (unsigned int idx = idx0; idx < nq0; idx += stride)
            {
                s_f0[idx] = f0[idx];
            }

            for (unsigned int idx = idx0; idx < nq1; idx += stride)
            {
                s_f1[idx] = f1[idx];
            }

            for (unsigned int idx = idx0; idx < nq2; idx += stride)
            {
                s_f2[idx] = f2[idx];
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

        unsigned int e = item_ct1.get_global_id(2);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nqTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * warpsize * iwarp;
            TData *deriv1      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv2 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv  = wsp + 3 * nqTot * nelmt + nqTot * warpsize * iwarp;

            StdAlignDerivBase3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt, w0, w1, w2, s_f0, s_f1, s_f1m,
                s_f2, dfptr, jacptr, inptr, deriv0, deriv1, deriv2);
            SumDerivTensor3DKernel<false, DEFORMED>(ilane, nq0, nq1, nq2, D0,
                                                    D1, D2, deriv0, deriv1,
                                                    deriv2, deriv);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                IProductWRTBaseHexSumFacKernel<false, true, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    deriv, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                TData *prod = wsp + (4 * nqTot + nq1 * nq2 + nq2) * nelmt +
                              nm2 * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<false, true, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, deriv, outptr, wsp0, wsp1, prod,
                    (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                TData *wsp2 = wsp + (4 * nqTot + nq1 * nq2 + nq2) * nelmt +
                              nm1 * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<false, true, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, deriv, outptr, wsp0, wsp1, wsp2,
                    (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                IProductWRTBasePyrSumFacKernel<false, true, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, deriv, outptr, wsp0, wsp1, (TData)1.0);
            }
            e += item_ct1.get_global_range(2);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset0, offset1, nmode0, nmode1, nmode2;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = nm0 * nm1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
            nmode0  = nm0;
            nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = nm0 * nm1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            offset0 = nm0 * nq1 * nq2;
            offset1 = nm0 * nm1 * nq2;
            nmode0  = nm0;
            nmode1  = nm1;
            nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }

        TData *deriv0   = shmemptr;
        TData *deriv1   = deriv0 + nqTot;
        TData *deriv2   = deriv1 + nqTot;
        TData *deriv    = deriv2 + nqTot;
        TData *s_wsp0   = deriv + nqTot;
        TData *s_wsp1   = s_wsp0 + offset0;
        TData *s_basis0 = s_wsp1 + offset1;
        TData *s_basis1 = s_basis0 + nmode0 * nq0;
        TData *s_basis2 = s_basis1 + nmode1 * nq1;

        // Copy to shared memory.
        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode0 * nq0;
             idx += item_ct1.get_local_range(2))
        {
            s_basis0[idx] = basis0[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode1 * nq1;
             idx += item_ct1.get_local_range(2))
        {
            s_basis1[idx] = basis1[idx];
        }

        for (unsigned int idx = item_ct1.get_local_id(2); idx < nmode2 * nq2;
             idx += item_ct1.get_local_range(2))
        {
            s_basis2[idx] = basis2[idx];
        }

        unsigned int e = item_ct1.get_group(2);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nqTot * e;
            TData *outptr       = out + nmTot * e;

            StdAlignDerivBase3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nelmt, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, inptr, deriv0, deriv1, deriv2, item_ct1);
            SumDerivTensor3DQPKernel<false, DEFORMED>(nq0, nq1, nq2, D0, D1, D2,
                                                      deriv0, deriv1, deriv2,
                                                      deriv, item_ct1);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                IProductWRTBaseHexSumFacQPKernel<false, true, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, deriv, outptr, s_wsp0, s_wsp1,
                    (TData)1.0, item_ct1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                IProductWRTBaseTetSumFacQPKernel<false, true, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, deriv,
                    outptr, s_wsp0, s_wsp1, (TData)1.0, item_ct1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                IProductWRTBasePrismSumFacQPKernel<false, true, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, deriv,
                    outptr, s_wsp0, s_wsp1, (TData)1.0, item_ct1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                IProductWRTBasePyrSumFacQPKernel<false, true, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, deriv, outptr,
                    s_wsp0, s_wsp1, (TData)1.0, item_ct1);
            }

            e += item_ct1.get_group_range(2);
        }
    }
}

// Launchers
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *dbasis0, const TData *w0,
    const TData *df, const TData *jac, const TData *in, TData *out, TData *wsp)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int shmemsize =
        IProductWRTDerivBaseSharedMemorySize<Implementation>(nq0, nm0);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nq0);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shmem.template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 IProductWRTDerivBase1DKernel<Implementation, DEFORMED>(
                     ncoord, nm0, nq0, nelmt, dbasis0, w0, df, jac, in, out,
                     wsp, shmemptr, item_ct1);
             });
     }).wait();
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int nm0, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase1DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const TData *dbasis0,
    const TData *w0, const TData *df, const TData *jac, const TData *in,
    TData *out, TData *wsp)
{
    IProductWRTDerivBase1DKernel<ExecSpace, Implementation, DEFORMED>(
        ncoord, nm0, nq0, nelmt, dbasis0, w0, df, jac, in, out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const bool isModified, const unsigned int *index0, const TData *basis0,
    const TData *basis1, const TData *D0, const TData *D1, const TData *w0,
    const TData *w1, const TData *f0, const TData *f1, const TData *df,
    const TData *jac, const TData *in, TData *out, TData *wsp)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
    const unsigned int shmemsize =
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nm0, nm1);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shmem.template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 IProductWRTDerivBase2DKernel<SHAPE_TYPE, Implementation,
                                              DEFORMED>(
                     ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified,
                     index0, basis0, basis1, D0, D1, w0, w1, f0, f1, df, jac,
                     in, out, wsp, shmemptr, item_ct1);
             });
     }).wait();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase2DKernel(
    const unsigned int ncoord, const unsigned int nelmt, const bool isModified,
    const unsigned int *index0, const TData *basis0, const TData *basis1,
    const TData *D0, const TData *D1, const TData *w0, const TData *w1,
    const TData *f0, const TData *f1, const TData *df, const TData *jac,
    const TData *in, TData *out, TData *wsp)
{
    IProductWRTDerivBase2DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                 DEFORMED>(
        ncoord, nm0, nm1, nq0, nq1, nelmt, isModified, index0, basis0, basis1,
        D0, D1, w0, w1, f0, f1, df, jac, in, out, wsp);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *D0, const TData *D1,
    const TData *D2, const TData *w0, const TData *w1, const TData *w2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *jac, const TData *in, TData *out, TData *wsp)
{
    sycl::queue &Q = SYCLQueue::GetInstance();

    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int shmemsize =
        IProductWRTDerivBaseSharedMemorySize<SHAPE_TYPE, Implementation>(
            nq0, nq1, nq2, nm0, nm1, nm2);
    const auto blocksize = GetSYCLBlockSize<Implementation>(nmTot);
    const auto gridsize  = GetSYCLGridSize<Implementation>(nelmt);

    Q.submit([=](sycl::handler &cgh) {
         sycl::local_accessor<TData, 1> shmem(sycl::range<1>(shmemsize), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(gridsize * blocksize, blocksize),
             [=](sycl::nd_item<3> item_ct1) {
                 TData *shmemptr =
                     shmem.template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
#pragma forceinline
                 IProductWRTDerivBase3DKernel<SHAPE_TYPE, Implementation,
                                              DEFORMED>(
                     nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified,
                     index0, index1, index2, basis0, basis1, basis2, D0, D1, D2,
                     w0, w1, w2, f0, f1, f1m, f2, df, jac, in, out, wsp,
                     shmemptr, item_ct1);
             });
     }).wait();
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nm0,
          unsigned int nm1, unsigned int nm2, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void IProductWRTDerivBase3DKernel(
    const unsigned int nelmt, const bool isModified, const unsigned int *index0,
    const unsigned int *index1, const unsigned int *index2, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *D0, const TData *D1,
    const TData *D2, const TData *w0, const TData *w1, const TData *w2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *jac, const TData *in, TData *out, TData *wsp)
{
    IProductWRTDerivBase3DKernel<SHAPE_TYPE, ExecSpace, Implementation,
                                 DEFORMED>(
        nm0, nm1, nm2, nq0, nq1, nq2, nelmt, isModified, index0, index1, index2,
        basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, df,
        jac, in, out, wsp);
}

} // namespace Nektar::Operators::detail

#endif
