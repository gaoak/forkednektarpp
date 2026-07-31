///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceSumFacTOPKernels.hpp
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

// get hold of memory sizing and routines related to collocation Helmholtz op
#include "Operators/ElmtOps/Helmholtz/HelmholtzDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DSumFacTOPKernel(
    const unsigned ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        TData tmp = advVel0[i] * deriv0[i];
        if (ncoord > 1)
        {
            tmp += advVel1[i] * deriv1[i];
        }
        if (ncoord > 2)
        {
            tmp += advVel2[i] * deriv2[i];
        }
        out[i] = scale * out[i] + tmp;
    }
    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0 * nq1; i += stride)
    {
        TData tmp = advVel0[i] * deriv0[i] + advVel1[i] * deriv1[i];
        if (ncoord == 3)
        {
            tmp += advVel2[i] * deriv2[i];
        }
        out[i] = scale * out[i] + tmp;
    }
    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0 * nq1 * nq2; i += stride)
    {
        out[i] = scale * out[i] + advVel0[i] * deriv0[i] +
                 advVel1[i] * deriv1[i] + advVel2[i] * deriv2[i];
    }
    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LinAdvDiffReaction1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT advVel0,
    const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData lambda,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *bwd   = (TData *)shmemptr;
    TData *deriv = bwd + nq0;

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr      = df + ndf * dfsize * e;
        const TData *jacptr     = jac + jacsize * e;
        const TData *inptr      = in + nm0 * e;
        TData *outptr           = out + nm0 * e;
        const TData *advVel0ptr = advVel0 + nq0 * e;
        const TData *advVel1ptr = ncoord > 1 ? advVel1 + nq0 * e : advVel1;
        const TData *advVel2ptr = ncoord > 2 ? advVel2 + nq0 * e : advVel2;

        BwdTransSegSumFacTOPKernel<false>(nm0, nq0, basis0, inptr, bwd,
                                          threadBlock);
        PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, D0, dfptr, bwd,
                                             deriv, threadBlock);
        AddAdvection1DSumFacTOPKernel(
            ncoord, nq0, advVel0ptr, advVel1ptr, advVel2ptr, deriv, deriv + nq0,
            deriv + 2 * nq0, bwd, lambda, threadBlock);
        ApplyMetric1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, w0, dfptr,
                                               jacptr, coeff, deriv, deriv, bwd,
                                               (TData)1.0, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<false, true>(nq0, D0, deriv, bwd,
                                                     (TData)1.0, threadBlock);
        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LinAdvDiffReaction2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const bool isModified,
    const unsigned int *NEK_RESTRICT index0, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT nodToMod,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT advVel0,
    const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData lambda,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = std::max(nm0 * nq1, nm1 * nq0);
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

    TData *metric   = (TData *)shmemptr;
    TData *bwd      = metric + 6;
    TData *deriv    = bwd + nqTot;
    TData *tmp      = deriv;
    TData *deriv0   = deriv;
    TData *deriv1   = deriv0 + nqTot;
    TData *s_wsp0   = deriv + ncoord * nqTot;
    TData *s_basis0 = s_wsp0 + offset;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

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
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacTOPKernel<false>(nm0, nm1, nq0, nq1, nqTot,
                                               s_basis0, s_basis1, tmp, bwd,
                                               s_wsp0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel<false>(nm0, nm1, nq0, nq1, nqTot,
                                              isModified, s_basis0, s_basis1,
                                              tmp, bwd, s_wsp0, threadBlock);
        }

        PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr, bwd, deriv,
            threadBlock);
        const TData *advVel0ptr = advVel0 + nqTot * e;
        const TData *advVel1ptr = advVel1 + nqTot * e;
        const TData *advVel2ptr = ncoord == 3 ? advVel2 + nqTot * e : advVel2;
        AddAdvection2DSumFacTOPKernel(
            ncoord, nq0, nq1, advVel0ptr, advVel1ptr, advVel2ptr, deriv,
            deriv + nqTot, deriv + 2 * nqTot, bwd, lambda, threadBlock);
        if constexpr (DEFORMED)
        {
            TData dmetric[6] = {0.0};
            ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr, coeff,
                deriv, deriv0, deriv1, dmetric, bwd, (TData)1.0, threadBlock);
        }
        else
        {
            ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr, coeff,
                deriv, deriv0, deriv1, metric, bwd, (TData)1.0, threadBlock);
        }
        SumDerivTensor2DSumFacTOPKernel<false, true>(
            nq0, nq1, D0, D1, deriv0, deriv1, bwd, (TData)1.0, threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, tmp, s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LinAdvDiffReaction3DSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const unsigned int *NEK_RESTRICT index0,
    const unsigned int *NEK_RESTRICT index1,
    const unsigned int *NEK_RESTRICT index2,
    const unsigned int *NEK_RESTRICT index3, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData lambda,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
        offset1 = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
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
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *metric   = (TData *)shmemptr;
    TData *bwd      = metric + 9;
    TData *deriv    = bwd + nqTot;
    TData *tmp      = deriv;
    TData *deriv0   = deriv;
    TData *deriv1   = deriv0 + nqTot;
    TData *deriv2   = deriv1 + nqTot;
    TData *s_wsp0   = deriv2 + nqTot;
    TData *s_wsp1   = s_wsp0 + offset0;
    TData *s_basis0 = s_wsp1 + offset1;
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
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        // Copy to shared memory.
        if constexpr (SHAPE_TYPE == LibUtilities::NodalTet ||
                      SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel<false>(nm0, nm1, nm2, nq0, nq1, nq2,
                                              nqTot, isModified, index0, index3,
                                              s_basis0, s_basis1, s_basis2, tmp,
                                              bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr, bwd,
            deriv, threadBlock);
        const TData *advVel0ptr = advVel0 + nqTot * e;
        const TData *advVel1ptr = advVel1 + nqTot * e;
        const TData *advVel2ptr = advVel2 + nqTot * e;
        AddAdvection3DSumFacTOPKernel(nq0, nq1, nq2, advVel0ptr, advVel1ptr,
                                      advVel2ptr, deriv0, deriv1, deriv2, bwd,
                                      lambda, threadBlock);
        if constexpr (DEFORMED)
        {
            TData dmetric[9] = {0.0};
            ApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nqTot, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, coeff, deriv, deriv0, deriv1, deriv2, dmetric, bwd,
                (TData)1.0, threadBlock);
        }
        else
        {
            ApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nqTot, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, coeff, deriv, deriv0, deriv1, deriv2, metric, bwd,
                (TData)1.0, threadBlock);
        }
        SumDerivTensor3DSumFacTOPKernel<false, true>(
            nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, bwd, (TData)1.0,
            threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, bwd, outptr, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <typename Implementation, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::
        type __LAUNCH_BOUNDS__(
            (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
            LinAdvDiffReaction1DKernelLauncher(
                const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
                const size_t nelmt, const TData *NEK_RESTRICT basis0,
                const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT w0,
                const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
                const TData *NEK_RESTRICT coeff,
                const TData *NEK_RESTRICT advVel0,
                const TData *NEK_RESTRICT advVel1,
                const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT in,
                TData *NEK_RESTRICT out,
                [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
                unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction1DSumFacTOPKernel<DEFORMED>(
        ncoord, sizeParam1D.nm0(), sizeParam1D.nq0(), nelmt, basis0, D0, w0, df,
        jac, coeff, advVel0, advVel1, advVel2, in, out, lambda, shmemptr,
        threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::
        type __LAUNCH_BOUNDS__(
            (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
            LinAdvDiffReaction2DKernelLauncher(
                const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
                const size_t nelmt, const bool isModified,
                const unsigned int *NEK_RESTRICT index0,
                const TData *NEK_RESTRICT basis0,
                const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT D0,
                const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT w0,
                const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
                const TData *NEK_RESTRICT f1,
                const TData *NEK_RESTRICT nodToMod,
                const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
                const TData *NEK_RESTRICT coeff,
                const TData *NEK_RESTRICT advVel0,
                const TData *NEK_RESTRICT advVel1,
                const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT in,
                TData *NEK_RESTRICT out,
                [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
                unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, sizeParam2D.nm0(), sizeParam2D.nm1(), sizeParam2D.nmTot(),
        sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, advVel0,
        advVel1, advVel2, in, out, lambda, shmemptr, threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::
        type __LAUNCH_BOUNDS__(
            (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
            LinAdvDiffReaction3DKernelLauncher(
                const TSizeParameter3D sizeParam3D, const size_t nelmt,
                const bool isModified, const unsigned int *NEK_RESTRICT index0,
                const unsigned int *NEK_RESTRICT index1,
                const unsigned int *NEK_RESTRICT index2,
                const unsigned int *NEK_RESTRICT index3,
                const TData *NEK_RESTRICT basis0,
                const TData *NEK_RESTRICT basis1,
                const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
                const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
                const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
                const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
                const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
                const TData *NEK_RESTRICT f2,
                const TData *NEK_RESTRICT nodToMod,
                const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
                const TData *NEK_RESTRICT coeff,
                const TData *NEK_RESTRICT advVel0,
                const TData *NEK_RESTRICT advVel1,
                const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT in,
                TData *NEK_RESTRICT out,
                [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
                unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nm0(), sizeParam3D.nm1(), sizeParam3D.nm2(),
        sizeParam3D.nmTot(), sizeParam3D.nq0(), sizeParam3D.nq1(),
        sizeParam3D.nq2(), nelmt, isModified, index0, index1, index2, index3,
        basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2,
        nodToMod, df, jac, coeff, advVel0, advVel1, advVel2, in, out, lambda,
        shmemptr, threadBlock);
}

#endif

} // namespace Nektar::Operators::detail
