///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceSumFacKernels.hpp
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

// get hold of memory sizing and routines related to collocation Helmholtz ops
#include "Operators/ElmtOps/Helmholtz/HelmholtzDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *__restrict__ advVel0, const TData *__restrict__ deriv0,
    TData *__restrict__ out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    // calculate advection term
    for (unsigned int i = 0u; i < nq0; ++i)
    {
        out[warpsize * i + ilane] =
            scale * out[warpsize * i + ilane] +
            advVel0[warpsize * i + ilane] * deriv0[warpsize * i + ilane];
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection2DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ deriv0, const TData *__restrict__ deriv1,
    TData *__restrict__ out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    // calculate advection term
    for (unsigned int i = 0u; i < nq0 * nq1; ++i)
    {
        out[warpsize * i + ilane] =
            scale * out[warpsize * i + ilane] +
            advVel0[warpsize * i + ilane] * deriv0[warpsize * i + ilane] +
            advVel1[warpsize * i + ilane] * deriv1[warpsize * i + ilane];
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection3DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *__restrict__ advVel0,
    const TData *__restrict__ advVel1, const TData *__restrict__ advVel2,
    const TData *__restrict__ deriv0, const TData *__restrict__ deriv1,
    const TData *__restrict__ deriv2, TData *__restrict__ out,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    // calculate advection term
    for (unsigned int i = 0u; i < nq0 * nq1 * nq2; ++i)
    {
        out[warpsize * i + ilane] =
            scale * out[warpsize * i + ilane] +
            advVel0[warpsize * i + ilane] * deriv0[warpsize * i + ilane] +
            advVel1[warpsize * i + ilane] * deriv1[warpsize * i + ilane] +
            advVel2[warpsize * i + ilane] * deriv2[warpsize * i + ilane];
    }
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DQPKernel(
    const unsigned int nq0, const TData *__restrict__ advVel0,
    const TData *__restrict__ deriv0, TData *__restrict__ out,
    const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        out[i] = scale * out[i] + advVel0[i] * deriv0[i];
    }
    localBarrier(threadBlock);
}

template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection2DQPKernel(
    const unsigned int nq0, const unsigned int nq1,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ deriv0, const TData *__restrict__ deriv1,
    TData *__restrict__ out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0 * nq1; i += stride)
    {
        out[i] =
            scale * out[i] + advVel0[i] * deriv0[i] + advVel1[i] * deriv1[i];
    }
    localBarrier(threadBlock);
}
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection3DQPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ advVel2, const TData *__restrict__ deriv0,
    const TData *__restrict__ deriv1, const TData *__restrict__ deriv2,
    TData *__restrict__ out, const TData scale, const TthreadBlock &threadBlock)
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

template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void LinAdvDiffReaction1DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const unsigned int nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda, TData *__restrict__ shmemptr,
    const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        unsigned int e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =

                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nm0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            TData *bwd         = wsp + nq0 * warpsize * iwarp;
            TData *deriv       = wsp + nq0 * nelmt + nq0 * warpsize * iwarp;
            BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, bwd);
            PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt, D0,
                                              dfptr, bwd, deriv);
            AddAdvection1DKernel(ilane, nq0, advVel0, deriv, bwd, lambda);
            ApplyMetric1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt, w0,
                                                dfptr, jacptr, coeff, deriv,
                                                bwd, deriv, (TData)1.0);
            SumDerivTensor1DKernel<true, DEFORMED>(ilane, nq0, D0, deriv, bwd);
            IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nq0, basis0, bwd, outptr, (TData)1.0);
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        TData *bwd   = shmemptr;
        TData *deriv = bwd + nq0;

        unsigned int e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nm0 * e;
            TData *outptr       = out + nm0 * e;

            BwdTransSegSumFacQPKernel(nm0, nq0, basis0, inptr, bwd,
                                      threadBlock);
            PhysDeriv1DSumFacQPKernel<DEFORMED>(ncoord, nq0, 1, D0, dfptr, bwd,
                                                deriv, threadBlock);
            AddAdvection1DQPKernel(nq0, advVel0, deriv, bwd, lambda,
                                   threadBlock);
            ApplyMetric1DSumFacQPKernel<DEFORMED>(
                ncoord, nq0, 1, w0, dfptr, jacptr, coeff, deriv, bwd, deriv,
                (TData)1.0, threadBlock);
            SumDerivTensor1DQPKernel<true, DEFORMED>(nq0, D0, deriv, bwd,
                                                     threadBlock);
            IProductWRTBaseSegSumFacQPKernel<false, false, DEFORMED>(
                nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);

            e += getBlockRange(threadBlock);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LinAdvDiffReaction2DKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp, const TData lambda,
    TData *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        TData *s_f0 = nullptr;
        TData *s_f1 = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
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

            localBarrier(threadBlock);
        }

        unsigned int e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            TData *deriv       = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv1 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 =
                    wsp + (1 + ncoord) * nqTot * nelmt + nq1 * warpsize * iwarp;
                BwdTransQuadSumFacKernel(ilane, nm0, nm1, nq0, nq1, basis0,
                                         basis1, inptr, bwd, wsp0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, inptr, bwd, wsp0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                TData *in1ptr = wsp + (1 + ncoord) * nqTot * nelmt +
                                nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + (1 + ncoord) * nqTot) * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;

                MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
                BwdTransTriSumFacKernel(ilane, nm0, nm1, nq0, nq1, isModified,
                                        basis0, basis1, in1ptr, bwd, wsp0);
            }

            PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt, D0, D1, s_f0, s_f1, dfptr, bwd,
                deriv);
            AddAdvection2DKernel(ilane, nq0, nq1, advVel0, advVel1, deriv0,
                                 deriv1, bwd, lambda);
            ApplyMetric2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt, w0, w1, s_f0, s_f1, dfptr,
                jacptr, coeff, deriv, bwd, deriv0, deriv1, (TData)1.0);
            SumDerivTensor2DKernel<true, DEFORMED>(ilane, nq0, nq1, D0, D1,
                                                   deriv0, deriv1, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                TData *wsp0 =
                    wsp + (1 + ncoord) * nqTot * nelmt + nq1 * warpsize * iwarp;
                IProductWRTBaseQuadSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, basis0, basis1, bwd, outptr,
                    wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                    outptr, wsp0, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                TData *out1ptr = wsp + (1 + ncoord) * nqTot * nelmt +
                                 nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt +
                              std::max(nq1, nm0) * warpsize * iwarp;
                IProductWRTBaseTriSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                    out1ptr, wsp0, (TData)1.0);
                // multiply by transpose  notToMod to transform coeffs
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
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

        TData *metric   = shmemptr;
        TData *bwd      = metric + 6;
        TData *deriv    = bwd + nqTot;
        TData *tmp      = deriv;
        TData *deriv0   = deriv;
        TData *deriv1   = deriv0 + nqTot;
        TData *s_wsp0   = deriv + ncoord * nqTot;
        TData *s_basis0 = s_wsp0 + offset;
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

        unsigned int e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * e;
            TData *outptr       = out + nmTot * e;

            if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                MatVecQPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
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

            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                BwdTransQuadSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                           s_basis1, tmp, bwd, s_wsp0,
                                           threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                               SHAPE_TYPE == LibUtilities::NodalTri)
            {
                BwdTransTriSumFacQPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                          s_basis0, s_basis1, tmp, bwd, s_wsp0,
                                          threadBlock);
            }

            PhysDeriv2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, 1, D0, D1, f0, f1, dfptr, bwd, deriv,
                threadBlock);
            AddAdvection2DQPKernel(nq0, nq1, advVel0, advVel1, deriv0, deriv1,
                                   bwd, lambda, threadBlock);
            if constexpr (DEFORMED)
            {
                TData dmetric[6];
                ApplyMetric2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, 1, w0, w1, f0, f1, dfptr, jacptr, coeff,
                    deriv, bwd, deriv0, deriv1, dmetric, (TData)1.0,
                    threadBlock);
            }
            else
            {
                ApplyMetric2DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, 1, w0, w1, f0, f1, dfptr, jacptr, coeff,
                    deriv, bwd, deriv0, deriv1, metric, (TData)1.0,
                    threadBlock);
            }
            SumDerivTensor2DQPKernel<true, DEFORMED>(nq0, nq1, D0, D1, deriv0,
                                                     deriv1, bwd, threadBlock);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                IProductWRTBaseQuadSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                    outptr, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                IProductWRTBaseTriSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, outptr, s_wsp0, (TData)1.0,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                IProductWRTBaseTriSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, tmp, s_wsp0, (TData)1.0,
                    threadBlock);
                // multiply by transpose nodToMod to convert coeffs
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LinAdvDiffReaction3DKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const bool isModified,
    [[maybe_unused]] const unsigned int *__restrict__ index0,
    [[maybe_unused]] const unsigned int *__restrict__ index1,
    [[maybe_unused]] const unsigned int *__restrict__ index2,
    [[maybe_unused]] const unsigned int *__restrict__ index3,
    const TData *__restrict__ basis0, const TData *__restrict__ basis1,
    const TData *__restrict__ basis2, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ w0, const TData *__restrict__ w1,
    const TData *__restrict__ w2, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ f1m,
    const TData *__restrict__ f2, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ advVel1, const TData *__restrict__ advVel2,
    const TData *__restrict__ in, TData *__restrict__ out,
    [[maybe_unused]] TData *__restrict__ wsp, const TData lambda,
    TData *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize =
            NektarSpaces::vector_width<TData>::value;

        TData *s_f0  = nullptr;
        TData *s_f1  = nullptr;
        TData *s_f1m = nullptr;
        TData *s_f2  = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                      SHAPE_TYPE == LibUtilities::NodalTet)
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

            localBarrier(threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
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

            localBarrier(threadBlock);
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

            localBarrier(threadBlock);
        }

        unsigned int e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const unsigned int ilane = e % warpsize;
            const unsigned int iwarp = e / warpsize;
            const TData *dfptr       = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nmTot * warpsize * iwarp;
            TData *outptr      = out + nmTot * warpsize * iwarp;
            TData *bwd         = wsp + nqTot * warpsize * iwarp;
            TData *deriv       = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv0      = wsp + nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv1 = wsp + 2 * nqTot * nelmt + nqTot * warpsize * iwarp;
            TData *deriv2 = wsp + 3 * nqTot * nelmt + nqTot * warpsize * iwarp;

            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                BwdTransHexSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        basis0, basis1, basis2, inptr, bwd,
                                        wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {

                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                TData *in1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (nmTot + 4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
                BwdTransTetSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        in1ptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)

            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          inptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                TData *in1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp +
                    (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                        nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
                BwdTransPrismSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                          isModified, basis0, basis1, basis2,
                                          in1ptr, bwd, wsp0, wsp1);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                BwdTransPyrSumFacKernel(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                        isModified, basis0, basis1, basis2,
                                        inptr, bwd, wsp0, wsp1);
            }
            PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt, D0, D1, D2, s_f0, s_f1, s_f1m,
                s_f2, dfptr, bwd, deriv);
            AddAdvection3DKernel(ilane, nq0, nq1, nq2, advVel0, advVel1,
                                 advVel2, deriv0, deriv1, deriv2, bwd, lambda);
            ApplyMetric3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt, w0, w1, w2, s_f0, s_f1, s_f1m,
                s_f2, dfptr, jacptr, coeff, deriv, bwd, deriv0, deriv1, deriv2,
                (TData)1.0);
            SumDerivTensor3DKernel<true, DEFORMED>(
                ilane, nq0, nq1, nq2, D0, D1, D2, deriv0, deriv1, deriv2, bwd);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              nq2 * warpsize * iwarp;
                IProductWRTBaseHexSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                    bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                TData *wsp0 =
                    wsp + 4 * nqTot * nelmt + nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 =
                    wsp + (4 * nqTot + nq1 * nq2 + std::max(nq2, nm0)) * nelmt +
                    nm2 * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                TData *out1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;

                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (nmTot + 4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 =
                    wsp +
                    (nmTot + 4 * nqTot + nq1 * nq2 + std::max(nq2, nm0)) *
                        nelmt +
                    nm2 * warpsize * iwarp;

                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, out1ptr, wsp0, wsp1, wsp2, (TData)1.0);
                // multiply by transpose  notToMod to transform coeffs
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 = wsp +
                              (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1) +
                               std::max(nq2, nm0)) *
                                  nelmt +
                              nm1 * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, wsp2, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                TData *out1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp +
                    (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                        nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                TData *wsp2 =
                    wsp +
                    (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1) +
                     std::max(nq2, nm0)) *
                        nelmt +
                    nm1 * warpsize * iwarp;
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, out1ptr, wsp0, wsp1, wsp2, (TData)1.0);
                // multiply by transpose  notToMod to transform coeffs
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                TData *wsp0 = wsp + 4 * nqTot * nelmt +
                              std::max(nq1 * nq2, nm0 * nm1) * warpsize * iwarp;
                TData *wsp1 =
                    wsp + (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt +
                    std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBasePyrSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0,
                     nmode2 = 0;
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

        TData *metric   = shmemptr;
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

        unsigned int e = getBlockIdx(threadBlock);
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
                MatVecQPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
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
                BwdTransHexSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          s_basis0, s_basis1, s_basis2, tmp,
                                          bwd, s_wsp0, s_wsp1, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                               SHAPE_TYPE == LibUtilities::NodalTet)
            {
                BwdTransTetSumFacQPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                          isModified, index0, index3, s_basis0,
                                          s_basis1, s_basis2, tmp, bwd, s_wsp0,
                                          s_wsp1, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                               SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                BwdTransPrismSumFacQPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                BwdTransPyrSumFacQPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
            }
            PhysDeriv3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, 1, D0, D1, D2, f0, f1, f1m, f2, dfptr, bwd,
                deriv, threadBlock);
            AddAdvection3DQPKernel(nq0, nq1, nq2, advVel0, advVel1, advVel2,
                                   deriv0, deriv1, deriv2, bwd, lambda,
                                   threadBlock);
            if constexpr (DEFORMED)
            {
                TData dmetric[9];
                ApplyMetric3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, 1, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                    jacptr, coeff, deriv, bwd, deriv0, deriv1, deriv2, dmetric,
                    (TData)1.0, threadBlock);
            }
            else
            {
                ApplyMetric3DSumFacQPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, 1, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                    jacptr, coeff, deriv, bwd, deriv0, deriv1, deriv2, metric,
                    (TData)1.0, threadBlock);
            }
            SumDerivTensor3DQPKernel<true, DEFORMED>(nq0, nq1, nq2, D0, D1, D2,
                                                     deriv0, deriv1, deriv2,
                                                     bwd, threadBlock);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                IProductWRTBaseHexSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                IProductWRTBaseTetSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                IProductWRTBaseTetSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    tmp, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
                // multiply by transpose nodToMod to convert coeffs
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                IProductWRTBasePrismSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                IProductWRTBasePrismSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    tmp, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
                // multiply by transpose nodToMod to convert coeffs
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                IProductWRTBasePyrSumFacQPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, bwd, outptr,
                    s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionDeviceOnHostSumFacKernelLaunchers.hpp"
#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionHIPCUDASumFacKernelLaunchers.hpp"
#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionSYCLSumFacKernelLaunchers.hpp"
