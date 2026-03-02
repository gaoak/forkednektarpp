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

// get hold of memory sizing and routines related to collocation Helmholtz op
#include "Operators/ElmtOps/Helmholtz/HelmholtzDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *__restrict__ advVel0, const TData *__restrict__ deriv0,
    TData *__restrict__ out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
    const size_t nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda,
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

        size_t e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const size_t ilane = e % warpsize;
            const size_t iwarp = e / warpsize;
            const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
            const TData *jacptr =
                DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
            const TData *inptr = in + nm0 * warpsize * iwarp;
            TData *outptr      = out + nm0 * warpsize * iwarp;
            TData *bwd         = wsp + nq0 * warpsize * iwarp;
            TData *deriv       = wsp + nq0 * nelmt + nq0 * warpsize * iwarp;
            BwdTransSegSumFacKernel(ilane, nm0, nq0, basis0, inptr, bwd);
            PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0,
                                              D0, dfptr, bwd, deriv);
            AddAdvection1DKernel(ilane, nq0, advVel0, deriv, bwd, lambda);
            ApplyMetric1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0,
                                                w0, dfptr, jacptr, coeff, deriv,
                                                deriv, bwd, (TData)1.0);
            SumDerivTensor1DKernel<true, DEFORMED>(ilane, nq0, D0, deriv, bwd);
            IProductWRTBaseSegSumFacKernel<false, false, DEFORMED>(
                ilane, nm0, nq0, basis0, bwd, outptr, (TData)1.0);
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        TData *bwd   = (TData *)shmemptr;
        TData *deriv = bwd + nq0;

        size_t e = getBlockIdx(threadBlock);
        while (e < nelmt)
        {
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nm0 * e;
            TData *outptr       = out + nm0 * e;

            BwdTransSegSumFacTOPKernel(nm0, nq0, basis0, inptr, bwd,
                                       threadBlock);
            PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, D0, dfptr,
                                                 bwd, deriv, threadBlock);
            AddAdvection1DQPKernel(nq0, advVel0, deriv, bwd, lambda,
                                   threadBlock);
            ApplyMetric1DSumFacTOPKernel<DEFORMED>(
                ncoord, nq0, nq0, w0, dfptr, jacptr, coeff, deriv, deriv, bwd,
                (TData)1.0, threadBlock);
            SumDerivTensor1DQPKernel<true, DEFORMED>(nq0, D0, deriv, bwd,
                                                     threadBlock);
            IProductWRTBaseSegSumFacTOPKernel<false, false, DEFORMED>(
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
    const size_t nelmt, const bool isModified,
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
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

        TData *s_f0 = nullptr;
        TData *s_f1 = nullptr;

        // Pre-compute factor.
        const unsigned int idx0   = getLocalIdx(threadBlock);
        const unsigned int stride = getLocalRange(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            s_f0 = (TData *)shmemptr;
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

        size_t e = getGlobalIdx(threadBlock);
        while (e < nelmt)
        {
            const size_t ilane = e % warpsize;
            const size_t iwarp = e / warpsize;
            const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
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
                ilane, ncoord, nq0, nq1, nelmt * nqTot, D0, D1, s_f0, s_f1,
                dfptr, bwd, deriv);
            AddAdvection2DKernel(ilane, nq0, nq1, advVel0, advVel1, deriv0,
                                 deriv1, bwd, lambda);
            ApplyMetric2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, ncoord, nq0, nq1, nelmt * nqTot, w0, w1, s_f0, s_f1,
                dfptr, jacptr, coeff, deriv, deriv0, deriv1, bwd, (TData)1.0);
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

                // Multiply by transpose notToMod to transform coeffs.
                MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr,
                                          outptr);
            }
            e += getGlobalRange(threadBlock);
        }
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
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
                MatVecQPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
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
                BwdTransQuadSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                            s_basis1, tmp, bwd, s_wsp0,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                               SHAPE_TYPE == LibUtilities::NodalTri)
            {
                BwdTransTriSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot,
                                           isModified, s_basis0, s_basis1, tmp,
                                           bwd, s_wsp0, threadBlock);
            }

            PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr, bwd, deriv,
                threadBlock);
            AddAdvection2DQPKernel(nq0, nq1, advVel0, advVel1, deriv0, deriv1,
                                   bwd, lambda, threadBlock);
            if constexpr (DEFORMED)
            {
                TData dmetric[6];
                ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr,
                    coeff, deriv, deriv0, deriv1, dmetric, bwd, (TData)1.0,
                    threadBlock);
            }
            else
            {
                ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr,
                    coeff, deriv, deriv0, deriv1, metric, bwd, (TData)1.0,
                    threadBlock);
            }
            SumDerivTensor2DQPKernel<true, DEFORMED>(nq0, nq1, D0, D1, deriv0,
                                                     deriv1, bwd, threadBlock);
            if constexpr (SHAPE_TYPE == LibUtilities::Quad)
            {
                IProductWRTBaseQuadSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                    outptr, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
            {
                IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, outptr, s_wsp0, (TData)1.0,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
            {
                IProductWRTBaseTriSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0,
                    s_basis0, s_basis1, bwd, tmp, s_wsp0, (TData)1.0,
                    threadBlock);

                // Multiply by transpose notToMod to transform coeffs.
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
    const unsigned int nq2, const size_t nelmt, const bool isModified,
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
    unsigned char *__restrict__ shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
            s_f0  = (TData *)shmemptr;
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
            s_f0 = (TData *)shmemptr;
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
            s_f0 = (TData *)shmemptr;
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

        size_t e = getGlobalIdx(threadBlock); // use size_t to prevent overflow
        while (e < nelmt)
        {
            const size_t ilane = e % warpsize;
            const size_t iwarp = e / warpsize;
            const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
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
                ilane, nq0, nq1, nq2, nelmt * nqTot, D0, D1, D2, s_f0, s_f1,
                s_f1m, s_f2, dfptr, bwd, deriv);
            AddAdvection3DKernel(ilane, nq0, nq1, nq2, advVel0, advVel1,
                                 advVel2, deriv0, deriv1, deriv2, bwd, lambda);
            ApplyMetric3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nelmt * nqTot, w0, w1, w2, s_f0, s_f1,
                s_f1m, s_f2, dfptr, jacptr, coeff, deriv, deriv0, deriv1,
                deriv2, bwd, (TData)1.0);
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
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                TData *out1ptr =
                    wsp + 4 * nqTot * nelmt + nmTot * warpsize * iwarp;
                TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt +
                              nq1 * nq2 * warpsize * iwarp;
                TData *wsp1 = wsp + (nmTot + 4 * nqTot + nq1 * nq2) * nelmt +
                              std::max(nq2, nm0) * warpsize * iwarp;
                IProductWRTBaseTetSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

                // Multiply by transpose notToMod to transform coeffs.
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
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
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
                IProductWRTBasePrismSumFacKernel<false, false, DEFORMED>(
                    ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0,
                    basis1, basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

                // Multiply by transpose notToMod to transform coeffs.
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
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
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
                BwdTransHexSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                           s_basis0, s_basis1, s_basis2, tmp,
                                           bwd, s_wsp0, s_wsp1, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                               SHAPE_TYPE == LibUtilities::NodalTet)
            {
                BwdTransTetSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                           isModified, index0, index3, s_basis0,
                                           s_basis1, s_basis2, tmp, bwd, s_wsp0,
                                           s_wsp1, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                               SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                BwdTransPrismSumFacTOPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                BwdTransPyrSumFacTOPKernel(
                    nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                    s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
            }
            PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr, bwd,
                deriv, threadBlock);
            AddAdvection3DQPKernel(nq0, nq1, nq2, advVel0, advVel1, advVel2,
                                   deriv0, deriv1, deriv2, bwd, lambda,
                                   threadBlock);
            if constexpr (DEFORMED)
            {
                TData dmetric[9];
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
            SumDerivTensor3DQPKernel<true, DEFORMED>(nq0, nq1, nq2, D0, D1, D2,
                                                     deriv0, deriv1, deriv2,
                                                     bwd, threadBlock);
            if constexpr (SHAPE_TYPE == LibUtilities::Hex)
            {
                IProductWRTBaseHexSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0,
                    s_basis1, s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0,
                    threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
            {
                IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
            {
                IProductWRTBaseTetSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    tmp, s_wsp1, s_wsp0, (TData)1.0, threadBlock);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
            {
                IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    outptr, s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
            {
                IProductWRTBasePrismSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, index2, s_basis0, s_basis1, s_basis2, bwd,
                    tmp, s_wsp1, s_wsp0, (TData)1.0, threadBlock);

                // Multiply by transpose notToMod to transform coeffs.
                MatVecQPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                            threadBlock);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                IProductWRTBasePyrSumFacTOPKernel<false, false, DEFORMED>(
                    nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified,
                    index0, index1, s_basis0, s_basis1, s_basis2, bwd, outptr,
                    s_wsp1, s_wsp0, (TData)1.0, threadBlock);
            }

            e += getBlockRange(threadBlock);
        }
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL void LinAdvDiffReaction1DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const TData *__restrict__ basis0,
    const TData *__restrict__ D0, const TData *__restrict__ w0,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ in, TData *__restrict__ out,
    TData *__restrict__ wsp, const TData lambda, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction1DKernel<Implementation, DEFORMED>(
        ncoord, nm0, nq0, nelmt, basis0, D0, w0, df, jac, coeff, advVel0, in,
        out, wsp, lambda, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(maxThreadPerBlock)
    LinAdvDiffReaction1DKernelLauncher(
        const unsigned int ncoord, const size_t nelmt,
        const TData *__restrict__ basis0, const TData *__restrict__ D0,
        const TData *__restrict__ w0, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ coeff,
        const TData *__restrict__ advVel0, const TData *__restrict__ in,
        TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction1DKernel<Implementation, DEFORMED>(
        ncoord, nm0, nq0, nelmt, basis0, D0, w0, df, jac, coeff, advVel0, in,
        out, wsp, lambda, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void LinAdvDiffReaction2DKernelLauncher(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const bool isModified,
    const unsigned int *__restrict__ index0, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ f0,
    const TData *__restrict__ f1, const TData *__restrict__ nodToMod,
    const TData *__restrict__ df, const TData *__restrict__ jac,
    const TData *__restrict__ coeff, const TData *__restrict__ advVel0,
    const TData *__restrict__ advVel1, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, advVel0,
        advVel1, in, out, wsp, lambda, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TthreadBlock,
          typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(maxThreadPerBlock)
    LinAdvDiffReaction2DKernelLauncher(
        const unsigned int ncoord, const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const TData *__restrict__ basis0, const TData *__restrict__ basis1,
        const TData *__restrict__ D0, const TData *__restrict__ D1,
        const TData *__restrict__ w0, const TData *__restrict__ w1,
        const TData *__restrict__ f0, const TData *__restrict__ f1,
        const TData *__restrict__ nodToMod, const TData *__restrict__ df,
        const TData *__restrict__ jac, const TData *__restrict__ coeff,
        const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
        const TData *__restrict__ in, TData *__restrict__ out,
        TData *__restrict__ wsp, const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, advVel0,
        advVel1, in, out, wsp, lambda, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL void LinAdvDiffReaction3DKernelLauncher(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const unsigned int *__restrict__ index0,
    const unsigned int *__restrict__ index1,
    const unsigned int *__restrict__ index2,
    const unsigned int *__restrict__ index3, const TData *__restrict__ basis0,
    const TData *__restrict__ basis1, const TData *__restrict__ basis2,
    const TData *__restrict__ D0, const TData *__restrict__ D1,
    const TData *__restrict__ D2, const TData *__restrict__ w0,
    const TData *__restrict__ w1, const TData *__restrict__ w2,
    const TData *__restrict__ f0, const TData *__restrict__ f1,
    const TData *__restrict__ f1m, const TData *__restrict__ f2,
    const TData *__restrict__ nodToMod, const TData *__restrict__ df,
    const TData *__restrict__ jac, const TData *__restrict__ coeff,
    const TData *__restrict__ advVel0, const TData *__restrict__ advVel1,
    const TData *__restrict__ advVel2, const TData *__restrict__ in,
    TData *__restrict__ out, TData *__restrict__ wsp, const TData lambda,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, advVel0, advVel1, advVel2, in, out,
        wsp, lambda, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(maxThreadPerBlock)
    LinAdvDiffReaction3DKernelLauncher(
        const size_t nelmt, const bool isModified,
        const unsigned int *__restrict__ index0,
        const unsigned int *__restrict__ index1,
        const unsigned int *__restrict__ index2,
        const unsigned int *__restrict__ index3,
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
        TData *__restrict__ wsp, const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    LinAdvDiffReaction3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, advVel0, advVel1, advVel2, in, out,
        wsp, lambda, shmemptr, threadBlock);
}

#endif

} // namespace Nektar::Operators::detail
