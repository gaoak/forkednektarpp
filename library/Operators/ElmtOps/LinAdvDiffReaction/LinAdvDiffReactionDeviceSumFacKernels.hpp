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

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

// get hold of memory sizing and routines related to collocation Helmholtz op
#include "Operators/ElmtOps/Helmholtz/HelmholtzDeviceSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    // calculate advection term
    for (unsigned int i = 0u; i < nq0; ++i)
    {
        TData tmp =
            advVel0[warpsize * i + ilane] * deriv0[warpsize * i + ilane];
        if (ncoord > 1)
        {
            tmp += advVel1[warpsize * i + ilane] * deriv1[warpsize * i + ilane];
        }
        if (ncoord > 2)
        {
            tmp += advVel2[warpsize * i + ilane] * deriv2[warpsize * i + ilane];
        }
        out[warpsize * i + ilane] = scale * out[warpsize * i + ilane] + tmp;
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection2DKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const TData *NEK_RESTRICT advVel0,
    const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
    const TData *NEK_RESTRICT deriv0, const TData *NEK_RESTRICT deriv1,
    const TData *NEK_RESTRICT deriv2, TData *NEK_RESTRICT out,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    // calculate advection term
    for (unsigned int i = 0u; i < nq0 * nq1; ++i)
    {
        TData tmp =
            advVel0[warpsize * i + ilane] * deriv0[warpsize * i + ilane] +
            advVel1[warpsize * i + ilane] * deriv1[warpsize * i + ilane];
        if (ncoord == 3)
        {
            tmp += advVel2[warpsize * i + ilane] * deriv2[warpsize * i + ilane];
        }
        out[warpsize * i + ilane] = scale * out[warpsize * i + ilane] + tmp;
    }
}

template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection3DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT advVel0,
    const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
    const TData *NEK_RESTRICT deriv0, const TData *NEK_RESTRICT deriv1,
    const TData *NEK_RESTRICT deriv2, TData *NEK_RESTRICT out,
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

template <typename Implementation, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    LinAdvDiffReaction1DKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT advVel0,
        const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData lambda,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nm0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nm0 * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nq0 * (nelmt * c + warpsize * iwarp);
        TData *deriv       = wsp + nq0 * nelmt * ncomp +
                       nq0 * (ncoord * nelmt * c + warpsize * iwarp);
        const TData *advVel0ptr = advVel0 + nq0 * warpsize * iwarp;
        const TData *advVel1ptr =
            ncoord > 1 ? advVel1 + nq0 * warpsize * iwarp : advVel1;
        const TData *advVel2ptr =
            ncoord > 2 ? advVel2 + nq0 * warpsize * iwarp : advVel2;

        BwdTransSegSumFacKernel<false>(ilane, nm0, nq0, basis0, inptr, bwd);
        PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0, D0,
                                          dfptr, bwd, deriv);
        AddAdvection1DKernel(ilane, ncoord, nq0, advVel0ptr, advVel1ptr,
                             advVel2ptr, deriv, deriv + nq0 * nelmt,
                             deriv + 2 * nq0 * nelmt, bwd, lambda);
        ApplyMetric1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0, w0,
                                            dfptr, jacptr, coeff, deriv, deriv,
                                            bwd, (TData)1.0);
        SumDerivTensor1DKernel<false, true>(ilane, nq0, D0, deriv, bwd,
                                            (TData)1.0);
        IProductWRTBaseSegSumFacKernel<false, false>(ilane, nm0, nq0, basis0,
                                                     bwd, outptr, (TData)1.0);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    LinAdvDiffReaction2DKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
        const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Pre-compute factor.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nqTot * (nelmt * c + warpsize * iwarp);
        TData *deriv       = wsp + nqTot * nelmt * ncomp +
                       nqTot * (ncoord * nelmt * c + warpsize * iwarp);
        TData *deriv0 = wsp + nqTot * nelmt * ncomp +
                        nqTot * (ncoord * nelmt * c + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * (ncomp + 1) +
                        nqTot * (ncoord * nelmt * c + warpsize * iwarp);
        TData *deriv2 = wsp + nqTot * nelmt * (ncomp + 2) +
                        nqTot * (ncoord * nelmt * c + warpsize * iwarp);
        const TData *advVel0ptr = advVel0 + nqTot * warpsize * iwarp;
        const TData *advVel1ptr = advVel1 + nqTot * warpsize * iwarp;
        const TData *advVel2ptr =
            ncoord == 3 ? advVel2 + nqTot * warpsize * iwarp : advVel2;

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          nq1 * (nelmt * c + warpsize * iwarp);
            BwdTransQuadSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1, basis0,
                                            basis1, inptr, bwd, wsp0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransTriSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1,
                                           isModified, basis0, basis1, inptr,
                                           bwd, wsp0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *in1ptr = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                            nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);

            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTriSumFacKernel<false>(ilane, nm0, nm1, nq0, nq1,
                                           isModified, basis0, basis1, in1ptr,
                                           bwd, wsp0);
        }

        PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, nelmt * nqTot, D0, D1, s_f0, s_f1, dfptr,
            bwd, deriv);
        AddAdvection2DKernel(ilane, ncoord, nq0, nq1, advVel0ptr, advVel1ptr,
                             advVel2ptr, deriv, deriv1, deriv2, bwd, lambda);
        ApplyMetric2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, nelmt * nqTot, w0, w1, s_f0, s_f1, dfptr,
            jacptr, coeff, deriv, deriv0, deriv1, bwd, (TData)1.0);
        SumDerivTensor2DKernel<false, true>(ilane, nq0, nq1, D0, D1, deriv0,
                                            deriv1, bwd);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          nq1 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseQuadSumFacKernel<false, false>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, bwd, outptr, wsp0,
                (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wsp0 = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<false, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                outptr, wsp0, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr = wsp + (1 + ncoord) * nqTot * nelmt * ncomp +
                             nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + ((1 + ncoord) * nqTot + nmTot) * nelmt * ncomp +
                          std::max(nq1, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<false, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, bwd,
                out1ptr, wsp0, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    LinAdvDiffReaction3DKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT advVel0,
        const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Pre-compute factor.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);
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

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nmTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        TData *bwd         = wsp + nqTot * (nelmt * c + warpsize * iwarp);
        TData *deriv       = wsp + nqTot * nelmt * ncomp +
                       nqTot * (3 * nelmt * c + warpsize * iwarp);
        TData *deriv0 = wsp + nqTot * nelmt * ncomp +
                        nqTot * (3 * nelmt * c + warpsize * iwarp);
        TData *deriv1 = wsp + nqTot * nelmt * (ncomp + 1) +
                        nqTot * (3 * nelmt * c + warpsize * iwarp);
        TData *deriv2 = wsp + nqTot * nelmt * (ncomp + 2) +
                        nqTot * (3 * nelmt * c + warpsize * iwarp);
        const TData *advVel0ptr = advVel0 + nqTot * warpsize * iwarp;
        const TData *advVel1ptr = advVel1 + nqTot * warpsize * iwarp;
        const TData *advVel2ptr = advVel2 + nqTot * warpsize * iwarp;

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            BwdTransHexSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           basis0, basis1, basis2, inptr, bwd,
                                           wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransTetSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           inptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *in1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                            nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransTetSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           in1ptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransPrismSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1,
                                             nq2, isModified, basis0, basis1,
                                             basis2, inptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *in1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                            nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 =
                wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                              nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            MatVecKernel(ilane, nmTot, nodToMod, inptr, in1ptr);
            BwdTransPrismSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1,
                                             nq2, isModified, basis0, basis1,
                                             basis2, in1ptr, bwd, wsp0, wsp1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            BwdTransPyrSumFacKernel<false>(ilane, nm0, nm1, nm2, nq0, nq1, nq2,
                                           isModified, basis0, basis1, basis2,
                                           inptr, bwd, wsp0, wsp1);
        }
        PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, nelmt * nqTot, D0, D1, D2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, bwd, deriv);
        AddAdvection3DKernel(ilane, nq0, nq1, nq2, advVel0ptr, advVel1ptr,
                             advVel2ptr, deriv0, deriv1, deriv2, bwd, lambda);
        ApplyMetric3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, nelmt * nqTot, w0, w1, w2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, jacptr, coeff, deriv, deriv0, deriv1, deriv2, bwd,
            (TData)1.0);
        SumDerivTensor3DKernel<true, true>(ilane, nq0, nq1, nq2, D0, D1, D2,
                                           deriv0, deriv1, deriv2, bwd);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseHexSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + 4 * nqTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData *out1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                             nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 = wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + nq1 * nq2) * nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr = wsp + 4 * nqTot * nelmt * ncomp +
                             nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0 =
                wsp + (nmTot + 4 * nqTot) * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp +
                          (nmTot + 4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) *
                              nelmt * ncomp +
                          std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, out1ptr, wsp0, wsp1, (TData)1.0);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<false, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 =
                wsp + 4 * nqTot * nelmt * ncomp +
                std::max(nq1 * nq2, nm0 * nm1) * (nelmt * c + warpsize * iwarp);
            TData *wsp1 =
                wsp +
                (4 * nqTot + std::max(nq1 * nq2, nm0 * nm1)) * nelmt * ncomp +
                std::max(nq2, nm0) * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePyrSumFacKernel<false, false>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, bwd, outptr, wsp0, wsp1, (TData)1.0);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
