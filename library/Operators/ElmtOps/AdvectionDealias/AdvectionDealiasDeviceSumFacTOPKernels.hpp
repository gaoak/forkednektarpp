///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasDeviceSumFacTOPKernels.hpp
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
// Description: Fused per-shape launchers and fine-grid product for 3/2-rule
// dealiased advection, SumFacTOP (block-per-element) convention.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

// The 3/2 dealiasing rule for one direction of an element. A direction whose
// quadrature sits one point below the first keeps that one-point gap on the
// fine grid rather than rounding on its own, so the templated switch over
// collapsed coordinates still matches a shape it has an instantiation for;
// passing nq0 for nq gives the first direction itself.
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionDealiasFineSize(
    const unsigned int nq0, const unsigned int nq)
{
    return (nq0 - nq == 1u) ? 3u * nq0 / 2u - 1u : 3u * nq / 2u;
}

// A tensor-contraction intermediate is reused by the interpolation
// (native->fine) and the projection (fine->native), which leave different
// numbers of points in it - a cross-axis product, not a same-axis sum - so it
// has to span whichever of the two is larger.
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionDealiasScratchSize(
    const unsigned int interp, const unsigned int project)
{
    return (interp > project) ? interp : project;
}

// Per-element workspace, in TData elements: the reference derivatives, the
// fine-grid buffers, the accumulators, then one tensor-contraction
// intermediate per contracted direction. The launchers lay the regions out in
// this order and the host sizes its allocation with the same call, so the two
// cannot disagree. The local pipeline fuses the interpolation with the
// advection product, so it carries one derivative, one fine-grid buffer and
// one accumulator per element; unfused, every component and direction needs
// its own. Only the pipeline changes the sizes - which tier of memory the
// regions then come from does not.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t AdvectionDealiasWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord, const unsigned int ncomp,
    const TPhysSizeParameter1D sizeParam1D, const bool localPipeline)
{
    const unsigned int nq0     = sizeParam1D.nq0();
    const unsigned int nq0Fine = AdvectionDealiasFineSize(nq0, nq0);

    const unsigned int nderiv = localPipeline ? 1u : ncomp;
    const unsigned int nfine =
        localPipeline ? 2u : (ncoord + ncomp * ncoord + ncomp);

    return (nderiv * ncoord * nq0 + nfine * nq0Fine) * nelmt;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t AdvectionDealiasWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord, const unsigned int ncomp,
    const TPhysSizeParameter2D sizeParam2D, const bool localPipeline)
{
    const unsigned int nq0     = sizeParam2D.nq0();
    const unsigned int nq0Fine = AdvectionDealiasFineSize(nq0, nq0);

    const unsigned int nderiv = localPipeline ? 1u : ncomp;
    const unsigned int nfine =
        localPipeline ? 2u : (ncoord + ncomp * ncoord + ncomp);

    const unsigned int nq1     = sizeParam2D.nq1();
    const unsigned int nq1Fine = AdvectionDealiasFineSize(nq0, nq1);
    const unsigned int wsp0Stride =
        AdvectionDealiasScratchSize(nq0Fine * nq1, nq0 * nq1Fine);

    return (nderiv * ncoord * nq0 * nq1 + nfine * nq0Fine * nq1Fine +
            wsp0Stride) *
           nelmt;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t AdvectionDealiasWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord, const unsigned int ncomp,
    const TPhysSizeParameter3D sizeParam3D, const bool localPipeline)
{
    const unsigned int nq0     = sizeParam3D.nq0();
    const unsigned int nq0Fine = AdvectionDealiasFineSize(nq0, nq0);

    const unsigned int nderiv = localPipeline ? 1u : ncomp;
    const unsigned int nfine =
        localPipeline ? 2u : (ncoord + ncomp * ncoord + ncomp);

    const unsigned int nq1        = sizeParam3D.nq1();
    const unsigned int nq2        = sizeParam3D.nq2();
    const unsigned int nq1Fine    = AdvectionDealiasFineSize(nq0, nq1);
    const unsigned int nq2Fine    = AdvectionDealiasFineSize(nq0, nq2);
    const unsigned int wsp0Stride = AdvectionDealiasScratchSize(
        nq0Fine * nq1 * nq2, nq0 * nq1Fine * nq2Fine);
    const unsigned int wsp1Stride = AdvectionDealiasScratchSize(
        nq0Fine * nq1Fine * nq2, nq0 * nq1 * nq2Fine);

    return (nderiv * ncoord * nq0 * nq1 * nq2 +
            nfine * nq0Fine * nq1Fine * nq2Fine + wsp0Stride + wsp1Stride) *
           nelmt;
}

// How much of the workspace block-local memory holds, in TData elements. The
// regions live at the end of the layout AdvectionDealiasWorkSpaceSize
// describes, so this is always a suffix of it: the fused pipeline claims the
// accumulator and the tensor-contraction intermediates, the middle tier the
// intermediates alone, and the last tier nothing. They earn it in that order -
// the intermediates are written and read once per contraction stage, and the
// accumulator once per direction plus once more by the projection.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionDealiasSharedWorkSpaceSize(
    const TPhysSizeParameter1D sizeParam1D, const bool localPipeline,
    const bool sharedScratch)
{
    if (!localPipeline && !sharedScratch)
    {
        return 0u;
    }

    const unsigned int nq0     = sizeParam1D.nq0();
    const unsigned int nq0Fine = AdvectionDealiasFineSize(nq0, nq0);

    // Seg contracts a single direction in place, so there is no scratch to
    // promote on its own.
    return localPipeline ? nq0Fine : 0u;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionDealiasSharedWorkSpaceSize(
    const TPhysSizeParameter2D sizeParam2D, const bool localPipeline,
    const bool sharedScratch)
{
    if (!localPipeline && !sharedScratch)
    {
        return 0u;
    }

    const unsigned int nq0     = sizeParam2D.nq0();
    const unsigned int nq0Fine = AdvectionDealiasFineSize(nq0, nq0);

    const unsigned int nq1     = sizeParam2D.nq1();
    const unsigned int nq1Fine = AdvectionDealiasFineSize(nq0, nq1);
    const unsigned int wsp0Stride =
        AdvectionDealiasScratchSize(nq0Fine * nq1, nq0 * nq1Fine);

    return (localPipeline ? nq0Fine * nq1Fine : 0u) + wsp0Stride;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionDealiasSharedWorkSpaceSize(
    const TPhysSizeParameter3D sizeParam3D, const bool localPipeline,
    const bool sharedScratch)
{
    if (!localPipeline && !sharedScratch)
    {
        return 0u;
    }

    const unsigned int nq0     = sizeParam3D.nq0();
    const unsigned int nq0Fine = AdvectionDealiasFineSize(nq0, nq0);

    const unsigned int nq1        = sizeParam3D.nq1();
    const unsigned int nq2        = sizeParam3D.nq2();
    const unsigned int nq1Fine    = AdvectionDealiasFineSize(nq0, nq1);
    const unsigned int nq2Fine    = AdvectionDealiasFineSize(nq0, nq2);
    const unsigned int wsp0Stride = AdvectionDealiasScratchSize(
        nq0Fine * nq1 * nq2, nq0 * nq1Fine * nq2Fine);
    const unsigned int wsp1Stride = AdvectionDealiasScratchSize(
        nq0Fine * nq1Fine * nq2, nq0 * nq1 * nq2Fine);

    return (localPipeline ? nq0Fine * nq1Fine * nq2Fine : 0u) + wsp0Stride +
           wsp1Stride;
}

// SumFacTOP assigns one element to each thread block, so block-local memory
// holds part of that element's workspace - and nothing else. The host picks
// the tier; this is what the pick costs.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
inline constexpr unsigned int AdvectionDealiasSharedMemorySize(
    const TPhysSizeParameter sizeParam, const bool localPipeline,
    const bool sharedScratch)
{
    return AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE, Implementation>(
        sizeParam, localPipeline, sharedScratch);
}

// Stage C: advVel/grad/out are already offset to this single element's
// data; every thread in the block cooperatively covers the nqTot points.
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AdvectionDealiasCombineSumFacTOPKernel(
    const unsigned int nqTot, const unsigned int coordDim,
    const TData *NEK_RESTRICT advVel, const size_t advVelOffset,
    const TData *NEK_RESTRICT grad, const size_t gradOffset,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int j = idx0; j < nqTot; j += stride)
    {
        TData tmp = advVel[j] * grad[j];
#pragma unroll
        for (unsigned int d = 1u; d < coordDim; ++d)
        {
            tmp += advVel[d * advVelOffset + j] * grad[d * gradOffset + j];
        }

        if constexpr (APPEND)
        {
            out[j] += scale * tmp;
        }
        else
        {
            out[j] = scale * tmp;
        }
    }

    localBarrier(threadBlock);
}

// Interpolate a scalar field and fuse the final tensor contraction with the
// pointwise multiply/accumulate used by dealiased advection. This avoids
// materialising a second complete field on the fine grid.
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransSegSumFacTOPMultiplyKernel(
    const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    const TData *NEK_RESTRICT multiplier, TData *NEK_RESTRICT out,
    const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        TData tmp = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p)
        {
            tmp += in[p] * basis0[p * nq0 + i];
        }

        if constexpr (APPEND)
        {
            out[i] += scale * multiplier[i] * tmp;
        }
        else
        {
            out[i] = scale * multiplier[i] * tmp;
        }
    }

    localBarrier(threadBlock);
}

// Interpolate a scalar field and fuse the final tensor contraction with the
// pointwise multiply/accumulate used by dealiased advection. This avoids
// materialising a second complete field on the fine grid.
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransQuadSumFacTOPMultiplyKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nq0,
    const unsigned int nq1, const unsigned int nqTot,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, const TData *NEK_RESTRICT multiplier,
    TData *NEK_RESTRICT out, const TData scale, TData *NEK_RESTRICT wsp,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

        if constexpr (APPEND)
        {
            out[idx] += scale * multiplier[idx] * tmp;
        }
        else
        {
            out[idx] = scale * multiplier[idx] * tmp;
        }
    }

    localBarrier(threadBlock);
}

// Interpolate a scalar field and fuse the final tensor contraction with the
// pointwise multiply/accumulate used by dealiased advection. This avoids
// materialising a second complete field on the fine grid.
template <bool APPEND, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void BwdTransHexSumFacTOPMultiplyKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nqTot, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, const TData *NEK_RESTRICT multiplier,
    TData *NEK_RESTRICT out, const TData scale, TData *NEK_RESTRICT wsp0,
    TData *NEK_RESTRICT wsp1, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nm1 * nm2; idx += stride)
    {
        const unsigned int q = idx % nm1;
        const unsigned int r = (idx / nm1) % nm2;
        const unsigned int i = idx / (nm1 * nm2);
        unsigned int cnt     = nm1 * nm0 * r + nm0 * q;
        TData tmp            = 0.0;
#pragma unroll
        for (unsigned int p = 0u; p < nm0; ++p, ++cnt)
        {
            tmp += in[cnt] * basis0[p * nq0 + i];
        }
        wsp0[idx] = tmp;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq1 * nm2; idx += stride)
    {
        const unsigned int r = idx % nm2;
        const unsigned int i = (idx / nm2) % nq0;
        const unsigned int j = idx / (nm2 * nq0);
        unsigned int cnt     = nm1 * nm2 * i + nm1 * r;
        TData tmp            = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nm1; ++q, ++cnt)
        {
            tmp += wsp0[cnt] * basis1[q * nq1 + j];
        }
        wsp1[idx] = tmp;
    }

    localBarrier(threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq1 * nq0);
        unsigned int cnt     = nq0 * nm2 * j + nm2 * i;
        TData tmp            = 0.0;
#pragma unroll
        for (unsigned int r = 0u; r < nm2; ++r, ++cnt)
        {
            tmp += wsp1[cnt] * basis2[r * nq2 + k];
        }

        if constexpr (APPEND)
        {
            out[idx] += scale * multiplier[idx] * tmp;
        }
        else
        {
            out[idx] = scale * multiplier[idx] * tmp;
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, bool LOCAL_PIPELINE, bool SHARED_SCRATCH,
          typename TPhysSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionDealiasKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const unsigned int ncomp, const unsigned int nhomo,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT Binterp0, const TData *NEK_RESTRICT Bproject0,
    const TData scale, const TData *NEK_RESTRICT in, const size_t inCompStride,
    const TData *NEK_RESTRICT advVel, const size_t advVelPlaneStride,
    TData *NEK_RESTRICT out, const size_t outCompStride,
    TData *NEK_RESTRICT elmtWsp, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int coordDim = sizeParam1D.ncoord();
    const unsigned int nq0      = sizeParam1D.nq0();
    const unsigned int nq0Fine  = AdvectionDealiasFineSize(nq0, nq0);

    const unsigned int nqTot     = nq0;
    const unsigned int nqTotFine = nq0Fine;
    const unsigned int dfsize    = DEFORMED ? nqTot : 1u;

    const size_t advVelCompStride = advVelPlaneStride * nhomo;

    // The workspace split, from the same two helpers the host sized the
    // allocation with: block-local memory holds a suffix of the layout, and
    // the prefix left over is what each element owns in the static
    // allocation.
    const unsigned int wspSize = static_cast<unsigned int>(
        AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
            1u, coordDim, ncomp, sizeParam1D, LOCAL_PIPELINE));
    const unsigned int sharedWspSize =
        AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE, Implementation>(
            sizeParam1D, LOCAL_PIPELINE, SHARED_SCRATCH);
    const unsigned int staticSize = wspSize - sharedWspSize;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr       = df + coordDim * dfsize * e;
        const TData *inGroup     = in + nqTot * e;
        const TData *advVelGroup = advVel + p * advVelPlaneStride + nqTot * e;
        TData *outGroup          = out + nqTot * e;

        TData *staticBase = elmtWsp + staticSize * (nelmt * p + e);
        TData *sharedBase = (TData *)shmemptr;

        if constexpr (LOCAL_PIPELINE)
        {
            // One derivative, one fine-grid buffer and one accumulator per
            // element: the interpolation is fused with the advection product,
            // so no gradient is ever materialised on the fine grid. Splitting
            // the bases here rather than behind a runtime select keeps each
            // pointer's address space known to the compiler, so the
            // block-local buffers compile to block-local loads and stores.
            TData *derivOut = staticBase;
            TData *fineTemp = staticBase + coordDim * nqTot;
            TData *combined = sharedBase;

            for (unsigned int c = 0u; c < ncomp; ++c)
            {
                PhysDeriv1DSumFacTOPKernel<DEFORMED>(
                    coordDim, nq0, nqTot, D0, dfptr,
                    inGroup + (c * nhomo + p) * inCompStride, derivOut,
                    threadBlock);

                for (unsigned int d = 0u; d < coordDim; ++d)
                {
                    BwdTransSegSumFacTOPKernel<false>(nq0, nq0Fine, Binterp0,
                                                      advVelGroup +
                                                          d * advVelCompStride,
                                                      fineTemp, threadBlock);

                    if (d == 0u)
                    {
                        BwdTransSegSumFacTOPMultiplyKernel<false>(
                            nq0, nq0Fine, Binterp0, derivOut + d * nqTot,
                            fineTemp, combined, scale, threadBlock);
                    }
                    else
                    {
                        BwdTransSegSumFacTOPMultiplyKernel<true>(
                            nq0, nq0Fine, Binterp0, derivOut + d * nqTot,
                            fineTemp, combined, scale, threadBlock);
                    }
                }

                BwdTransSegSumFacTOPKernel<APPEND>(
                    nq0Fine, nq0, Bproject0, combined,
                    outGroup + (c * nhomo + p) * outCompStride, threadBlock);
            }
        }
        else
        {
            // Every fine-grid field is materialised in the static workspace
            // and combined in a separate pass. The tensor-contraction
            // intermediates still earn block-local memory on their own where
            // they fit, which is the only thing SHARED_SCRATCH changes.
            TData *derivGroup = staticBase;
            TData *fineGroup  = derivGroup + ncomp * coordDim * nqTot;
            TData *combinedGroup =
                fineGroup + (coordDim + ncomp * coordDim) * nqTotFine;

            for (unsigned int d = 0u; d < coordDim; ++d)
            {
                BwdTransSegSumFacTOPKernel<false>(
                    nq0, nq0Fine, Binterp0, advVelGroup + d * advVelCompStride,
                    fineGroup + d * nqTotFine, threadBlock);
            }

            for (unsigned int c = 0u; c < ncomp; ++c)
            {
                TData *derivOut = derivGroup + c * coordDim * nqTot;
                TData *gradFine =
                    fineGroup + (coordDim + c * coordDim) * nqTotFine;
                TData *combinedOut = combinedGroup + c * nqTotFine;

                PhysDeriv1DSumFacTOPKernel<DEFORMED>(
                    coordDim, nq0, nqTot, D0, dfptr,
                    inGroup + (c * nhomo + p) * inCompStride, derivOut,
                    threadBlock);

                for (unsigned int d = 0u; d < coordDim; ++d)
                {
                    BwdTransSegSumFacTOPKernel<false>(
                        nq0, nq0Fine, Binterp0, derivOut + d * nqTot,
                        gradFine + d * nqTotFine, threadBlock);
                }

                AdvectionDealiasCombineSumFacTOPKernel<false>(
                    nqTotFine, coordDim, fineGroup, nqTotFine, gradFine,
                    nqTotFine, combinedOut, scale, threadBlock);

                BwdTransSegSumFacTOPKernel<APPEND>(
                    nq0Fine, nq0, Bproject0, combinedOut,
                    outGroup + (c * nhomo + p) * outCompStride, threadBlock);
            }
        }

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, bool LOCAL_PIPELINE, bool SHARED_SCRATCH,
          typename TPhysSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionDealiasKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const unsigned int ncomp, const unsigned int nhomo,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT Binterp0,
    const TData *NEK_RESTRICT Binterp1, const TData *NEK_RESTRICT Bproject0,
    const TData *NEK_RESTRICT Bproject1, const TData scale,
    const TData *NEK_RESTRICT in, const size_t inCompStride,
    const TData *NEK_RESTRICT advVel, const size_t advVelPlaneStride,
    TData *NEK_RESTRICT out, const size_t outCompStride,
    TData *NEK_RESTRICT elmtWsp, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int coordDim  = sizeParam2D.ncoord();
    const unsigned int nq0       = sizeParam2D.nq0();
    const unsigned int nq1       = sizeParam2D.nq1();
    const unsigned int nq0Fine   = AdvectionDealiasFineSize(nq0, nq0);
    const unsigned int nq1Fine   = AdvectionDealiasFineSize(nq0, nq1);
    const unsigned int nqTot     = nq0 * nq1;
    const unsigned int nqTotFine = nq0Fine * nq1Fine;
    const unsigned int ndf       = 2u * coordDim;
    const unsigned int dfsize    = DEFORMED ? nqTot : 1u;

    const size_t advVelCompStride = advVelPlaneStride * nhomo;

    // The workspace split, from the same two helpers the host sized the
    // allocation with: block-local memory holds a suffix of the layout, and
    // the prefix left over is what each element owns in the static
    // allocation.
    const unsigned int wspSize = static_cast<unsigned int>(
        AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
            1u, coordDim, ncomp, sizeParam2D, LOCAL_PIPELINE));
    const unsigned int sharedWspSize =
        AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE, Implementation>(
            sizeParam2D, LOCAL_PIPELINE, SHARED_SCRATCH);
    const unsigned int staticSize = wspSize - sharedWspSize;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr       = df + ndf * dfsize * e;
        const TData *inGroup     = in + nqTot * e;
        const TData *advVelGroup = advVel + p * advVelPlaneStride + nqTot * e;
        TData *outGroup          = out + nqTot * e;

        TData *staticBase = elmtWsp + staticSize * (nelmt * p + e);
        TData *sharedBase = (TData *)shmemptr;

        if constexpr (LOCAL_PIPELINE)
        {
            // One derivative, one fine-grid buffer and one accumulator per
            // element: the interpolation is fused with the advection product,
            // so no gradient is ever materialised on the fine grid. Splitting
            // the bases here rather than behind a runtime select keeps each
            // pointer's address space known to the compiler, so the
            // block-local buffers compile to block-local loads and stores.
            TData *derivOut = staticBase;
            TData *fineTemp = staticBase + coordDim * nqTot;
            TData *combined = sharedBase;
            TData *wsp0     = combined + nqTotFine;

            for (unsigned int c = 0u; c < ncomp; ++c)
            {
                PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    coordDim, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr,
                    inGroup + (c * nhomo + p) * inCompStride, derivOut,
                    threadBlock);

                for (unsigned int d = 0u; d < coordDim; ++d)
                {
                    BwdTransQuadSumFacTOPKernel<false>(
                        nq0, nq1, nq0Fine, nq1Fine, nqTotFine, Binterp0,
                        Binterp1, advVelGroup + d * advVelCompStride, fineTemp,
                        wsp0, threadBlock);

                    if (d == 0u)
                    {
                        BwdTransQuadSumFacTOPMultiplyKernel<false>(
                            nq0, nq1, nq0Fine, nq1Fine, nqTotFine, Binterp0,
                            Binterp1, derivOut + d * nqTot, fineTemp, combined,
                            scale, wsp0, threadBlock);
                    }
                    else
                    {
                        BwdTransQuadSumFacTOPMultiplyKernel<true>(
                            nq0, nq1, nq0Fine, nq1Fine, nqTotFine, Binterp0,
                            Binterp1, derivOut + d * nqTot, fineTemp, combined,
                            scale, wsp0, threadBlock);
                    }
                }

                BwdTransQuadSumFacTOPKernel<APPEND>(
                    nq0Fine, nq1Fine, nq0, nq1, nqTot, Bproject0, Bproject1,
                    combined, outGroup + (c * nhomo + p) * outCompStride, wsp0,
                    threadBlock);
            }
        }
        else
        {
            // Every fine-grid field is materialised in the static workspace
            // and combined in a separate pass. The tensor-contraction
            // intermediates still earn block-local memory on their own where
            // they fit, which is the only thing SHARED_SCRATCH changes.
            TData *derivGroup = staticBase;
            TData *fineGroup  = derivGroup + ncomp * coordDim * nqTot;
            TData *combinedGroup =
                fineGroup + (coordDim + ncomp * coordDim) * nqTotFine;
            // if constexpr, not a select: a pointer that could be either
            // would have to be addressed generically.
            TData *wsp0 = nullptr;
            if constexpr (SHARED_SCRATCH)
            {
                wsp0 = sharedBase;
            }
            else
            {
                wsp0 = combinedGroup + ncomp * nqTotFine;
            }

            for (unsigned int d = 0u; d < coordDim; ++d)
            {
                BwdTransQuadSumFacTOPKernel<false>(
                    nq0, nq1, nq0Fine, nq1Fine, nqTotFine, Binterp0, Binterp1,
                    advVelGroup + d * advVelCompStride,
                    fineGroup + d * nqTotFine, wsp0, threadBlock);
            }

            for (unsigned int c = 0u; c < ncomp; ++c)
            {
                TData *derivOut = derivGroup + c * coordDim * nqTot;
                TData *gradFine =
                    fineGroup + (coordDim + c * coordDim) * nqTotFine;
                TData *combinedOut = combinedGroup + c * nqTotFine;

                PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    coordDim, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr,
                    inGroup + (c * nhomo + p) * inCompStride, derivOut,
                    threadBlock);

                for (unsigned int d = 0u; d < coordDim; ++d)
                {
                    BwdTransQuadSumFacTOPKernel<false>(
                        nq0, nq1, nq0Fine, nq1Fine, nqTotFine, Binterp0,
                        Binterp1, derivOut + d * nqTot,
                        gradFine + d * nqTotFine, wsp0, threadBlock);
                }

                AdvectionDealiasCombineSumFacTOPKernel<false>(
                    nqTotFine, coordDim, fineGroup, nqTotFine, gradFine,
                    nqTotFine, combinedOut, scale, threadBlock);

                BwdTransQuadSumFacTOPKernel<APPEND>(
                    nq0Fine, nq1Fine, nq0, nq1, nqTot, Bproject0, Bproject1,
                    combinedOut, outGroup + (c * nhomo + p) * outCompStride,
                    wsp0, threadBlock);
            }
        }

        e += getBlockRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, bool LOCAL_PIPELINE, bool SHARED_SCRATCH,
          typename TPhysSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionDealiasKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const unsigned int ncomp, const unsigned int nhomo,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT Binterp0, const TData *NEK_RESTRICT Binterp1,
    const TData *NEK_RESTRICT Binterp2, const TData *NEK_RESTRICT Bproject0,
    const TData *NEK_RESTRICT Bproject1, const TData *NEK_RESTRICT Bproject2,
    const TData scale, const TData *NEK_RESTRICT in, const size_t inCompStride,
    const TData *NEK_RESTRICT advVel, const size_t advVelPlaneStride,
    TData *NEK_RESTRICT out, const size_t outCompStride,
    TData *NEK_RESTRICT elmtWsp, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0            = sizeParam3D.nq0();
    const unsigned int nq1            = sizeParam3D.nq1();
    const unsigned int nq2            = sizeParam3D.nq2();
    const unsigned int nq0Fine        = AdvectionDealiasFineSize(nq0, nq0);
    const unsigned int nq1Fine        = AdvectionDealiasFineSize(nq0, nq1);
    const unsigned int nq2Fine        = AdvectionDealiasFineSize(nq0, nq2);
    constexpr unsigned int coordDim3D = 3u;
    const unsigned int nqTot          = nq0 * nq1 * nq2;
    const unsigned int nqTotFine      = nq0Fine * nq1Fine * nq2Fine;
    constexpr unsigned int ndf        = 9u;
    const unsigned int dfsize         = DEFORMED ? nqTot : 1u;
    // wsp1 starts past wsp0, so it needs the same span
    // AdvectionDealiasWorkSpaceSize gave it.
    const unsigned int wsp0Stride = AdvectionDealiasScratchSize(
        nq0Fine * nq1 * nq2, nq0 * nq1Fine * nq2Fine);

    const size_t advVelCompStride = advVelPlaneStride * nhomo;

    // The workspace split, from the same two helpers the host sized the
    // allocation with: block-local memory holds a suffix of the layout, and
    // the prefix left over is what each element owns in the static
    // allocation.
    const unsigned int wspSize = static_cast<unsigned int>(
        AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
            1u, coordDim3D, ncomp, sizeParam3D, LOCAL_PIPELINE));
    const unsigned int sharedWspSize =
        AdvectionDealiasSharedWorkSpaceSize<SHAPE_TYPE, Implementation>(
            sizeParam3D, LOCAL_PIPELINE, SHARED_SCRATCH);
    const unsigned int staticSize = wspSize - sharedWspSize;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr       = df + ndf * dfsize * e;
        const TData *inGroup     = in + nqTot * e;
        const TData *advVelGroup = advVel + p * advVelPlaneStride + nqTot * e;
        TData *outGroup          = out + nqTot * e;

        TData *staticBase = elmtWsp + staticSize * (nelmt * p + e);
        TData *sharedBase = (TData *)shmemptr;

        if constexpr (LOCAL_PIPELINE)
        {
            // One derivative, one fine-grid buffer and one accumulator per
            // element: the interpolation is fused with the advection product,
            // so no gradient is ever materialised on the fine grid. Splitting
            // the bases here rather than behind a runtime select keeps each
            // pointer's address space known to the compiler, so the
            // block-local buffers compile to block-local loads and stores.
            TData *derivOut = staticBase;
            TData *fineTemp = staticBase + coordDim3D * nqTot;
            TData *combined = sharedBase;
            TData *wsp0     = combined + nqTotFine;
            TData *wsp1     = wsp0 + wsp0Stride;

            for (unsigned int c = 0u; c < ncomp; ++c)
            {
                PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr,
                    inGroup + (c * nhomo + p) * inCompStride, derivOut,
                    threadBlock);

                for (unsigned int d = 0u; d < coordDim3D; ++d)
                {
                    BwdTransHexSumFacTOPKernel<false>(
                        nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, nqTotFine,
                        Binterp0, Binterp1, Binterp2,
                        advVelGroup + d * advVelCompStride, fineTemp, wsp0,
                        wsp1, threadBlock);

                    if (d == 0u)
                    {
                        BwdTransHexSumFacTOPMultiplyKernel<false>(
                            nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, nqTotFine,
                            Binterp0, Binterp1, Binterp2, derivOut + d * nqTot,
                            fineTemp, combined, scale, wsp0, wsp1, threadBlock);
                    }
                    else
                    {
                        BwdTransHexSumFacTOPMultiplyKernel<true>(
                            nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, nqTotFine,
                            Binterp0, Binterp1, Binterp2, derivOut + d * nqTot,
                            fineTemp, combined, scale, wsp0, wsp1, threadBlock);
                    }
                }

                BwdTransHexSumFacTOPKernel<APPEND>(
                    nq0Fine, nq1Fine, nq2Fine, nq0, nq1, nq2, nqTot, Bproject0,
                    Bproject1, Bproject2, combined,
                    outGroup + (c * nhomo + p) * outCompStride, wsp0, wsp1,
                    threadBlock);
            }
        }
        else
        {
            // Every fine-grid field is materialised in the static workspace
            // and combined in a separate pass. The tensor-contraction
            // intermediates still earn block-local memory on their own where
            // they fit, which is the only thing SHARED_SCRATCH changes.
            TData *derivGroup = staticBase;
            TData *fineGroup  = derivGroup + ncomp * coordDim3D * nqTot;
            TData *combinedGroup =
                fineGroup + (coordDim3D + ncomp * coordDim3D) * nqTotFine;
            // if constexpr, not a select: a pointer that could be either
            // would have to be addressed generically.
            TData *wsp0 = nullptr;
            if constexpr (SHARED_SCRATCH)
            {
                wsp0 = sharedBase;
            }
            else
            {
                wsp0 = combinedGroup + ncomp * nqTotFine;
            }
            TData *wsp1 = wsp0 + wsp0Stride;

            for (unsigned int d = 0u; d < coordDim3D; ++d)
            {
                BwdTransHexSumFacTOPKernel<false>(
                    nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, nqTotFine,
                    Binterp0, Binterp1, Binterp2,
                    advVelGroup + d * advVelCompStride,
                    fineGroup + d * nqTotFine, wsp0, wsp1, threadBlock);
            }

            for (unsigned int c = 0u; c < ncomp; ++c)
            {
                TData *derivOut = derivGroup + c * coordDim3D * nqTot;
                TData *gradFine =
                    fineGroup + (coordDim3D + c * coordDim3D) * nqTotFine;
                TData *combinedOut = combinedGroup + c * nqTotFine;

                PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                    nq0, nq1, nq2, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr,
                    inGroup + (c * nhomo + p) * inCompStride, derivOut,
                    threadBlock);

                for (unsigned int d = 0u; d < coordDim3D; ++d)
                {
                    BwdTransHexSumFacTOPKernel<false>(
                        nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, nqTotFine,
                        Binterp0, Binterp1, Binterp2, derivOut + d * nqTot,
                        gradFine + d * nqTotFine, wsp0, wsp1, threadBlock);
                }

                AdvectionDealiasCombineSumFacTOPKernel<false>(
                    nqTotFine, coordDim3D, fineGroup, nqTotFine, gradFine,
                    nqTotFine, combinedOut, scale, threadBlock);

                BwdTransHexSumFacTOPKernel<APPEND>(
                    nq0Fine, nq1Fine, nq2Fine, nq0, nq1, nq2, nqTot, Bproject0,
                    Bproject1, Bproject2, combinedOut,
                    outGroup + (c * nhomo + p) * outCompStride, wsp0, wsp1,
                    threadBlock);
            }
        }

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
