///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasDeviceSumFacKernels.hpp
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
// dealiased advection, SumFac (warp-per-element) convention.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

// Per-warp-group workspace, in TData elements: the reference derivatives, the
// fine-grid fields, the products, then one tensor-contraction intermediate per
// contracted direction. The launchers below lay the regions out in that order,
// and the host sizes its static allocation with the same call.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t AdvectionDealiasWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord, const unsigned int ncomp,
    const TPhysSizeParameter1D sizeParam1D,
    [[maybe_unused]] const bool localPipeline)
{
    const unsigned int nq0     = sizeParam1D.nq0();
    const unsigned int nq0Fine = 3u * nq0 / 2u;

    return (ncomp * ncoord * nq0 +
            (ncoord + ncomp * ncoord + ncomp) * nq0Fine) *
           nelmt;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t AdvectionDealiasWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord, const unsigned int ncomp,
    const TPhysSizeParameter2D sizeParam2D,
    [[maybe_unused]] const bool localPipeline)
{
    const unsigned int nq0     = sizeParam2D.nq0();
    const unsigned int nq0Fine = 3u * nq0 / 2u;

    const unsigned int nq1 = sizeParam2D.nq1();
    const unsigned int nq1Fine =
        (nq0 - nq1 == 1u) ? nq0Fine - 1u : 3u * nq1 / 2u;

    return (ncomp * ncoord * nq0 * nq1 +
            (ncoord + ncomp * ncoord + ncomp) * nq0Fine * nq1Fine +
            (nq1 + nq1Fine)) *
           nelmt;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t AdvectionDealiasWorkSpaceSize(
    const size_t nelmt, const unsigned int ncoord, const unsigned int ncomp,
    const TPhysSizeParameter3D sizeParam3D,
    [[maybe_unused]] const bool localPipeline)
{
    const unsigned int nq0     = sizeParam3D.nq0();
    const unsigned int nq0Fine = 3u * nq0 / 2u;

    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();
    const unsigned int nq1Fine =
        (nq0 - nq1 == 1u) ? nq0Fine - 1u : 3u * nq1 / 2u;
    const unsigned int nq2Fine =
        (nq0 - nq2 == 1u) ? nq0Fine - 1u : 3u * nq2 / 2u;

    return (ncomp * ncoord * nq0 * nq1 * nq2 +
            (ncoord + ncomp * ncoord + ncomp) * nq0Fine * nq1Fine * nq2Fine +
            (nq1 * nq2 + nq1Fine * nq2Fine) + (nq2 + nq2Fine)) *
           nelmt;
}

// SumFac keeps its whole workspace in the static allocation - shared memory
// there holds the collapsed-coordinate broadcast tables instead.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int AdvectionDealiasSharedWorkSpaceSize(
    [[maybe_unused]] const TPhysSizeParameter sizeParam,
    [[maybe_unused]] const bool localPipeline,
    [[maybe_unused]] const bool sharedScratch)
{
    return 0u;
}

// Shared memory holds the collapsed-coordinate broadcast tables the reused
// PhysDeriv kernels read, so the requirement is PhysDeriv's own.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
inline constexpr unsigned int AdvectionDealiasSharedMemorySize(
    const TPhysSizeParameter sizeParam,
    [[maybe_unused]] const bool localPipeline,
    [[maybe_unused]] const bool sharedScratch)
{
    return PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
}

// Stage C: advVel/grad/out are already offset to this warp-group's data;
// ilane selects the lane (element) within the warp.
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void AdvectionDealiasCombineSumFacKernel(
    const unsigned int ilane, const unsigned int nqTot,
    const unsigned int coordDim, const TData *NEK_RESTRICT advVel,
    const size_t advVelOffset, const TData *NEK_RESTRICT grad,
    const size_t gradOffset, TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int j = 0u; j < nqTot; ++j)
    {
        TData tmp = advVel[warpsize * j + ilane] * grad[warpsize * j + ilane];
#pragma unroll
        for (unsigned int d = 1u; d < coordDim; ++d)
        {
            // advVelOffset/gradOffset are already warpsize-scaled strides
            // (matching the caller's nqTotFine*warpsize), so only the `j`
            // term needs the extra warpsize factor here - not the whole
            // (d*offset+j) sum.
            tmp += advVel[d * advVelOffset + warpsize * j + ilane] *
                   grad[d * gradOffset + warpsize * j + ilane];
        }

        if constexpr (APPEND)
        {
            out[warpsize * j + ilane] += scale * tmp;
        }
        else
        {
            out[warpsize * j + ilane] = scale * tmp;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, bool LOCAL_PIPELINE, bool SHARED_SCRATCH,
          typename TPhysSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void AdvectionDealiasKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const unsigned int ncomp, const unsigned int nhomo,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT Binterp0, const TData *NEK_RESTRICT Bproject0,
    const TData scale, const TData *NEK_RESTRICT in, const size_t inCompStride,
    const TData *NEK_RESTRICT advVel, const size_t advVelPlaneStride,
    TData *NEK_RESTRICT out, const size_t outCompStride,
    TData *NEK_RESTRICT elmtWsp, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    const unsigned int coordDim = sizeParam1D.ncoord();
    const unsigned int nq0      = sizeParam1D.nq0();
    const unsigned int nq0Fine  = 3u * nq0 / 2u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    const unsigned int nqTot        = nq0;
    const unsigned int nqTotFine    = nq0Fine;
    const unsigned int dfsize       = DEFORMED ? nqTot : 1u;

    const size_t advVelCompStride = advVelPlaneStride * nhomo;

    // Per-warp-group stride of the workspace laid out below.
    const unsigned int wspSize = static_cast<unsigned int>(
        AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
            1u, coordDim, ncomp, sizeParam1D, LOCAL_PIPELINE));

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *dfptr   = df + coordDim * dfsize * warpsize * iwarp;
        const TData *inGroup = in + nqTot * warpsize * iwarp;
        const TData *advVelGroup =
            advVel + p * advVelPlaneStride + nqTot * warpsize * iwarp;
        TData *outGroup = out + nqTot * warpsize * iwarp;
        // Per-warp-group workspace, laid out in this order by both the host
        // and the kernel: the reference derivatives, the fine-grid fields,
        // the products, then one tensor-contraction intermediate per
        // contracted direction.
        TData *wsp        = elmtWsp + wspSize * (nelmt * p + warpsize * iwarp);
        TData *derivGroup = wsp;
        TData *fineGroup  = derivGroup + ncomp * coordDim * nqTot * warpsize;
        TData *combinedGroup =
            fineGroup + (coordDim + ncomp * coordDim) * nqTotFine * warpsize;

        for (unsigned int d = 0u; d < coordDim; ++d)
        {
            BwdTransSegSumFacKernel<false>(ilane, nq0, nq0Fine, Binterp0,
                                           advVelGroup + d * advVelCompStride,
                                           fineGroup +
                                               d * nqTotFine * warpsize);
        }

        // One pass per component: reference derivatives on the native grid,
        // interpolation of those derivatives to the fine grid, the advection
        // product there, and the Galerkin projection back.
        for (unsigned int c = 0u; c < ncomp; ++c)
        {
            TData *derivOut = derivGroup + c * coordDim * nqTot * warpsize;
            TData *gradFine =
                fineGroup + (coordDim + c * coordDim) * nqTotFine * warpsize;
            TData *combinedOut = combinedGroup + c * nqTotFine * warpsize;

            PhysDeriv1DSumFacKernel<DEFORMED>(
                ilane, coordDim, nq0, nqTot * warpsize, D0, dfptr,
                inGroup + (c * nhomo + p) * inCompStride, derivOut);

            for (unsigned int d = 0u; d < coordDim; ++d)
            {
                BwdTransSegSumFacKernel<false>(ilane, nq0, nq0Fine, Binterp0,
                                               derivOut + d * nqTot * warpsize,
                                               gradFine +
                                                   d * nqTotFine * warpsize);
            }

            AdvectionDealiasCombineSumFacKernel<false>(
                ilane, nqTotFine, coordDim, fineGroup, nqTotFine * warpsize,
                gradFine, nqTotFine * warpsize, combinedOut, scale);

            TData *outptr = outGroup + (c * nhomo + p) * outCompStride;
            if constexpr (APPEND)
            {
                BwdTransSegSumFacKernel<true>(ilane, nq0Fine, nq0, Bproject0,
                                              combinedOut, outptr);
            }
            else
            {
                BwdTransSegSumFacKernel<false>(ilane, nq0Fine, nq0, Bproject0,
                                               combinedOut, outptr);
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, bool LOCAL_PIPELINE, bool SHARED_SCRATCH,
          typename TPhysSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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
    TData *NEK_RESTRICT elmtWsp, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int coordDim = sizeParam2D.ncoord();
    const unsigned int nq0      = sizeParam2D.nq0();
    const unsigned int nq1      = sizeParam2D.nq1();
    const unsigned int nq0Fine  = 3u * nq0 / 2u;
    const unsigned int nq1Fine =
        (nq0 - nq1 == 1u) ? nq0Fine - 1u : 3u * nq1 / 2u;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    const unsigned int nqTot        = nq0 * nq1;
    const unsigned int nqTotFine    = nq0Fine * nq1Fine;
    const unsigned int ndf          = 2u * coordDim;
    const unsigned int dfsize       = DEFORMED ? nqTot : 1u;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
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

    const size_t advVelCompStride = advVelPlaneStride * nhomo;

    // Per-warp-group stride of the workspace laid out below.
    const unsigned int wspSize = static_cast<unsigned int>(
        AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
            1u, coordDim, ncomp, sizeParam2D, LOCAL_PIPELINE));

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *dfptr   = df + ndf * dfsize * warpsize * iwarp;
        const TData *inGroup = in + nqTot * warpsize * iwarp;
        const TData *advVelGroup =
            advVel + p * advVelPlaneStride + nqTot * warpsize * iwarp;
        TData *outGroup = out + nqTot * warpsize * iwarp;
        // Per-warp-group workspace, laid out in this order by both the host
        // and the kernel: the reference derivatives, the fine-grid fields,
        // the products, then one tensor-contraction intermediate per
        // contracted direction.
        TData *wsp        = elmtWsp + wspSize * (nelmt * p + warpsize * iwarp);
        TData *derivGroup = wsp;
        TData *fineGroup  = derivGroup + ncomp * coordDim * nqTot * warpsize;
        TData *combinedGroup =
            fineGroup + (coordDim + ncomp * coordDim) * nqTotFine * warpsize;
        TData *wsp0 = combinedGroup + ncomp * nqTotFine * warpsize;

        for (unsigned int d = 0u; d < coordDim; ++d)
        {
            BwdTransQuadSumFacKernel<false>(
                ilane, nq0, nq1, nq0Fine, nq1Fine, Binterp0, Binterp1,
                advVelGroup + d * advVelCompStride,
                fineGroup + d * nqTotFine * warpsize, wsp0);
        }

        // One pass per component: reference derivatives on the native grid,
        // interpolation of those derivatives to the fine grid, the advection
        // product there, and the Galerkin projection back.
        for (unsigned int c = 0u; c < ncomp; ++c)
        {
            TData *derivOut = derivGroup + c * coordDim * nqTot * warpsize;
            TData *gradFine =
                fineGroup + (coordDim + c * coordDim) * nqTotFine * warpsize;
            TData *combinedOut = combinedGroup + c * nqTotFine * warpsize;

            PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, coordDim, nq0, nq1, nqTot * warpsize, D0, D1, s_f0, s_f1,
                dfptr, inGroup + (c * nhomo + p) * inCompStride, derivOut);

            for (unsigned int d = 0u; d < coordDim; ++d)
            {
                BwdTransQuadSumFacKernel<false>(
                    ilane, nq0, nq1, nq0Fine, nq1Fine, Binterp0, Binterp1,
                    derivOut + d * nqTot * warpsize,
                    gradFine + d * nqTotFine * warpsize, wsp0);
            }

            AdvectionDealiasCombineSumFacKernel<false>(
                ilane, nqTotFine, coordDim, fineGroup, nqTotFine * warpsize,
                gradFine, nqTotFine * warpsize, combinedOut, scale);

            TData *outptr = outGroup + (c * nhomo + p) * outCompStride;
            if constexpr (APPEND)
            {
                BwdTransQuadSumFacKernel<true>(ilane, nq0Fine, nq1Fine, nq0,
                                               nq1, Bproject0, Bproject1,
                                               combinedOut, outptr, wsp0);
            }
            else
            {
                BwdTransQuadSumFacKernel<false>(ilane, nq0Fine, nq1Fine, nq0,
                                                nq1, Bproject0, Bproject1,
                                                combinedOut, outptr, wsp0);
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, bool LOCAL_PIPELINE, bool SHARED_SCRATCH,
          typename TPhysSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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
    TData *NEK_RESTRICT elmtWsp, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0     = sizeParam3D.nq0();
    const unsigned int nq1     = sizeParam3D.nq1();
    const unsigned int nq2     = sizeParam3D.nq2();
    const unsigned int nq0Fine = 3u * nq0 / 2u;
    const unsigned int nq1Fine =
        (nq0 - nq1 == 1u) ? nq0Fine - 1u : 3u * nq1 / 2u;
    const unsigned int nq2Fine =
        (nq0 - nq2 == 1u) ? nq0Fine - 1u : 3u * nq2 / 2u;
    constexpr unsigned int warpsize   = NektarSpaces::Device::warpSize;
    constexpr unsigned int coordDim3D = 3u;
    const unsigned int nqTot          = nq0 * nq1 * nq2;
    const unsigned int nqTotFine      = nq0Fine * nq1Fine * nq2Fine;
    constexpr unsigned int ndf        = 9u;
    const unsigned int dfsize         = DEFORMED ? nqTot : 1u;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
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

    const size_t advVelCompStride = advVelPlaneStride * nhomo;

    // Per-warp-group stride of the workspace laid out below.
    const unsigned int wspSize = static_cast<unsigned int>(
        AdvectionDealiasWorkSpaceSize<SHAPE_TYPE, Implementation>(
            1u, coordDim3D, ncomp, sizeParam3D, LOCAL_PIPELINE));

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int p = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;

        const TData *dfptr   = df + ndf * dfsize * warpsize * iwarp;
        const TData *inGroup = in + nqTot * warpsize * iwarp;
        const TData *advVelGroup =
            advVel + p * advVelPlaneStride + nqTot * warpsize * iwarp;
        TData *outGroup = out + nqTot * warpsize * iwarp;
        // Per-warp-group workspace, laid out in this order by both the host
        // and the kernel: the reference derivatives, the fine-grid fields,
        // the products, then one tensor-contraction intermediate per
        // contracted direction.
        TData *wsp        = elmtWsp + wspSize * (nelmt * p + warpsize * iwarp);
        TData *derivGroup = wsp;
        TData *fineGroup  = derivGroup + ncomp * coordDim3D * nqTot * warpsize;
        TData *combinedGroup = fineGroup + (coordDim3D + ncomp * coordDim3D) *
                                               nqTotFine * warpsize;
        TData *wsp0 = combinedGroup + ncomp * nqTotFine * warpsize;
        TData *wsp1 = wsp0 + (nq1 * nq2 + nq1Fine * nq2Fine) * warpsize;

        // The direction count is a compile time constant, so this is
        // unrolled rather than run.
#pragma unroll
        for (unsigned int d = 0u; d < coordDim3D; ++d)
        {
            BwdTransHexSumFacKernel<false>(
                ilane, nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, Binterp0,
                Binterp1, Binterp2, advVelGroup + d * advVelCompStride,
                fineGroup + d * nqTotFine * warpsize, wsp0, wsp1);
        }

        // One pass per component: reference derivatives on the native grid,
        // interpolation of those derivatives to the fine grid, the advection
        // product there, and the Galerkin projection back.
        for (unsigned int c = 0u; c < ncomp; ++c)
        {
            TData *derivOut = derivGroup + c * coordDim3D * nqTot * warpsize;
            TData *gradFine = fineGroup + (coordDim3D + c * coordDim3D) *
                                              nqTotFine * warpsize;
            TData *combinedOut = combinedGroup + c * nqTotFine * warpsize;

            PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
                ilane, nq0, nq1, nq2, nqTot * warpsize, D0, D1, D2, s_f0, s_f1,
                s_f1m, s_f2, dfptr, inGroup + (c * nhomo + p) * inCompStride,
                derivOut);

#pragma unroll
            for (unsigned int d = 0u; d < coordDim3D; ++d)
            {
                BwdTransHexSumFacKernel<false>(
                    ilane, nq0, nq1, nq2, nq0Fine, nq1Fine, nq2Fine, Binterp0,
                    Binterp1, Binterp2, derivOut + d * nqTot * warpsize,
                    gradFine + d * nqTotFine * warpsize, wsp0, wsp1);
            }

            AdvectionDealiasCombineSumFacKernel<false>(
                ilane, nqTotFine, coordDim3D, fineGroup, nqTotFine * warpsize,
                gradFine, nqTotFine * warpsize, combinedOut, scale);

            TData *outptr = outGroup + (c * nhomo + p) * outCompStride;
            if constexpr (APPEND)
            {
                BwdTransHexSumFacKernel<true>(
                    ilane, nq0Fine, nq1Fine, nq2Fine, nq0, nq1, nq2, Bproject0,
                    Bproject1, Bproject2, combinedOut, outptr, wsp0, wsp1);
            }
            else
            {
                BwdTransHexSumFacKernel<false>(
                    ilane, nq0Fine, nq1Fine, nq2Fine, nq0, nq1, nq2, Bproject0,
                    Bproject1, Bproject2, combinedOut, outptr, wsp0, wsp1);
            }
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
