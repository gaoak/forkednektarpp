///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceSumFacKernels.hpp
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
// Description: Device SumFac kernels of the lift-and-project element operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceSumFacKernels.hpp
 * @brief Device SumFac kernels of the lift-and-project operator: the direct
 * lift of the Pointwise form, the fused launches of the StdInvMass form, and
 * the scales the separate-operator route takes.
 *
 * @details
 * As the other SumFac Device kernels, one thread per element: the block is
 * at warpSize, element `e` is lane `ilane = e % warpSize` of element group
 * `iwarp = e / warpSize`, consecutive threads touch consecutive entries, and
 * each thread loops over its element's points or modes. Each kernel is
 * restricted to Implementation SumFac; its SumFacTOP counterpart has the
 * same name and arguments, so that the caller launches
 * `Kernel<Implementation, ...>` for either.
 *
 * The fused launches are built from the per-element functions of the
 * IProductWRTPhysTrace, IProductWRTBase and BwdTrans SumFac kernels, as
 * the Mass kernels chain BwdTrans and IProductWRTBase:
 * - PhysLiftProjectKernelLauncher: the StdInvMass form with a diagonal
 *   standard inverse mass, in one launch;
 * - PhysLiftProjectLiftBasisTransposeKernelLauncher and
 *   PhysLiftProjectScaleBwdTransKernelLauncher: the StdInvMass form with a
 *   dense standard inverse mass, either side of the batched product.
 * Their intermediates live in the global workspace, one region per
 * intermediate laid out as a block at warpSize.
 *
 * PhysLiftProjectPointwiseKernel and PhysLiftProjectInvMassScaleKernel
 * serve the separate-operator route, which takes the nodal shapes and the
 * Pointwise blocks the direct lift does not.
 *
 * PhysLiftProjectDirectLiftKernelLauncher, the Pointwise form's direct
 * lift, takes one thread per point of each element instead, lanes first, a
 * thread block being a warp as for the other kernels. The quadrature
 * weights are read
 * once per point and broadcast across the lanes, as the basis is.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>

#include <type_traits>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief Per-element size of the first sum-factorisation stage scratch of
 * the fused SumFac kernels, which the basis transpose and the backward
 * transform share: the larger of what IProductWRTBase and BwdTrans take for
 * the shape. None in one dimension.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysLiftProjectStage0Size(
    [[maybe_unused]] const TSizeParameter sizeParam)
{
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        const unsigned int nq1 = sizeParam.nq1();
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return std::max(nq1, sizeParam.nm1());
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            return std::max(nq1, sizeParam.nm0());
        }
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const unsigned int nm0  = sizeParam.nm0();
        const unsigned int nm1  = sizeParam.nm1();
        const unsigned int nq12 = sizeParam.nq1() * sizeParam.nq2();
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return std::max(nq12, nm1 * sizeParam.nm2());
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return std::max(nq12, (2u * nm1 - nm0 + 1u) * nm0 / 2u);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::Pyr ||
                           SHAPE_TYPE == LibUtilities::NodalPyr)
        {
            return std::max(nq12, nm0 * nm1);
        }
    }
    else
    {
        return 0u;
    }
}

/**
 * @brief Per-element size of the second sum-factorisation stage scratch of
 * the fused SumFac kernels, as PhysLiftProjectStage0Size(). Three dimensions
 * only.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysLiftProjectStage1Size(
    [[maybe_unused]] const TSizeParameter sizeParam)
{
    if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const unsigned int nq2 = sizeParam.nq2();
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return std::max(nq2, sizeParam.nm2());
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::Pyr ||
                           SHAPE_TYPE == LibUtilities::NodalPyr)
        {
            return std::max(nq2, sizeParam.nm0());
        }
    }
    else
    {
        return 0u;
    }
}

/**
 * @brief Dynamic shared memory of the fused SumFac kernels: none, their
 * intermediates living in the global workspace.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectKernel KERNEL, typename TData, typename TSizeParameter,
          typename TTraceSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
inline constexpr unsigned int PhysLiftProjectSharedMemorySize(
    [[maybe_unused]] const TSizeParameter sizeParam,
    [[maybe_unused]] const TTraceSizeParameter traceSizeParam,
    [[maybe_unused]] const unsigned int elmtsPerBlock)
{
    return 0u;
}

/// The lift of the two end values of a segment onto its points, in place
/// on @p vol at warpSize (one dimension has no lift function of its own).
template <typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectSegLiftKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *NEK_RESTRICT nbasis0, const bool endPtsCollocated0,
    const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT vol)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (endPtsCollocated0)
    {
        vol[ilane] += trace[ilane];
        vol[warpsize * (nq0 - 1) + ilane] += trace[warpsize + ilane];
    }
    else
    {
        for (unsigned int p = 0; p < nq0; ++p)
        {
            vol[warpsize * p + ilane] +=
                nbasis0[2 * p] * trace[ilane] +
                nbasis0[2 * p + 1] * trace[warpsize + ilane];
        }
    }
}

/// Copy @p in into @p vol, both at warpSize, unless they are the same
/// storage, which they are when the input was interleaved into the
/// workspace.
template <typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectCopyInKernel(
    const unsigned int ilane, const unsigned int nqTot, const TData *in,
    TData *vol)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (in != vol)
    {
        for (unsigned int q = 0; q < nqTot; ++q)
        {
            vol[warpsize * q + ilane] = in[warpsize * q + ilane];
        }
    }
}

/**
 * @brief The Pointwise form: `out (+)= scale / (w_q J) * in` at every point,
 * zero on the padding elements, whose Jacobian is zero. Grid dimension 1 is
 * the component.
 *
 * @param   weights  The quadrature weights at every point (WeightsKey at
 *                   width one), the same for every lane.
 * @param   jac      The Jacobians (JacobianKey at warpSize), one per
 *                   element, or one per point when @p DEFORMED.
 */
template <typename Implementation, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectPointwiseKernel(
    const size_t nelmt, const unsigned int nqTot, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int c = getBlockIdx<1>(threadBlock);
    for (size_t e = getGlobalIdx<0>(threadBlock); e < nelmt;
         e += getGlobalRange<0>(threadBlock))
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const size_t offset = nqTot * (nelmt * c + warpsize * iwarp) + ilane;
        for (unsigned int q = 0; q < nqTot; ++q)
        {
            TData jacobian;
            if constexpr (DEFORMED)
            {
                jacobian = jac[(iwarp * nqTot + q) * warpsize + ilane];
            }
            else
            {
                jacobian = jac[e];
            }
            const TData factor = SafeDivide(scale, weights[q] * jacobian);
            const size_t i     = offset + static_cast<size_t>(q) * warpsize;
            if constexpr (APPEND)
            {
                out[i] += factor * in[i];
            }
            else
            {
                out[i] = factor * in[i];
            }
        }
    }
}

/**
 * @brief The inverse mass of an affine element with the scale: `out[m] =
 * scale / J * diag[m] * in[m]`, the diagonal of the standard inverse mass
 * @p diag (eInvMassStdMatDiagonal) left out unless @p DIAG, zero on the
 * padding elements, whose Jacobian is zero. @p in may be @p out. Grid
 * dimension 1 is the component.
 *
 * @param   jac  The Jacobians (JacobianKey at warpSize), one per element.
 */
template <typename Implementation, bool DIAG, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectInvMassScaleKernel(
    const size_t nelmt, const unsigned int nmTot, const TData scale,
    const TData *NEK_RESTRICT diag, const TData *NEK_RESTRICT jac,
    const TData *in, TData *out, const TthreadBlock &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int c = getBlockIdx<1>(threadBlock);
    for (size_t e = getGlobalIdx<0>(threadBlock); e < nelmt;
         e += getGlobalRange<0>(threadBlock))
    {
        const size_t ilane  = e % warpsize;
        const size_t iwarp  = e / warpsize;
        const size_t offset = nmTot * (nelmt * c + warpsize * iwarp) + ilane;
        const TData factor  = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nmTot; ++m)
        {
            const size_t i = offset + static_cast<size_t>(m) * warpsize;
            if constexpr (DIAG)
            {
                out[i] = factor * diag[m] * in[i];
            }
            else
            {
                out[i] = factor * in[i];
            }
        }
    }
}

/**
 * @brief The StdInvMass form with a diagonal standard inverse mass in one
 * dimension. See the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        [[maybe_unused]] const TTraceSizeParameter1D traceSizeParam1D,
        const size_t nelmt, const TData scale, const bool lift,
        [[maybe_unused]] const bool isModified,
        [[maybe_unused]] const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const bool endPtsCollocated0, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT diag, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out, TData *vol,
        [[maybe_unused]] TData *NEK_RESTRICT coef,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        const TData *inptr = in + nq0 * slot;
        TData *outptr      = out + nq0 * slot;
        TData *volptr      = vol + nq0 * slot;

        PhysLiftProjectCopyInKernel(ilane, nq0, inptr, volptr);
        if (lift)
        {
            PhysLiftProjectSegLiftKernel(ilane, nq0, nbasis0, endPtsCollocated0,
                                         trace + 2 * slot, volptr);
        }

        TData *coefptr = coef + nm0 * slot;
        IProductWRTBaseSegSumFacKernel<false, false>(
            ilane, nm0, nq0, basis0, volptr, coefptr, TData(1.0));
        const TData factor = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nm0; ++m)
        {
            coefptr[warpsize * m + ilane] *= factor * diag[m];
        }
        BwdTransSegSumFacKernel<APPEND>(ilane, nm0, nq0, basis0, coefptr,
                                        outptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a diagonal standard inverse mass in two
 * dimensions. See the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const TData scale, const bool lift, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac, const bool isCollocated0,
        const bool isCollocated1, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT diag, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out, TData *vol,
        [[maybe_unused]] TData *NEK_RESTRICT coef, TData *NEK_RESTRICT wsp0,
        TData *NEK_RESTRICT lift0, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int tnq00 = traceSizeParam2D.nq00();
    const unsigned int tnq10 = traceSizeParam2D.nq10();
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    const unsigned int wsp0Size =
        PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(sizeParam2D);
    const unsigned int lift0Size =
        2u * IProductWRTPhysTraceEdgeModeBlockSize(traceSizeParam2D);

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        const TData *inptr = in + nqTot * slot;
        TData *outptr      = out + nqTot * slot;
        TData *volptr      = vol + nqTot * slot;

        PhysLiftProjectCopyInKernel(ilane, nqTot, inptr, volptr);
        if (lift)
        {
            IProductWRTPhysTrace2DKernel<SHAPE_TYPE, false, true>(
                ilane, nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0,
                tbasis1, tw00, tw10, tjac + numDataJac * warpsize * iwarp,
                lift0 + lift0Size * slot, trace + nqTotTrace * slot, volptr,
                isCollocated0, isCollocated1, endPtsCollocated0,
                endPtsCollocated1);
        }

        TData *coefptr = coef + nmTot * slot;
        TData *wsp0ptr = wsp0 + wsp0Size * slot;
        IProductWRTBase2DSumFacKernel<SHAPE_TYPE, false, false>(
            ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, volptr,
            coefptr, wsp0ptr, TData(1.0));
        const TData factor = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nmTot; ++m)
        {
            coefptr[warpsize * m + ilane] *= factor * diag[m];
        }
        BwdTrans2DSumFacKernel<SHAPE_TYPE, APPEND>(ilane, nm0, nm1, nq0, nq1,
                                                   isModified, basis0, basis1,
                                                   coefptr, outptr, wsp0ptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a diagonal standard inverse mass of a
 * block of affine three-dimensional elements in one launch, one thread per
 * element at warpSize.
 *
 * Each thread copies its element's inner products into @p vol (unless the
 * input was interleaved into @p vol already), lifts the trace onto them,
 * applies the basis transpose into @p coef, `coef *= scale / J * diag` and
 * the backward transform into @p out, appending when @p APPEND.
 *
 * @p in, @p trace, @p out and the workspace regions @p vol, @p coef,
 * @p wsp0, @p wsp1, @p lift0 and @p lift1 are at warpSize, each region
 * holding its per-element size for every element and component, laid out
 * as a block (`size * (nelmt * c + warpSize * iwarp)` to the element's
 * group). The lift tables and flags are those of IProductWRTPhysTrace, the
 * bases and the index tables those of IProductWRTBase and BwdTrans; the
 * index tables are read only by the SumFacTOP counterpart.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const TData scale, const bool lift, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
        const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
        const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
        const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
        const TData *NEK_RESTRICT tjac, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diag,
        const TData *in, const TData *NEK_RESTRICT trace,
        TData *NEK_RESTRICT out, TData *vol,
        [[maybe_unused]] TData *NEK_RESTRICT coef, TData *NEK_RESTRICT wsp0,
        TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT lift0,
        TData *NEK_RESTRICT lift1, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int tnq00 = traceSizeParam3D.nq00();
    const unsigned int tnq01 = traceSizeParam3D.nq01();
    const unsigned int tnq10 = traceSizeParam3D.nq10();
    const unsigned int tnq11 = traceSizeParam3D.nq11();
    const unsigned int tnq20 = traceSizeParam3D.nq20();
    const unsigned int tnq21 = traceSizeParam3D.nq21();
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    const unsigned int wsp0Size =
        PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(sizeParam3D);
    const unsigned int wsp1Size =
        PhysLiftProjectStage1Size<SHAPE_TYPE, Implementation>(sizeParam3D);
    const unsigned int lift0Size =
        IProductWRTPhysTraceFaceModeBlockSize(traceSizeParam3D);
    const unsigned int lift1Size =
        IProductWRTPhysTraceFaceScratchSize(traceSizeParam3D);

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        const TData *inptr = in + nqTot * slot;
        TData *outptr      = out + nqTot * slot;
        TData *volptr      = vol + nqTot * slot;

        PhysLiftProjectCopyInKernel(ilane, nqTot, inptr, volptr);
        if (lift)
        {
            IProductWRTPhysTrace3DFaceKernel<SHAPE_TYPE, false, true>(
                ilane, nq0, nq1, nq2, nbasis0, nbasis1, nbasis2, tnq00, tnq01,
                tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10,
                tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20,
                tw21, tjac + numDataJac * warpsize * iwarp,
                lift0 + lift0Size * slot, lift1 + lift1Size * slot,
                trace + nqTotTrace * slot, volptr, isCollocated00,
                isCollocated01, isCollocated10, isCollocated11, isCollocated20,
                isCollocated21, endPtsCollocated0, endPtsCollocated1,
                endPtsCollocated2);
        }

        TData *coefptr = coef + nmTot * slot;
        TData *wsp0ptr = wsp0 + wsp0Size * slot;
        TData *wsp1ptr = wsp1 + wsp1Size * slot;
        IProductWRTBase3DSumFacKernel<SHAPE_TYPE, false, false>(
            ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
            basis2, volptr, coefptr, wsp0ptr, wsp1ptr, TData(1.0));
        const TData factor = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nmTot; ++m)
        {
            coefptr[warpsize * m + ilane] *= factor * diag[m];
        }
        BwdTrans3DSumFacKernel<SHAPE_TYPE, APPEND>(
            ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
            basis2, coefptr, outptr, wsp0ptr, wsp1ptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, before
 * the product: the lift and the basis transpose into @p coef, in one
 * launch. One dimension; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        [[maybe_unused]] const TTraceSizeParameter1D traceSizeParam1D,
        const size_t nelmt, const bool lift,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const bool endPtsCollocated0, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT coef, TData *vol,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        const TData *inptr = in + nq0 * slot;
        TData *coefptr     = coef + nm0 * slot;

        TData *volptr = vol + nq0 * slot;
        PhysLiftProjectCopyInKernel(ilane, nq0, inptr, volptr);
        if (lift)
        {
            PhysLiftProjectSegLiftKernel(ilane, nq0, nbasis0, endPtsCollocated0,
                                         trace + 2 * slot, volptr);
        }
        IProductWRTBaseSegSumFacKernel<false, false>(
            ilane, nm0, nq0, basis0, volptr, coefptr, TData(1.0));

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form before the product in two dimensions;
 * see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const bool lift, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac, const bool isCollocated0,
        const bool isCollocated1, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT coef, TData *vol,
        TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT lift0,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int tnq00 = traceSizeParam2D.nq00();
    const unsigned int tnq10 = traceSizeParam2D.nq10();
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    const unsigned int wsp0Size =
        PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(sizeParam2D);
    const unsigned int lift0Size =
        2u * IProductWRTPhysTraceEdgeModeBlockSize(traceSizeParam2D);

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        const TData *inptr = in + nqTot * slot;
        TData *coefptr     = coef + nmTot * slot;

        TData *volptr = vol + nqTot * slot;
        PhysLiftProjectCopyInKernel(ilane, nqTot, inptr, volptr);
        if (lift)
        {
            IProductWRTPhysTrace2DKernel<SHAPE_TYPE, false, true>(
                ilane, nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0,
                tbasis1, tw00, tw10, tjac + numDataJac * warpsize * iwarp,
                lift0 + lift0Size * slot, trace + nqTotTrace * slot, volptr,
                isCollocated0, isCollocated1, endPtsCollocated0,
                endPtsCollocated1);
        }
        IProductWRTBase2DSumFacKernel<SHAPE_TYPE, false, false>(
            ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, volptr,
            coefptr, wsp0 + wsp0Size * slot, TData(1.0));

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, before
 * the product, of a block of three-dimensional affine elements in one
 * launch: each thread copies its element's inner products into @p vol,
 * lifts the trace onto them and applies the basis transpose into @p coef,
 * at warpSize. The arguments are those of PhysLiftProjectKernelLauncher().
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const bool lift, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
        const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
        const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
        const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
        const TData *NEK_RESTRICT tjac, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const TData *in, const TData *NEK_RESTRICT trace,
        TData *NEK_RESTRICT coef, TData *vol, TData *NEK_RESTRICT wsp0,
        TData *NEK_RESTRICT wsp1, TData *NEK_RESTRICT lift0,
        TData *NEK_RESTRICT lift1, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int tnq00 = traceSizeParam3D.nq00();
    const unsigned int tnq01 = traceSizeParam3D.nq01();
    const unsigned int tnq10 = traceSizeParam3D.nq10();
    const unsigned int tnq11 = traceSizeParam3D.nq11();
    const unsigned int tnq20 = traceSizeParam3D.nq20();
    const unsigned int tnq21 = traceSizeParam3D.nq21();
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    const unsigned int wsp0Size =
        PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(sizeParam3D);
    const unsigned int wsp1Size =
        PhysLiftProjectStage1Size<SHAPE_TYPE, Implementation>(sizeParam3D);
    const unsigned int lift0Size =
        IProductWRTPhysTraceFaceModeBlockSize(traceSizeParam3D);
    const unsigned int lift1Size =
        IProductWRTPhysTraceFaceScratchSize(traceSizeParam3D);

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        const TData *inptr = in + nqTot * slot;
        TData *coefptr     = coef + nmTot * slot;

        TData *volptr = vol + nqTot * slot;
        PhysLiftProjectCopyInKernel(ilane, nqTot, inptr, volptr);
        if (lift)
        {
            IProductWRTPhysTrace3DFaceKernel<SHAPE_TYPE, false, true>(
                ilane, nq0, nq1, nq2, nbasis0, nbasis1, nbasis2, tnq00, tnq01,
                tnq10, tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10,
                tbasis11, tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20,
                tw21, tjac + numDataJac * warpsize * iwarp,
                lift0 + lift0Size * slot, lift1 + lift1Size * slot,
                trace + nqTotTrace * slot, volptr, isCollocated00,
                isCollocated01, isCollocated10, isCollocated11, isCollocated20,
                isCollocated21, endPtsCollocated0, endPtsCollocated1,
                endPtsCollocated2);
        }
        IProductWRTBase3DSumFacKernel<SHAPE_TYPE, false, false>(
            ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
            basis2, volptr, coefptr, wsp0 + wsp0Size * slot,
            wsp1 + wsp1Size * slot, TData(1.0));

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, after
 * the product: `coef *= scale / J` in place and the backward transform
 * into @p out, appending when @p APPEND, in one launch, one thread per
 * element at warpSize. One dimension; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        [[maybe_unused]] const TTraceSizeParameter1D traceSizeParam1D,
        const size_t nelmt, const TData scale,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT jac,
        TData *NEK_RESTRICT coef, TData *NEK_RESTRICT out,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        TData *coefptr     = coef + nm0 * slot;
        TData *outptr      = out + nq0 * slot;

        const TData factor = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nm0; ++m)
        {
            coefptr[warpsize * m + ilane] *= factor;
        }
        BwdTransSegSumFacKernel<APPEND>(ilane, nm0, nq0, basis0, coefptr,
                                        outptr);

        e += getGlobalRange<0>(threadBlock);
    }
}

/// The dense StdInvMass form after the product in two dimensions; see the
/// three-dimensional overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        [[maybe_unused]] const TTraceSizeParameter2D traceSizeParam2D,
        const size_t nelmt, const TData scale, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT coef,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int wsp0Size =
        PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(sizeParam2D);

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        TData *coefptr     = coef + nmTot * slot;
        TData *outptr      = out + nqTot * slot;

        const TData factor = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nmTot; ++m)
        {
            coefptr[warpsize * m + ilane] *= factor;
        }
        BwdTrans2DSumFacKernel<SHAPE_TYPE, APPEND>(
            ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, coefptr,
            outptr, wsp0 + wsp0Size * slot);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, after
 * the product, of a block of three-dimensional affine elements: each
 * thread scales its element's coefficients in @p coef by `scale / J` in
 * place and applies the backward transform into @p out, appending when
 * @p APPEND, at warpSize.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const TTraceSizeParameter3D traceSizeParam3D,
        const size_t nelmt, const TData scale, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT jac,
        TData *NEK_RESTRICT coef, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int wsp0Size =
        PhysLiftProjectStage0Size<SHAPE_TYPE, Implementation>(sizeParam3D);
    const unsigned int wsp1Size =
        PhysLiftProjectStage1Size<SHAPE_TYPE, Implementation>(sizeParam3D);

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const size_t slot  = nelmt * c + warpsize * iwarp;
        TData *coefptr     = coef + nmTot * slot;
        TData *outptr      = out + nqTot * slot;

        const TData factor = SafeDivide(scale, jac[e]);
        for (unsigned int m = 0; m < nmTot; ++m)
        {
            coefptr[warpsize * m + ilane] *= factor;
        }
        BwdTrans3DSumFacKernel<SHAPE_TYPE, APPEND>(
            ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
            basis2, coefptr, outptr, wsp0 + wsp0Size * slot,
            wsp1 + wsp1Size * slot);

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The factor `scale / (w J)` of the Pointwise form at point @p pt of
 * element @p e, whose point index is @p i; zero on the padding elements,
 * whose Jacobian is zero. The weight is read once per point, the same for
 * every lane.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData PhysLiftProjectDirectLiftFactorKernel(
    const size_t i, const size_t e, const unsigned int pt, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac)
{
    TData jacobian;
    if constexpr (DEFORMED)
    {
        jacobian = jac[i];
    }
    else
    {
        jacobian = jac[e];
    }
    return SafeDivide(scale, weights[pt] * jacobian);
}

/// `out[ientry] (+)= factor * val`.
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectDirectLiftStoreKernel(
    TData *out, const size_t ientry, const TData factor, const TData val)
{
    if constexpr (APPEND)
    {
        out[ientry] += factor * val;
    }
    else
    {
        out[ientry] = factor * val;
    }
}

/**
 * @brief The lift of the traces of a one-point element onto its point: the
 * sum over its traces, in the packed order, of each trace's integral of the
 * trace values times its Jacobian, times the end-point interpolant of the
 * point along the trace's normal direction. A one-point direction's
 * interpolant onto the trace points is one, so no in-trace interpolant is
 * applied.
 *
 * @tparam SHAPE_TYPE  Tri, Tet, Prism or Pyr.
 * @tparam DEFORMED    The trace Jacobians are per trace point.
 *
 * @param   tnq      Trace points of a trace in each normal direction and
 *                   in-trace direction, `tnq[2 * d + t]` (`tnq[d]` in two
 *                   dimensions, one in-trace direction).
 * @param   nbasis   End-point interpolant of each direction, one row.
 * @param   tw       In-trace weights, in the order of @p tnq.
 * @param   tjacptr  The element's trace Jacobians, at its lane.
 * @param   traceptr The element's trace values of the component, at its
 *                   lane.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData PhysLiftProjectOnePointLiftKernel(
    const unsigned int *tnq, const TData *const *nbasis, const TData *const *tw,
    const TData *NEK_RESTRICT tjacptr, const TData *NEK_RESTRICT traceptr)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr unsigned int dim      = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
    // The traces of each direction, taken at compile time: the host table
    // cannot be indexed at run time in device code.
    constexpr unsigned int nface0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned int nface1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int nface2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];
    const unsigned int nfaces[3] = {nface0, nface1, nface2};
    TData lift                   = 0.0;
    unsigned int k               = 0;
    unsigned int iface           = 0;
#pragma unroll
    for (unsigned int d = 0; d < dim; ++d)
    {
        const unsigned int tnq0 = (dim == 3) ? tnq[2 * d] : tnq[d];
        const unsigned int tnq1 = (dim == 3) ? tnq[2 * d + 1] : 1u;
        const TData *tw0        = (dim == 3) ? tw[2 * d] : tw[d];
        const TData *tw1        = (dim == 3) ? tw[2 * d + 1] : nullptr;
#pragma unroll
        for (unsigned int f = 0; f < nfaces[d]; ++f, ++iface)
        {
            TData sum = 0.0;
            for (unsigned int j = 0; j < tnq1; ++j)
            {
                for (unsigned int i = 0; i < tnq0; ++i, ++k)
                {
                    TData faceJac;
                    if constexpr (DEFORMED)
                    {
                        faceJac = tjacptr[warpsize * k];
                    }
                    else
                    {
                        faceJac = tjacptr[warpsize * iface];
                    }
                    TData val = traceptr[warpsize * k] * faceJac * tw0[i];
                    if (dim == 3)
                    {
                        val *= tw1[j];
                    }
                    sum += val;
                }
            }
            lift += sum * nbasis[d][f];
        }
    }
    return lift;
}

/**
 * @brief The Pointwise form of a block of segments on SumFac, the trace
 * lifted directly onto every point: `out (+)= scale / (w J) * (in + h_0(p)
 * g_0 + h_1(p) g_1)`. See the three-dimensional launcher; a segment's
 * traces are points, with no Jacobian or weight, and @p tjac is not read.
 *
 * One thread per volume point of each element, over every component, the
 * block at warpSize: thread `i = warpSize * (nqTot * iwarp + pt) + ilane`
 * takes point @em pt of element `warpSize * iwarp + ilane`, so that
 * consecutive threads take the same point of consecutive elements. A thread
 * block is a warp, as in the other SumFac kernels.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter1D>()))
    PhysLiftProjectDirectLiftKernelLauncher(
        const TTraceSizeParameter1D traceSizeParam1D, const size_t nelmt,
        const unsigned int ncomp, const TData scale, const bool lift,
        const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
        const TData *in, const TData *NEK_RESTRICT trace, TData *out,
        const TthreadBlock &threadBlock)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    const unsigned int nqTot        = traceSizeParam1D.nm0();
    const size_t compSize           = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize      = 2u * nelmt;

    for (size_t i = getGlobalIdx(threadBlock); i < compSize;
         i += getGlobalRange(threadBlock))
    {
        // Point index i = warpsize * (nqTot * iwarp + p) + ilane: point p of
        // element warpsize * iwarp + ilane, lanes first.
        const unsigned int ilane = i % warpsize;
        const size_t ipoint      = i / warpsize;
        const size_t iwarp       = ipoint / nqTot;
        const unsigned int p     = ipoint % nqTot;
        const size_t e           = warpsize * iwarp + ilane;

        const TData factor = PhysLiftProjectDirectLiftFactorKernel<DEFORMED>(
            i, e, p, scale, weights, jac);
        const TData h0 = nbasis0[2u * p];
        const TData h1 = nbasis0[2u * p + 1u];

        const size_t traceOffset = 2u * warpsize * iwarp + ilane;
        for (unsigned int c = 0; c < ncomp; ++c)
        {
            const size_t ientry = compSize * c + i;
            TData val           = in[ientry];
            if (lift)
            {
                const TData *traceptr = trace + compTraceSize * c + traceOffset;
                val += h0 * traceptr[0] + h1 * traceptr[warpsize];
            }
            PhysLiftProjectDirectLiftStoreKernel<APPEND>(out, ientry, factor,
                                                         val);
        }
    }
}

/**
 * @brief The Pointwise form of a block of two-dimensional elements on SumFac,
 * the trace lifted directly onto every point. See the three-dimensional
 * launcher: quadrilaterals whose edges have the volume points, each point
 * adding its four edges' contributions (the two direction-0 edges, then the
 * two direction-1 edges), and triangles of one point.
 *
 * One thread per volume point of each element, over every component, the
 * block at warpSize: thread `i = warpSize * (nqTot * iwarp + pt) + ilane`
 * takes point @em pt of element `warpSize * iwarp + ilane`, so that
 * consecutive threads take the same point of consecutive elements. A thread
 * block is a warp, as in the other SumFac kernels.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    PhysLiftProjectDirectLiftKernelLauncher(
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const unsigned int ncomp, const TData scale, const bool lift,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac, const TData *NEK_RESTRICT weights,
        const TData *NEK_RESTRICT jac, const TData *in,
        const TData *NEK_RESTRICT trace, TData *out,
        const TthreadBlock &threadBlock)
{
    static_assert(SHAPE_TYPE == LibUtilities::Quad ||
                      SHAPE_TYPE == LibUtilities::Tri,
                  "The direct lift takes quadrilaterals and triangles.");

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr bool isCollapsed      = SHAPE_TYPE != LibUtilities::Quad;
    constexpr unsigned int nedge = LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The trace size parameter's nm0() and nm1() are the volume quadrature
    // point counts here (#m_nm), as in the trace operators.
    const unsigned int nq0   = traceSizeParam2D.nm0();
    const unsigned int nq1   = traceSizeParam2D.nm1();
    const unsigned int nqTot = nq0 * nq1;
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac = DEFORMED ? nqTotTrace : nedge;

    const size_t compSize      = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize = static_cast<size_t>(nqTotTrace) * nelmt;

    for (size_t i = getGlobalIdx(threadBlock); i < compSize;
         i += getGlobalRange(threadBlock))
    {
        // Point index i = warpsize * (nqTot * iwarp + pt) + ilane: point pt of
        // element warpsize * iwarp + ilane, lanes first.
        const unsigned int ilane = i % warpsize;
        const size_t ipoint      = i / warpsize;
        const size_t iwarp       = ipoint / nqTot;
        const unsigned int pt    = ipoint % nqTot;
        const size_t e           = warpsize * iwarp + ilane;

        const TData factor = PhysLiftProjectDirectLiftFactorKernel<DEFORMED>(
            i, e, pt, scale, weights, jac);
        const TData *tjacptr = tjac + numDataJac * warpsize * iwarp + ilane;
        const size_t traceOffset =
            static_cast<size_t>(nqTotTrace) * warpsize * iwarp + ilane;

        if constexpr (!isCollapsed)
        {
            // The point of each edge beside this point, the edge's weight
            // there and the end-point interpolant along its normal.
            const unsigned int p        = pt % nq0;
            const unsigned int q        = pt / nq0;
            const unsigned int k[nedge] = {q, nq1 + q, 2u * nq1 + p,
                                           2u * nq1 + nq0 + p};
            const TData w[nedge]        = {tw00[q], tw00[q], tw10[p], tw10[p]};
            const TData h[nedge] = {nbasis0[2u * p], nbasis0[2u * p + 1u],
                                    nbasis1[2u * q], nbasis1[2u * q + 1u]};
            // Each edge's factor, formed once for every component.
            TData coef[nedge];
#pragma unroll
            for (unsigned int iedge = 0; iedge < nedge; ++iedge)
            {
                TData edgeJac;
                if constexpr (DEFORMED)
                {
                    edgeJac = tjacptr[warpsize * k[iedge]];
                }
                else
                {
                    edgeJac = tjacptr[warpsize * iedge];
                }
                coef[iedge] = edgeJac * w[iedge] * h[iedge];
            }
            for (unsigned int c = 0; c < ncomp; ++c)
            {
                const size_t ientry = compSize * c + i;
                TData val           = in[ientry];
                if (lift)
                {
                    const TData *traceptr =
                        trace + compTraceSize * c + traceOffset;
#pragma unroll
                    for (unsigned int iedge = 0; iedge < nedge; ++iedge)
                    {
                        val += traceptr[warpsize * k[iedge]] * coef[iedge];
                    }
                }
                PhysLiftProjectDirectLiftStoreKernel<APPEND>(out, ientry,
                                                             factor, val);
            }
        }
        else
        {
            const unsigned int tnq[2] = {traceSizeParam2D.nq00(),
                                         traceSizeParam2D.nq10()};
            const TData *nbasis[2]    = {nbasis0, nbasis1};
            const TData *tw[2]        = {tw00, tw10};
            for (unsigned int c = 0; c < ncomp; ++c)
            {
                const size_t ientry = compSize * c + i;
                TData val           = in[ientry];
                if (lift)
                {
                    val +=
                        PhysLiftProjectOnePointLiftKernel<SHAPE_TYPE, DEFORMED>(
                            tnq, nbasis, tw, tjacptr,
                            trace + compTraceSize * c + traceOffset);
                }
                PhysLiftProjectDirectLiftStoreKernel<APPEND>(out, ientry,
                                                             factor, val);
            }
        }
    }
}

/**
 * @brief The Pointwise form of a block of three-dimensional elements on SumFac,
 * the trace lifted directly onto every point, over every component: `out
 * (+)= scale / (w J) * (in + L g)`, zero on the padding elements, whose
 * Jacobian is zero.
 *
 * @details
 * Hexahedra whose faces have the volume points in both in-face directions:
 * the lift of face @em f onto volume point (p, q, r) is then the face's
 * value at the point beside it, times the face's Jacobian there, its two
 * in-face weights there and the end-point interpolant of the point along
 * the face's normal direction, \f$h(\pm 1)\f$, a delta where the rule has
 * its end points. Each point adds its six faces' contributions directly,
 * with no scratch and no zeroing of a volume field, in the packed trace
 * order (direction 0's low and high faces, then direction 1's and
 * direction 2's: the hexahedron's faces 4, 2, 1, 3, 0 and 5), each formed as
 * the face-by-face lift of IProductWRTPhysTrace forms it.
 *
 * Tetrahedra, prisms and pyramids of one point: every face contributes its
 * integral of the trace values times its Jacobian, times the one-point
 * end-point interpolant (PhysLiftProjectOnePointLiftKernel()).
 *
 * Element `e = warpSize * iwarp + ilane` has its point @em pt at
 * `nqTot * (nelmt * c + warpSize * iwarp) + warpSize * pt + ilane` in
 * component @em c, which is `nqTot * nelmt * c + i` for the point's index
 * `i = warpSize * (nqTot * iwarp + pt) + ilane`: consecutive threads take
 * the same point of consecutive elements, so every read of a point, a trace
 * value or a Jacobian is coalesced across the lanes. Its trace values are
 * at `nqTotTrace * (nelmt * c + warpSize * iwarp) + warpSize * k + ilane`
 * for trace point @em k, the faces packed as above and each face's points
 * in its in-face directions in the volume's order. A thread block is a
 * warp, as in the other SumFac kernels.
 *
 * @tparam SHAPE_TYPE  Hex, Tet, Prism or Pyr.
 * @tparam APPEND      Accumulate onto @p out.
 * @tparam DEFORMED    The Jacobians are per point.
 *
 * @param   traceSizeParam3D  Volume point counts per direction (nm0(),
 *                            nm1(), nm2()) and trace point counts per face
 *                            pair.
 * @param   nelmt     Elements in the block, including padding.
 * @param   ncomp     Components (times homogeneous modes) of the block.
 * @param   scale     Factor the result is scaled by.
 * @param   lift      Lift @p trace; without it only the scale is applied.
 * @param   nbasis0,nbasis1,nbasis2  End-point interpolant of each
 *                    direction, `nbasis[p * ntrace + side]`.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21  Weights of each face pair's
 *                    in-face directions (FetchTraceTables()).
 * @param   tjac      Trace Jacobians (JacobianLocTraceKey at warpSize), one
 *                    per face, or one per trace point when @p DEFORMED.
 * @param   weights   Quadrature weights at every point (WeightsKey at width
 *                    one), the same for every lane.
 * @param   jac       Jacobians (JacobianKey at warpSize), one per element,
 *                    or one per point when @p DEFORMED.
 * @param   in        Inner products; may be @p out.
 * @param   trace     Element-local trace values.
 * @param   out       Result.
 * @param   threadBlock  The thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    PhysLiftProjectDirectLiftKernelLauncher(
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const unsigned int ncomp, const TData scale, const bool lift,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT nbasis2, const TData *NEK_RESTRICT tw00,
        const TData *NEK_RESTRICT tw01, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tw11, const TData *NEK_RESTRICT tw20,
        const TData *NEK_RESTRICT tw21, const TData *NEK_RESTRICT tjac,
        const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
        const TData *in, const TData *NEK_RESTRICT trace, TData *out,
        const TthreadBlock &threadBlock)
{
    static_assert(SHAPE_TYPE == LibUtilities::Hex ||
                      SHAPE_TYPE == LibUtilities::Tet ||
                      SHAPE_TYPE == LibUtilities::Prism ||
                      SHAPE_TYPE == LibUtilities::Pyr,
                  "The direct lift takes hexahedra, tetrahedra, prisms and "
                  "pyramids.");

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;
    constexpr bool isCollapsed      = SHAPE_TYPE != LibUtilities::Hex;
    constexpr unsigned int nface = LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The trace size parameter's nm0()..nm2() are the volume quadrature
    // point counts here (#m_nm), as in the trace operators.
    const unsigned int nq0   = traceSizeParam3D.nm0();
    const unsigned int nq1   = traceSizeParam3D.nm1();
    const unsigned int nq2   = traceSizeParam3D.nm2();
    const unsigned int nq01  = nq0 * nq1;
    const unsigned int nqTot = nq01 * nq2;
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac = DEFORMED ? nqTotTrace : nface;

    const size_t compSize      = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize = static_cast<size_t>(nqTotTrace) * nelmt;

    for (size_t i = getGlobalIdx(threadBlock); i < compSize;
         i += getGlobalRange(threadBlock))
    {
        // Point index i = warpsize * (nqTot * iwarp + pt) + ilane: point pt of
        // element warpsize * iwarp + ilane, lanes first.
        const unsigned int ilane = i % warpsize;
        const size_t ipoint      = i / warpsize;
        const size_t iwarp       = ipoint / nqTot;
        const unsigned int pt    = ipoint % nqTot;
        const size_t e           = warpsize * iwarp + ilane;

        const TData factor = PhysLiftProjectDirectLiftFactorKernel<DEFORMED>(
            i, e, pt, scale, weights, jac);
        const TData *tjacptr = tjac + numDataJac * warpsize * iwarp + ilane;
        const size_t traceOffset =
            static_cast<size_t>(nqTotTrace) * warpsize * iwarp + ilane;

        if constexpr (!isCollapsed)
        {
            const unsigned int p = pt % nq0;
            const unsigned int q = (pt / nq0) % nq1;
            const unsigned int r = pt / nq01;

            // The point of each face beside this point, the face's in-face
            // weights there and the end-point interpolant along its normal.
            const unsigned int fsz0     = nq1 * nq2;
            const unsigned int fsz1     = nq0 * nq2;
            const unsigned int off1     = 2u * fsz0;
            const unsigned int off2     = off1 + 2u * fsz1;
            const unsigned int k[nface] = {
                r * nq1 + q,        fsz0 + r * nq1 + q,
                off1 + r * nq0 + p, off1 + fsz1 + r * nq0 + p,
                off2 + q * nq0 + p, off2 + nq01 + q * nq0 + p};
            const TData w0[nface] = {tw00[q], tw00[q], tw10[p],
                                     tw10[p], tw20[p], tw20[p]};
            const TData w1[nface] = {tw01[r], tw01[r], tw11[r],
                                     tw11[r], tw21[q], tw21[q]};
            const TData h[nface]  = {nbasis0[2u * p], nbasis0[2u * p + 1u],
                                     nbasis1[2u * q], nbasis1[2u * q + 1u],
                                     nbasis2[2u * r], nbasis2[2u * r + 1u]};
            // Each face's factor, formed once for every component.
            TData coef[nface];
#pragma unroll
            for (unsigned int iface = 0; iface < nface; ++iface)
            {
                TData faceJac;
                if constexpr (DEFORMED)
                {
                    faceJac = tjacptr[warpsize * k[iface]];
                }
                else
                {
                    faceJac = tjacptr[warpsize * iface];
                }
                coef[iface] = faceJac * w0[iface] * w1[iface] * h[iface];
            }
            for (unsigned int c = 0; c < ncomp; ++c)
            {
                const size_t ientry = compSize * c + i;
                TData val           = in[ientry];
                if (lift)
                {
                    const TData *traceptr =
                        trace + compTraceSize * c + traceOffset;
#pragma unroll
                    for (unsigned int iface = 0; iface < nface; ++iface)
                    {
                        val += traceptr[warpsize * k[iface]] * coef[iface];
                    }
                }
                PhysLiftProjectDirectLiftStoreKernel<APPEND>(out, ientry,
                                                             factor, val);
            }
        }
        else
        {
            const unsigned int tnq[6] = {
                traceSizeParam3D.nq00(), traceSizeParam3D.nq01(),
                traceSizeParam3D.nq10(), traceSizeParam3D.nq11(),
                traceSizeParam3D.nq20(), traceSizeParam3D.nq21()};
            const TData *nbasis[3] = {nbasis0, nbasis1, nbasis2};
            const TData *tw[6]     = {tw00, tw01, tw10, tw11, tw20, tw21};
            for (unsigned int c = 0; c < ncomp; ++c)
            {
                const size_t ientry = compSize * c + i;
                TData val           = in[ientry];
                if (lift)
                {
                    val +=
                        PhysLiftProjectOnePointLiftKernel<SHAPE_TYPE, DEFORMED>(
                            tnq, nbasis, tw, tjacptr,
                            trace + compTraceSize * c + traceOffset);
                }
                PhysLiftProjectDirectLiftStoreKernel<APPEND>(out, ientry,
                                                             factor, val);
            }
        }
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
