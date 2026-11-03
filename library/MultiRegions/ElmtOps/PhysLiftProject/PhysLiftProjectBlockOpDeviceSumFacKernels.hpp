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
 * @brief Device SumFac kernels of the lift-and-project operator: the fused
 * launches of each form, and the scales the separate-operator route takes.
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
 * - PhysLiftProjectKernelLauncher: the Pointwise form, or the StdInvMass
 *   form with a diagonal standard inverse mass, in one launch;
 * - PhysLiftProjectLiftBasisTransposeKernelLauncher and
 *   PhysLiftProjectScaleBwdTransKernelLauncher: the StdInvMass form with a
 *   dense standard inverse mass, either side of the batched product.
 * Their intermediates live in the global workspace, one region per
 * intermediate laid out as a block at warpSize.
 *
 * PhysLiftProjectPointwiseKernel and PhysLiftProjectInvMassScaleKernel
 * serve the separate-operator route, which takes the nodal shapes.
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
                           SHAPE_TYPE == LibUtilities::Pyr)
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
                           SHAPE_TYPE == LibUtilities::Pyr)
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
          PhysLiftProjectKernel KERNEL, PhysLiftProjectForm FORM,
          typename TData, typename TSizeParameter, typename TTraceSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
inline constexpr unsigned int PhysLiftProjectSharedMemorySize(
    [[maybe_unused]] const TSizeParameter sizeParam,
    [[maybe_unused]] const TTraceSizeParameter traceSizeParam)
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

/// The Pointwise form from @p vol: `out (+)= scale / (w_q J) * vol` at
/// every point of the element of lane @p ilane in group @p iwarp.
template <bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectPointScaleKernel(
    const unsigned int ilane, const size_t iwarp, const size_t e,
    const unsigned int nqTot, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT vol, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

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
        const TData factor =
            SafeDivide(scale, weights[q * warpsize + ilane] * jacobian);
        const unsigned int i = warpsize * q + ilane;
        if constexpr (APPEND)
        {
            out[i] += factor * vol[i];
        }
        else
        {
            out[i] = factor * vol[i];
        }
    }
}

/**
 * @brief The Pointwise form: `out (+)= scale / (w_q J) * in` at every point,
 * zero on the padding elements, whose Jacobian is zero. Grid dimension 1 is
 * the component.
 *
 * @param   weights  The quadrature weights at every point (WeightsKey at
 *                   warpSize).
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
            const TData factor =
                SafeDivide(scale, weights[q * warpsize + ilane] * jacobian);
            const size_t i = offset + static_cast<size_t>(q) * warpsize;
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
 * @brief The fused Pointwise and diagonal-inverse-mass forms in one
 * dimension. See the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectForm FORM, bool APPEND, bool DEFORMED,
          typename TSizeParameter1D, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
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
        const bool endPtsCollocated0,
        [[maybe_unused]] const TData *NEK_RESTRICT weights,
        const TData *NEK_RESTRICT jac,
        [[maybe_unused]] const TData *NEK_RESTRICT diag, const TData *in,
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

        if constexpr (FORM == PhysLiftProjectForm::Pointwise)
        {
            PhysLiftProjectPointScaleKernel<APPEND, DEFORMED>(
                ilane, iwarp, e, nq0, scale, weights, jac, volptr, outptr);
        }
        else
        {
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
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The fused Pointwise and diagonal-inverse-mass forms in two
 * dimensions. See the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectForm FORM, bool APPEND, bool DEFORMED,
          typename TSizeParameter2D, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
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
        const bool endPtsCollocated1,
        [[maybe_unused]] const TData *NEK_RESTRICT weights,
        const TData *NEK_RESTRICT jac,
        [[maybe_unused]] const TData *NEK_RESTRICT diag, const TData *in,
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
        DEFORMED ? nqTotTrace : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

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
            IProductWRTPhysTrace2DKernel<SHAPE_TYPE, DEFORMED, true>(
                ilane, nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0,
                tbasis1, tw00, tw10, tjac + numDataJac * warpsize * iwarp,
                lift0 + lift0Size * slot, trace + nqTotTrace * slot, volptr,
                isCollocated0, isCollocated1, endPtsCollocated0,
                endPtsCollocated1);
        }

        if constexpr (FORM == PhysLiftProjectForm::Pointwise)
        {
            PhysLiftProjectPointScaleKernel<APPEND, DEFORMED>(
                ilane, iwarp, e, nqTot, scale, weights, jac, volptr, outptr);
        }
        else
        {
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
            BwdTrans2DSumFacKernel<SHAPE_TYPE, APPEND>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, coefptr,
                outptr, wsp0ptr);
        }

        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief The Pointwise form, or the StdInvMass form with a diagonal
 * standard inverse mass, of a block of three-dimensional elements in one
 * launch, one thread per element at warpSize.
 *
 * Each thread copies its element's inner products into @p vol (unless the
 * input was interleaved into @p vol already), lifts the trace onto them,
 * and then
 * - Pointwise: `out (+)= scale / (w J) * vol`;
 * - StdInvMass: the basis transpose into @p coef, `coef *= scale / J * diag`
 *   and the backward transform into @p out, appending when @p APPEND.
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
          PhysLiftProjectForm FORM, bool APPEND, bool DEFORMED,
          typename TSizeParameter3D, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
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
        [[maybe_unused]] const TData *NEK_RESTRICT weights,
        const TData *NEK_RESTRICT jac,
        [[maybe_unused]] const TData *NEK_RESTRICT diag, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out, TData *vol,
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
        DEFORMED ? nqTotTrace : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

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
            IProductWRTPhysTrace3DFaceKernel<SHAPE_TYPE, DEFORMED, true>(
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

        if constexpr (FORM == PhysLiftProjectForm::Pointwise)
        {
            PhysLiftProjectPointScaleKernel<APPEND, DEFORMED>(
                ilane, iwarp, e, nqTot, scale, weights, jac, volptr, outptr);
        }
        else
        {
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
        }

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

#endif

} // namespace Nektar::MultiRegions::detail
