///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernels of the lift-and-project element
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the lift-and-project operator: the
 * fused launches of each form, and the diagonal inverse mass the
 * separate-operator route takes; the Pointwise kernel at width one of
 * that route is shared with the StdMat implementation
 * (PhysLiftProjectBlockOpDeviceKernels.hpp).
 *
 * @details
 * As the other SumFacTOP Device kernels, the block at width one, each
 * thread block striding over the elements and its threads over the
 * element's points or modes, so that consecutive threads touch consecutive
 * entries. Each kernel has the name and arguments of its SumFac
 * counterpart (PhysLiftProjectBlockOpDeviceSumFacKernels.hpp), restricted to
 * Implementation SumFacTOP, so that the caller launches
 * `Kernel<Implementation, ...>` for either.
 *
 * The fused launches chain the per-element functions of the
 * IProductWRTPhysTrace, IProductWRTBase and BwdTrans SumFacTOP kernels on
 * one shared-memory buffer per thread block, as the Mass kernels chain
 * BwdTrans and IProductWRTBase: the element's inner products, its
 * coefficients, the padded rows the basis transpose reads (conflict-free,
 * as in Mass), two stage scratches shared by the lift, the basis transpose
 * and the backward transform, and the basis tables, staged once per
 * thread block; each kernel reserves only the buffers of the stages it runs
 * (PhysLiftProjectSharedLayout()), and its launch is sized from the same
 * sizes. The block size is GetDeviceBlockSize() of the mode count,
 * as for Mass, the threads striding over points and modes alike.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>

#include <type_traits>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief Rows of the basis table of direction @p dir the SumFacTOP
 * kernels of IProductWRTBase and BwdTrans index, as Mass stages them: the
 * mode count, or the combined mode count where the table is indexed by
 * it. None in one dimension, whose table is read from global memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysLiftProjectBasisRows(
    [[maybe_unused]] const TSizeParameter sizeParam,
    [[maybe_unused]] const unsigned int dir)
{
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        if (dir == 0)
        {
            return sizeParam.nm0();
        }
        return (SHAPE_TYPE == LibUtilities::Quad) ? sizeParam.nm1()
                                                  : sizeParam.nmTot();
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const unsigned int nm0   = sizeParam.nm0();
        const unsigned int nm1   = sizeParam.nm1();
        const unsigned int nm2   = sizeParam.nm2();
        const unsigned int nmTot = sizeParam.nmTot();
        if (dir == 0)
        {
            return nm0;
        }
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return (dir == 1) ? nm1 : nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return (dir == 1)
                       ? (2u * nm1 - nm0 + 1u) * nm0 / 2u
                       : nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            return (dir == 1) ? nm1 : (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            return (dir == 1)
                       ? nm1
                       : nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        }
    }
    else
    {
        return 0u;
    }
}

/**
 * @brief Size of sum-factorisation stage scratch @p stage (zero or one) of
 * a fused SumFacTOP kernel that runs the lift (@p LIFT), the basis
 * transpose (@p TRANSPOSE) and the backward transform (@p BWD) or some of
 * them: the largest of what the IProductWRTPhysTrace, IProductWRTBase and
 * BwdTrans SumFacTOP kernels it runs ask of it, the last two as the Mass
 * kernels size them, the basis transpose's with its padded rows. The
 * collapsed shapes' basis transpose takes the two scratches the other way
 * round (the IProductWRTBase3DSumFacTOPKernel() calls), so that each stage
 * holds the same intermediate for both transforms.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool LIFT, bool TRANSPOSE,
          bool BWD, typename TData, typename TSizeParameter,
          typename TTraceSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysLiftProjectStageSize(
    [[maybe_unused]] const TSizeParameter sizeParam,
    [[maybe_unused]] const TTraceSizeParameter traceSizeParam,
    [[maybe_unused]] const unsigned int stage)
{
    unsigned int size = 0u;
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        if (stage == 0)
        {
            const unsigned int nm0 = sizeParam.nm0();
            const unsigned int nq1 = sizeParam.nq1();
            if constexpr (LIFT)
            {
                size =
                    2u * IProductWRTPhysTraceEdgeModeBlockSize(traceSizeParam);
            }
            if constexpr (TRANSPOSE)
            {
                size = std::max(
                    size,
                    nm0 * nq1 +
                        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                            sizeParam));
            }
            if constexpr (BWD)
            {
                size = std::max(size, (SHAPE_TYPE == LibUtilities::Quad)
                                          ? sizeParam.nm1() * sizeParam.nq0()
                                          : nm0 * nq1);
            }
        }
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const unsigned int nm0  = sizeParam.nm0();
        const unsigned int nm1  = sizeParam.nm1();
        const unsigned int nm2  = sizeParam.nm2();
        const unsigned int nq0  = sizeParam.nq0();
        const unsigned int nq1  = sizeParam.nq1();
        const unsigned int nq2  = sizeParam.nq2();
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        if constexpr (LIFT)
        {
            size = (stage == 0)
                       ? IProductWRTPhysTraceFaceModeBlockSize(traceSizeParam)
                       : IProductWRTPhysTraceFaceScratchSize(traceSizeParam);
        }
        unsigned int transpose = 0u, bwd = 0u;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            transpose = (stage == 0) ? nm0 * nq1 * nq2 : nm0 * nm1 * nq2;
            bwd       = (stage == 0) ? nq0 * nm1 * nm2 : nq0 * nq1 * nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            transpose = (stage == 0) ? nm01 * nq2 : nm0 * nq1 * nq2;
            bwd       = transpose;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::Pyr)
        {
            transpose = (stage == 0) ? nm0 * nm1 * nq2 : nm0 * nq1 * nq2;
            bwd       = transpose;
        }
        if constexpr (TRANSPOSE)
        {
            size = std::max(
                size,
                transpose + IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                                sizeParam));
        }
        if constexpr (BWD)
        {
            size = std::max(size, bwd);
        }
    }
    return size;
}

/**
 * @brief Shared memory one block of a fused SumFacTOP kernel takes, buffer
 * by buffer, in TData values, each rounded up to 16 bytes; zero for a
 * buffer the kernel does not use. The kernels carve their buffers out in
 * this order with no gaps, from the same sizes the launch is sized with.
 */
struct PhysLiftProjectSharedSizes
{
    /// The element's coefficients.
    unsigned int coef = 0u;
    /// Its inner products.
    unsigned int vol = 0u;
    /// Their copy with padded rows, which the basis transpose reads.
    unsigned int ipb = 0u;
    /// The two stage scratches of the lift and the transforms.
    unsigned int wsp0 = 0u;
    unsigned int wsp1 = 0u;
    /// The staged basis tables.
    unsigned int basis = 0u;

    NEK_HOSTDEVICE_INLINE constexpr unsigned int Total() const
    {
        return coef + vol + ipb + wsp0 + wsp1 + basis;
    }
};

/**
 * @brief The shared-memory buffers of fused SumFacTOP kernel @p KERNEL for
 * a block of form @p FORM (PhysLiftProjectSharedSizes): the fused launch
 * of the Pointwise form takes the inner products and the lift's scratch,
 * that of the StdInvMass form every buffer; the launch before the dense
 * product everything but the coefficients, and the one after it the
 * coefficients, the backward transform's scratch and the basis tables.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, PhysLiftProjectKernel KERNEL,
          PhysLiftProjectForm FORM, typename TData, typename TSizeParameter,
          typename TTraceSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr PhysLiftProjectSharedSizes
PhysLiftProjectSharedLayout(const TSizeParameter sizeParam,
                            const TTraceSizeParameter traceSizeParam)
{
    // The stages the kernel runs.
    constexpr bool PROJECT = KERNEL == PhysLiftProjectKernel::Fused &&
                             FORM != PhysLiftProjectForm::Pointwise;
    constexpr bool LIFT = KERNEL != PhysLiftProjectKernel::ScaleBwdTrans;
    constexpr bool TRANSPOSE =
        PROJECT || KERNEL == PhysLiftProjectKernel::LiftBasisTranspose;
    constexpr bool BWD =
        PROJECT || KERNEL == PhysLiftProjectKernel::ScaleBwdTrans;
    constexpr bool MULTID = !IsSizeParameter1D_v<TSizeParameter>;

    unsigned int nqTot = sizeParam.nq0();
    unsigned int basis = 0u;
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        nqTot *= sizeParam.nq1();
        basis = PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam, 0) *
                    sizeParam.nq0() +
                PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam, 1) *
                    sizeParam.nq1();
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        nqTot *= sizeParam.nq1() * sizeParam.nq2();
        basis = PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam, 0) *
                    sizeParam.nq0() +
                PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam, 1) *
                    sizeParam.nq1() +
                PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam, 2) *
                    sizeParam.nq2();
    }

    PhysLiftProjectSharedSizes size;
    if constexpr (BWD)
    {
        size.coef = IProductWRTBaseAlignedSize<TData>(sizeParam.nmTot());
    }
    if constexpr (LIFT)
    {
        size.vol = IProductWRTBaseAlignedSize<TData>(nqTot);
    }
    if constexpr (TRANSPOSE && MULTID)
    {
        size.ipb = IProductWRTBaseAlignedSize<TData>(
            IProductWRTBasePaddedInputSize<TData>(sizeParam));
    }
    size.wsp0 = IProductWRTBaseAlignedSize<TData>(
        PhysLiftProjectStageSize<SHAPE_TYPE, LIFT, TRANSPOSE, BWD, TData>(
            sizeParam, traceSizeParam, 0));
    size.wsp1 = IProductWRTBaseAlignedSize<TData>(
        PhysLiftProjectStageSize<SHAPE_TYPE, LIFT, TRANSPOSE, BWD, TData>(
            sizeParam, traceSizeParam, 1));
    if constexpr ((TRANSPOSE || BWD) && MULTID)
    {
        size.basis = IProductWRTBaseAlignedSize<TData>(basis);
    }
    return size;
}

/**
 * @brief Dynamic shared memory one block of fused SumFacTOP kernel
 * @p KERNEL needs, in TData values: one element resident at a time, its
 * buffers (PhysLiftProjectSharedLayout()) back to back.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectKernel KERNEL, PhysLiftProjectForm FORM,
          typename TData, typename TSizeParameter, typename TTraceSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
inline constexpr unsigned int PhysLiftProjectSharedMemorySize(
    const TSizeParameter sizeParam, const TTraceSizeParameter traceSizeParam)
{
    return PhysLiftProjectSharedLayout<SHAPE_TYPE, KERNEL, FORM, TData>(
               sizeParam, traceSizeParam)
        .Total();
}

/// The Pointwise form from @p s_vol: `out (+)= scale / (w_q J) * s_vol`
/// at every point of element @p e. No barrier follows: each thread reads
/// here only the points of @p s_vol it alone writes in the next element's
/// StageToSharedMemory(), which strides them the same way.
template <bool APPEND, bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectPointScaleTOPKernel(
    const size_t e, const unsigned int nqTot, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT s_vol, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    for (unsigned int q = getLocalIdx<0>(threadBlock); q < nqTot;
         q += getLocalRange<0>(threadBlock))
    {
        TData jacobian;
        if constexpr (DEFORMED)
        {
            jacobian = jac[e * nqTot + q];
        }
        else
        {
            jacobian = jac[e];
        }
        const TData factor = SafeDivide(scale, weights[q] * jacobian);
        if constexpr (APPEND)
        {
            out[q] += factor * s_vol[q];
        }
        else
        {
            out[q] = factor * s_vol[q];
        }
    }
}

/// `s_coef *= factor * diag`, or by @p factor alone without @p DIAG, then
/// a barrier.
template <bool DIAG, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectCoefScaleTOPKernel(
    const unsigned int nmTot, const TData factor,
    [[maybe_unused]] const TData *NEK_RESTRICT diag, TData *NEK_RESTRICT s_coef,
    const TthreadBlock &threadBlock)
{
    for (unsigned int m = getLocalIdx<0>(threadBlock); m < nmTot;
         m += getLocalRange<0>(threadBlock))
    {
        if constexpr (DIAG)
        {
            s_coef[m] *= factor * diag[m];
        }
        else
        {
            s_coef[m] *= factor;
        }
    }
    localBarrier(threadBlock);
}

/// The lift of a segment's two end values onto @p s_vol, then a barrier
/// (one dimension has no lift function of its own).
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectSegLiftTOPKernel(
    const unsigned int nq0, const TData *NEK_RESTRICT nbasis0,
    const bool endPtsCollocated0, const TData *NEK_RESTRICT trace,
    TData *NEK_RESTRICT s_vol, const TthreadBlock &threadBlock)
{
    for (unsigned int p = getLocalIdx<0>(threadBlock); p < nq0;
         p += getLocalRange<0>(threadBlock))
    {
        if (endPtsCollocated0)
        {
            if (p == 0u)
            {
                s_vol[p] += trace[0];
            }
            if (p == nq0 - 1u)
            {
                s_vol[p] += trace[1];
            }
        }
        else
        {
            s_vol[p] +=
                nbasis0[2 * p] * trace[0] + nbasis0[2 * p + 1] * trace[1];
        }
    }
    localBarrier(threadBlock);
}

/// The padded copy of @p s_vol the basis transpose reads (Mass's layout),
/// or @p s_vol itself where the rows need no padding.
template <typename TSizeParameter, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static const TData *PhysLiftProjectPadKernel(
    const TSizeParameter sizeParam, const TData *NEK_RESTRICT s_vol,
    TData *NEK_RESTRICT s_ipb, const TthreadBlock &threadBlock)
{
    const unsigned int nq0 = sizeParam.nq0();
    const unsigned int nq1 = sizeParam.nq1();
    const unsigned int ldi = IProductWRTBaseRowStride<TData>(nq0);
    if (ldi == nq0)
    {
        return s_vol;
    }
    unsigned int nqTot = nq0 * nq1;
    if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        nqTot *= sizeParam.nq2();
    }
    for (unsigned int idx = getLocalIdx<0>(threadBlock); idx < nqTot;
         idx += getLocalRange<0>(threadBlock))
    {
        const unsigned int i   = idx % nq0;
        const unsigned int row = idx / nq0;
        s_ipb[ldi * row + i]   = s_vol[idx];
    }
    localBarrier(threadBlock);
    return s_ipb;
}

/**
 * @brief The inverse mass of an affine element with the scale at width one:
 * `out[m] = scale / J * diag[m] * in[m]`, @p diag left out unless @p DIAG,
 * zero on the padding elements. @p in may be @p out. Grid dimension 1 is
 * the component.
 */
template <typename Implementation, bool DIAG, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectInvMassScaleKernel(
    const size_t nelmt, const unsigned int nmTot, const TData scale,
    const TData *NEK_RESTRICT diag, const TData *NEK_RESTRICT jac,
    const TData *in, TData *out, const TthreadBlock &threadBlock)
{
    const unsigned int c = getBlockIdx<1>(threadBlock);
    for (size_t e = getBlockIdx<0>(threadBlock); e < nelmt;
         e += getBlockRange<0>(threadBlock))
    {
        const size_t offset = (nelmt * c + e) * nmTot;
        const TData factor  = SafeDivide(scale, jac[e]);
        for (unsigned int m = getLocalIdx<0>(threadBlock); m < nmTot;
             m += getLocalRange<0>(threadBlock))
        {
            if constexpr (DIAG)
            {
                out[offset + m] = factor * diag[m] * in[offset + m];
            }
            else
            {
                out[offset + m] = factor * in[offset + m];
            }
        }
    }
}

/**
 * @brief The fused Pointwise and diagonal-inverse-mass forms in one
 * dimension at width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectForm FORM, bool APPEND, bool DEFORMED,
          typename TSizeParameter1D, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        const TTraceSizeParameter1D traceSizeParam1D, const size_t nelmt,
        const TData scale, const bool lift,
        [[maybe_unused]] const bool isModified,
        [[maybe_unused]] const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const bool endPtsCollocated0,
        [[maybe_unused]] const TData *NEK_RESTRICT weights,
        const TData *NEK_RESTRICT jac,
        [[maybe_unused]] const TData *NEK_RESTRICT diag, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT coef,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                    FORM, TData>(sizeParam1D, traceSizeParam1D);
    TData *s_coef = (TData *)shmemptr;
    TData *s_vol  = s_coef + size.coef;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nq0 * slot;
        TData *outptr      = out + nq0 * slot;

        // Copy to shared memory.
        StageToSharedMemory(nq0, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            PhysLiftProjectSegLiftTOPKernel(nq0, nbasis0, endPtsCollocated0,
                                            trace + 2 * slot, s_vol,
                                            threadBlock);
        }

        if constexpr (FORM == PhysLiftProjectForm::Pointwise)
        {
            PhysLiftProjectPointScaleTOPKernel<APPEND, DEFORMED>(
                e, nq0, scale, weights, jac, s_vol, outptr, threadBlock);
        }
        else
        {
            IProductWRTBaseSegSumFacTOPKernel<false, false>(
                nm0, nq0, basis0, s_vol, s_coef, TData(1.0), threadBlock);
            PhysLiftProjectCoefScaleTOPKernel<true>(
                nm0, SafeDivide(scale, jac[e]), diag, s_coef, threadBlock);
            BwdTransSegSumFacTOPKernel<APPEND>(nm0, nq0, basis0, s_coef, outptr,
                                               threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The fused Pointwise and diagonal-inverse-mass forms in two
 * dimensions at width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectForm FORM, bool APPEND, bool DEFORMED,
          typename TSizeParameter2D, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const TData scale, const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
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
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT lift0, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int tnq00 = traceSizeParam2D.nq00();
    const unsigned int tnq10 = traceSizeParam2D.nq10();
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                    FORM, TData>(sizeParam2D, traceSizeParam2D);

    const unsigned int nmode0 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam2D, 0);
    const unsigned int nmode1 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam2D, 1);

    const unsigned int numDataJac =
        DEFORMED ? nqTotTrace : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    TData *s_coef   = (TData *)shmemptr;
    TData *s_vol    = s_coef + size.coef;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *s_basis0 = s_wsp0 + size.wsp0;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

    if constexpr (FORM != PhysLiftProjectForm::Pointwise)
    {
        // Copy to shared memory.
        StageToSharedMemory(nmode0 * nq0, basis0, s_basis0, threadBlock);
        StageToSharedMemory(nmode1 * nq1, basis1, s_basis1, threadBlock);
    }

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *outptr      = out + nqTot * slot;

        // Copy to shared memory.
        StageToSharedMemory(nqTot, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            IProductWRTPhysTrace2DTOPKernel<SHAPE_TYPE, DEFORMED, true>(
                nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0, tbasis1,
                tw00, tw10, tjac + numDataJac * e, s_wsp0,
                trace + nqTotTrace * slot, s_vol, isCollocated0, isCollocated1,
                endPtsCollocated0, endPtsCollocated1, threadBlock);
        }

        if constexpr (FORM == PhysLiftProjectForm::Pointwise)
        {
            PhysLiftProjectPointScaleTOPKernel<APPEND, DEFORMED>(
                e, nqTot, scale, weights, jac, s_vol, outptr, threadBlock);
        }
        else
        {
            const TData *ipb = PhysLiftProjectPadKernel(sizeParam2D, s_vol,
                                                        s_ipb, threadBlock);
            IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, ipb, s_coef, s_wsp0, (TData)1.0, threadBlock);
            PhysLiftProjectCoefScaleTOPKernel<true>(
                nmTot, SafeDivide(scale, jac[e]), diag, s_coef, threadBlock);
            BwdTrans2DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
                nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1,
                s_coef, outptr, s_wsp0, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The Pointwise form, or the StdInvMass form with a diagonal
 * standard inverse mass, of a block of three-dimensional elements in one
 * launch, one element per thread block at width one.
 *
 * The block's threads load the element's inner products into shared
 * memory, coalesced, lift the trace onto them there, and then
 * - Pointwise: write `out (+)= scale / (w J) * vol`, coalesced;
 * - StdInvMass: apply the basis transpose into the shared coefficients
 *   from the padded rows Mass uses, scale them by `scale / J * diag`, and
 *   apply the backward transform into @p out, appending when @p APPEND.
 * One shared-memory buffer (PhysLiftProjectSharedLayout()) serves every
 * stage: the lift's scratch is the two stage scratches of the basis
 * transpose and the backward transform, and the basis tables are staged
 * once per thread block, as Mass does. The arguments are those of the
 * SumFac counterpart; @p vol, @p coef and the global workspaces are not
 * read, the intermediates living in shared memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectForm FORM, bool APPEND, bool DEFORMED,
          typename TSizeParameter3D, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const TData scale, const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
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
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT wsp1,
        [[maybe_unused]] TData *NEK_RESTRICT lift0,
        [[maybe_unused]] TData *NEK_RESTRICT lift1, unsigned char *shmemptr,
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
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int tnq00 = traceSizeParam3D.nq00();
    const unsigned int tnq01 = traceSizeParam3D.nq01();
    const unsigned int tnq10 = traceSizeParam3D.nq10();
    const unsigned int tnq11 = traceSizeParam3D.nq11();
    const unsigned int tnq20 = traceSizeParam3D.nq20();
    const unsigned int tnq21 = traceSizeParam3D.nq21();
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                    FORM, TData>(sizeParam3D, traceSizeParam3D);

    const unsigned int nmode0 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 0);
    const unsigned int nmode1 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 1);
    const unsigned int nmode2 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 2);

    const unsigned int numDataJac =
        DEFORMED ? nqTotTrace : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    TData *s_coef   = (TData *)shmemptr;
    TData *s_vol    = s_coef + size.coef;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *s_wsp1   = s_wsp0 + size.wsp0;
    TData *s_basis0 = s_wsp1 + size.wsp1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

    if constexpr (FORM != PhysLiftProjectForm::Pointwise)
    {
        // Copy to shared memory.
        StageToSharedMemory(nmode0 * nq0, basis0, s_basis0, threadBlock);
        StageToSharedMemory(nmode1 * nq1, basis1, s_basis1, threadBlock);
        StageToSharedMemory(nmode2 * nq2, basis2, s_basis2, threadBlock);
    }

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *outptr      = out + nqTot * slot;

        // Copy to shared memory.
        StageToSharedMemory(nqTot, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            IProductWRTPhysTrace3DFaceTOPKernel<SHAPE_TYPE, DEFORMED, true>(
                nq0, nq1, nq2, nbasis0, nbasis1, nbasis2, tnq00, tnq01, tnq10,
                tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
                tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21,
                tjac + numDataJac * e, s_wsp0, s_wsp1,
                trace + nqTotTrace * slot, s_vol, isCollocated00,
                isCollocated01, isCollocated10, isCollocated11, isCollocated20,
                isCollocated21, endPtsCollocated0, endPtsCollocated1,
                endPtsCollocated2, threadBlock);
        }

        if constexpr (FORM == PhysLiftProjectForm::Pointwise)
        {
            PhysLiftProjectPointScaleTOPKernel<APPEND, DEFORMED>(
                e, nqTot, scale, weights, jac, s_vol, outptr, threadBlock);
        }
        else
        {
            const TData *ipb   = PhysLiftProjectPadKernel(sizeParam3D, s_vol,
                                                          s_ipb, threadBlock);
            constexpr bool hex = (SHAPE_TYPE == LibUtilities::Hex);
            IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb, s_coef,
                hex ? s_wsp0 : s_wsp1, hex ? s_wsp1 : s_wsp0, (TData)1.0,
                threadBlock);
            PhysLiftProjectCoefScaleTOPKernel<true>(
                nmTot, SafeDivide(scale, jac[e]), diag, s_coef, threadBlock);
            BwdTrans3DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
                s_basis0, s_basis1, s_basis2, s_coef, outptr, s_wsp0, s_wsp1,
                threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form before the product in one dimension at
 * width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *vol, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    TData *s_vol = (TData *)shmemptr;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nq0 * slot;
        TData *coefptr     = coef + nm0 * slot;

        // Copy to shared memory.
        StageToSharedMemory(nq0, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            PhysLiftProjectSegLiftTOPKernel(nq0, nbasis0, endPtsCollocated0,
                                            trace + 2 * slot, s_vol,
                                            threadBlock);
        }
        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, s_vol, coefptr, TData(1.0), threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form before the product in two dimensions
 * at width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac, const bool isCollocated0,
        const bool isCollocated1, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT lift0, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

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

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE,
                                    PhysLiftProjectKernel::LiftBasisTranspose,
                                    PhysLiftProjectForm::StdInvMass, TData>(
            sizeParam2D, traceSizeParam2D);

    const unsigned int nmode0 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam2D, 0);
    const unsigned int nmode1 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam2D, 1);

    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    TData *s_vol    = (TData *)shmemptr;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *s_basis0 = s_wsp0 + size.wsp0;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

    // Copy to shared memory.
    StageToSharedMemory(nmode0 * nq0, basis0, s_basis0, threadBlock);
    StageToSharedMemory(nmode1 * nq1, basis1, s_basis1, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *coefptr     = coef + nmTot * slot;

        // Copy to shared memory.
        StageToSharedMemory(nqTot, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            IProductWRTPhysTrace2DTOPKernel<SHAPE_TYPE, false, true>(
                nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0, tbasis1,
                tw00, tw10, tjac + numDataJac * e, s_wsp0,
                trace + nqTotTrace * slot, s_vol, isCollocated0, isCollocated1,
                endPtsCollocated0, endPtsCollocated1, threadBlock);
        }
        const TData *ipb =
            PhysLiftProjectPadKernel(sizeParam2D, s_vol, s_ipb, threadBlock);
        IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
            nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
            s_basis1, ipb, coefptr, s_wsp0, (TData)1.0, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, before
 * the product, of a block of three-dimensional affine elements in one
 * launch, one element per thread block at width one: the inner products
 * loaded into shared memory, the trace lifted onto them there and the
 * basis transpose written to @p coef, coalesced. The shared memory holds
 * no coefficients (PhysLiftProjectSharedLayout()).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
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
        TData *NEK_RESTRICT coef, [[maybe_unused]] TData *vol,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT wsp1,
        [[maybe_unused]] TData *NEK_RESTRICT lift0,
        [[maybe_unused]] TData *NEK_RESTRICT lift1, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

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

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE,
                                    PhysLiftProjectKernel::LiftBasisTranspose,
                                    PhysLiftProjectForm::StdInvMass, TData>(
            sizeParam3D, traceSizeParam3D);

    const unsigned int nmode0 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 0);
    const unsigned int nmode1 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 1);
    const unsigned int nmode2 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 2);

    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    TData *s_vol    = (TData *)shmemptr;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *s_wsp1   = s_wsp0 + size.wsp0;
    TData *s_basis0 = s_wsp1 + size.wsp1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    StageToSharedMemory(nmode0 * nq0, basis0, s_basis0, threadBlock);
    StageToSharedMemory(nmode1 * nq1, basis1, s_basis1, threadBlock);
    StageToSharedMemory(nmode2 * nq2, basis2, s_basis2, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *coefptr     = coef + nmTot * slot;

        // Copy to shared memory.
        StageToSharedMemory(nqTot, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            IProductWRTPhysTrace3DFaceTOPKernel<SHAPE_TYPE, false, true>(
                nq0, nq1, nq2, nbasis0, nbasis1, nbasis2, tnq00, tnq01, tnq10,
                tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
                tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21,
                tjac + numDataJac * e, s_wsp0, s_wsp1,
                trace + nqTotTrace * slot, s_vol, isCollocated00,
                isCollocated01, isCollocated10, isCollocated11, isCollocated20,
                isCollocated21, endPtsCollocated0, endPtsCollocated1,
                endPtsCollocated2, threadBlock);
        }
        const TData *ipb =
            PhysLiftProjectPadKernel(sizeParam3D, s_vol, s_ipb, threadBlock);
        constexpr bool hex = (SHAPE_TYPE == LibUtilities::Hex);
        IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
            index1, index2, s_basis0, s_basis1, s_basis2, ipb, coefptr,
            hex ? s_wsp0 : s_wsp1, hex ? s_wsp1 : s_wsp0, (TData)1.0,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form after the product in one dimension at
 * width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
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
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    TData *s_coef = (TData *)shmemptr;

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot = nelmt * c + e;
        TData *coefptr    = coef + nm0 * slot;
        TData *outptr     = out + nq0 * slot;

        // Copy to shared memory.
        StageToSharedMemory(nm0, coefptr, s_coef, threadBlock);
        localBarrier(threadBlock);

        PhysLiftProjectCoefScaleTOPKernel<false>(nm0, SafeDivide(scale, jac[e]),
                                                 (const TData *)nullptr, s_coef,
                                                 threadBlock);
        BwdTransSegSumFacTOPKernel<APPEND>(nm0, nq0, basis0, s_coef, outptr,
                                           threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form after the product in two dimensions at
 * width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const TData scale, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT coef,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE,
                                    PhysLiftProjectKernel::ScaleBwdTrans,
                                    PhysLiftProjectForm::StdInvMass, TData>(
            sizeParam2D, traceSizeParam2D);

    const unsigned int nmode0 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam2D, 0);
    const unsigned int nmode1 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam2D, 1);

    TData *s_coef   = (TData *)shmemptr;
    TData *s_wsp0   = s_coef + size.coef;
    TData *s_basis0 = s_wsp0 + size.wsp0;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

    // Copy to shared memory.
    StageToSharedMemory(nmode0 * nq0, basis0, s_basis0, threadBlock);
    StageToSharedMemory(nmode1 * nq1, basis1, s_basis1, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot = nelmt * c + e;
        TData *coefptr    = coef + nmTot * slot;
        TData *outptr     = out + nqTot * slot;

        // Copy to shared memory.
        StageToSharedMemory(nmTot, coefptr, s_coef, threadBlock);
        localBarrier(threadBlock);

        PhysLiftProjectCoefScaleTOPKernel<false>(
            nmTot, SafeDivide(scale, jac[e]), (const TData *)nullptr, s_coef,
            threadBlock);
        BwdTrans2DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
            nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1, s_coef,
            outptr, s_wsp0, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, after
 * the product, of a block of three-dimensional affine elements in one
 * launch, one element per thread block at width one: the coefficients
 * loaded into shared memory, coalesced, scaled by `scale / J` there, and
 * the backward transform written to @p out, appending when @p APPEND. The
 * shared memory holds neither the inner products nor their padded copy,
 * and its scratch is sized for the backward transform alone
 * (PhysLiftProjectSharedLayout()); the lift's size parameter is passed for
 * the arguments to match the other launches'.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const TData scale, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT jac,
        TData *NEK_RESTRICT coef, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT wsp1, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE,
                                    PhysLiftProjectKernel::ScaleBwdTrans,
                                    PhysLiftProjectForm::StdInvMass, TData>(
            sizeParam3D, traceSizeParam3D);

    const unsigned int nmode0 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 0);
    const unsigned int nmode1 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 1);
    const unsigned int nmode2 =
        PhysLiftProjectBasisRows<SHAPE_TYPE>(sizeParam3D, 2);

    TData *s_coef   = (TData *)shmemptr;
    TData *s_wsp0   = s_coef + size.coef;
    TData *s_wsp1   = s_wsp0 + size.wsp0;
    TData *s_basis0 = s_wsp1 + size.wsp1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    StageToSharedMemory(nmode0 * nq0, basis0, s_basis0, threadBlock);
    StageToSharedMemory(nmode1 * nq1, basis1, s_basis1, threadBlock);
    StageToSharedMemory(nmode2 * nq2, basis2, s_basis2, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot = nelmt * c + e;
        TData *coefptr    = coef + nmTot * slot;
        TData *outptr     = out + nqTot * slot;

        // Copy to shared memory.
        StageToSharedMemory(nmTot, coefptr, s_coef, threadBlock);
        localBarrier(threadBlock);

        PhysLiftProjectCoefScaleTOPKernel<false>(
            nmTot, SafeDivide(scale, jac[e]), (const TData *)nullptr, s_coef,
            threadBlock);
        BwdTrans3DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
            nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
            s_basis0, s_basis1, s_basis2, s_coef, outptr, s_wsp0, s_wsp1,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
