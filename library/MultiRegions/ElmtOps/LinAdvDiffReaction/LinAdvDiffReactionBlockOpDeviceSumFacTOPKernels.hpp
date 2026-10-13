///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: SumFacTOP device kernels of the linearised
// advection-diffusion-reaction operator, one element per thread block.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOpDeviceSumFacTOPKernels.hpp
 * @brief SumFacTOP device kernels of the linearised
 * advection-diffusion-reaction operator: one element per thread block,
 * with every intermediate staged in shared memory.
 *
 * @details
 * These are the kernels the Device/SumFacTOP block implementation
 * launches (see LinAdvDiffReactionBlockOpDeviceSumFac.hpp); the SumFac
 * counterparts, where one thread owns a whole element, are in
 * LinAdvDiffReactionBlockOpDeviceSumFacKernels.hpp. Everything below is guarded
 * by NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY, so it exists only in
 * translation units compiled for a device back-end.
 *
 * One thread block evaluates one element, striding over the block's
 * elements; within an element the block's threads are indexed over the
 * work items of the current stage -- output points or modes -- and
 * synchronise between stages. The field data is therefore read
 * non-interleaved (one element's values contiguous), and the whole
 * per-element working set lives in the dynamic shared memory the launch
 * reserves: in two and three dimensions the reference-space diffusion
 * metric first, then the physical values, the derivative components,
 * the shape's sum-factorisation intermediates and the basis tables,
 * which are copied in once at the start because every element of the
 * block shares them. The layout is exactly the one
 * HelmholtzSharedMemorySize accounts for, this family adding no
 * intermediates of its own.
 *
 * A second grid dimension carries the field-component (or
 * homogeneous-mode) index: the launchers read it as `c` from the thread
 * block's second index. The `in` and `out` arrays hold every
 * component's data together as consecutive regions of `nelmt` elements,
 * so component `c`'s data starts `c * nelmt` elements' worth in, the
 * usual per-element offset applying inside that region; the
 * shared-memory layout above is per element only and does not depend on
 * `c`. The advection velocity is shared by the variables but not by the
 * planes, so the 1D and 2D launchers read it on plane `c % nhomo`; a 3D
 * field has a single plane.
 *
 * The stage sequence is the same as the SumFac drivers' -- backward
 * transform, derivatives, advection, diffusion metric with quadrature
 * weighting, summed transposed derivative, inner product -- built from
 * the SumFacTOP kernels of the BwdTrans, PhysDeriv, Helmholtz and
 * IProductWRTBase families with this family's AddAdvection kernels
 * spliced in, and with the reaction term folded into the advection step
 * so that the metric stage runs with a unit coefficient. The PhysDeriv
 * stage already applies the collapsed-coordinate factors and the
 * derivative factors, so the advection kernels see Cartesian
 * derivatives.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

// get hold of memory sizing and routines related to collocation Helmholtz op
#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzBlockOpDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Segment advection kernel: add the advection term to the scaled
 * physical values of the block's element.
 *
 * The threads of the block share the element's quadrature points
 * between them and each writes
 * \f$\mathbf{V}\cdot\nabla u + \mathrm{scale}\cdot u\f$ for the points
 * it owns, the derivatives arriving already in Cartesian form from the
 * PhysDeriv stage. The closing barrier makes the result visible to the
 * next stage. All pointers address one element's contiguous values; the
 * higher velocity and derivative components are read only when
 * @p ncoord allows.
 *
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   ncoord  Coordinate dimension: how many components are used.
 * @param   nq0     Quadrature points of the element.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   deriv0,deriv1,deriv2        Cartesian derivatives of
 *                                      \f$u\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out: the
 *                      advection-plus-reaction values.
 * @param   scale   Reaction coefficient \f$\lambda\f$.
 * @param   threadBlock Back-end thread-block handle supplying the local
 *                      thread index, its stride and the barrier.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DSumFacTOPKernel(
    const unsigned ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

/**
 * @brief Two-dimensional advection kernel: add the advection term to
 * the scaled physical values of the block's element.
 *
 * As AddAdvection1DSumFacTOPKernel, over the element's nq0 * nq1
 * points. The third velocity and derivative component is read only for
 * a surface element embedded in three dimensions (@p ncoord == 3).
 *
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   ncoord  Coordinate dimension (2 or 3).
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   deriv0,deriv1,deriv2        Cartesian derivatives of
 *                                      \f$u\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out: the
 *                      advection-plus-reaction values.
 * @param   scale   Reaction coefficient \f$\lambda\f$.
 * @param   threadBlock Back-end thread-block handle.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

/**
 * @brief Three-dimensional advection kernel: add the advection term to
 * the scaled physical values of the block's element.
 *
 * As AddAdvection2DSumFacTOPKernel, over the element's
 * nq0 * nq1 * nq2 points. There is no coordinate-dimension argument: a
 * three-dimensional element has coordinate dimension three, so all
 * three components always contribute.
 *
 * @tparam TthreadBlock    Back-end thread-block handle type.
 * @tparam TData           Floating-point type of the field data.
 *
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   deriv0,deriv1,deriv2        Cartesian derivatives of
 *                                      \f$u\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out: the
 *                      advection-plus-reaction values.
 * @param   scale   Reaction coefficient \f$\lambda\f$.
 * @param   threadBlock Back-end thread-block handle.
 */
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void AddAdvection3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
    const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT deriv0,
    const TData *NEK_RESTRICT deriv1, const TData *NEK_RESTRICT deriv2,
    TData *NEK_RESTRICT out, const TData scale, const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int i = idx0; i < nq0 * nq1 * nq2; i += stride)
    {
        out[i] = scale * out[i] + advVel0[i] * deriv0[i] +
                 advVel1[i] * deriv1[i] + advVel2[i] * deriv2[i];
    }
    localBarrier(threadBlock);
}

/**
 * @brief Entry point and driver of the 1D SumFacTOP launch: apply the
 * whole operator to the block's segments, one element per thread block.
 *
 * @details
 * The enable_if constrains this overload to the SumFacTOP tag, so that
 * the block implementation's single call site resolves to this launcher
 * or to its SumFac namesake in LinAdvDiffReactionBlockOpDeviceSumFacKernels.hpp
 * according to the Implementation tag it was instantiated with; a
 * static_assert then rejects anything but the 1D size parameters,
 * runtime or compile-time. The launch bounds come from
 * GetMaxThreadPerBlock for the tag and size-parameter pair.
 *
 * After unpacking @p sizeParam1D, thread blocks stride over the block's
 * elements; for each one the shared memory is used as the physical
 * values followed by @p ncoord derivative components, `nq0` values
 * apiece, which fits within the `4 * nq0` HelmholtzSharedMemorySize
 * reserves for a segment. The element's coefficients and the basis
 * table are read straight from global memory, and the 1D metric kernel
 * takes no shared metric slot. The stage sequence of the file notes then
 * runs with the block's threads sharing each stage's work items. No
 * global-memory workspace is needed, which is why @p wsp is unused.
 *
 * @tparam SHAPE_TYPE          Element shape (segments only).
 * @tparam Implementation      MultiRegions::SumFacTOP.
 * @tparam DEFORMED            Geometry is deformed.
 * @tparam TSizeParameter1D    Runtime or compile-time 1D size
 *                             parameter.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam1D     Element mode and point counts.
 * @param   ncoord          Coordinate dimension: velocity components,
 *                          derivative components and rank of the
 *                          symmetric diffusivity tensor, whose packed
 *                          lower triangle therefore has
 *                          ncoord * (ncoord + 1) / 2 entries.
 * @param   nelmt           Elements in the block, padding included.
 * @param   nhomo           Homogeneous planes of the field.
 * @param   isModified      First-direction basis is eModified_A; unused
 *                          for segments.
 * @param   basis0          1D basis table.
 * @param   D0              1D collocation derivative matrix.
 * @param   w0              1D quadrature weights.
 * @param   nodToMod        Nodal-to-modal matrix; unused for segments.
 * @param   df              Derivative factors of the block, one
 *                          element's values contiguous.
 * @param   jac             Jacobians of the block.
 * @param   coeff           Packed diffusivity tensor (see
 *                          LinAdvDiffReactionOp::SetDiffCoeff).
 * @param   advVel0,advVel1,advVel2     Velocity components of the
 *                                      block; the higher ones are read
 *                                      only when @p ncoord allows.
 * @param   in              Coefficients of the block.
 * @param   out             Output coefficients of the block,
 *                          overwritten.
 * @param   wsp             Global-memory workspace; unused by this
 *                          strategy, which stages everything in shared
 *                          memory.
 * @param   lambda          Reaction coefficient.
 * @param   shmemptr        Dynamic shared memory, laid out as above.
 * @param   threadBlock     Back-end thread-block handle supplying the
 *                          block index, the block stride and the local
 *                          thread indexing in its first dimension, and
 *                          the component index in its second.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    LinAdvDiffReactionKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const unsigned int nhomo,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT advVel0,
        const TData *NEK_RESTRICT advVel1, const TData *NEK_RESTRICT advVel2,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    TData *bwd   = (TData *)shmemptr;
    TData *deriv = bwd + nq0;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nm0 * nelmt * c + nm0 * e;
        TData *outptr       = out + nm0 * nelmt * c + nm0 * e;
        // The advection velocity is shared by the variables but not by the
        // planes, and the plane this slice sits on is c % nhomo.
        const size_t advelplane = nq0 * (nelmt * (c % nhomo) + e);
        const TData *advVel0ptr = advVel0 + advelplane;
        const TData *advVel1ptr = (ncoord > 1) ? advVel1 + advelplane : advVel1;
        const TData *advVel2ptr = (ncoord > 2) ? advVel2 + advelplane : advVel2;

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

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Entry point and driver of the 2D SumFacTOP launch: apply the
 * whole operator to the block's quadrilaterals or (nodal) triangles, one
 * element per thread block.
 *
 * @details
 * Constrained and bounded exactly as the 1D overload.
 *
 * After unpacking @p sizeParam2D, the shared memory is carved up once,
 * into the six reference-space diffusion-metric entries, the physical
 * values, @p ncoord derivative components of nqTot values each, the
 * shape's sum-factorisation intermediate and the two basis tables; the
 * basis tables are copied in before the element loop, being the same for
 * every element. The element's coefficients are staged over the
 * derivative slots at the start of each iteration -- directly for the
 * modal shapes, through the nodal-to-modal MatVecSumFacTOPKernel for
 * NodalTri -- and the same slots take the result of the inner product
 * back for the transposed conversion. The collapsed-coordinate factor
 * tables are read from global memory, not staged.
 *
 * The metric slots are used only when the geometry is regular, in which
 * case the entries are constant over the element and the block's
 * threads share them; when it is deformed they vary point by point and
 * the driver hands the metric kernel a thread-local array instead, which
 * is what the two branches around ApplyMetric2DSumFacTOPKernel select
 * between.
 *
 * The direction-1 basis table is indexed by the combined (p, q) mode for
 * the triangular shapes and by the direction-1 mode alone for
 * quadrilaterals, which is what the two shapes' differing shared-memory
 * sizes reflect.
 *
 * @tparam SHAPE_TYPE          Quad, Tri or NodalTri.
 * @tparam Implementation      MultiRegions::SumFacTOP.
 * @tparam DEFORMED            Geometry is deformed.
 * @tparam TSizeParameter2D    Runtime or compile-time 2D size
 *                             parameter.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam2D     Element mode and point counts.
 * @param   ncoord          Coordinate dimension (2 or 3).
 * @param   nelmt           Elements in the block, padding included.
 * @param   nhomo           Homogeneous planes of the field.
 * @param   isModified      First-direction basis is eModified_A.
 * @param   index0          Mode-index table of the triangular shapes,
 *                          mapping a flat mode index to its direction-0
 *                          mode.
 * @param   basis0,basis1   1D basis tables.
 * @param   D0,D1           1D collocation derivative matrices.
 * @param   w0,w1           1D quadrature weights.
 * @param   f0,f1           Collapsed-coordinate factors
 *                          \f$(1+\eta_0)/2\f$ and \f$2/(1-\eta_1)\f$;
 *                          triangular shapes only.
 * @param   nodToMod        Nodal-to-modal matrix; nodal shapes only.
 * @param   df              Derivative factors of the block, one
 *                          element's values contiguous and factor-major
 *                          within it.
 * @param   jac             Jacobians of the block.
 * @param   coeff           Packed diffusivity tensor.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   in              Coefficients of the block.
 * @param   out             Output coefficients of the block,
 *                          overwritten.
 * @param   wsp             Global-memory workspace; unused.
 * @param   lambda          Reaction coefficient.
 * @param   shmemptr        Dynamic shared memory, laid out as above.
 * @param   threadBlock     Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    LinAdvDiffReactionKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const unsigned int nhomo, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT advVel0, const TData *NEK_RESTRICT advVel1,
        const TData *NEK_RESTRICT advVel2, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData lambda, unsigned char *shmemptr,
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
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

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
        // The advection velocity is shared by the variables but not by the
        // planes, and the plane this slice sits on is c % nhomo.
        const size_t advelplane = nqTot * (nelmt * (c % nhomo) + e);
        const TData *advVel0ptr = advVel0 + advelplane;
        const TData *advVel1ptr = advVel1 + advelplane;
        const TData *advVel2ptr =
            (ncoord == 3) ? advVel2 + advelplane : advVel2;
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

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief Entry point and driver of the 3D SumFacTOP launch: apply the
 * whole operator to the block's hexahedra, (nodal) tetrahedra, (nodal)
 * prisms or pyramids, one element per thread block.
 *
 * @details
 * Constrained and bounded exactly as the 1D overload. The coordinate
 * dimension is accepted but not used: a three-dimensional element has
 * coordinate dimension three.
 *
 * As the 2D overload, with nine metric entries instead of six, three
 * derivative components, two sum-factorisation intermediates and three
 * basis tables; the nodal-to-modal conversion applies to NodalTet and
 * NodalPrism. The per-shape sizes computed at the top give the extents
 * of those slots. The direction-0 table always has nm0 rows; a collapsed
 * direction is indexed by a combined mode instead, so the tetrahedron's
 * direction-1 table has the triangular direction-0/1 count
 * `(2 * nm1 - nm0 + 1) * nm0 / 2`, the prism's direction-2 table the
 * corresponding direction-0/2 count, and the tetrahedron's and the
 * pyramid's direction-2 tables `nmTot` rows plus a further
 * `nm0 * (nm2 - nm1 + 1) * (nm2 - nm1) / 2`. The hexahedron uses the
 * per-direction mode counts throughout. The collapsed shapes hand the
 * two intermediates to their inner-product kernel in swapped order.
 *
 * @tparam SHAPE_TYPE          Hex, Tet, NodalTet, Prism, NodalPrism or
 *                             Pyr.
 * @tparam Implementation      MultiRegions::SumFacTOP.
 * @tparam DEFORMED            Geometry is deformed.
 * @tparam TSizeParameter3D    Runtime or compile-time 3D size
 *                             parameter.
 * @tparam TthreadBlock        Back-end thread-block handle type.
 * @tparam TData               Floating-point type of the field data.
 *
 * @param   sizeParam3D     Element mode and point counts.
 * @param   ncoord          Coordinate dimension; unused.
 * @param   nelmt           Elements in the block, padding included.
 * @param   nhomo           Homogeneous planes; unused, a 3D field has
 *                          one.
 * @param   isModified      First-direction basis is eModified_A.
 * @param   index0,index1,index2,index3     Mode-index tables of the
 *                                          collapsed shapes, mapping a
 *                                          flat mode index back to the
 *                                          individual mode numbers;
 *                                          which ones are needed
 *                                          depends on the shape and on
 *                                          the stage.
 * @param   basis0,basis1,basis2    1D basis tables.
 * @param   D0,D1,D2        1D collocation derivative matrices.
 * @param   w0,w1,w2        1D quadrature weights.
 * @param   f0              \f$(1+\eta_0)/2\f$, per direction-0 point.
 * @param   f1              \f$(1+\eta_1)/2\f$, per direction-1 point.
 * @param   f1m             \f$2/(1-\eta_1)\f$, per direction-1 point.
 * @param   f2              \f$2/(1-\eta_2)\f$, per direction-2 point.
 * @param   nodToMod        Nodal-to-modal matrix; nodal shapes only.
 * @param   df              Derivative factors of the block, one
 *                          element's values contiguous and factor-major
 *                          within it.
 * @param   jac             Jacobians of the block.
 * @param   coeff           Packed diffusivity tensor.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   in              Coefficients of the block.
 * @param   out             Output coefficients of the block,
 *                          overwritten.
 * @param   wsp             Global-memory workspace; unused.
 * @param   lambda          Reaction coefficient.
 * @param   shmemptr        Dynamic shared memory, laid out as above.
 * @param   threadBlock     Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    LinAdvDiffReactionKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        [[maybe_unused]] const unsigned int nhomo, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
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
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

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

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
