///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOpDeviceSumFacKernels.hpp
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
// Description: SumFac device kernels of the linearised
// advection-diffusion-reaction operator, one element per warp lane.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOpDeviceSumFacKernels.hpp
 * @brief SumFac device kernels of the linearised
 * advection-diffusion-reaction operator: one element per thread, with
 * the elements of a warp interleaved lane by lane and the stage
 * intermediates held in a global-memory workspace.
 *
 * @details
 * These are the kernels the Device/SumFac block implementation launches
 * (see LinAdvDiffReactionBlockOpDeviceSumFac.hpp); the SumFacTOP counterparts,
 * where a whole thread block cooperates on one element, are in
 * LinAdvDiffReactionBlockOpDeviceSumFacTOPKernels.hpp. Everything below is
 * guarded by NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY, so it exists
 * only in translation units compiled for a device back-end.
 *
 * One thread evaluates one element. The block storage is reshaped to
 * interleave width warpSize before the launch, so that the elements of
 * a warp are interleaved: element `e` is handled by lane
 * `ilane = e % warpSize` of warp `iwarp = e / warpSize`, and value `n`
 * of that element sits at offset `warpSize * n + ilane` within its
 * warp's group. The LinAdvDiffReactionKernelLauncher overloads stride
 * over the elements of the block, slicing the flat workspace, the
 * geometric data and the advection velocity per warp as they go.
 *
 * A second grid dimension carries the field-component (or
 * homogeneous-mode) index: the launchers read it as `c`, and the
 * component count as `ncomp`, from the thread block's second index and
 * range. The `in` and `out` arrays and every named workspace region
 * hold every component's data together: each region's size is
 * multiplied by `ncomp`, and component `c`'s slice within it starts at
 * `c` times the region's single-component size, with the warp-interleave
 * offset applied inside that slice. The advection velocity is the
 * exception: it is shared by the variables but not by the planes, so the
 * 1D and 2D launchers read it on plane `c % nhomo`; a 3D field has a
 * single plane.
 *
 * Each launcher is the corresponding Helmholtz kernel with this family's
 * advection kernel spliced in, and reuses the Helmholtz workspace and
 * shared-memory sizing (HelmholtzWorkSpaceSize,
 * HelmholtzSharedMemorySize) unchanged. Per element it runs
 * -# the shape's BwdTrans sum-factorisation kernel, giving \f$u\f$ at
 *    the quadrature points in the workspace slice `bwd`, preceded by the
 *    nodal-to-modal conversion (MatVecKernel) for the nodal shapes;
 * -# PhysDeriv{1,2,3}DSumFacKernel, which applies the
 *    collapsed-coordinate factors and the derivative factors, so the
 *    `deriv` slices hold Cartesian derivatives, one component per
 *    coordinate direction;
 * -# AddAdvection{1,2,3}DSumFacKernel (below), overwriting `bwd` with
 *    \f$\mathbf{V}\cdot\nabla u + \lambda u\f$;
 * -# ApplyMetric{1,2,3}DSumFacKernel of the Helmholtz family, which
 *    forms the diffusion flux in the reference directions and applies
 *    the quadrature weights and the Jacobian to it and to `bwd` --
 *    called with a unit coefficient, the reaction term having been
 *    folded in at the previous step;
 * -# SumDerivTensor{1,2,3}DKernel of the PhysDeriv family, adding the
 *    contracted flux onto `bwd`;
 * -# the shape's IProductWRTBase sum-factorisation kernel, followed by
 *    the transposed nodal conversion for the nodal shapes.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

// get hold of memory sizing and routines related to collocation Helmholtz op
#include <MultiRegions/ElmtOps/Helmholtz/HelmholtzBlockOpDeviceSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Segment advection kernel: add the advection term to the scaled
 * physical values of one thread's element.
 *
 * The derivatives arrive already in Cartesian form from the PhysDeriv
 * stage, so the kernel only dots them with the velocity and folds in the
 * reaction term, leaving
 * \f$\mathbf{V}\cdot\nabla u + \mathrm{scale}\cdot u\f$ in @p out. All
 * pointers address the whole warp's interleaved data, from which
 * @p ilane selects this thread's element. The higher velocity and
 * derivative components are read only when @p ncoord allows.
 *
 * @tparam TData   Floating-point type of the field data.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   ncoord  Coordinate dimension: how many components are used.
 * @param   nq0     Quadrature points of the element.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   deriv0,deriv1,deriv2        Cartesian derivatives of
 *                                      \f$u\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out: the
 *                      advection-plus-reaction values.
 * @param   scale   Reaction coefficient \f$\lambda\f$.
 */
template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection1DSumFacKernel(
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

/**
 * @brief Two-dimensional advection kernel: add the advection term to
 * the scaled physical values of one thread's element.
 *
 * As AddAdvection1DSumFacKernel, over the element's nq0 * nq1 points.
 * The third velocity and derivative component is read only for a
 * surface element embedded in three dimensions (@p ncoord == 3).
 *
 * @tparam TData   Floating-point type of the field data.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   ncoord  Coordinate dimension (2 or 3).
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   deriv0,deriv1,deriv2        Cartesian derivatives of
 *                                      \f$u\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out: the
 *                      advection-plus-reaction values.
 * @param   scale   Reaction coefficient \f$\lambda\f$.
 */
template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection2DSumFacKernel(
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

/**
 * @brief Three-dimensional advection kernel: add the advection term to
 * the scaled physical values of one thread's element.
 *
 * As AddAdvection2DSumFacKernel, over the element's nq0 * nq1 * nq2
 * points. There is no coordinate-dimension argument: a
 * three-dimensional element has coordinate dimension three, so all
 * three components always contribute.
 *
 * @tparam TData   Floating-point type of the field data.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nq0,nq1,nq2     Quadrature points per direction.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   deriv0,deriv1,deriv2        Cartesian derivatives of
 *                                      \f$u\f$; unchanged.
 * @param[in,out] out   In: \f$u\f$ at the quadrature points; out: the
 *                      advection-plus-reaction values.
 * @param   scale   Reaction coefficient \f$\lambda\f$.
 */
template <typename TData>
NEK_DEVICE_INLINE static void AddAdvection3DSumFacKernel(
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

/**
 * @brief Entry point and driver of the 1D SumFac launch: apply the whole
 * operator to the block's segments, one element per thread.
 *
 * @details
 * The enable_if constrains this overload to the SumFac tag, so that the
 * block implementation's single call site resolves to this launcher or
 * to its SumFacTOP namesake in
 * LinAdvDiffReactionBlockOpDeviceSumFacTOPKernels.hpp according to the
 * Implementation tag it was instantiated with; a static_assert then
 * rejects anything but the 1D size parameters, runtime or compile-time.
 * The launch bounds come from GetMaxThreadPerBlock for the tag and
 * size-parameter pair.
 *
 * After unpacking @p sizeParam1D, threads grid-stride over the block's
 * elements, and for each element slice the flat workspace, the
 * geometric data and the velocity down to the owning warp (and, for the
 * velocity, plane) before
 * running the stage sequence described in the file notes. The workspace
 * holds, per element and warp-interleaved, the physical values followed
 * by @p ncoord derivative components lying `nq0 * nelmt` apart -- ncomp
 * times what HelmholtzWorkSpaceSize reserves for a segment, laid out as
 * the file notes describe. The velocity pointers of the higher
 * components are offset to the warp only when @p ncoord has them.
 *
 * @tparam SHAPE_TYPE          Element shape (segments only).
 * @tparam Implementation      MultiRegions::SumFac.
 * @tparam DEFORMED            Geometry is deformed, so the Jacobian and
 *                             the derivative factors vary point by
 *                             point.
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
 * @param   df              Derivative factors of the block,
 *                          warp-interleaved.
 * @param   jac             Jacobians of the block, warp-interleaved.
 * @param   coeff           Packed diffusivity tensor (see
 *                          LinAdvDiffReactionOp::SetDiffCoeff).
 * @param   advVel0,advVel1,advVel2     Velocity components of the
 *                                      block; the higher ones are read
 *                                      only when @p ncoord allows.
 * @param   in              Coefficients of the block.
 * @param   out             Output coefficients of the block,
 *                          overwritten.
 * @param   wsp             Global-memory workspace, laid out as above.
 * @param   lambda          Reaction coefficient.
 * @param   shmemptr        Dynamic shared memory; not used by the
 *                          segment kernel.
 * @param   threadBlock     Back-end thread-block handle supplying the
 *                          global thread index and grid stride in its
 *                          first dimension, and the component index and
 *                          count in its second.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
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
        // The advection velocity is shared by the variables but not by the
        // planes, and the plane this slice sits on is c % nhomo.
        const size_t advelplane =
            nq0 * (nelmt * (c % nhomo) + warpsize * iwarp);
        const TData *advVel0ptr = advVel0 + advelplane;
        const TData *advVel1ptr = (ncoord > 1) ? advVel1 + advelplane : advVel1;
        const TData *advVel2ptr = (ncoord > 2) ? advVel2 + advelplane : advVel2;

        BwdTransSegSumFacKernel<false>(ilane, nm0, nq0, basis0, inptr, bwd);
        PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, nelmt * nq0, D0,
                                          dfptr, bwd, deriv);
        AddAdvection1DSumFacKernel(ilane, ncoord, nq0, advVel0ptr, advVel1ptr,
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

/**
 * @brief Entry point and driver of the 2D SumFac launch: apply the whole
 * operator to the block's quadrilaterals or (nodal) triangles, one
 * element per thread.
 *
 * @details
 * Constrained and bounded exactly as the 1D overload. The mode-index
 * table @p index0 is accepted for signature compatibility with the
 * SumFacTOP launcher and is not used: only the SumFacTOP kernels need
 * it.
 *
 * As the 1D overload, with two additions. The workspace carries, per
 * element and warp-interleaved, the physical values, @p ncoord
 * derivative components and then the shape's sum-factorisation
 * intermediate, the derivative components lying `nqTot * nelmt` apart
 * within the component's slice; NodalTri inserts a further
 * `nmTot * nelmt` region ahead of that intermediate for the
 * modal-converted coefficients, which the nodal-to-modal MatVecKernel
 * writes before the backward transform and the transposed conversion
 * reads back after the inner product. And for the triangular shapes the
 * two collapsed-coordinate factor tables are staged in shared memory up
 * front, since every element of the block uses the same ones.
 *
 * @tparam SHAPE_TYPE          Quad, Tri or NodalTri.
 * @tparam Implementation      MultiRegions::SumFac.
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
 * @param   index0          Mode-index table; unused here.
 * @param   basis0,basis1   1D basis tables.
 * @param   D0,D1           1D collocation derivative matrices.
 * @param   w0,w1           1D quadrature weights.
 * @param   f0,f1           Collapsed-coordinate factors
 *                          \f$(1+\eta_0)/2\f$ and \f$2/(1-\eta_1)\f$;
 *                          triangular shapes only.
 * @param   nodToMod        Nodal-to-modal matrix; nodal shapes only.
 * @param   df              Derivative factors of the block,
 *                          warp-interleaved.
 * @param   jac             Jacobians of the block, warp-interleaved.
 * @param   coeff           Packed diffusivity tensor.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   in              Coefficients of the block.
 * @param   out             Output coefficients of the block,
 *                          overwritten.
 * @param   wsp             Global-memory workspace, laid out as above.
 * @param   lambda          Reaction coefficient.
 * @param   shmemptr        Dynamic shared memory, used for the factor
 *                          tables of the triangular shapes.
 * @param   threadBlock     Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    LinAdvDiffReactionKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const unsigned int nhomo, const bool isModified,
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
        // The advection velocity is shared by the variables but not by the
        // planes, and the plane this slice sits on is c % nhomo.
        const size_t advelplane =
            nqTot * (nelmt * (c % nhomo) + warpsize * iwarp);
        const TData *advVel0ptr = advVel0 + advelplane;
        const TData *advVel1ptr = advVel1 + advelplane;
        const TData *advVel2ptr =
            (ncoord == 3) ? advVel2 + advelplane : advVel2;

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
        AddAdvection2DSumFacKernel(ilane, ncoord, nq0, nq1, advVel0ptr,
                                   advVel1ptr, advVel2ptr, deriv, deriv1,
                                   deriv2, bwd, lambda);
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

/**
 * @brief Entry point and driver of the 3D SumFac launch: apply the whole
 * operator to the block's hexahedra, (nodal) tetrahedra, (nodal) prisms
 * or pyramids, one element per thread.
 *
 * @details
 * Constrained and bounded exactly as the 1D overload. The four
 * mode-index tables are accepted for signature compatibility with the
 * SumFacTOP launcher and are not used, and the coordinate dimension is
 * accepted but not used either: a three-dimensional element has
 * coordinate dimension three.
 *
 * As the 2D overload. The workspace carries, per element and
 * warp-interleaved, the physical values and the three derivative
 * components, then the shape's two sum-factorisation intermediates, with
 * the nodal shapes again inserting an `nmTot * nelmt` region for the
 * modal-converted coefficients ahead of them. The collapsed shapes stage
 * the factor tables they need in shared memory up front: the tetrahedron
 * all four, the prism @p f0 and @p f2, the pyramid @p f0, @p f1 and
 * @p f2; the hexahedron none.
 *
 * @note For the tetrahedral shapes the second intermediate is sliced
 * with the per-element stride `max(nq2, nm0)` while
 * HelmholtzWorkSpaceSize reserves `nq2` per element for it; the two
 * agree only while `nm0 <= nq2`. The prism and pyramid reserve
 * `max(nq2, nm0)` and match their slicing.
 *
 * @tparam SHAPE_TYPE          Hex, Tet, NodalTet, Prism, NodalPrism or
 *                             Pyr.
 * @tparam Implementation      MultiRegions::SumFac.
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
 * @param   index0,index1,index2,index3     Mode-index tables; unused
 *                                          here.
 * @param   basis0,basis1,basis2    1D basis tables.
 * @param   D0,D1,D2        1D collocation derivative matrices.
 * @param   w0,w1,w2        1D quadrature weights.
 * @param   f0              \f$(1+\eta_0)/2\f$, per direction-0 point.
 * @param   f1              \f$(1+\eta_1)/2\f$, per direction-1 point.
 * @param   f1m             \f$2/(1-\eta_1)\f$, per direction-1 point.
 * @param   f2              \f$2/(1-\eta_2)\f$, per direction-2 point.
 * @param   nodToMod        Nodal-to-modal matrix; nodal shapes only.
 * @param   df              Derivative factors of the block,
 *                          warp-interleaved.
 * @param   jac             Jacobians of the block, warp-interleaved.
 * @param   coeff           Packed diffusivity tensor.
 * @param   advVel0,advVel1,advVel2     Velocity components.
 * @param   in              Coefficients of the block.
 * @param   out             Output coefficients of the block,
 *                          overwritten.
 * @param   wsp             Global-memory workspace, laid out as above.
 * @param   lambda          Reaction coefficient.
 * @param   shmemptr        Dynamic shared memory, used for the factor
 *                          tables of the collapsed shapes.
 * @param   threadBlock     Back-end thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    LinAdvDiffReactionKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
        [[maybe_unused]] const unsigned int nhomo, const bool isModified,
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
        AddAdvection3DSumFacKernel(ilane, nq0, nq1, nq2, advVel0ptr, advVel1ptr,
                                   advVel2ptr, deriv0, deriv1, deriv2, bwd,
                                   lambda);
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

} // namespace Nektar::MultiRegions::detail
