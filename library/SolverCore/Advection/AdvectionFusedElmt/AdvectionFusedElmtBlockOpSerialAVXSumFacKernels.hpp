///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtBlockOpSerialAVXSumFacKernels.hpp
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
// Description: Serial/AVX kernels of the fused element-local advection
// operator: one element group through the pass.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionFusedElmtBlockOpSerialAVXSumFacKernels.hpp
 * @brief Serial/AVX kernels of the fused element-local advection operator:
 * one SIMD vector of elements per call.
 *
 * @details
 * AdvectionFusedElmtGroupKernel() composes the sum-factorised kernels of the
 * separate element operators around the flux policy:
 * - the state on the traces (PhysTraceExtractDirLinesKernel() for traces
 *   whose tangential rules are the volume rules, followed by
 *   PhysInterpEdgeKernel() or PhysInterpFaceKernel() otherwise);
 * - the policy's advective flux at every point (AdvectionFusedElmtFlux());
 * - IProductWRTDerivBase's metric alignment with the transposed derivative
 *   contractions of all the variables at once
 *   (SumDerivTensorLinesKernelND(), negated).
 *
 * The trace extraction and the derivative inner product are also the
 * advection-diffusion pass's
 * (AdvDiffFusedElmtBlockOpSerialAVXSumFacKernels.hpp), which calls them here
 * between its viscous stages.
 *
 * The basis data the kernels read are the ones the separate operators fetch
 * from the data warehouse, gathered per block in AdvectionFusedElmtBasisData
 * by the block operator (AdvectionFusedElmtBlockOpSerialAVXSumFac.hpp): the
 * fused pass applies the same operators and differs from the separate chain
 * only in rounding.
 */

#pragma once

#include <algorithm>

#include <LibUtilities/BasicUtils/ShapeType.hpp>
// Declares the matrix-vector kernel the IProductWRTBase kernels below call.
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <MultiRegions/ElmtOps/ElmtHelper.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractBlockOpSerialAVXSumFacKernels.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtKernels.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The basis data of one block the fused kernels read, as the separate
 * operators fetch them from the data warehouse.
 *
 * @tparam TData      Scalar type of the data.
 * @tparam simd_type  SIMD vector type of the lane-broadcast data.
 */
template <typename TData, typename simd_type> struct AdvectionFusedElmtBasisData
{
    /// Differentiation matrices and quadrature weights per direction.
    const TData *derivMatrix[3] = {};
    const TData *weights[3]     = {};
    /// The collapsed-coordinate factors, as PhysDeriv and
    /// IProductWRTDerivBase fetch them (unused on a segment, quadrilateral
    /// or hexahedron but part of the kernel interface).
    const simd_type *factors[4] = {};
    /// The extraction's tables, as PhysTraceExtract fetches them: the
    /// end-point interpolant of each direction, then the in-trace
    /// interpolants of the trace pairs of each direction.
    const TData *traceBasis[9] = {};
};

/**
 * @brief The workspaces of one block operator the advection kernels write,
 * each sized for one element group.
 *
 * @tparam simd_type  SIMD vector type.
 */
template <typename simd_type> struct AdvectionFusedElmtWorkspace
{
    /// The flux vector and the aligned flux vector of one group, component
    /// d of variable n at `n * varStride + d * nqTot`.
    simd_type *flux, *aligned;
    /// Distance between consecutive variables of the vector workspaces
    /// (these two, and the gradient and viscous flux of the
    /// advection-diffusion pass), at least `NDIM * nqTot`: padded so that
    /// the variables do not start at the same offset within a page.
    unsigned int varStride;
    /// IProductWRTDerivBase's derivative-factor scratch.
    simd_type *dfWsp;
    /// The scratch of a trace interpolated in both tangential directions.
    simd_type *interpWsp;
    /// The traces of one group at the volume tangential points, one field
    /// after another, `nqTotTraceVol` each: what a trace is interpolated
    /// from when its tangential rule is not the volume rule.
    simd_type *stage;
};

// -----------------------------------------------------------------------------
//  THE SEPARATE OPERATORS' KERNELS ON ONE GROUP'S FIELDS
// -----------------------------------------------------------------------------

/// IProductWRTDerivBase's metric alignment of one field's flux vector
/// `in[dir * nqTot]` into `out[dir * nqTot]`.
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          typename TPhysSize, typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvectionFusedElmtAlignDerivField(
    const TPhysSize sizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const simd_type *df, const simd_type *jac, const simd_type *in,
    simd_type *out, simd_type *dfWsp)
{
    const unsigned int nqTot = sizeParam.nqTot();
    if constexpr (NDIM == 1)
    {
        MultiRegions::detail::StdAlignDerivBase1D<DEFORMED>(
            sizeParam.nq0(), 1, df, dfWsp, nqTot, in, out, jac,
            basisData.weights[0]);
    }
    else if constexpr (NDIM == 2)
    {
        simd_type *planes[2] = {out, out + nqTot};
        MultiRegions::detail::StdAlignDerivBase2D<SHAPE, DEFORMED>(
            sizeParam.nq0(), sizeParam.nq1(), 2, df, dfWsp, nqTot, in, planes,
            basisData.factors[0], basisData.factors[1], jac,
            basisData.weights[0], basisData.weights[1]);
    }
    else if constexpr (NDIM == 3)
    {
        simd_type *planes[3] = {out, out + nqTot, out + 2 * nqTot};
        MultiRegions::detail::StdAlignDerivBase3D<SHAPE, DEFORMED>(
            sizeParam.nq0(), sizeParam.nq1(), sizeParam.nq2(), df, dfWsp, nqTot,
            basisData.factors[0], basisData.factors[1], basisData.factors[2],
            basisData.factors[3], jac, basisData.weights[0],
            basisData.weights[1], basisData.weights[2], in, planes);
    }
}

/**
 * @brief The register-block length of the line-blocked kernels for the size
 * parameter @p TPhysSize: the largest point count of its directions when the
 * sizes are compile-time, so the line buffers fit the element exactly and the
 * blocked path serves every compile-time size; @p MAXQ, the buffer the
 * runtime sizes are given, otherwise, beyond which those kernels take the
 * per-field path.
 */
template <typename TPhysSize, unsigned int MAXQ>
static constexpr unsigned int AdvectionFusedElmtLineLength()
{
    if constexpr (MultiRegions::detail::IsTemplatedPhysSizeParameter1D<
                      TPhysSize>::value)
    {
        return TPhysSize::nq0();
    }
    else if constexpr (MultiRegions::detail::IsTemplatedPhysSizeParameter2D<
                           TPhysSize>::value)
    {
        return std::max(TPhysSize::nq0(), TPhysSize::nq1());
    }
    else if constexpr (MultiRegions::detail::IsTemplatedPhysSizeParameter3D<
                           TPhysSize>::value)
    {
        return std::max({TPhysSize::nq0(), TPhysSize::nq1(), TPhysSize::nq2()});
    }
    else
    {
        return MAXQ;
    }
}

/// PhysTraceExtractDirLinesKernel() on the traces of direction @p DIR of
/// one element group: the direction's end-point table and flag come from the
/// block's basis data and packing, and the traces' offsets from
/// @p traceOffset. @p in (state form) or @p tensor with the trace normals
/// @p traceNormals (contract form) to @p out, per variable.
template <LibUtilities::ShapeType SHAPE, unsigned int DIR, bool CONTRACT,
          unsigned int NDIM, unsigned int NVAR, typename TPhysSize,
          typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvectionFusedElmtExtractDirLines(
    const TPhysSize sizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const AdvectionFusedElmtFacePacking &packing,
    const unsigned int *traceOffset, const simd_type *const in[NVAR],
    const simd_type *tensor, const unsigned int varStride,
    const simd_type (*traceNormals)[NDIM], simd_type *const out[NVAR])
{
    constexpr unsigned int NF =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE][DIR];
    unsigned int offsets[NF];
    for (unsigned int f = 0; f < NF; ++f)
    {
        offsets[f] = traceOffset[packing.faceSlot[DIR][f]];
    }
    unsigned int nq1 = 1, nq2 = 1;
    if constexpr (NDIM >= 2)
    {
        nq1 = sizeParam.nq1();
    }
    if constexpr (NDIM == 3)
    {
        nq2 = sizeParam.nq2();
    }
    MultiRegions::detail::PhysTraceExtractDirLinesKernel<SHAPE, DIR, CONTRACT,
                                                         NDIM, NVAR>(
        sizeParam.nq0(), nq1, nq2, basisData.traceBasis[DIR],
        packing.endPts[DIR], offsets, in, tensor, varStride, traceNormals, out);
}

/**
 * @brief The traces of direction @p DIR of one element group, all variables
 * together, onto their trace rules.
 *
 * In state form (@p CONTRACT false) it takes @p in; in contract form it
 * takes the direction components of @p tensor and contracts them with the
 * normal of each trace (@p traceNormals) in the same pass, the form the
 * advection-diffusion pass forms its normal viscous flux on the traces with.
 *
 * A trace whose tangential rules are the volume rules, and every trace of a
 * segment, is written straight to @p out by the line-blocked kernel.
 * Otherwise the line-blocked kernel writes the traces at the volume
 * tangential points to @p stage, and PhysInterpEdgeKernel() (two
 * dimensions) or PhysInterpFaceKernel() (three) takes them onto the trace
 * rule in the directions that differ, variable by variable: a small dense
 * interpolation on trace data already in cache.
 *
 * @param   stage      Traces at the volume tangential points, per variable,
 *                     `nqTotTraceVol` each.
 * @param   interpWsp  PhysInterpFaceKernel()'s scratch, for a trace that
 *                     interpolates in both tangential directions.
 */
template <LibUtilities::ShapeType SHAPE, unsigned int DIR, bool CONTRACT,
          unsigned int NDIM, unsigned int NVAR, typename TPhysSize,
          typename TTraceSize, typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvectionFusedElmtExtractDir(
    const TPhysSize sizeParam, [[maybe_unused]] const TTraceSize traceSizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const AdvectionFusedElmtFacePacking &packing,
    const simd_type *const in[NVAR], const simd_type *tensor,
    const unsigned int varStride, const simd_type (*traceNormals)[NDIM],
    simd_type *const out[NVAR], [[maybe_unused]] simd_type *stage,
    [[maybe_unused]] simd_type *interpWsp)
{
    bool tanColloc0 = true, tanColloc1 = true;
    if constexpr (NDIM == 2)
    {
        tanColloc0 = packing.colloc[DIR];
    }
    else if constexpr (NDIM == 3)
    {
        tanColloc0 = packing.colloc[2 * DIR];
        tanColloc1 = packing.colloc[2 * DIR + 1];
    }
    if (tanColloc0 && tanColloc1)
    {
        AdvectionFusedElmtExtractDirLines<SHAPE, DIR, CONTRACT, NDIM, NVAR>(
            sizeParam, basisData, packing, packing.faceOff, in, tensor,
            varStride, traceNormals, out);
        return;
    }
    if constexpr (NDIM >= 2)
    {
        simd_type *staged[NVAR];
        for (unsigned int v = 0; v < NVAR; ++v)
        {
            staged[v] = stage + v * packing.nqTotTraceVol;
        }
        AdvectionFusedElmtExtractDirLines<SHAPE, DIR, CONTRACT, NDIM, NVAR>(
            sizeParam, basisData, packing, packing.faceOffVol, in, tensor,
            varStride, traceNormals, staged);

        // The direction's traces lie consecutively in both layouts, the
        // lower tangential index fastest in each.
        constexpr unsigned int NF =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE][DIR];
        const unsigned int first = packing.faceSlot[DIR][0];
        for (unsigned int v = 0; v < NVAR; ++v)
        {
            if constexpr (NDIM == 2)
            {
                MultiRegions::detail::PhysInterpEdgeKernel(
                    NF, DIR == 0 ? sizeParam.nq1() : sizeParam.nq0(),
                    DIR == 0 ? traceSizeParam.nq00() : traceSizeParam.nq10(),
                    basisData.traceBasis[NDIM + DIR],
                    staged[v] + packing.faceOffVol[first],
                    out[v] + packing.faceOff[first]);
            }
            else if constexpr (NDIM == 3)
            {
                MultiRegions::detail::PhysInterpFaceKernel(
                    NF, DIR == 0 ? sizeParam.nq1() : sizeParam.nq0(),
                    DIR == 2 ? sizeParam.nq1() : sizeParam.nq2(),
                    DIR == 0   ? traceSizeParam.nq00()
                    : DIR == 1 ? traceSizeParam.nq10()
                               : traceSizeParam.nq20(),
                    DIR == 0   ? traceSizeParam.nq01()
                    : DIR == 1 ? traceSizeParam.nq11()
                               : traceSizeParam.nq21(),
                    basisData.traceBasis[NDIM + 2 * DIR],
                    basisData.traceBasis[NDIM + 2 * DIR + 1], interpWsp,
                    staged[v] + packing.faceOffVol[first],
                    out[v] + packing.faceOff[first], tanColloc0, tanColloc1);
            }
        }
    }
}

/// AdvectionFusedElmtExtractDir() for every direction, state form.
template <LibUtilities::ShapeType SHAPE, unsigned int NDIM, unsigned int NVAR,
          typename TPhysSize, typename TTraceSize, typename TData,
          typename simd_type>
NEK_FORCE_INLINE static void AdvectionFusedElmtExtractAllDir(
    const TPhysSize sizeParam, const TTraceSize traceSizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const AdvectionFusedElmtFacePacking &packing,
    const simd_type *const in[NVAR], simd_type *const out[NVAR],
    simd_type *stage, simd_type *interpWsp)
{
    // Typed nulls, so that the SIMD type still deduces.
    const simd_type *noTensor         = nullptr;
    const simd_type(*noNormals)[NDIM] = nullptr;
    AdvectionFusedElmtExtractDir<SHAPE, 0, false, NDIM, NVAR>(
        sizeParam, traceSizeParam, basisData, packing, in, noTensor, 0,
        noNormals, out, stage, interpWsp);
    if constexpr (NDIM >= 2)
    {
        AdvectionFusedElmtExtractDir<SHAPE, 1, false, NDIM, NVAR>(
            sizeParam, traceSizeParam, basisData, packing, in, noTensor, 0,
            noNormals, out, stage, interpWsp);
    }
    if constexpr (NDIM == 3)
    {
        AdvectionFusedElmtExtractDir<SHAPE, 2, false, NDIM, NVAR>(
            sizeParam, traceSizeParam, basisData, packing, in, noTensor, 0,
            noNormals, out, stage, interpWsp);
    }
}

// -----------------------------------------------------------------------------
//  STAGES
// -----------------------------------------------------------------------------

/**
 * @brief The negated derivative inner product of the flux vector of one
 * element group: the metric alignment variable by variable into
 * `wsp.aligned`, then the transposed derivative contractions of all the
 * variables together, line-blocked, straight into @p out.
 *
 * @tparam LINE  The register-block length (AdvectionFusedElmtLineLength()).
 *
 * @param   df,jac  The group's derivative factors and Jacobian.
 * @param   wsp     The block operator's workspaces; `wsp.flux` holds the
 *                  flux vector.
 * @param   out     Out: the negated derivative inner product per variable.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          unsigned int NVAR, unsigned int LINE, typename TPhysSize,
          typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvectionFusedElmtFluxDerivative(
    const TPhysSize sizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const simd_type *df, const simd_type *jac,
    const AdvectionFusedElmtWorkspace<simd_type> &wsp,
    simd_type *const out[NVAR])
{
    for (unsigned int n = 0; n < NVAR; ++n)
    {
        AdvectionFusedElmtAlignDerivField<SHAPE, DEFORMED, NDIM>(
            sizeParam, basisData, df, jac, wsp.flux + n * wsp.varStride,
            wsp.aligned + n * wsp.varStride, wsp.dfWsp);
    }
    SumDerivTensorLinesKernelND<true, NDIM, NVAR, LINE>(
        sizeParam, wsp.aligned, wsp.varStride, basisData.derivMatrix, out);
}

// -----------------------------------------------------------------------------
//  GROUP KERNEL
// -----------------------------------------------------------------------------

/**
 * @brief One element group through the fused advection pass.
 *
 * -# State trace. Every trace is one interpolation along its normal, done
 *    for all the variables and every trace of a direction per line; a trace
 *    whose tangential rule is not the volume rule is then interpolated onto
 *    it from a staging buffer (AdvectionFusedElmtExtractDir()).
 * -# The flux vector `-advFlux`, pointwise, from the policy
 *    (AdvectionFusedElmtFlux()).
 * -# Its derivative inner product, negated
 *    (AdvectionFusedElmtFluxDerivative()).
 *
 * @tparam SHAPE      Shape of the block's elements.
 * @tparam DEFORMED   Geometric factors vary point by point.
 * @tparam MAXQ       Points per direction the line-blocked kernels hold in
 *                    registers.
 * @tparam Physics    The flux policy (AdvectionFusedElmtKernels.hpp).
 *
 * @param   sizeParam       Physical-space size parameter.
 * @param   traceSizeParam  Trace size parameter.
 * @param   basisData       The block's basis data.
 * @param   packing         The block's trace packing.
 * @param   physics         The flux policy.
 * @param   df,jac          The group's derivative factors and Jacobian.
 * @param   state           The state per variable, `nqTot` each.
 * @param   aux             The auxiliary fields per component, `nqTot`
 *                          each, or null.
 * @param   trace           Out: the state on the traces per variable.
 * @param   out             Out: the negated derivative inner product per
 *                          variable.
 * @param   wsp             The block operator's workspaces.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int MAXQ,
          typename Physics, typename TPhysSize, typename TTraceSize,
          typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvectionFusedElmtGroupKernel(
    const TPhysSize sizeParam, const TTraceSize traceSizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const AdvectionFusedElmtFacePacking &packing, const Physics &physics,
    const simd_type *df, const simd_type *jac, const simd_type *const *state,
    [[maybe_unused]] const simd_type *const *aux, simd_type *const *trace,
    simd_type *const *out, const AdvectionFusedElmtWorkspace<simd_type> &wsp)
{
    constexpr unsigned int NDIM = Physics::NDIM;
    constexpr unsigned int NVAR = Physics::NVAR;
    constexpr unsigned int NAUX = Physics::NAUX;
    constexpr unsigned int LINE =
        AdvectionFusedElmtLineLength<TPhysSize, MAXQ>();
    const unsigned int nqTot = sizeParam.nqTot();

    // 1. State trace.
    AdvectionFusedElmtExtractAllDir<SHAPE, NDIM, NVAR>(
        sizeParam, traceSizeParam, basisData, packing, state, trace, wsp.stage,
        wsp.interpWsp);

    // 2. The flux vector, pointwise.
    for (unsigned int i = 0; i < nqTot; ++i)
    {
        simd_type statePt[NVAR], fluxPt[NDIM][NVAR];
        simd_type auxPt[NAUX > 0 ? NAUX : 1];
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            statePt[n] = state[n][i];
        }
        if constexpr (NAUX > 0)
        {
            for (unsigned int a = 0; a < NAUX; ++a)
            {
                auxPt[a] = aux[a][i];
            }
        }
        AdvectionFusedElmtFlux(physics, statePt, NAUX > 0 ? auxPt : nullptr,
                               fluxPt);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                wsp.flux[n * wsp.varStride + d * nqTot + i] = fluxPt[d][n];
            }
        }
    }

    // 3. Its derivative inner product, negated.
    AdvectionFusedElmtFluxDerivative<SHAPE, DEFORMED, NDIM, NVAR, LINE>(
        sizeParam, basisData, df, jac, wsp, out);
}

} // namespace Nektar::SolverCore::detail
