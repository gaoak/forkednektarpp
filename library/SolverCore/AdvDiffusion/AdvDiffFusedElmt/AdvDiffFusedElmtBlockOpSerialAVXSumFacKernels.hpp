///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffFusedElmtBlockOpSerialAVXSumFacKernels.hpp
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
// Description: Serial/AVX kernels of the fused element-local advection-
// diffusion operator: one element group through the pass.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvDiffFusedElmtBlockOpSerialAVXSumFacKernels.hpp
 * @brief Serial/AVX kernels of the fused element-local advection-diffusion
 * operator: one SIMD vector of elements per call.
 *
 * @details
 * AdvDiffFusedElmtGroupKernel() composes the sum-factorised kernels of the
 * separate element operators around the flux policy:
 * - the collapsed-coordinate derivatives of all the variables at once
 *   (PhysDerivTensorLinesKernelND()) with PhysDeriv's metric;
 * - the state on the traces, as the advection pass extracts it
 *   (AdvectionFusedElmtExtractAllDir());
 * - the policy's fluxes at every point (AdvDiffFusedElmtFlux());
 * - the derivative inner product of the flux vector, as the advection pass
 *   forms it (AdvectionFusedElmtFluxDerivative());
 * - each element's outward normal viscous flux on its traces, by contracting
 *   the viscous flux with the trace normal and extracting the result, and
 *   the gradient on the traces where it is asked for.
 *
 * The stages shared with the advection pass are the kernels of
 * SolverCore/Advection/AdvectionFusedElmt; this file holds the viscous
 * ones.
 */

#pragma once

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpSerialAVXSumFacKernels.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtKernels.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOpSerialAVXSumFacKernels.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The workspaces of one block operator the advection-diffusion
 * kernels write, each sized for one element group: the advection pass's,
 * with the viscous stages'.
 *
 * @tparam simd_type  SIMD vector type.
 */
template <typename simd_type>
struct AdvDiffFusedElmtWorkspace : AdvectionFusedElmtWorkspace<simd_type>
{
    /// The gradient and the viscous flux of one group, laid out as the flux
    /// vector: component d of variable n at `n * varStride + d * nqTot`.
    simd_type *grad, *viscFlux;
    /// The directional components of the viscous flux on the traces,
    /// `NDIM * nqTotTrace`, for deformed elements.
    simd_type *traceComps;
    /// The gradient on the traces of one group, `NVAR * NDIM * nqTotTrace`,
    /// component `n * NDIM + d`.
    simd_type *gradTrace;
};

/// PhysDeriv's metric kernel on the collapsed-coordinate derivatives of one
/// field, in place: `deriv[dir * nqTot]` holds direction @c dir.
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int NDIM,
          typename TPhysSize, typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvDiffFusedElmtPhysDerivField(
    const TPhysSize sizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const simd_type *df, simd_type *deriv)
{
    const unsigned int nqTot = sizeParam.nqTot();
    if constexpr (NDIM == 1)
    {
        simd_type *planes[3] = {deriv, nullptr, nullptr};
        MultiRegions::detail::PhysDeriv1DKernel<SHAPE, DEFORMED>(
            sizeParam.nq0(), 1, df, planes);
    }
    else if constexpr (NDIM == 2)
    {
        simd_type *planes[3] = {deriv, deriv + nqTot, nullptr};
        MultiRegions::detail::PhysDeriv2DKernel<SHAPE, DEFORMED>(
            sizeParam.nq0(), sizeParam.nq1(), 2, basisData.factors[0],
            basisData.factors[1], df, planes);
    }
    else if constexpr (NDIM == 3)
    {
        MultiRegions::detail::PhysDeriv3DKernel<SHAPE, DEFORMED>(
            sizeParam.nq0(), sizeParam.nq1(), sizeParam.nq2(),
            basisData.factors[0], basisData.factors[1], basisData.factors[2],
            basisData.factors[3], df, deriv, deriv + nqTot, deriv + 2 * nqTot);
    }
}

/// AdvectionFusedElmtExtractDir() in contract form for direction @p DIR:
/// the viscous flux contracted with the normal of each of the direction's
/// traces, one normal per trace (affine elements).
template <LibUtilities::ShapeType SHAPE, unsigned int DIR, unsigned int NDIM,
          unsigned int NVAR, typename TPhysSize, typename TTraceSize,
          typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvDiffFusedElmtContractDir(
    const TPhysSize sizeParam, const TTraceSize traceSizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const AdvectionFusedElmtFacePacking &packing,
    const simd_type *const normals[NDIM], const simd_type *viscFlux,
    const unsigned int varStride, simd_type *const normalVisc[NVAR],
    simd_type *stage, simd_type *interpWsp)
{
    const simd_type *const *noState = nullptr;
    simd_type traceNormals[2][NDIM];
    for (unsigned int k = 0; k < packing.nFaceDir[DIR]; ++k)
    {
        const unsigned int slot = packing.faceSlot[DIR][k];
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            traceNormals[k][d] = normals[d][packing.faceOff[slot]];
        }
    }
    AdvectionFusedElmtExtractDir<SHAPE, DIR, true, NDIM, NVAR>(
        sizeParam, traceSizeParam, basisData, packing, noState, viscFlux,
        varStride, traceNormals, normalVisc, stage, interpWsp);
}

/**
 * @brief One element group through the fused advection-diffusion pass.
 *
 * -# The gradient. The collapsed-coordinate derivatives of the variables
 *    are formed together, a line of inputs for all variables held in
 *    registers against all the line's outputs; the metric follows variable
 *    by variable.
 * -# State trace (AdvectionFusedElmtExtractAllDir()).
 * -# Fluxes, pointwise, from the policy (AdvDiffFusedElmtFlux()).
 * -# Derivative inner product of the flux vector, negated
 *    (AdvectionFusedElmtFluxDerivative()).
 * -# The outward normal viscous flux on the traces. With one normal per
 *    trace (affine elements) the contraction with the trace normal goes into
 *    the same line pass as the interpolation along the normal; with a normal
 *    per point the viscous flux's directional components are extracted as
 *    the state trace is and the contraction follows pointwise.
 * -# The gradient on the traces into the workspace, when asked for
 *    (component `n * NDIM + d` the derivative of variable n along direction
 *    d).
 *
 * @tparam SHAPE      Shape of the block's elements.
 * @tparam DEFORMED   Geometric factors vary point by point; otherwise every
 *                    trace has one normal, and the normal viscous flux is
 *                    contracted before it is extracted.
 * @tparam MAXQ       Points per direction the line-blocked kernels hold in
 *                    registers.
 * @tparam Physics    The flux policy (AdvDiffFusedElmtKernels.hpp).
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
 * @param   normals         Each element's outward unit normals on its
 *                          traces, per coordinate, `nqTotTrace` each.
 * @param   trace           Out: the state on the traces per variable.
 * @param   normalVisc      Out: the outward normal viscous flux on the
 *                          traces per variable.
 * @param   out             Out: the negated derivative inner product per
 *                          variable.
 * @param   writeGradTrace  Write the gradient on the traces to
 *                          `wsp.gradTrace`.
 * @param   wsp             The block operator's workspaces.
 */
template <LibUtilities::ShapeType SHAPE, bool DEFORMED, unsigned int MAXQ,
          typename Physics, typename TPhysSize, typename TTraceSize,
          typename TData, typename simd_type>
NEK_FORCE_INLINE static void AdvDiffFusedElmtGroupKernel(
    const TPhysSize sizeParam, const TTraceSize traceSizeParam,
    const AdvectionFusedElmtBasisData<TData, simd_type> &basisData,
    const AdvectionFusedElmtFacePacking &packing, const Physics &physics,
    const simd_type *df, const simd_type *jac, const simd_type *const *state,
    [[maybe_unused]] const simd_type *const *aux,
    const simd_type *const *normals, simd_type *const *trace,
    simd_type *const *normalVisc, simd_type *const *out,
    const bool writeGradTrace, const AdvDiffFusedElmtWorkspace<simd_type> &wsp)
{
    constexpr unsigned int NDIM = Physics::NDIM;
    constexpr unsigned int NVAR = Physics::NVAR;
    constexpr unsigned int NAUX = Physics::NAUX;
    constexpr unsigned int LINE =
        AdvectionFusedElmtLineLength<TPhysSize, MAXQ>();
    const unsigned int nqTot      = sizeParam.nqTot();
    const unsigned int nqTotTrace = packing.nqTotTrace;

    // 1. The gradient.
    PhysDerivTensorLinesKernelND<NDIM, NVAR, LINE>(
        sizeParam, state, basisData.derivMatrix, wsp.grad, wsp.varStride);
    for (unsigned int n = 0; n < NVAR; ++n)
    {
        AdvDiffFusedElmtPhysDerivField<SHAPE, DEFORMED, NDIM>(
            sizeParam, basisData, df, wsp.grad + n * wsp.varStride);
    }

    // 2. State trace.
    AdvectionFusedElmtExtractAllDir<SHAPE, NDIM, NVAR>(
        sizeParam, traceSizeParam, basisData, packing, state, trace, wsp.stage,
        wsp.interpWsp);

    // 3. Fluxes, pointwise.
    for (unsigned int i = 0; i < nqTot; ++i)
    {
        simd_type statePt[NVAR], gradPt[NDIM][NVAR], fluxPt[NDIM][NVAR],
            viscFluxPt[NDIM][NVAR];
        simd_type auxPt[NAUX > 0 ? NAUX : 1];
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            statePt[n] = state[n][i];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                gradPt[d][n] = wsp.grad[n * wsp.varStride + d * nqTot + i];
            }
        }
        if constexpr (NAUX > 0)
        {
            for (unsigned int a = 0; a < NAUX; ++a)
            {
                auxPt[a] = aux[a][i];
            }
        }
        AdvDiffFusedElmtFlux(physics, statePt, gradPt,
                             NAUX > 0 ? auxPt : nullptr, fluxPt, viscFluxPt);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                wsp.flux[n * wsp.varStride + d * nqTot + i] = fluxPt[d][n];
                wsp.viscFlux[n * wsp.varStride + d * nqTot + i] =
                    viscFluxPt[d][n];
            }
        }
    }

    // 4. Derivative inner product of the flux vector, negated.
    AdvectionFusedElmtFluxDerivative<SHAPE, DEFORMED, NDIM, NVAR, LINE>(
        sizeParam, basisData, df, jac, wsp, out);

    // 5. Outward normal viscous flux on the traces.
    if constexpr (!DEFORMED)
    {
        AdvDiffFusedElmtContractDir<SHAPE, 0, NDIM, NVAR>(
            sizeParam, traceSizeParam, basisData, packing, normals,
            wsp.viscFlux, wsp.varStride, normalVisc, wsp.stage, wsp.interpWsp);
        if constexpr (NDIM >= 2)
        {
            AdvDiffFusedElmtContractDir<SHAPE, 1, NDIM, NVAR>(
                sizeParam, traceSizeParam, basisData, packing, normals,
                wsp.viscFlux, wsp.varStride, normalVisc, wsp.stage,
                wsp.interpWsp);
        }
        if constexpr (NDIM == 3)
        {
            AdvDiffFusedElmtContractDir<SHAPE, 2, NDIM, NVAR>(
                sizeParam, traceSizeParam, basisData, packing, normals,
                wsp.viscFlux, wsp.varStride, normalVisc, wsp.stage,
                wsp.interpWsp);
        }
    }
    else
    {
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            const simd_type *comps[NDIM];
            simd_type *traceComps[NDIM];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                comps[d]      = wsp.viscFlux + n * wsp.varStride + d * nqTot;
                traceComps[d] = wsp.traceComps + d * nqTotTrace;
            }
            AdvectionFusedElmtExtractAllDir<SHAPE, NDIM, NDIM>(
                sizeParam, traceSizeParam, basisData, packing, comps,
                traceComps, wsp.stage, wsp.interpWsp);
            for (unsigned int pt = 0; pt < nqTotTrace; ++pt)
            {
                simd_type sum = normals[0][pt] * traceComps[0][pt];
                for (unsigned int d = 1; d < NDIM; ++d)
                {
                    sum.fma(normals[d][pt], traceComps[d][pt]);
                }
                normalVisc[n][pt] = sum;
            }
        }
    }

    // 6. The gradient on the traces, when asked for.
    if (writeGradTrace)
    {
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const simd_type *gradComps[NVAR];
            simd_type *gradTraceComps[NVAR];
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                gradComps[n]      = wsp.grad + n * wsp.varStride + d * nqTot;
                gradTraceComps[n] = wsp.gradTrace + (n * NDIM + d) * nqTotTrace;
            }
            AdvectionFusedElmtExtractAllDir<SHAPE, NDIM, NVAR>(
                sizeParam, traceSizeParam, basisData, packing, gradComps,
                gradTraceComps, wsp.stage, wsp.interpWsp);
        }
    }
}

} // namespace Nektar::SolverCore::detail
