///////////////////////////////////////////////////////////////////////////////
//
// File: DiffTraceFluxCFEKernels.hpp
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
// Description: Kernels for the Diffusion CFE Volume fluxes
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DiffusionCFEVolFlux/DiffusionCFEVolFluxKernels.hpp"
#include "MultiRegions/DataWarehouse/TraceDataWarehouse.hpp"
#include <LibUtilities/BasicUtils/SafeDivide.hpp>

namespace Nektar::detail
{
/**
 * @brief Viscous numerical flux for the compressible Navier-Stokes equations,
 * evaluated over one padded trace block and accumulated onto the negated
 * inviscid flux already there.
 *
 * At each trace point the kernel builds the average conserved state across the
 * trace, forms a penalty-corrected gradient trace, and contracts the viscous
 * flux tensor with the interface normal:
 *
 * 1. Average the conserved variables. The energy component is averaged through
 *    the *internal* energy rather than directly, so that the kinetic energy
 *    implied by the averaged momentum is consistent with the averaged density.
 * 2. Form the jump, weighted according to @p IsInterior.
 * 3. For each derivative direction, take the mean of the two gradient traces
 *    and add the jump scaled by the normal and the penalty factor.
 * 4. Evaluate transport properties from the averaged state (internal energy,
 *    temperature, then dynamic viscosity via Sutherland's law when
 *    @p isMuVariable) and accumulate the normal viscous flux.
 *
 * @p IsInterior selects the averaging and jump weights, matching the interior
 * penalty convention used elsewhere in the redesign:
 *
 * | Quantity   | `IsInterior == true` | `IsInterior == false` |
 * |------------|----------------------|-----------------------|
 * | average    | \f$(u_f + u_b)/2\f$  | \f$u_b\f$             |
 * | jump       | \f$u_b - u_f\f$      | \f$2(u_b - u_f)\f$    |
 *
 * Data layout: all arrays are component-major with @p npts between consecutive
 * components; the gradient arrays carry `ndim * nvarComps` components,
 * variable outer and derivative direction inner. The loop runs one vector
 * of points per iteration, as DiffusionCFEVolFluxKernel() does: for AVX a
 * `tinysimd::simd<TData>` lane per point, and the scalar type otherwise.
 *
 * @tparam ExecSpace     Execution space the loop runs in.
 * @tparam IsInterior    True for interior traces, false for boundary traces.
 * @tparam EqnOfSParams  Equation-of-state parameter pack.
 * @tparam TData         Floating-point representation.
 *
 * @param EoS               - Equation-of-state parameters.
 * @param npts              - Padded number of points in the block, a whole
 *                            number of vectors.
 * @param ndim              - Spatial dimension.
 * @param nvarComps         - Number of conserved variables. Must equal
 *                            `ndim + 2`; asserted.
 * @param prandtl           - Prandtl number.
 * @param muRef             - Reference dynamic viscosity.
 * @param isMuVariable      - Apply Sutherland's law rather than constant
 *                            viscosity.
 * @param oneOverTStar      - Reciprocal reference temperature.
 * @param tRatioSutherland  - Sutherland temperature over reference
 *                            temperature.
 * @param normbase          - Trace normals, @p ndim components.
 * @param penaltyFactorBase - Per-point geometric penalty factor.
 * @param fwdbase           - Forward-side conserved state.
 * @param bwdbase           - Backward-side conserved state.
 * @param derivfwdbase      - Forward-side gradient of the conserved state.
 * @param derivbwdbase      - Backward-side gradient.
 * @param fluxbase          - The inviscid flux on entry; on exit its
 *                            negative with the viscous flux accumulated on
 *                            top.
 * @param streamID          - Stream the launch runs in; reset to the default
 * afterwards.
 *
 * @note The per-point temporaries are fixed at five entries, which is exactly
 *       `ndim + 2` in 3D. The assertion on @p nvarComps is what keeps this
 *       safe.
 */
template <typename ExecSpace, bool IsInterior, typename EqnOfSParams,
          typename TData>
NEK_FORCE_INLINE static void DiffuseTraceFluxKernel(
    const EqnOfSParams EoS, const unsigned npts, const unsigned ndim,
    const unsigned nvarComps, const TData prandtl, const TData muRef,
    const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *normbase,
    const TData *penaltyFactorBase, const TData *viscousEnergyWeightBase,
    const TData *fwdbase, const TData *bwdbase, const TData *derivfwdbase,
    const TData *derivbwdbase, TData *fluxbase, const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    ASSERTL0(nvarComps == ndim + 2,
             "number of componeents is not equal to the dimension plus two")

    // One vector lane per trace point: vec_t = tinysimd::simd<TData> for
    // AVX, vec_t = TData otherwise. Every array is component-major with
    // npts between components, and npts is a whole number of vectors, so a
    // component stride in vectors is npts / vec_width.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    ASSERTL1(npts % vec_width == 0,
             "The padded trace block is not a whole number of vectors");

    const size_t groupsize = npts / vec_width;
    const size_t vecStride = npts / vec_width;

    const auto normvec    = reinterpret_cast<const vec_t *>(normbase);
    const auto penaltyvec = reinterpret_cast<const vec_t *>(penaltyFactorBase);
    const auto engyWvec =
        reinterpret_cast<const vec_t *>(viscousEnergyWeightBase);
    const auto fwdvec      = reinterpret_cast<const vec_t *>(fwdbase);
    const auto bwdvec      = reinterpret_cast<const vec_t *>(bwdbase);
    const auto derivfwdvec = reinterpret_cast<const vec_t *>(derivfwdbase);
    const auto derivbwdvec = reinterpret_cast<const vec_t *>(derivbwdbase);
    auto fluxvec           = reinterpret_cast<vec_t *>(fluxbase);

    // One averaging rule everywhere, {w} = (w+ + w-)/2, as defined in
    // Zhen-Guo Yan, "Efficient implicit spectral/hp element DG techniques for
    // compressible flows", PhD thesis, Imperial College London, 2021, Ch. 2
    // "Symmetric interior penalty Galerkin method", Eq. (2.17). A boundary
    // needs no special weight here because the caller hands in an exterior
    // state already reflected about the imposed condition; see
    // AdvDiffTraceFluxCFEOpImpl::m_gloT1Diff.
    const vec_t bWeightAver = vec_t(0.5);
    const vec_t fWeightAver = vec_t(0.5);
    const vec_t bWeightJump = (IsInterior) ? vec_t(1.0) : vec_t(0.0);
    const vec_t fWeightJump = (IsInterior) ? vec_t(1.0) : vec_t(2.0);

    // Assemble the numerical flux on the trace a vector of points at a time:
    // average state, jump, penalty-corrected derivative trace, then normal
    // viscous flux.
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            vec_t fwdTmp[5]  = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                                vec_t(0.0)};
            vec_t bwdTmp[5]  = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                                vec_t(0.0)};
            vec_t averTmp[5] = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                                vec_t(0.0)};
            vec_t jumpTmp[5] = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                                vec_t(0.0)};
            vec_t qTmp[5]    = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                                vec_t(0.0)};
            vec_t outTmp[5]  = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                                vec_t(0.0)};

            const unsigned nEngy = nvarComps - 1;

            for (unsigned f = 0; f < nvarComps; ++f)
            {
                fwdTmp[f] = fwdvec[f * vecStride + i];
                bwdTmp[f] = bwdvec[f * vecStride + i];
            }

            for (unsigned f = 0; f < nEngy; ++f)
            {
                averTmp[f] = fWeightAver * fwdTmp[f] + bWeightAver * bwdTmp[f];
            }

            vec_t lInternal = vec_t(0.0);
            vec_t rInternal = vec_t(0.0);
            vec_t aInternal = vec_t(0.0);
            for (unsigned d = 1; d < nEngy; ++d)
            {
                lInternal += fwdTmp[d] * fwdTmp[d];
                rInternal += bwdTmp[d] * bwdTmp[d];
                aInternal += averTmp[d] * averTmp[d];
            }

            lInternal      = fwdTmp[nEngy] - vec_t(0.5) * lInternal / fwdTmp[0];
            rInternal      = bwdTmp[nEngy] - vec_t(0.5) * rInternal / bwdTmp[0];
            averTmp[nEngy] = fWeightAver * lInternal + bWeightAver * rInternal +
                             vec_t(0.5) * aInternal / averTmp[0];

            for (unsigned f = 0; f < nvarComps; ++f)
            {
                jumpTmp[f] = (averTmp[f] - fwdTmp[f]) * fWeightJump +
                             (bwdTmp[f] - averTmp[f]) * bWeightJump;
            }

            // The inviscid flux is already in place; it goes to the other
            // side of the equation here, in the same pass that accumulates
            // the viscous contribution on top of it.
            for (unsigned f = 0; f < nvarComps; ++f)
            {
                fluxvec[f * vecStride + i] = -fluxvec[f * vecStride + i];
            }

            const vec_t penalty = penaltyvec[i];

            const vec_t e = GetInternalEnergy(ndim, averTmp[0], averTmp + 1,
                                              averTmp[ndim + 1]);
            const vec_t viscousEnergyWeight =
                IsInterior ? vec_t(1.0) : engyWvec[i];

            const vec_t temperature = GetTemperature(EoS, averTmp[0], e);

            const vec_t mu = GetDynamicViscosity(
                temperature, vec_t(muRef), isMuVariable, vec_t(oneOverTStar),
                vec_t(tRatioSutherland));

            for (unsigned derivDir = 0; derivDir < ndim; ++derivDir)
            {
                for (unsigned f = 0; f < nvarComps; ++f)
                {
                    // The extraction gives the raw derivative traces of
                    // both sides, so the average is formed here.
                    qTmp[f] =
                        vec_t(0.5) *
                            (derivfwdvec[(f * ndim + derivDir) * vecStride +
                                         i] +
                             derivbwdvec[(f * ndim + derivDir) * vecStride +
                                         i]) +
                        normvec[derivDir * vecStride + i] * jumpTmp[f] *
                            penalty;
                }

                for (unsigned fluxDir = 0; fluxDir < ndim; ++fluxDir)
                {
                    // Contract the viscous flux tensor with the interface
                    // normal to obtain the scalar trace contribution.
                    GetViscousFluxBilinearFormKernel(
                        ndim, fluxDir, derivDir, averTmp, qTmp, mu,
                        vec_t(EoS.gamma()), vec_t(prandtl), outTmp);

                    // At a no-slip wall the viscous work u.tau vanishes, so
                    // the energy component of this flux is purely the heat
                    // flux q.n = k dT/dn. An adiabatic wall is therefore
                    // imposed by suppressing exactly this term, which is what
                    // a zero weight does; every other boundary and the whole
                    // interior carry weight one.
                    const vec_t normal = normvec[fluxDir * vecStride + i];
                    for (unsigned f = 0; f < nEngy; ++f)
                    {
                        fluxvec[f * vecStride + i] += normal * outTmp[f];
                    }
                    fluxvec[nEngy * vecStride + i] +=
                        viscousEnergyWeight * normal * outTmp[nEngy];
                }
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

/**
 * @brief The averaged state of one trace point: the conserved variables
 * averaged, and the energy as the average of the two internal energies plus
 * the kinetic energy of the averaged momentum, so that the kinetic energy
 * the averaged momentum implies is consistent with the averaged density.
 *
 * @param fwd, bwd  The two states, NDIM+2 entries each.
 * @param aver      Out: the averaged state.
 * @return 1/rho of the averaged state.
 */
template <unsigned int NDIM, typename vec_t>
NEK_HOSTDEVICE_INLINE static vec_t AveragedTraceState(const vec_t *fwd,
                                                      const vec_t *bwd,
                                                      vec_t *aver)
{
    constexpr unsigned int nEngy = NDIM + 1;
    const vec_t half             = vec_t(0.5);
    for (unsigned f = 0; f < nEngy; ++f)
    {
        aver[f] = half * (fwd[f] + bwd[f]);
    }
    vec_t lInternal = vec_t(0.0), rInternal = vec_t(0.0),
          aInternal = vec_t(0.0);
    for (unsigned d = 1; d < nEngy; ++d)
    {
        lInternal += fwd[d] * fwd[d];
        rInternal += bwd[d] * bwd[d];
        aInternal += aver[d] * aver[d];
    }
    // Three reciprocals serve every division of the trace flux; zero on the
    // zero padding of a block.
    const vec_t invL   = SafeDivide(vec_t(1.0), fwd[0]);
    const vec_t invR   = SafeDivide(vec_t(1.0), bwd[0]);
    const vec_t invRho = SafeDivide(vec_t(1.0), aver[0]);
    lInternal          = fwd[nEngy] - half * lInternal * invL;
    rInternal          = bwd[nEngy] - half * rInternal * invR;
    aver[nEngy] =
        half * lInternal + half * rInternal + half * aInternal * invRho;
    return invRho;
}

/**
 * @brief The viscous trace flux of one interior point in the
 * average-of-normal-fluxes form, in registers.
 *
 * The viscous part is the mean of the two elements' own normal viscous
 * fluxes, each contracted with that element's outward normal before it
 * reached the trace, plus the interior-penalty term of
 * DiffuseTraceFluxKernel() alone: `n . G({u}) : (n (jump) penalty)`, with
 * the same averaged state and jump. On entry @p flux holds the inviscid
 * flux, which is negated and the viscous part accumulated on top.
 *
 * @param fwd, bwd        The two states, NDIM+2 entries each.
 * @param nFfwd           The forward element's outward normal viscous flux.
 * @param nFbwd           The backward element's outward normal viscous flux,
 *                        taken with that element's outward normal, so its
 *                        negative is the flux through @p normal.
 * @param normal          The trace normal (the forward element's outward).
 * @param penalty         The IP penalty factor of this point.
 * @param flux            In: inviscid flux. Out: -inviscid + viscous.
 */
template <unsigned int NDIM, typename EqnOfSParams, typename vec_t,
          typename TData>
NEK_HOSTDEVICE_INLINE static void AvgNormalFluxTracePoint(
    const EqnOfSParams &EoS, const vec_t *fwd, const vec_t *bwd,
    const vec_t *nFfwd, const vec_t *nFbwd, const vec_t *normal,
    const vec_t penalty, const TData prandtl, const TData muRef,
    const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, vec_t *flux)
{
    constexpr unsigned int NVAR  = NDIM + 2;
    constexpr unsigned int nEngy = NDIM + 1;
    const vec_t half             = vec_t(0.5);
    const vec_t gammaOverPr      = vec_t(TData(EoS.gamma() / prandtl));

    vec_t averTmp[NVAR], jumpTmp[NVAR];

    const vec_t invRho = AveragedTraceState<NDIM>(fwd, bwd, averTmp);
    for (unsigned f = 0; f < NVAR; ++f)
    {
        jumpTmp[f] = bwd[f] - fwd[f];
        flux[f]    = -flux[f] + half * (nFfwd[f] - nFbwd[f]);
    }

    const vec_t e = GetInternalEnergyFromInvDensity(NDIM, invRho, averTmp + 1,
                                                    averTmp[NDIM + 1]);
    const vec_t temperature = GetTemperature(EoS, averTmp[0], e);
    const vec_t mu =
        GetDynamicViscosity(temperature, vec_t(muRef), isMuVariable,
                            vec_t(oneOverTStar), vec_t(tRatioSutherland));

    // The penalty gradient is rank one, n (x) [u] tau, so the viscous
    // tensor it produces contracts with n in closed form (|n| = 1):
    //   d_d u_j = n_d tau a_j,   a_j = ([rho u_j] - u_j [rho]) / rho
    //   d_d e   = n_d tau b,     b   = ([E] - E/rho [rho]) / rho - u . a
    //   n . tau_j = mu tau (a_j + n_j (n . a) / 3)
    //   n . (tau u + k grad T) = u . (n . tau) + mu gamma/Pr tau b
    // which is the viscous flux of GetViscousFluxBilinearFormKernel() on
    // that gradient, contracted with n, without forming the tensor.
    const vec_t eTotal = averTmp[nEngy] * invRho;
    vec_t u[NDIM], a[NDIM];
    vec_t nDotA = vec_t(0.0), uDotA = vec_t(0.0);
    for (unsigned j = 0; j < NDIM; ++j)
    {
        u[j] = averTmp[j + 1] * invRho;
        a[j] = (jumpTmp[j + 1] - u[j] * jumpTmp[0]) * invRho;
        nDotA += normal[j] * a[j];
        uDotA += u[j] * a[j];
    }
    const vec_t b     = (jumpTmp[nEngy] - eTotal * jumpTmp[0]) * invRho - uDotA;
    const vec_t muTau = mu * penalty;
    const vec_t thirdNDotA = nDotA * vec_t(TData(1.0) / TData(3.0));
    vec_t work             = vec_t(0.0);
    for (unsigned j = 0; j < NDIM; ++j)
    {
        const vec_t tj = muTau * (a[j] + normal[j] * thirdNDotA);
        flux[j + 1] += tj;
        work += u[j] * tj;
    }
    flux[nEngy] += work + muTau * gammaOverPr * b;
}

/**
 * @brief AvgNormalFluxTracePoint() over one padded interior trace block,
 * for a compile-time dimension. The layout is DiffuseTraceFluxKernel()'s,
 * with the two normal-flux arrays carrying `ndim + 2` components each in
 * place of the gradients.
 */
template <typename ExecSpace, unsigned int NDIM, typename EqnOfSParams,
          typename TData>
NEK_FORCE_INLINE static void AvgNormalFluxTraceKernelND(
    const EqnOfSParams EoS, const unsigned npts, const TData prandtl,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *normbase,
    const TData *penaltyFactorBase, const TData *fwdbase, const TData *bwdbase,
    const TData *nFfwdbase, const TData *nFbwdbase, TData *fluxbase,
    const unsigned int streamID)
{
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize = npts / vec_width;
    const size_t vecStride = npts / vec_width;
    const auto normvec     = reinterpret_cast<const vec_t *>(normbase);
    const auto penaltyvec  = reinterpret_cast<const vec_t *>(penaltyFactorBase);
    const auto fwdvec      = reinterpret_cast<const vec_t *>(fwdbase);
    const auto bwdvec      = reinterpret_cast<const vec_t *>(bwdbase);
    const auto nFfwdvec    = reinterpret_cast<const vec_t *>(nFfwdbase);
    const auto nFbwdvec    = reinterpret_cast<const vec_t *>(nFbwdbase);
    auto fluxvec           = reinterpret_cast<vec_t *>(fluxbase);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            // Declared here, not in the enclosing function: MSVC does not
            // take a constant of the enclosing scope as a constant
            // expression inside the lambda.
            constexpr unsigned int NVAR = NDIM + 2;

            vec_t fwdTmp[NVAR], bwdTmp[NVAR], fluxTmp[NVAR];
            vec_t nFfwd[NVAR], nFbwd[NVAR];
            vec_t normal[NDIM];
            for (unsigned f = 0; f < NVAR; ++f)
            {
                fwdTmp[f]  = fwdvec[f * vecStride + i];
                bwdTmp[f]  = bwdvec[f * vecStride + i];
                nFfwd[f]   = nFfwdvec[f * vecStride + i];
                nFbwd[f]   = nFbwdvec[f * vecStride + i];
                fluxTmp[f] = fluxvec[f * vecStride + i];
            }
            for (unsigned d = 0; d < NDIM; ++d)
            {
                normal[d] = normvec[d * vecStride + i];
            }

            AvgNormalFluxTracePoint<NDIM>(EoS, fwdTmp, bwdTmp, nFfwd, nFbwd,
                                          normal, penaltyvec[i], prandtl, muRef,
                                          isMuVariable, oneOverTStar,
                                          tRatioSutherland, fluxTmp);

            for (unsigned f = 0; f < NVAR; ++f)
            {
                fluxvec[f * vecStride + i] = fluxTmp[f];
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/// AvgNormalFluxTraceKernelND() dispatched on the dimension, @p ndim + 2
/// variables.
template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void AvgNormalFluxTraceKernel(
    const EqnOfSParams EoS, const unsigned npts, const unsigned ndim,
    const TData prandtl, const TData muRef, const bool isMuVariable,
    const TData oneOverTStar, const TData tRatioSutherland,
    const TData *normbase, const TData *penaltyFactorBase, const TData *fwdbase,
    const TData *bwdbase, const TData *nFfwdbase, const TData *nFbwdbase,
    TData *fluxbase, const unsigned int streamID = 0)
{
    switch (ndim)
    {
        case 1:
            AvgNormalFluxTraceKernelND<ExecSpace, 1>(
                EoS, npts, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, normbase, penaltyFactorBase, fwdbase, bwdbase,
                nFfwdbase, nFbwdbase, fluxbase, streamID);
            break;
        case 2:
            AvgNormalFluxTraceKernelND<ExecSpace, 2>(
                EoS, npts, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, normbase, penaltyFactorBase, fwdbase, bwdbase,
                nFfwdbase, nFbwdbase, fluxbase, streamID);
            break;
        case 3:
            AvgNormalFluxTraceKernelND<ExecSpace, 3>(
                EoS, npts, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, normbase, penaltyFactorBase, fwdbase, bwdbase,
                nFfwdbase, nFbwdbase, fluxbase, streamID);
            break;
        default:
            NEKERROR(ErrorUtil::efatal,
                     "The average-of-normal-fluxes viscous trace flux is for "
                     "one, two and three dimensions.");
    }
}

/**
 * @brief Reflect the boundary ghost state the diffusion sees, where needed.
 *
 * @p gloT1Diff must already hold a copy of the Riemann solver's exterior state.
 * On a trace whose ghost is not already a reflection of the interior, this
 * replaces it with 2g - Q+, so that the half-and-half average the viscous
 * kernel takes reproduces the imposed value g. Traces flagged reflected are
 * left as they are - see AdvDiffTraceFluxCFEOpImpl::m_gloT1Diff for why the
 * distinction lives in the state rather than in an averaging weight.
 *
 * Only the points in use are touched; the padding keeps the plain copy, which
 * is what the kernel that walks the whole padded block expects to find there.
 *
 * @tparam ExecSpace         Execution space the loop runs in.
 * @tparam TData             Floating-point representation.
 *
 * @param numTrace           Traces in the block.
 * @param npTot              Points on one trace.
 * @param npTBlock           Padded component stride of the block.
 * @param numflux            Flux components.
 * @param ghostIsReflected   Per trace, non-zero where the ghost already
 *                           reflects. A device array of `unsigned` rather than
 *                           of `bool`, since the host holds it in a
 *                           `std::vector<bool>` whose bits have no addresses to
 *                           copy from.
 * @param gloT0              Interior state.
 * @param gloT1              Riemann solver's exterior state.
 * @param gloT1Diff          Diffusion's exterior state, overwritten in place.

 * @param streamID           Stream the launch runs in; reset to the default
 afterwards.*/
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void ReflectDiffusionGhostStateKernel(
    const size_t numTrace, const size_t npTot, const size_t npTBlock,
    const unsigned numflux, const unsigned *ghostIsReflected,
    const TData *gloT0, const TData *gloT1, TData *gloT1Diff,
    const unsigned int streamID = 0)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    const size_t npTrace = numTrace * npTot;

    Nektar::parallel_for<ExecSpace>(
        0u, npTrace * numflux, NEKTAR_LAMBDA(const size_t idx) {
            const size_t f = idx / npTrace;
            const size_t r = idx - f * npTrace;
            const size_t t = r / npTot;

            if (ghostIsReflected[t])
            {
                return;
            }

            const size_t o = f * npTBlock + r;

            gloT1Diff[o] = TData(2.0) * gloT1[o] - gloT0[o];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
