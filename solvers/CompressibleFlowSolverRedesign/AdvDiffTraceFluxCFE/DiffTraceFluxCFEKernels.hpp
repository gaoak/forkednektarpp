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

#include <DiffusionCFEVolFlux/CFEViscosityParams.hpp>
#include <DiffusionCFEVolFlux/DiffusionCFEVolFluxKernels.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>
#include <SolverCore/TraceFlux/TraceFluxKernels.hpp>

#include <type_traits>

namespace Nektar::detail
{

/**
 * @brief The equation of state and the viscosity law the interior trace flux
 * kernels take.
 */
template <typename EoSParamType, typename TData> struct ViscousTraceParams
{
    EoSParamType EoS;
    CFEViscosityParams<TData> visc;
};

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
 * @brief The interior-penalty viscous trace flux of one point, in registers.
 *
 * Forms the averaged state (average of the internal energies plus the
 * kinetic energy of the averaged momentum), the jump, the IP gradient
 * `(grad_fwd + grad_bwd)/2 + n (jump) penalty`, the viscous flux tensor
 * through ViscousFluxTensorKernel(), and contracts it with the normal.
 * On entry @p flux holds the inviscid flux, which is negated and the viscous
 * part accumulated on top, exactly as DiffuseTraceFluxKernelLauncher() does.
 *
 * @param fwd, bwd        The two states, NDIM+2 entries each.
 * @param dfwd, dbwd      The two gradients, [direction][component].
 * @param normal          The trace normal.
 * @param penalty         The IP penalty factor of this point.
 * @param energyWeight    The viscous energy-flux weight (1 in the interior).
 * @param flux            In: inviscid flux. Out: -inviscid + viscous.
 */
template <unsigned int NDIM, bool IsInterior, typename EqnOfSParams,
          typename vec_t, typename TData>
NEK_HOSTDEVICE_INLINE static void IPViscousTraceFluxPoint(
    const EqnOfSParams &EoS, const vec_t *fwd, const vec_t *bwd,
    const vec_t (*dfwd)[NDIM + 2], const vec_t (*dbwd)[NDIM + 2],
    const vec_t *normal, const vec_t penalty, const vec_t energyWeight,
    const TData prandtl, const TData muRef, const bool isMuVariable,
    const TData oneOverTStar, const TData tRatioSutherland, vec_t *flux)
{
    constexpr unsigned int NVAR  = NDIM + 2;
    constexpr unsigned int nEngy = NDIM + 1;
    const vec_t half             = vec_t(0.5);
    const vec_t bWeightJump      = (IsInterior) ? vec_t(1.0) : vec_t(0.0);
    const vec_t fWeightJump      = (IsInterior) ? vec_t(1.0) : vec_t(2.0);
    const vec_t gammaOverPr      = vec_t(TData(EoS.gamma() / prandtl));

    vec_t averTmp[NVAR], jumpTmp[NVAR];
    vec_t qTmp[NDIM][NVAR];
    vec_t outTmp[NDIM][NVAR];

    const vec_t invRho = AveragedTraceState<NDIM>(fwd, bwd, averTmp);
    for (unsigned f = 0; f < NVAR; ++f)
    {
        jumpTmp[f] = (averTmp[f] - fwd[f]) * fWeightJump +
                     (bwd[f] - averTmp[f]) * bWeightJump;
        flux[f] = -flux[f];
    }

    const vec_t internalEnergy = GetInternalEnergyFromInvDensity(
        NDIM, invRho, averTmp + 1, averTmp[NDIM + 1]);
    const vec_t temperature = GetTemperature(EoS, averTmp[0], internalEnergy);
    const vec_t mu =
        GetDynamicViscosity(temperature, vec_t(muRef), isMuVariable,
                            vec_t(oneOverTStar), vec_t(tRatioSutherland));

    for (unsigned d = 0; d < NDIM; ++d)
    {
        for (unsigned f = 0; f < NVAR; ++f)
        {
            qTmp[d][f] = half * (dfwd[d][f] + dbwd[d][f]) +
                         normal[d] * jumpTmp[f] * penalty;
        }
    }

    ViscousFluxTensorKernel<NDIM>(averTmp, invRho, qTmp, mu, gammaOverPr,
                                  outTmp);

    for (unsigned f = 0; f < nEngy; ++f)
    {
        vec_t sum = vec_t(0.0);
        for (unsigned d = 0; d < NDIM; ++d)
        {
            sum += normal[d] * outTmp[d][f];
        }
        flux[f] += sum;
    }
    vec_t sumE = vec_t(0.0);
    for (unsigned d = 0; d < NDIM; ++d)
    {
        sumE += normal[d] * outTmp[d][nEngy];
    }
    flux[nEngy] += energyWeight * sumE;
}

/**
 * @brief The viscous trace flux of one interior point in the
 * average-of-normal-fluxes form, in registers.
 *
 * The viscous part is the mean of the two elements' own normal viscous
 * fluxes, each contracted with that element's outward normal before it
 * reached the trace, plus the interior-penalty term of
 * IPViscousTraceFluxPoint() alone: `n . G({u}) : (n (jump) penalty)`, with
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

    const vec_t internalEnergy = GetInternalEnergyFromInvDensity(
        NDIM, invRho, averTmp + 1, averTmp[NDIM + 1]);
    const vec_t temperature = GetTemperature(EoS, averTmp[0], internalEnergy);
    const vec_t mu =
        GetDynamicViscosity(temperature, vec_t(muRef), isMuVariable,
                            vec_t(oneOverTStar), vec_t(tRatioSutherland));

    // The penalty gradient is rank one, n (x) [u] tau, so the viscous
    // tensor it produces contracts with n in closed form (|n| = 1):
    //   d_d u_j = n_d tau a_j,   a_j = ([rho u_j] - u_j [rho]) / rho
    //   d_d e   = n_d tau b,     b   = ([E] - E/rho [rho]) / rho - u . a
    //   n . tau_j = mu tau (a_j + n_j (n . a) / 3)
    //   n . (tau u + k grad T) = u . (n . tau) + mu gamma/Pr tau b
    // which is ViscousFluxTensorKernel() on that gradient, without the
    // tensor.
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
 * @brief The combined inviscid and viscous flux of one interior trace point,
 * as the in-place trace flux kernels take it (the point flux of
 * SolverCore/TraceFlux/TraceFluxKernels.hpp): the Riemann flux through the
 * normal, then IPViscousTraceFluxPoint() with the two gradients as the
 * second input, or under @p AVG_NORMAL_FLUX AvgNormalFluxTracePoint() with
 * the two outward normal viscous fluxes.
 *
 * @tparam AVG_NORMAL_FLUX  The average-of-normal-fluxes form: the second
 *                       input is the outward normal viscous flux, one
 *                       component per state component, not the gradient.
 * @tparam RiemannKernel Flux function, as a class template on execution space,
 *                       equation-of-state parameters and dimension.
 * @tparam ExecSpace     Execution space; selects the SIMD width.
 * @tparam EoSParamType  Equation-of-state parameter pack.
 * @tparam NDIM          Spatial dimension.
 * @tparam TData         Floating-point representation.
 */
template <bool AVG_NORMAL_FLUX,
          template <typename, typename, unsigned int> typename RiemannKernel,
          typename ExecSpace, typename EoSParamType, unsigned int NDIM,
          typename TData>
struct ViscousPointFlux
{
    static constexpr unsigned numVar       = NDIM + 2;
    static constexpr unsigned secondPerVar = AVG_NORMAL_FLUX ? 1 : NDIM;
    static constexpr bool usesPenalty      = true;

    ViscousTraceParams<EoSParamType, TData> params;

    template <typename vec_t>
    NEK_HOSTDEVICE_INLINE void operator()(const vec_t *fwd, const vec_t *bwd,
                                          const vec_t *second0,
                                          const vec_t *second1,
                                          const vec_t *normal,
                                          const vec_t penalty,
                                          vec_t *flux) const
    {
        const auto &visc = params.visc;
        RiemannKernel<ExecSpace, EoSParamType, NDIM>()(params.EoS, 1, fwd, bwd,
                                                       normal, flux);
        if constexpr (AVG_NORMAL_FLUX)
        {
            AvgNormalFluxTracePoint<NDIM>(
                params.EoS, fwd, bwd, second0, second1, normal, penalty,
                visc.prandtl, visc.muRef, visc.isMuVariable, visc.oneOverTstar,
                visc.tRatioSutherland, flux);
        }
        else
        {
            // The gradient field holds component n's derivatives together;
            // the point flux takes them direction by direction.
            vec_t dfwd[NDIM][numVar], dbwd[NDIM][numVar];
            for (unsigned n = 0; n < numVar; ++n)
            {
                for (unsigned d = 0; d < NDIM; ++d)
                {
                    dfwd[d][n] = second0[n * NDIM + d];
                    dbwd[d][n] = second1[n * NDIM + d];
                }
            }
            IPViscousTraceFluxPoint<NDIM, true>(
                params.EoS, fwd, bwd, dfwd, dbwd, normal, penalty, vec_t(1.0),
                visc.prandtl, visc.muRef, visc.isMuVariable, visc.oneOverTstar,
                visc.tRatioSutherland, flux);
        }
    }
};

/**
 * @brief AvgNormalFluxTracePoint() over one padded interior trace block,
 * for a compile-time dimension. The layout is DiffuseTraceFluxKernelND()'s,
 * with the two normal-flux arrays carrying `nvarComps` components each in
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
NEK_FORCE_INLINE static void AvgNormalFluxTraceKernelLauncher(
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
 * @brief DiffuseTraceFluxKernelLauncher() for a compile-time dimension: the IP
 * gradient of every point is formed for all directions at once and the
 * viscous flux tensor taken through ViscousFluxTensorKernel(), then
 * contracted with the normal.
 */
template <typename ExecSpace, bool IsInterior, unsigned int NDIM,
          typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void DiffuseTraceFluxKernelND(
    const EqnOfSParams EoS, const unsigned npts, const TData prandtl,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *normbase,
    const TData *penaltyFactorBase, const TData *viscousEnergyWeightBase,
    const TData *fwdbase, const TData *bwdbase, const TData *derivfwdbase,
    const TData *derivbwdbase, TData *fluxbase, const unsigned int streamID)
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
    const auto engyWvec =
        reinterpret_cast<const vec_t *>(viscousEnergyWeightBase);
    const auto fwdvec      = reinterpret_cast<const vec_t *>(fwdbase);
    const auto bwdvec      = reinterpret_cast<const vec_t *>(bwdbase);
    const auto derivfwdvec = reinterpret_cast<const vec_t *>(derivfwdbase);
    const auto derivbwdvec = reinterpret_cast<const vec_t *>(derivbwdbase);
    auto fluxvec           = reinterpret_cast<vec_t *>(fluxbase);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            // Declared here, not in the enclosing function: MSVC does not
            // take a constant of the enclosing scope as a constant
            // expression inside the lambda.
            constexpr unsigned int NVAR = NDIM + 2;

            vec_t fwdTmp[NVAR], bwdTmp[NVAR], fluxTmp[NVAR];
            vec_t dfwd[NDIM][NVAR], dbwd[NDIM][NVAR];
            vec_t normal[NDIM];
            for (unsigned f = 0; f < NVAR; ++f)
            {
                fwdTmp[f]  = fwdvec[f * vecStride + i];
                bwdTmp[f]  = bwdvec[f * vecStride + i];
                fluxTmp[f] = fluxvec[f * vecStride + i];
                for (unsigned d = 0; d < NDIM; ++d)
                {
                    dfwd[d][f] = derivfwdvec[(f * NDIM + d) * vecStride + i];
                    dbwd[d][f] = derivbwdvec[(f * NDIM + d) * vecStride + i];
                }
            }
            for (unsigned d = 0; d < NDIM; ++d)
            {
                normal[d] = normvec[d * vecStride + i];
            }
            const vec_t viscousEnergyWeight =
                IsInterior ? vec_t(1.0) : engyWvec[i];

            IPViscousTraceFluxPoint<NDIM, IsInterior>(
                EoS, fwdTmp, bwdTmp, dfwd, dbwd, normal, penaltyvec[i],
                viscousEnergyWeight, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, fluxTmp);

            for (unsigned f = 0; f < NVAR; ++f)
            {
                fluxvec[f * vecStride + i] = fluxTmp[f];
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

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
 * @p IsInterior selects the jump weights, matching the interior penalty
 * convention used elsewhere in the redesign: on an interior trace the jump
 * is \f$(u_b - \{u\}) + (\{u\} - u_f)\f$, on a boundary trace
 * \f$2(\{u\} - u_f)\f$, with \f$\{u\}\f$ the averaged state of step 1.
 *
 * Data layout: all arrays are component-major with @p npts between consecutive
 * components; the gradient arrays carry `ndim * nvarComps` components,
 * variable outer and derivative direction inner. The loop runs one vector
 * of points per iteration, as DiffusionCFEVolFluxKernelLauncher() does: for AVX
 * a `tinysimd::simd<TData>` lane per point, and the scalar type otherwise.
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
 *                            afterwards.
 */
template <typename ExecSpace, bool IsInterior, typename EqnOfSParams,
          typename TData>
NEK_FORCE_INLINE static void DiffuseTraceFluxKernelLauncher(
    const EqnOfSParams EoS, const unsigned npts, const unsigned ndim,
    const TData prandtl, const TData muRef, const bool isMuVariable,
    const TData oneOverTStar, const TData tRatioSutherland,
    const TData *normbase, const TData *penaltyFactorBase,
    const TData *viscousEnergyWeightBase, const TData *fwdbase,
    const TData *bwdbase, const TData *derivfwdbase, const TData *derivbwdbase,
    TData *fluxbase, const unsigned int streamID = 0)
{
    switch (ndim)
    {
        case 1:
            DiffuseTraceFluxKernelND<ExecSpace, IsInterior, 1>(
                EoS, npts, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, normbase, penaltyFactorBase,
                viscousEnergyWeightBase, fwdbase, bwdbase, derivfwdbase,
                derivbwdbase, fluxbase, streamID);
            break;
        case 2:
            DiffuseTraceFluxKernelND<ExecSpace, IsInterior, 2>(
                EoS, npts, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, normbase, penaltyFactorBase,
                viscousEnergyWeightBase, fwdbase, bwdbase, derivfwdbase,
                derivbwdbase, fluxbase, streamID);
            break;
        case 3:
            DiffuseTraceFluxKernelND<ExecSpace, IsInterior, 3>(
                EoS, npts, prandtl, muRef, isMuVariable, oneOverTStar,
                tRatioSutherland, normbase, penaltyFactorBase,
                viscousEnergyWeightBase, fwdbase, bwdbase, derivfwdbase,
                derivbwdbase, fluxbase, streamID);
            break;
        default:
            NEKERROR(ErrorUtil::efatal, "The viscous trace flux is for one, "
                                        "two and three dimensions.");
    }
}

/**
 * @brief The boundary exterior state the diffusion sees, in one pass.
 *
 * A copy of the Riemann solver's exterior state @p gloT1, except on the
 * points of a trace whose ghost is not already a reflection of the interior,
 * where it is 2g - Q+, so that the half-and-half average the viscous kernel
 * takes reproduces the imposed value g. Traces whose ghost reflects keep the
 * copy - see AdvDiffTraceFluxCFEOpImpl::m_gloT1Diff for why the distinction
 * lives in the state rather than in an averaging weight.
 *
 * The whole padded block is written, the padding with the plain copy, which
 * is what the kernel that walks the whole padded block expects to find there.
 *
 * @tparam ExecSpace         Execution space the loop runs in.
 * @tparam TData             Floating-point representation.
 *
 * @param npTBlock           Padded component stride of the block.
 * @param numflux            Flux components.
 * @param reflectGhost       Per point of the padded block, non-zero where the
 *                           ghost is to be reflected: on the points of a
 *                           trace whose ghost does not already reflect; zero
 *                           on the others and on the padding. Built once per
 *                           boundary block.
 * @param gloT0              Interior state.
 * @param gloT1              Riemann solver's exterior state.
 * @param gloT1Diff          Diffusion's exterior state, written.
 * @param streamID           Stream the launch runs in; reset to the default
 *                           afterwards.
 */
template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void DiffusionGhostStateKernelLauncher(
    const size_t npTBlock, const unsigned numflux, const unsigned *reflectGhost,
    const TData *gloT0, const TData *gloT1, TData *gloT1Diff,
    const unsigned int streamID)
{
    if constexpr (!std::is_same_v<ExecSpace, NektarSpaces::Device>)
    {
        // Component by component over the contiguous points, which the
        // compiler vectorises; the per-point form strides by the block.
        for (unsigned f = 0; f < numflux; ++f)
        {
            const TData *const in0 = gloT0 + f * npTBlock;
            const TData *const in1 = gloT1 + f * npTBlock;
            TData *const out       = gloT1Diff + f * npTBlock;
            for (size_t i = 0; i < npTBlock; ++i)
            {
                out[i] =
                    reflectGhost[i] ? TData(2.0) * in1[i] - in0[i] : in1[i];
            }
        }
        return;
    }

    // One point of the padded block per thread, over every component:
    // consecutive threads read consecutive points of each component.
    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, npTBlock, NEKTAR_LAMBDA(const size_t i) {
            const bool reflect = reflectGhost[i];
            for (unsigned f = 0; f < numflux; ++f)
            {
                const size_t idx = f * npTBlock + i;
                gloT1Diff[idx] =
                    reflect ? TData(2.0) * gloT1[idx] - gloT0[idx] : gloT1[idx];
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
