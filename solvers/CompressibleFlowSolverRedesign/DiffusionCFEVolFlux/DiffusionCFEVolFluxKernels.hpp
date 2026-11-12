///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionCFEVolFluxKernels.hpp
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

#include <LibUtilities/LoopExecution/LoopExecution.hpp>

// header should appear after loop execution.hpp
#include <EquationOfState/VariableConverters.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>

namespace Nektar::detail
{

/**
 * @brief The viscous flux tensor of one point, in stress-tensor form.
 *
 * With the velocity gradients and the internal-energy gradient recovered
 * from the conservative ones once,
 *
 *     d_i u_j   = (q_i[j+1] - u_j q_i[0]) / rho
 *     d_i e     = (q_i[E] - (E/rho) q_i[0]) / rho - u_j d_i u_j
 *     tau_ij    = mu (d_i u_j + d_j u_i) - (2/3) mu delta_ij div u
 *     F_i       = (0, tau_i1 .. tau_iN, u_j tau_ij + mu (gamma/Pr) d_i e).
 *
 * The caller passes the inverse density it has already formed, and with
 * @p NDIM a template parameter every loop unrolls. The IP penalty, when the
 * caller has added it to @p grad, is carried through unchanged, since the
 * tensor is linear in the gradients.
 *
 * @param inAverage   The state (rho, rho u_j, E), NDIM+2 entries.
 * @param invRho      1 / rho.
 * @param grad        The gradients, grad[d][f] = d_d of component f.
 * @param mu          The dynamic viscosity at this point.
 * @param gammaOverPr gamma / Pr.
 * @param out         out[i][f]: component f of the flux in direction i.
 */
template <unsigned int NDIM, typename TData>
NEK_HOSTDEVICE_INLINE static void ViscousFluxTensorKernel(
    const TData *inAverage, const TData invRho, const TData (*grad)[NDIM + 2],
    const TData mu, const TData gammaOverPr, TData (*out)[NDIM + 2])
{
    const TData twoThird = TData(2.0) / TData(3.0);
    const TData eTotal   = inAverage[NDIM + 1] * invRho;

    TData vel[NDIM];
    for (unsigned int j = 0; j < NDIM; ++j)
    {
        vel[j] = inAverage[j + 1] * invRho;
    }

    TData du[NDIM][NDIM]; // du[d][j] = d_d u_j
    TData de[NDIM];       // d_d e_int
    TData div = TData(0.0);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        TData uDu = TData(0.0);
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            du[d][j] = (grad[d][j + 1] - vel[j] * grad[d][0]) * invRho;
            uDu += vel[j] * du[d][j];
        }
        de[d] = (grad[d][NDIM + 1] - eTotal * grad[d][0]) * invRho - uDu;
        div += du[d][d];
    }

    const TData muDiv = twoThird * mu * div;
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        out[i][0]  = TData(0.0);
        TData work = TData(0.0);
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            TData tau = mu * (du[i][j] + du[j][i]);
            if (i == j)
            {
                tau -= muDiv;
            }
            out[i][j + 1] = tau;
            work += vel[j] * tau;
        }
        out[i][NDIM + 1] = work + mu * gammaOverPr * de[i];
    }
}

/**
 * @brief The Euler flux tensor of one point, laid out as
 * ViscousFluxTensorKernel() lays out the viscous one: `fluxTensor[d][f]` is
 * component f of the flux in direction d,
 *
 *     F_d = (rho u_d, rho u_a u_d + p delta_ad, (E + p) u_d).
 *
 * @param state       The state (rho, rho u_j, E), NDIM+2 entries.
 * @param vel         The velocity.
 * @param pressure    The pressure.
 * @param fluxTensor  Out: the flux tensor.
 */
template <unsigned int NDIM, typename TData>
NEK_HOSTDEVICE_INLINE static void EulerFluxTensorKernel(
    const TData state[NDIM + 2], const TData vel[NDIM], const TData pressure,
    TData fluxTensor[NDIM][NDIM + 2])
{
    const TData ePlusP = state[NDIM + 1] + pressure;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        fluxTensor[d][0] = state[1 + d];
        for (unsigned int a = 0; a < NDIM; ++a)
        {
            fluxTensor[d][1 + a] = state[1 + a] * vel[d];
        }
        fluxTensor[d][1 + d] += pressure;
        fluxTensor[d][NDIM + 1] = ePlusP * vel[d];
    }
}

/**
 * @brief The viscous volume flux tensor of the compressible Navier-Stokes
 * equations, pointwise from the conservative state and its gradient, for a
 * compile-time dimension @p NDIM and so @p NDIM + 2 variables.
 *
 * With @p TOTAL the same pass also writes @p totbase, the viscous flux plus
 * @p advScale times the Euler flux, formed from the state already loaded and
 * the internal energy already computed: with `advScale = -1` this is the
 * total volume flux `F_v - F_a` that an advection-diffusion right-hand side
 * integrates, while @p outbase keeps the viscous flux on its own.
 *
 * @tparam NDIM    Spatial dimension.
 * @tparam APPEND  Accumulate the viscous flux onto @p outbase; not with
 *                 @p TOTAL.
 * @tparam TOTAL   Write the total flux into @p totbase as well.
 */
template <unsigned int NDIM, bool APPEND, bool TOTAL, typename ExecSpace,
          typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void DiffusionCFEVolFluxKernelND(
    const EqnOfSParams EoS, const size_t npts, const size_t inStride,
    const size_t derivStride, const size_t outStride, const TData prandtl,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *inbase, const TData *qbase,
    TData *outbase, TData *totbase, const TData advScale,
    const unsigned int streamID)
{
    static_assert(!(APPEND && TOTAL),
                  "The total flux is written alongside a fresh viscous flux");

    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize      = npts / vec_width;
    const auto invec            = reinterpret_cast<const vec_t *>(inbase);
    const auto qvec             = reinterpret_cast<const vec_t *>(qbase);
    auto outvec                 = reinterpret_cast<vec_t *>(outbase);
    auto totvec                 = reinterpret_cast<vec_t *>(totbase);
    const size_t inVecStride    = inStride / vec_width;
    const size_t derivVecStride = derivStride / vec_width;
    const size_t outVecStride   = outStride / vec_width;
    const vec_t gammaOverPr     = vec_t(EoS.gamma()) / vec_t(prandtl);

    Nektar::LoopExecutionSetStreamID(streamID);
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            // Restated inside the lambda: MSVC does not take a captured
            // constexpr local as a constant expression for an array bound.
            constexpr unsigned int NVAR = NDIM + 2;
            vec_t inTmp[NVAR];
            vec_t qTmp[NDIM][NVAR];
            vec_t outTmp[NDIM][NVAR];
            vec_t advTmp[NDIM][NVAR];

            for (unsigned int f = 0; f < NVAR; ++f)
            {
                inTmp[f] = invec[f * inVecStride + i];
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    qTmp[d][f] = qvec[(f * NDIM + d) * derivVecStride + i];
                }
            }

            // Zero on the zero padding, and with it the temperature, every
            // velocity and gradient, and so both fluxes.
            const vec_t invRho = SafeDivide(vec_t(TData(1)), inTmp[0]);
            const vec_t e      = GetInternalEnergyFromInvDensity(
                NDIM, invRho, inTmp + 1, inTmp[NDIM + 1]);
            const vec_t temperature = GetTemperature(EoS, inTmp[0], e);
            const vec_t mu          = GetDynamicViscosity(
                temperature, vec_t(muRef), isMuVariable, vec_t(oneOverTStar),
                vec_t(tRatioSutherland));

            ViscousFluxTensorKernel<NDIM>(inTmp, invRho, qTmp, mu, gammaOverPr,
                                          outTmp);

            // Plain if, not if constexpr: nvcc refuses an extended lambda
            // whose first capture of a variable sits inside an if-constexpr
            // branch. The template constants still fold the dead branches.
            if (TOTAL)
            {
                vec_t vel[NDIM];
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    vel[d] = inTmp[1 + d] * invRho;
                }
                EulerFluxTensorKernel<NDIM>(
                    inTmp, vel, GetPressure(EoS, inTmp[0], e), advTmp);
            }

            for (unsigned int f = 0; f < NVAR; ++f)
            {
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    const size_t idx = (f * NDIM + d) * outVecStride + i;
                    if (APPEND)
                    {
                        outvec[idx] += outTmp[d][f];
                    }
                    else
                    {
                        outvec[idx] = outTmp[d][f];
                    }
                    if (TOTAL)
                    {
                        totvec[idx] = outTmp[d][f] + advTmp[d][f] * advScale;
                    }
                }
            }
        });
    Nektar::LoopExecutionSetStreamID(0);
}

/// DiffusionCFEVolFluxKernelND() dispatched on the dimension, @p nDim + 2
/// variables.
template <bool APPEND, bool TOTAL, typename ExecSpace, typename EqnOfSParams,
          typename TData>
NEK_FORCE_INLINE static void DiffusionCFEVolFluxKernelLauncher(
    const EqnOfSParams EoS, const size_t npts, const unsigned int nDim,
    const unsigned int nvarComps, const size_t inStride,
    const size_t derivStride, const size_t outStride, const TData prandtl,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *inbase, const TData *qbase,
    TData *outbase, TData *totbase, const TData advScale,
    const unsigned int streamID)
{
    ASSERTL0(nvarComps == nDim + 2,
             "The viscous volume flux needs the dimension plus two variables.");
    switch (nDim)
    {
        case 1:
            DiffusionCFEVolFluxKernelND<1, APPEND, TOTAL, ExecSpace>(
                EoS, npts, inStride, derivStride, outStride, prandtl, muRef,
                isMuVariable, oneOverTStar, tRatioSutherland, inbase, qbase,
                outbase, totbase, advScale, streamID);
            break;
        case 2:
            DiffusionCFEVolFluxKernelND<2, APPEND, TOTAL, ExecSpace>(
                EoS, npts, inStride, derivStride, outStride, prandtl, muRef,
                isMuVariable, oneOverTStar, tRatioSutherland, inbase, qbase,
                outbase, totbase, advScale, streamID);
            break;
        case 3:
            DiffusionCFEVolFluxKernelND<3, APPEND, TOTAL, ExecSpace>(
                EoS, npts, inStride, derivStride, outStride, prandtl, muRef,
                isMuVariable, oneOverTStar, tRatioSutherland, inbase, qbase,
                outbase, totbase, advScale, streamID);
            break;
        default:
            NEKERROR(ErrorUtil::efatal,
                     "The viscous volume flux is for one, two and three "
                     "dimensions.");
    }
}

} // namespace Nektar::detail