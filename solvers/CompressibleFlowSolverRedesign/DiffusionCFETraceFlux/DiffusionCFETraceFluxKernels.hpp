///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionCFETraceFluxKernels.hpp
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
#include "Operators/Common/DataWarehouse/TraceDataWarehouse.hpp"

namespace Nektar::detail
{
template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void DiffuseTraceFluxKernel(
    const EqnOfSParams EoS, const size_t npts, const unsigned int ndim,
    const unsigned int nvarComps, const size_t traceStride,
    const size_t derivStride, const TData prandtl, const TData muRef,
    const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *normbase,
    const TData *bwdWeightAverBase, const TData *bwdWeightJumpBase,
    const TData *lengthRecipBase, const TData *penaltyFactorBase,
    const TData *fwdbase, const TData *bwdbase, const TData *derivfwdbase,
    const TData *derivbwdbase, TData *averbase, TData *jumpbase,
    TData *fluxbase)
{
    ASSERTL0(nvarComps == ndim + 2,
             "number of componeents is not equal to the dimension plus two")

    // Assemble the numerical flux on the trace pointwise:
    // average state, jump, penalty-corrected derivative trace, then normal
    // viscous flux.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    const size_t groupsize = npts / vec_width;
    const auto normvec     = reinterpret_cast<const vec_t *>(normbase);
    const auto bwdAverVec  = reinterpret_cast<const vec_t *>(bwdWeightAverBase);
    const auto bwdJumpVec  = reinterpret_cast<const vec_t *>(bwdWeightJumpBase);
    const auto lengthVec   = reinterpret_cast<const vec_t *>(lengthRecipBase);
    const auto penaltyVec  = reinterpret_cast<const vec_t *>(penaltyFactorBase);
    const auto fwdvec      = reinterpret_cast<const vec_t *>(fwdbase);
    const auto bwdvec      = reinterpret_cast<const vec_t *>(bwdbase);
    const auto derivFwdVec = reinterpret_cast<const vec_t *>(derivfwdbase);
    const auto derivBwdVec = reinterpret_cast<const vec_t *>(derivbwdbase);
    auto avervec           = reinterpret_cast<vec_t *>(averbase);
    auto jumpvec           = reinterpret_cast<vec_t *>(jumpbase);
    auto fluxvec           = reinterpret_cast<vec_t *>(fluxbase);
    const size_t traceVecStride = traceStride / vec_width;
    const size_t derivVecStride = derivStride / vec_width;

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

            const unsigned int nEngy = nvarComps - 1;
            const vec_t bWeightAver  = bwdAverVec[i];
            const vec_t bWeightJump  = bwdJumpVec[i];
            const vec_t fWeightAver  = vec_t(1.0) - bWeightAver;
            const vec_t fWeightJump  = vec_t(2.0) - bWeightJump;

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                fwdTmp[f] = fwdvec[f * traceVecStride + i];
                bwdTmp[f] = bwdvec[f * traceVecStride + i];
            }

            for (unsigned int f = 0; f < nEngy; ++f)
            {
                averTmp[f] = fWeightAver * fwdTmp[f] + bWeightAver * bwdTmp[f];
            }

            vec_t lInternal = vec_t(0.0);
            vec_t rInternal = vec_t(0.0);
            vec_t aInternal = vec_t(0.0);
            for (unsigned int d = 1; d < nEngy; ++d)
            {
                lInternal += fwdTmp[d] * fwdTmp[d];
                rInternal += bwdTmp[d] * bwdTmp[d];
                aInternal += averTmp[d] * averTmp[d];
            }

            lInternal      = fwdTmp[nEngy] - vec_t(0.5) * lInternal / fwdTmp[0];
            rInternal      = bwdTmp[nEngy] - vec_t(0.5) * rInternal / bwdTmp[0];
            averTmp[nEngy] = fWeightAver * lInternal + bWeightAver * rInternal +
                             vec_t(0.5) * aInternal / averTmp[0];

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                jumpTmp[f] = (averTmp[f] - fwdTmp[f]) * fWeightJump +
                             (bwdTmp[f] - averTmp[f]) * bWeightJump;

                avervec[f * traceVecStride + i] = averTmp[f];
                jumpvec[f * traceVecStride + i] = jumpTmp[f];
                fluxvec[f * traceVecStride + i] = vec_t(0.0);
            }

            const vec_t penalty = penaltyVec[i] * lengthVec[i];

            const vec_t e = GetInternalEnergy(ndim, averTmp[0], averTmp + 1,
                                              averTmp[ndim + 1]);
            const vec_t temperature = GetTemperature(EoS, averTmp[0], e);

            const vec_t mu = GetDynamicViscosity(
                temperature, vec_t(muRef), isMuVariable, vec_t(oneOverTStar),
                vec_t(tRatioSutherland));

            for (unsigned int derivDir = 0; derivDir < ndim; ++derivDir)
            {
                for (unsigned int f = 0; f < nvarComps; ++f)
                {
                    // Legacy stores 0.5*dq^+ and 0.5*dq^-, then adds the
                    // two work arrays after the exchange stage. In this
                    // serial field path the extraction still gives raw
                    // derivative traces, so the kernel applies the same
                    // half-scaling before summing.
                    qTmp[f] = vec_t(0.5) * (derivFwdVec[(f * ndim + derivDir) *
                                                            derivVecStride +
                                                        i] +
                                            derivBwdVec[(f * ndim + derivDir) *
                                                            derivVecStride +
                                                        i]) +
                              normvec[derivDir * traceVecStride + i] *
                                  jumpTmp[f] * penalty;
                }

                for (unsigned int fluxDir = 0; fluxDir < ndim; ++fluxDir)
                {
                    // Contract the viscous flux tensor with the interface
                    // normal to obtain the scalar trace contribution.
                    GetViscousFluxBilinearFormKernel(
                        ndim, fluxDir, derivDir, averTmp, qTmp, mu,
                        vec_t(EoS.gamma()), vec_t(prandtl), outTmp);

                    for (unsigned int f = 0; f < nvarComps; ++f)
                    {
                        fluxvec[f * traceVecStride + i] +=
                            normvec[fluxDir * traceVecStride + i] * outTmp[f];
                    }
                }
            }
        });
}

} // namespace Nektar::detail
