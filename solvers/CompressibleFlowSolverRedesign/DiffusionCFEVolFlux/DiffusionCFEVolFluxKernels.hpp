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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

// header should appear after loop execution.hpp
#include "EquationOfState/VariableConverters.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
NEK_DEVICE_INLINE static void GetViscousFluxBilinearFormKernel(
    const unsigned int nDim, const unsigned int fluxDirection,
    const unsigned int derivDirection, const TData *inAverage,
    const TData *inJump, const TData mu, const TData gamma, const TData prandtl,
    TData *outarray)
{
    // Viscous bilinear form for one flux-direction / derivative-direction
    // pair in conservative variables.
    const unsigned int nDimPlusOne  = nDim + 1;
    const unsigned int fluxPlusOne  = fluxDirection + 1;
    const unsigned int derivPlusOne = derivDirection + 1;

    const TData gammaOverPr         = gamma / prandtl;
    const TData oneMinusGammaOverPr = TData(1.0) - gammaOverPr;

    const TData oneThird  = TData(1.0) / TData(3.0);
    const TData twoThird  = TData(2.0) / TData(3.0);
    const TData fourThird = TData(4.0) / TData(3.0);

    if (derivDirection == fluxDirection)
    {
        outarray[0] = TData(0.0);

        const TData invRho = TData(1.0) / inAverage[0];

        TData u[3]  = {TData(0.0), TData(0.0), TData(0.0)};
        TData u2[3] = {TData(0.0), TData(0.0), TData(0.0)};
        TData u2sum = TData(0.0);

        for (unsigned int d = 0; d < nDim; ++d)
        {
            u[d]  = inAverage[d + 1] * invRho;
            u2[d] = u[d] * u[d];
            u2sum += u2[d];
        }

        TData eMinusU2 = inAverage[nDimPlusOne] * invRho - u2sum;
        TData nu       = mu * invRho;

        TData tmp1 = (TData(oneThird) * u2[fluxDirection] + u2sum +
                      gammaOverPr * eMinusU2) *
                     inJump[0];
        TData tmp2 = gammaOverPr * inJump[nDimPlusOne] - tmp1;

        TData outTmpE = TData(0.0);
        for (unsigned int d = 0; d < nDim; ++d)
        {
            const unsigned int dPlusOne = d + 1;
            TData outTmpD = (inJump[dPlusOne] - u[d] * inJump[0]) * nu;

            outTmpE += oneMinusGammaOverPr * u[d] * inJump[dPlusOne];

            if (d == fluxDirection)
            {
                outTmpD *= TData(fourThird);
                outTmpE +=
                    TData(oneThird) * u[fluxDirection] * inJump[fluxPlusOne];
            }

            outarray[dPlusOne] = outTmpD;
        }

        outarray[nDimPlusOne] = (outTmpE + tmp2) * nu;
    }
    else
    {
        outarray[0] = TData(0.0);

        const TData invRho = TData(1.0) / inAverage[0];

        TData u[3] = {TData(0.0), TData(0.0), TData(0.0)};

        for (unsigned int d = 0; d < nDim; ++d)
        {
            const unsigned int dPlusOne = d + 1;
            u[d]                        = inAverage[dPlusOne] * invRho;
            outarray[dPlusOne]          = TData(0.0);
        }

        TData nu = mu * invRho;

        TData tmp1 = (u[derivDirection] * inJump[0] - inJump[derivPlusOne]) *
                     TData(twoThird);
        outarray[fluxPlusOne] = nu * tmp1;

        tmp1 = inJump[fluxPlusOne] - u[fluxDirection] * inJump[0];
        outarray[derivPlusOne] = nu * tmp1;

        tmp1 =
            TData(oneThird) * u[fluxDirection] * u[derivDirection] * inJump[0];
        TData tmp2 = TData(twoThird) * u[fluxDirection] * inJump[derivPlusOne];
        outarray[nDimPlusOne] =
            nu * (u[derivDirection] * inJump[fluxPlusOne] - (tmp1 + tmp2));
    }
}

template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void DiffusionCFEVolFluxKernel(
    const EqnOfSParams EoS, const size_t npts, const unsigned int nDim,
    const unsigned int nvarComps, const size_t inStride,
    const size_t derivStride, const size_t outStride, const TData prandtl,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *inbase, const TData *qbase,
    TData *outbase, const unsigned int streamID)
{
    ASSERTL0(nvarComps == nDim + 2,
             "number of componeents is not equal to the dimension plus two")

    // Assemble the volume viscous flux tensor pointwise from the conservative
    // state and its physical derivatives.
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
    const size_t inVecStride    = inStride / vec_width;
    const size_t derivVecStride = derivStride / vec_width;
    const size_t outVecStride   = outStride / vec_width;

    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            vec_t inTmp[5]  = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                               vec_t(0.0)};
            vec_t qTmp[5]   = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                               vec_t(0.0)};
            vec_t outTmp[5] = {vec_t(0.0), vec_t(0.0), vec_t(0.0), vec_t(0.0),
                               vec_t(0.0)};

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                inTmp[f] = invec[f * inVecStride + i];
            }

            const vec_t e =
                GetInternalEnergy(nDim, inTmp[0], inTmp + 1, inTmp[nDim + 1]);
            const vec_t temperature = GetTemperature(EoS, inTmp[0], e);

            const vec_t mu = GetDynamicViscosity(
                temperature, vec_t(muRef), isMuVariable, vec_t(oneOverTStar),
                vec_t(tRatioSutherland));

            for (unsigned int fluxDir = 0; fluxDir < nDim; ++fluxDir)
            {
                for (unsigned int f = 0; f < nvarComps; ++f)
                {
                    outvec[(f * nDim + fluxDir) * outVecStride + i] =
                        vec_t(0.0);
                }

                for (unsigned int derivDir = 0; derivDir < nDim; ++derivDir)
                {
                    for (unsigned int f = 0; f < nvarComps; ++f)
                    {
                        qTmp[f] =
                            qvec[(f * nDim + derivDir) * derivVecStride + i];
                    }

                    GetViscousFluxBilinearFormKernel(
                        nDim, fluxDir, derivDir, inTmp, qTmp, mu,
                        vec_t(EoS.gamma()), vec_t(prandtl), outTmp);

                    for (unsigned int f = 0; f < nvarComps; ++f)
                    {
                        outvec[(f * nDim + fluxDir) * outVecStride + i] +=
                            outTmp[f];
                    }
                }
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::Operators::detail
