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

#include "Operators/LoopExecution/LoopExecution.hpp"

// header should appear after loop execution.hpp
#include "EquationOfState/VariableConverters.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
NEK_DEVICE_INLINE static void GetViscousFluxBilinearFormKernel(
    const unsigned nDim, const unsigned fluxDirection,
    const unsigned derivDirection, const TData *inAverage, const TData *inJump,
    const TData mu, const TData gamma, const TData prandtl, TData *outarray)
{
    // Viscous bilinear form for one flux-direction / derivative-direction
    // pair in conservative variables.
    const unsigned nDimPlusOne  = nDim + 1;
    const unsigned fluxPlusOne  = fluxDirection + 1;
    const unsigned derivPlusOne = derivDirection + 1;

    const TData gammaOverPr         = gamma / prandtl;
    const TData oneMinusGammaOverPr = TData(1.0) - gammaOverPr;

    constexpr TData oneThird  = 1.0 / 3.0;
    constexpr TData twoThird  = 2.0 / 3.0;
    constexpr TData fourThird = 4.0 / 3.0;

    if (derivDirection == fluxDirection)
    {
        outarray[0] = TData(0.0);

        const TData invRho = TData(1.0) / inAverage[0];

        TData u[3]  = {TData(0.0), TData(0.0), TData(0.0)};
        TData u2[3] = {TData(0.0), TData(0.0), TData(0.0)};
        TData u2sum = TData(0.0);

        for (unsigned d = 0; d < nDim; ++d)
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
        for (unsigned d = 0; d < nDim; ++d)
        {
            const unsigned dPlusOne = d + 1;
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

        for (unsigned d = 0; d < nDim; ++d)
        {
            const unsigned dPlusOne = d + 1;
            u[d]                    = inAverage[dPlusOne] * invRho;
            outarray[dPlusOne]      = TData(0.0);
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
    const EqnOfSParams EoS, const unsigned npts, const unsigned nDim,
    const unsigned nvarComps, const unsigned inStride,
    const unsigned derivStride, const unsigned outStride, const TData prandtl,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *inbase, const TData *qbase,
    TData *outbase)
{
    ASSERTL0(nvarComps == nDim + 2,
             "number of componeents is not equal to the dimension plus two")

    // Assemble the volume viscous flux tensor pointwise from the conservative
    // state and its physical derivatives.
    Nektar::parallel_for<ExecSpace>(
        0u, npts, NEKTAR_LAMBDA(const size_t i) {
            TData inTmp[5]  = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                               TData(0.0)};
            TData qTmp[5]   = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                               TData(0.0)};
            TData outTmp[5] = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                               TData(0.0)};

            for (unsigned f = 0; f < nvarComps; ++f)
            {
                inTmp[f] = inbase[f * inStride + i];
            }

            const TData e =
                GetInternalEnergy(nDim, inTmp[0], inTmp + 1, inTmp[nDim + 1]);
            const TData temperature = GetTemperature(EoS, inTmp[0], e);

            const TData mu =
                GetDynamicViscosity(temperature, muRef, isMuVariable,
                                    oneOverTStar, tRatioSutherland);

            for (unsigned fluxDir = 0; fluxDir < nDim; ++fluxDir)
            {
                for (unsigned f = 0; f < nvarComps; ++f)
                {
                    outbase[(f * nDim + fluxDir) * outStride + i] = TData(0.0);
                }

                for (unsigned derivDir = 0; derivDir < nDim; ++derivDir)
                {
                    for (unsigned f = 0; f < nvarComps; ++f)
                    {
                        qTmp[f] =
                            qbase[(f * nDim + derivDir) * derivStride + i];
                    }

                    GetViscousFluxBilinearFormKernel(
                        nDim, fluxDir, derivDir, inTmp, qTmp, mu, EoS.gamma(),
                        prandtl, outTmp);

                    for (unsigned f = 0; f < nvarComps; ++f)
                    {
                        outbase[(f * nDim + fluxDir) * outStride + i] +=
                            outTmp[f];
                    }
                }
            }
        });
}

} // namespace Nektar::Operators::detail
