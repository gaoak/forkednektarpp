///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionIPKernels.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{
template <typename TData>
NEK_DEVICE_INLINE static TData DiffusionIPGetInternalEnergy(
    const unsigned int ndim, const TData *in)
{
    // Internal energy per unit mass from the conservative state.
    TData invRho = TData(1.0) / in[0];
    TData dynE   = TData(0.0);

    for (unsigned int d = 0; d < ndim; ++d)
    {
        dynE += in[d + 1] * in[d + 1];
    }

    dynE *= TData(0.5) * invRho;
    return (in[ndim + 1] - dynE) * invRho;
}

template <typename TData>
NEK_DEVICE_INLINE static TData DiffusionIPGetTemperature(
    const unsigned int ndim, const TData *in, const TData gamma,
    const TData gasConstant)
{
    // Ideal-gas temperature from conservative variables.
    return DiffusionIPGetInternalEnergy(ndim, in) * (gamma - TData(1.0)) /
           gasConstant;
}

template <typename TData>
NEK_DEVICE_INLINE static TData DiffusionIPGetDynamicViscosity(
    const TData temperature, const TData muRef, const bool isMuVariable,
    const TData oneOverTStar, const TData tRatioSutherland)
{
    // Constant-viscosity or Sutherland-law viscosity model.
    if (!isMuVariable)
    {
        return muRef;
    }

    const TData onePlusC = TData(1.0) + tRatioSutherland;
    const TData ratio    = temperature * oneOverTStar;

    return muRef * ratio * std::sqrt(ratio) * onePlusC /
           (ratio + tRatioSutherland);
}

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

    constexpr double oneThird  = 1.0 / 3.0;
    constexpr double twoThird  = 2.0 / 3.0;
    constexpr double fourThird = 4.0 / 3.0;

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

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void DiffuseVolumeFluxKernel(
    const unsigned int npts, const unsigned int ndim,
    const unsigned int nvarComps, const unsigned int inStride,
    const unsigned int derivStride, const unsigned int outStride,
    const TData gamma, const TData prandtl, const TData gasConstant,
    const TData muRef, const bool isMuVariable, const TData oneOverTStar,
    const TData tRatioSutherland, const TData *inbase, const TData *qbase,
    TData *outbase)
{
    if (nvarComps != ndim + 2)
    {
        return;
    }

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

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                inTmp[f] = inbase[f * inStride + i];
            }

            const TData temperature =
                DiffusionIPGetTemperature(ndim, inTmp, gamma, gasConstant);
            const TData mu =
                DiffusionIPGetDynamicViscosity(temperature, muRef, isMuVariable,
                                               oneOverTStar, tRatioSutherland);

            for (unsigned int fluxDir = 0; fluxDir < ndim; ++fluxDir)
            {
                for (unsigned int f = 0; f < nvarComps; ++f)
                {
                    outbase[(f * ndim + fluxDir) * outStride + i] = TData(0.0);
                }

                for (unsigned int derivDir = 0; derivDir < ndim; ++derivDir)
                {
                    for (unsigned int f = 0; f < nvarComps; ++f)
                    {
                        qTmp[f] =
                            qbase[(f * ndim + derivDir) * derivStride + i];
                    }

                    GetViscousFluxBilinearFormKernel(ndim, fluxDir, derivDir,
                                                     inTmp, qTmp, mu, gamma,
                                                     prandtl, outTmp);

                    for (unsigned int f = 0; f < nvarComps; ++f)
                    {
                        outbase[(f * ndim + fluxDir) * outStride + i] +=
                            outTmp[f];
                    }
                }
            }
        });
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void DiffuseTraceFluxKernel(
    const unsigned int npts, const unsigned int ndim,
    const unsigned int nvarComps, const unsigned int traceStride,
    const unsigned int derivStride, const TData gamma, const TData prandtl,
    const TData gasConstant, const TData muRef, const bool isMuVariable,
    const TData oneOverTStar, const TData tRatioSutherland,
    const TData *normbase, const double *bwdWeightAverBase,
    const double *bwdWeightJumpBase, const double *lengthRecipBase,
    const double *penaltyFactorBase, const TData *fwdbase, const TData *bwdbase,
    const TData *derivfwdbase, const TData *derivbwdbase, TData *averbase,
    TData *jumpbase, TData *fluxbase)
{
    if (nvarComps != ndim + 2)
    {
        return;
    }

    // Assemble the numerical flux on the trace pointwise:
    // average state, jump, penalty-corrected derivative trace, then normal
    // viscous flux.
    Nektar::parallel_for<ExecSpace>(
        0u, npts, NEKTAR_LAMBDA(const size_t i) {
            TData fwdTmp[5]  = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                                TData(0.0)};
            TData bwdTmp[5]  = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                                TData(0.0)};
            TData averTmp[5] = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                                TData(0.0)};
            TData jumpTmp[5] = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                                TData(0.0)};
            TData qTmp[5]    = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                                TData(0.0)};
            TData outTmp[5]  = {TData(0.0), TData(0.0), TData(0.0), TData(0.0),
                                TData(0.0)};

            const unsigned int nEngy = nvarComps - 1;
            const TData bWeightAver  = static_cast<TData>(bwdWeightAverBase[i]);
            const TData bWeightJump  = static_cast<TData>(bwdWeightJumpBase[i]);
            const TData fWeightAver  = TData(1.0) - bWeightAver;
            const TData fWeightJump  = TData(2.0) - bWeightJump;

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                fwdTmp[f] = fwdbase[f * traceStride + i];
                bwdTmp[f] = bwdbase[f * traceStride + i];
            }

            for (unsigned int f = 0; f < nEngy; ++f)
            {
                averTmp[f] = fWeightAver * fwdTmp[f] + bWeightAver * bwdTmp[f];
            }

            TData lInternal = TData(0.0);
            TData rInternal = TData(0.0);
            TData aInternal = TData(0.0);
            for (unsigned int d = 1; d < nEngy; ++d)
            {
                lInternal += fwdTmp[d] * fwdTmp[d];
                rInternal += bwdTmp[d] * bwdTmp[d];
                aInternal += averTmp[d] * averTmp[d];
            }

            lInternal      = fwdTmp[nEngy] - TData(0.5) * lInternal / fwdTmp[0];
            rInternal      = bwdTmp[nEngy] - TData(0.5) * rInternal / bwdTmp[0];
            averTmp[nEngy] = fWeightAver * lInternal + bWeightAver * rInternal +
                             TData(0.5) * aInternal / averTmp[0];

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                jumpTmp[f] = (averTmp[f] - fwdTmp[f]) * fWeightJump +
                             (bwdTmp[f] - averTmp[f]) * bWeightJump;

                averbase[f * traceStride + i] = averTmp[f];
                jumpbase[f * traceStride + i] = jumpTmp[f];
                fluxbase[f * traceStride + i] = TData(0.0);
            }

            const TData penalty =
                static_cast<TData>(penaltyFactorBase[i] * lengthRecipBase[i]);

            const TData temperature =
                DiffusionIPGetTemperature(ndim, averTmp, gamma, gasConstant);
            const TData mu =
                DiffusionIPGetDynamicViscosity(temperature, muRef, isMuVariable,
                                               oneOverTStar, tRatioSutherland);

            for (unsigned int derivDir = 0; derivDir < ndim; ++derivDir)
            {
                for (unsigned int f = 0; f < nvarComps; ++f)
                {
                    // Legacy stores 0.5*dq^+ and 0.5*dq^-, then adds the
                    // two work arrays after the exchange stage. In this
                    // serial field path the extraction still gives raw
                    // derivative traces, so the kernel applies the same
                    // half-scaling before summing.
                    qTmp[f] =
                        TData(0.5) *
                            (derivfwdbase[(f * ndim + derivDir) * derivStride +
                                          i] +
                             derivbwdbase[(f * ndim + derivDir) * derivStride +
                                          i]) +
                        normbase[derivDir * traceStride + i] * jumpTmp[f] *
                            penalty;
                }

                for (unsigned int fluxDir = 0; fluxDir < ndim; ++fluxDir)
                {
                    // Contract the viscous flux tensor with the interface
                    // normal to obtain the scalar trace contribution.
                    GetViscousFluxBilinearFormKernel(ndim, fluxDir, derivDir,
                                                     averTmp, qTmp, mu, gamma,
                                                     prandtl, outTmp);

                    for (unsigned int f = 0; f < nvarComps; ++f)
                    {
                        fluxbase[f * traceStride + i] +=
                            normbase[fluxDir * traceStride + i] * outTmp[f];
                    }
                }
            }
        });
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CopyBwdDerivTraceFromFwdOnBndKernel(
    const size_t nBndPts, const unsigned int nComps,
    const unsigned int compStride, const size_t *bndTraceOffset,
    const TData *fwdbase, TData *bwdbase)
{
    // Derivative traces do not have physical boundary data of their own. On
    // physical boundaries legacy DiffusionIP uses dq^- = dq^+ before the IP
    // penalty term is added, while periodic traces are handled separately.
    Nektar::parallel_for<ExecSpace>(
        0u, nBndPts, NEKTAR_LAMBDA(const size_t i) {
            const size_t offset = bndTraceOffset[i];
            for (unsigned int c = 0; c < nComps; ++c)
            {
                bwdbase[c * compStride + offset] =
                    fwdbase[c * compStride + offset];
            }
        });
}

} // namespace Nektar::Operators::detail
