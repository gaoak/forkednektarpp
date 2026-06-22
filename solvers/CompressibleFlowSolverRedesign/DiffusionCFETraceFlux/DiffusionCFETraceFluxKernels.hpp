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

namespace Nektar::Operators::detail
{
template <typename ExecSpace, typename EqnOfSParams, typename TData>
NEK_FORCE_INLINE static void DiffuseTraceFluxKernel(
    const EqnOfSParams EoS, const unsigned int npts, const unsigned int ndim,
    const unsigned int nvarComps, const unsigned int traceStride,
    const unsigned int derivStride, const TData prandtl, const TData muRef,
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

            const TData e = GetInternalEnergy(ndim, averTmp[0], averTmp + 1,
                                              averTmp[ndim + 1]);
            const TData temperature = GetTemperature(EoS, averTmp[0], e);

            const TData mu =
                GetDynamicViscosity(temperature, muRef, isMuVariable,
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
                    GetViscousFluxBilinearFormKernel(
                        ndim, fluxDir, derivDir, averTmp, qTmp, mu, EoS.gamma(),
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

} // namespace Nektar::Operators::detail
