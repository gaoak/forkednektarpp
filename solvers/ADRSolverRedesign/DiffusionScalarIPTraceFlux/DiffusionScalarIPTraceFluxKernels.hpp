///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionScalarIPTraceFluxKernels.hpp
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
// Description: Scalar interior penalty diffusion kernels.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/LoopExecution/LoopExecution.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void DiffuseScalarTraceFluxKernel(
    const size_t npts, const unsigned int ndim, const unsigned int nvarComps,
    const size_t traceStride, const size_t derivStride,
    const TData penaltyCoeff, const TData *diffCoeff, const TData *normbase,
    const TData *bwdWeightAverBase, const TData *bwdWeightJumpBase,
    const TData *lengthRecipBase, const TData *penaltyFactorBase,
    const TData *fwdbase, const TData *bwdbase, const TData *derivfwdbase,
    const TData *derivbwdbase, TData *averbase, TData *jumpbase,
    TData *fluxbase)
{
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
            const auto diffCoeffMapPtr = GetDiffCoeffMapPtr(ndim);
            const vec_t bWeightAver    = bwdAverVec[i];
            const vec_t bWeightJump    = bwdJumpVec[i];

            const vec_t fWeightAver = vec_t(1.0) - bWeightAver;
            const vec_t fWeightJump = vec_t(2.0) - bWeightJump;

            const vec_t penalty =
                vec_t(penaltyCoeff) * penaltyVec[i] * lengthVec[i];

            // Effective diffusion strength in the trace normal direction:
            //
            //     normalDiffusionStrength = n^T D n
            //
            // For isotropic diffusion D = I and unit normals, this is 1.
            // For anisotropic/tensor diffusion, this scales the penalty
            // according to how strongly diffusion acts across the current face.
            vec_t normalDiffusionStrength = vec_t(0.0);
            for (unsigned int n0 = 0; n0 < ndim; ++n0)
            {
                const vec_t normal0 = normvec[n0 * traceVecStride + i];

                for (unsigned int n1 = 0; n1 < ndim; ++n1)
                {
                    normalDiffusionStrength +=
                        normal0 *
                        vec_t(diffCoeff[diffCoeffMapPtr[n0 * ndim + n1]]) *
                        normvec[n1 * traceVecStride + i];
                }
            }

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                const vec_t uFwd = fwdvec[f * traceVecStride + i];

                const vec_t uBwd = bwdvec[f * traceVecStride + i];

                const vec_t uAverage = fWeightAver * uFwd + bWeightAver * uBwd;

                // This is u_bwd - u_fwd.
                const vec_t bwdMinusFwd = (uAverage - uFwd) * fWeightJump +
                                          (uBwd - uAverage) * bWeightJump;

                avervec[f * traceVecStride + i] = uAverage;
                jumpvec[f * traceVecStride + i] = bwdMinusFwd;
                // normalAverageGradientFlux = {D grad(u)} · n
                // Since D is symmetric, this is equivalently: n^T D {grad(u)}
                vec_t normalAverageGradientFlux = vec_t(0.0);

                for (unsigned int d = 0; d < ndim; ++d)
                {

                    const vec_t avgDeriv =
                        fWeightAver *
                            derivFwdVec[(f * ndim + d) * derivVecStride + i] +
                        bWeightAver *
                            derivBwdVec[(f * ndim + d) * derivVecStride + i];

                    vec_t normalDiffCoeff = vec_t(0.0);
                    for (unsigned int n = 0; n < ndim; ++n)
                    {
                        const unsigned int coeffId =
                            diffCoeffMapPtr[n * ndim + d];
                        normalDiffCoeff += normvec[n * traceVecStride + i] *
                                           vec_t(diffCoeff[coeffId]);
                    }

                    normalAverageGradientFlux += normalDiffCoeff * avgDeriv;
                }

                // Combine flux avg(D grad u).n - C11 * (u_fwd - u_bwd) on the
                // fwd side.
                fluxvec[f * traceVecStride + i] =
                    normalAverageGradientFlux +
                    penalty * normalDiffusionStrength * bwdMinusFwd;
            }
        });
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void AddScalarSymmetricTraceFluxCoeffKernel(
    const size_t nelmt, const size_t nelmtWithPadding,
    const unsigned int nTraces, const unsigned int nLocTracePts,
    const unsigned int nCoeffs, const unsigned int nDim,
    const unsigned int nComps, const size_t coeffStride,
    const unsigned int coeffInterleaveWidth,
    const unsigned int targetTraceBlock, const size_t traceBlockCompSize,
    const TData *diffCoeff, const TData *traceJumpBase,
    const TData *traceNormalBase, const unsigned int *traceBlockIdBase,
    const size_t *traceOffsetBase, const unsigned int *nqOffsetBase,
    const unsigned int *orientationMapsBase,
    const size_t *orientationMapsOffsetBase, const TData *derivBaseTraceBase,
    TData *outBase)
{
    const size_t nWork =
        static_cast<size_t>(nelmtWithPadding) * nComps * nCoeffs;

    Nektar::parallel_for<ExecSpace>(
        0u, nWork, NEKTAR_LAMBDA(const size_t i) {
            const auto diffCoeffMapPtr = GetDiffCoeffMapPtr(nDim);
            const unsigned int coeff   = i % nCoeffs;
            const unsigned int comp    = (i / nCoeffs) % nComps;
            const size_t el            = i / (nCoeffs * nComps);

            TData val = TData(0.0);

            if (el < nelmt)
            {
                for (unsigned int t = 0; t < nTraces; ++t)
                {
                    const size_t key = t * nelmtWithPadding + el;

                    if (traceBlockIdBase[key] != targetTraceBlock)
                    {
                        continue;
                    }

                    const size_t orientBase =
                        orientationMapsOffsetBase[el * nTraces + t];
                    const unsigned int nqOffset = nqOffsetBase[t];
                    const unsigned int nqNext =
                        (t + 1 < nTraces) ? nqOffsetBase[t + 1] : nLocTracePts;

                    for (unsigned int q = 0; q < nqNext - nqOffset; ++q)
                    {
                        const unsigned int locTracePt = nqOffset + q;
                        const unsigned int orientedPoint =
                            orientationMapsBase[orientBase + q];

                        const size_t traceBlockPoint =
                            traceOffsetBase[key] + orientedPoint;
                        const size_t traceIdx =
                            comp * traceBlockCompSize + traceBlockPoint;
                        const TData jump = traceJumpBase[traceIdx];

                        for (unsigned int d = 0; d < nDim; ++d)
                        {
                            TData diffNormal = TData(0.0);
                            for (unsigned int n = 0; n < nDim; ++n)
                            {
                                diffNormal +=
                                    diffCoeff[diffCoeffMapPtr[d * nDim + n]] *
                                    traceNormalBase[n * traceBlockCompSize +
                                                    traceBlockPoint];
                            }

                            val += derivBaseTraceBase[(d * nCoeffs + coeff) *
                                                          nLocTracePts +
                                                      locTracePt] *
                                   diffNormal * jump;
                        }
                    }
                }

                val *= TData(-0.5);
            }

            const size_t elmtGroup = el / coeffInterleaveWidth;
            const size_t elmtLane  = el % coeffInterleaveWidth;
            const size_t coeffIdx = elmtGroup * nCoeffs * coeffInterleaveWidth +
                                    coeff * coeffInterleaveWidth + elmtLane;

            outBase[comp * coeffStride + coeffIdx] += val;
        });
}

} // namespace Nektar::Operators::detail
