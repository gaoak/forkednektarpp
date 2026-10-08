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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::detail
{

/**
 * @brief Interior-penalty numerical flux for scalar diffusion, evaluated over
 * one padded trace block.
 *
 * For each trace point and field component the flux is
 *
 * \f[
 *   \hat{f} = \mathbf{n}^{T} D \{\nabla u\}
 *           + C\,(\mathbf{n}^{T} D \mathbf{n})\,[u]
 * \f]
 *
 * where \f$\{\cdot\}\f$ is the average across the trace, \f$[u]\f$ the jump,
 * and \f$C\f$ the product of @p penaltyCoeff with the per-point geometric
 * factor @p penaltyFactorBase. Resolving the diffusion tensor in the normal
 * direction in both terms is what makes the penalty scale correctly for
 * anisotropic \f$D\f$.
 *
 * The average is the half-and-half \f$\{w\} = (w^+ + w^-)/2\f$ of Zhen-Guo Yan,
 * *Efficient implicit spectral/hp element DG techniques for compressible
 * flows*, PhD thesis, Imperial College London, 2021, Ch. 2, Eq. (2.17), and it
 * is used on boundary traces as well as interior ones. @p IsInterior therefore
 * affects only how the jump is distributed:
 *
 * | Quantity          | `IsInterior == true`      | `IsInterior == false` |
 * |-------------------|---------------------------|-----------------------|
 * | average \f$\{u\}\f$ | \f$(u_f + u_b)/2\f$     | \f$(u_f + u_b)/2\f$   |
 * | jump \f$[u]\f$      | \f$u_b - u_f\f$         | \f$u_b - u_f\f$       |
 * | average gradient    | \f$(g_f + g_b)/2\f$     | \f$(g_f + g_b)/2\f$   |
 *
 * A single averaging rule only imposes the right condition if the backward
 * state on a boundary is a *ghost* rather than the boundary value itself: to
 * impose \f$u = g\f$ the caller must pass \f$u_b = 2g - u_f\f$, whose average
 * with the interior is \f$g\f$ and whose jump is \f$2(g - u_f)\f$. That is the
 * caller's responsibility - see DiffusionScalarIPTraceFluxOpImpl.
 *
 * Legacy did this the other way round, passing \f$g\f$ unmodified and giving
 * the boundary its own weights (average \f$u_b\f$, jump \f$2(u_b - u_f)\f$).
 * The two are algebraically identical. Carrying the condition in the state
 * keeps one rule in the kernel, which matters once boundary conditions differ
 * in *which* exterior state they build - a reflective wall already supplies a
 * ghost and must not be reflected again.
 *
 * A Neumann component has no imposed value to reflect about and is passed
 * through with \f$u_b = u_f\f$; the caller likewise passes the interior
 * gradient as both sides where no gradient condition applies, which leaves the
 * gradient term equal to the interior value.
 *
 * Data layout: all arrays are component-major with @p traceStride between
 * consecutive components of the same block, and the gradient arrays carry
 * `ndim * nvarComps` components ordered component-fastest-outer, direction
 * inner.
 *
 * @tparam ExecSpace   Execution space; selects the SIMD width used to walk the
 *                     block.
 * @tparam IsInterior  True for interior traces, false for boundary traces.
 * @tparam TData       Floating-point representation.
 *
 * @param npts              - Padded number of points in the block. Must be an
 *                            exact multiple of the vector width.
 * @param ndim              - Coordinate dimension.
 * @param nvarComps         - Number of field components.
 * @param traceStride       - Stride between components within the block.
 * @param penaltyCoeff      - User penalty coefficient, `IPPenaltyCoeff`.
 * @param diffCoeff         - Upper triangle of the symmetric diffusion tensor.
 * @param normbase          - Trace normals, @p ndim components.
 * @param penaltyFactorBase - Per-point geometric penalty factor.
 * @param fwdbase           - Forward-side solution values.
 * @param bwdbase           - Backward-side solution values.
 * @param derivfwdbase      - Forward-side solution gradient.
 * @param derivbwdbase      - Backward-side solution gradient.
 * @param fluxbase          - Output flux, @p nvarComps components.
 *
 */
template <typename ExecSpace, bool IsInterior, typename TData>
NEK_FORCE_INLINE static void DiffuseScalarTraceFluxKernel(
    const size_t npts, const unsigned int ndim, const unsigned int nvarComps,
    const size_t traceStride, const TData penaltyCoeff, const TData *diffCoeff,
    const TData *normbase, const TData *penaltyFactorBase, const TData *fwdbase,
    const TData *bwdbase, const TData *derivfwdbase, const TData *derivbwdbase,
    TData *fluxbase)
{
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TData>::width
            : 1;

    ASSERTL1(npts % vec_width == 0,
             "Routine assumes blksize is exact integer multiple of vec_width");

    const size_t groupsize = npts / vec_width;
    const auto normvec     = reinterpret_cast<const vec_t *>(normbase);
    const auto penaltyVec  = reinterpret_cast<const vec_t *>(penaltyFactorBase);
    const auto fwdvec      = reinterpret_cast<const vec_t *>(fwdbase);
    const auto bwdvec      = reinterpret_cast<const vec_t *>(bwdbase);
    const auto derivFwdVec = reinterpret_cast<const vec_t *>(derivfwdbase);
    const auto derivBwdVec = reinterpret_cast<const vec_t *>(derivbwdbase);
    auto fluxvec           = reinterpret_cast<vec_t *>(fluxbase);

    const size_t traceVecStride = traceStride / vec_width;

    // One averaging rule everywhere - see the note above on the exterior
    // state. Legacy instead collapsed the boundary average onto the backward
    // side (bWeightAver 1, fWeightAver 0); the two agree once the caller
    // supplies a reflected ghost.
    const TData bWeightAver = TData(0.5);
    const TData fWeightAver = TData(0.5);
    const TData bWeightJump = (IsInterior) ? TData(1.0) : TData(0.0);
    const TData fWeightJump = (IsInterior) ? TData(1.0) : TData(2.0);

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            const vec_t penalty = vec_t(penaltyCoeff) * penaltyVec[i];

            // Effective diffusion strength in the trace normal
            // direction:
            //
            //     normalDiffusionStrength = n^T D n
            //
            // For isotropic diffusion D = I and unit normals, this
            // is 1. For anisotropic/tensor diffusion, this scales the
            // penalty according to how strongly diffusion acts across
            // the current face.
            vec_t normalDiffusionStrength = vec_t(0.0);
            for (unsigned int n0 = 0; n0 < ndim; ++n0)
            {
                const vec_t normal0 = normvec[n0 * traceVecStride + i];

                for (unsigned int n1 = 0; n1 < ndim; ++n1)
                {
                    normalDiffusionStrength +=
                        normal0 *
                        vec_t(diffCoeff[LibUtilities::GetDiffCoeffMap(
                            ndim, n0 * ndim + n1)]) *
                        normvec[n1 * traceVecStride + i];
                }
            }

            for (unsigned int f = 0; f < nvarComps; ++f)
            {
                const vec_t uFwd     = fwdvec[f * traceVecStride + i];
                const vec_t uBwd     = bwdvec[f * traceVecStride + i];
                const vec_t uAverage = fWeightAver * uFwd + bWeightAver * uBwd;

                // This is u_bwd - u_fwd.
                const vec_t bwdMinusFwd = (uAverage - uFwd) * fWeightJump +
                                          (uBwd - uAverage) * bWeightJump;

                // normalAverageGradientFlux = {D grad(u)} · n
                // Since D is symmetric, this is equivalently: n^T D
                // {grad(u)}
                vec_t normalAverageGradientFlux = vec_t(0.0);

                for (unsigned int d = 0; d < ndim; ++d)
                {
                    const vec_t avgDeriv =
                        fWeightAver *
                            derivFwdVec[(f * ndim + d) * traceVecStride + i] +
                        bWeightAver *
                            derivBwdVec[(f * ndim + d) * traceVecStride + i];

                    vec_t normalDiffCoeff = vec_t(0.0);
                    for (unsigned int n = 0; n < ndim; ++n)
                    {
                        const unsigned int coeffId =
                            LibUtilities::GetDiffCoeffMap(ndim, n * ndim + d);
                        normalDiffCoeff += normvec[n * traceVecStride + i] *
                                           vec_t(diffCoeff[coeffId]);
                    }

                    normalAverageGradientFlux += normalDiffCoeff * avgDeriv;
                }

                // Combine flux avg(D grad u).n - C11 * (u_fwd -
                // u_bwd) on the fwd side.
                fluxvec[f * traceVecStride + i] =
                    normalAverageGradientFlux +
                    penalty * normalDiffusionStrength * bwdMinusFwd;
            }
        });

    Nektar::LoopExecutionSetStreamID(0);
}

} // namespace Nektar::detail
