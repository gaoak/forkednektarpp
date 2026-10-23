///////////////////////////////////////////////////////////////////////////////
//
// File: AverageSolverKernels.hpp
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

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

/**
 * @brief The average of the two states' fluxes through the trace normal, in
 * Cartesian components (as the legacy AverageSolver).
 */
template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct AverageSolverKernel
{
    /**
     * @param blksize  Points in the block; component c of the states, the
     *                 normal and the flux lies at offset `c * blksize`.
     * @param normal   The unit trace normal, @p NDIM components.
     */
    template <typename TScalar>
    NEK_HOSTDEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                          const size_t blksize,
                                          const TScalar *fwd,
                                          const TScalar *bwd,
                                          const TScalar *normal, TScalar *flux)
    {
        // Explicit vectorisation for AVX backend, vec_t =
        // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
        using vec_t =
            typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                                  TScalar>::type;
        constexpr unsigned int vec_width =
            (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                ? tinysimd::simd<TScalar>::width
                : 1;

        const vec_t oneHalf = 0.5;

        // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
        const size_t groupsize = blksize / vec_width;
        const auto fwdvec      = reinterpret_cast<const vec_t *>(fwd);
        const auto bwdvec      = reinterpret_cast<const vec_t *>(bwd);
        const auto nvec        = reinterpret_cast<const vec_t *>(normal);
        auto fluxvec           = reinterpret_cast<vec_t *>(flux);

        const vec_t rhoL = fwdvec[0];
        const vec_t rhoR = bwdvec[0];

        vec_t n[NDIM], mL[NDIM], mR[NDIM];
        vec_t unL = 0.0, unR = 0.0, qL2 = 0.0, qR2 = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            n[d]            = nvec[d * groupsize];
            mL[d]           = fwdvec[(1u + d) * groupsize];
            mR[d]           = bwdvec[(1u + d) * groupsize];
            const vec_t uLd = mL[d] / rhoL;
            const vec_t uRd = mR[d] / rhoR;
            unL += uLd * n[d];
            unR += uRd * n[d];
            qL2 += mL[d] * uLd;
            qR2 += mR[d] * uRd;
        }

        const vec_t EL = fwdvec[(1u + NDIM) * groupsize];
        const vec_t ER = bwdvec[(1u + NDIM) * groupsize];
        const vec_t eL = (EL - oneHalf * qL2) / rhoL;
        const vec_t eR = (ER - oneHalf * qR2) / rhoR;
        const vec_t pL = GetPressure(EoS, rhoL, eL);
        const vec_t pR = GetPressure(EoS, rhoR, eR);

        fluxvec[0] = oneHalf * (rhoL * unL + rhoR * unR);
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            fluxvec[(1u + d) * groupsize] =
                oneHalf * (mL[d] * unL + mR[d] * unR + (pL + pR) * n[d]);
        }
        fluxvec[(1u + NDIM) * groupsize] =
            oneHalf * (unL * (EL + pL) + unR * (ER + pR));
    }
};

} // namespace Nektar::detail
