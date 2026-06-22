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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename EqnOfStParams, unsigned int NDIM>
struct AverageSolverKernel
{
    template <typename TScalar>
    NEK_DEVICE_INLINE void operator()(const EqnOfStParams &EoS,
                                      const size_t blksize, const TScalar *fwd,
                                      const TScalar *bwd, TScalar *flux)
    {
        using std::abs;
        using std::sqrt;

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
        auto fluxvec           = reinterpret_cast<vec_t *>(flux);

        // Density
        const vec_t rhoL = fwdvec[0];
        const vec_t rhoR = bwdvec[0];

        // Velocities and kinetic energy terms
        vec_t uL[NDIM];
        vec_t uR[NDIM];
        vec_t qL2 = 0.0;
        vec_t qR2 = 0.0;
#pragma unroll
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const vec_t rhouL = fwdvec[(1u + d) * groupsize];
            const vec_t rhouR = bwdvec[(1u + d) * groupsize];

            uL[d] = rhouL / rhoL;
            uR[d] = rhouR / rhoR;

            qL2 += rhouL * uL[d];
            qR2 += rhouR * uR[d];
        }

        // Internal energy per unit mass
        const vec_t EL = fwdvec[(1u + NDIM) * groupsize];
        const vec_t ER = bwdvec[(1u + NDIM) * groupsize];
        const vec_t eL = (EL - oneHalf * qL2) / rhoL;
        const vec_t eR = (ER - oneHalf * qR2) / rhoR;

        // Pressure
        const vec_t pL = GetPressure(EoS, rhoL, eL);
        const vec_t pR = GetPressure(EoS, rhoR, eR);

        // Average Riemann fluxes
        fluxvec[0] = oneHalf * (rhoR * uR[0] + rhoL * uL[0]);
        fluxvec[1u * groupsize] =
            0.5 * ((rhoR * uR[0] * uR[0] + pR) + (rhoL * uL[0] * uL[0] + pL));
#pragma unroll
        for (unsigned int d = 1; d < NDIM; ++d)
        {
            fluxvec[(1u + d) * groupsize] =
                oneHalf * (rhoR * uR[0] * uR[d] + rhoL * uL[0] * uL[d]);
        }
        fluxvec[(1u + NDIM) * groupsize] =
            oneHalf * (uR[0] * (ER + pR) + uL[0] * (EL + pL));
    }
};

} // namespace Nektar::Operators::detail
