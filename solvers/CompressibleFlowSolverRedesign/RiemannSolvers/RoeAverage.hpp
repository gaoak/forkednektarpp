///////////////////////////////////////////////////////////////////////////////
//
// File: RoeAverage.hpp
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
// Description: Roe averages of the two states of a Riemann problem.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::detail
{

/**
 * @brief The two sides' velocities, internal energies, pressures and
 * enthalpies at one point, and the Roe averages of the velocity, the
 * enthalpy and the sound speed: what every form of the Roe flux starts from.
 */
template <unsigned int NDIM, typename vec_t> struct RoeAverageState
{
    vec_t invRhoL, invRhoR;
    vec_t uL[NDIM], uR[NDIM];
    vec_t EL, ER, eL, eR, pL, pR, hL, hR;
    vec_t srL, srR, srLR;
    vec_t uRoe[NDIM];
    vec_t URoe2, hRoe, cRoe;
};

/**
 * @brief Fill @p s from the two conserved states @p l and @p r, whose
 * components lie @p stride vectors apart.
 */
template <unsigned int NDIM, typename EqnOfStParams, typename vec_t>
NEK_HOSTDEVICE_INLINE static void RoeAverages(const EqnOfStParams &EoS,
                                              const vec_t *l, const vec_t *r,
                                              const size_t stride,
                                              RoeAverageState<NDIM, vec_t> &s)
{
    using std::sqrt;
    const vec_t half = vec_t(0.5);

    const vec_t rhoL = l[0], rhoR = r[0];

    // One reciprocal per density and per Roe denominator in place of a
    // division at every use.
    s.invRhoL = vec_t(1.0) / rhoL;
    s.invRhoR = vec_t(1.0) / rhoR;
    vec_t qL2 = vec_t(0.0), qR2 = vec_t(0.0);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        const vec_t rhouL = l[(1u + d) * stride];
        const vec_t rhouR = r[(1u + d) * stride];
        s.uL[d]           = rhouL * s.invRhoL;
        s.uR[d]           = rhouR * s.invRhoR;
        qL2 += rhouL * s.uL[d];
        qR2 += rhouR * s.uR[d];
    }
    s.EL = l[(1u + NDIM) * stride];
    s.ER = r[(1u + NDIM) * stride];
    s.eL = (s.EL - half * qL2) * s.invRhoL;
    s.eR = (s.ER - half * qR2) * s.invRhoR;
    s.pL = GetPressure(EoS, rhoL, s.eL);
    s.pR = GetPressure(EoS, rhoR, s.eR);
    s.hL = (s.EL + s.pL) * s.invRhoL;
    s.hR = (s.ER + s.pR) * s.invRhoR;

    s.srL               = sqrt(rhoL);
    s.srR               = sqrt(rhoR);
    s.srLR              = s.srL + s.srR;
    const vec_t invSrLR = vec_t(1.0) / s.srLR;
    s.URoe2             = vec_t(0.0);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        s.uRoe[d] = (s.srL * s.uL[d] + s.srR * s.uR[d]) * invSrLR;
        s.URoe2 += s.uRoe[d] * s.uRoe[d];
    }
    s.hRoe = (s.srL * s.hL + s.srR * s.hR) * invSrLR;
    s.cRoe = GetRoeSoundSpeed(EoS, rhoL, s.pL, s.eL, s.hL, s.srL, rhoR, s.pR,
                              s.eR, s.hR, s.srR, s.hRoe, s.URoe2, s.srLR);
}

} // namespace Nektar::detail
