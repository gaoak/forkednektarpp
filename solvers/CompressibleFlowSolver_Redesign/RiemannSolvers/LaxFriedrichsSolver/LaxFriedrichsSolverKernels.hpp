///////////////////////////////////////////////////////////////////////////////
//
// File: LaxFriedrichsSolverKernels.hpp
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
#include <LibUtilities/BasicUtils/NekInline.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

namespace Nektar::Operators::detail
{

template <typename TData>
NEK_DEVICE_INLINE TData GetPressure(const TData &rho, const TData &e)
{
    // Ideal gas law: P = (gamma - 1) * rho * e
    const TData gamma = 1.4; // Specific heat ratio for air
    return (gamma - 1) * rho * e;
}

template <typename TData>
NEK_DEVICE_INLINE TData GetRoeSoundSpeed(
    [[maybe_unused]] const TData &rhoL, [[maybe_unused]] const TData &pL,
    [[maybe_unused]] const TData &eL, [[maybe_unused]] const TData &HL,
    [[maybe_unused]] const TData &srL, [[maybe_unused]] const TData &rhoR,
    [[maybe_unused]] const TData &pR, [[maybe_unused]] const TData &eR,
    [[maybe_unused]] const TData &HR, [[maybe_unused]] const TData &srR,
    const TData &HRoe, const TData &URoe2, [[maybe_unused]] const TData &srLR)
{
    // Specific heat ratio for air
    const TData gamma = 1.4;
    // Calculate sound speed using ideal gas relation
    return std::sqrt((gamma - 1.0) * (HRoe - 0.5 * URoe2));
}

template <typename ExecSpace, typename TData, unsigned int NDIM>
NEK_FORCE_INLINE static void LaxFriedrichsSolverKernel(const size_t blksize,
                                                       const TData *fwd,
                                                       const TData *bwd,
                                                       TData *flux)
{
    // Layout: [rho | m0 | m1 | m2 | E] but only first (NDIM) moment exist.
    const TData *rhoL = fwd + 0u * blksize;
    const TData *rhoR = bwd + 0u * blksize;

    const TData *EL = fwd + (1u + NDIM) * blksize;
    const TData *ER = bwd + (1u + NDIM) * blksize;

    // Output blocks
    TData *rhof = flux + 0u * blksize;
    TData *Ef   = flux + (1u + NDIM) * blksize;

    Nektar::parallel_for<ExecSpace>(
        0u, blksize, NEKTAR_LAMBDA(const size_t i) {
            // Velocities (in rotated frame: m0 is normal)
            TData uL[3] = {0, 0, 0}, uR[3] = {0, 0, 0};
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                uL[d] = (fwd + (1u + d) * blksize)[i] / rhoL[i];
                uR[d] = (bwd + (1u + d) * blksize)[i] / rhoR[i];
            }

            // Kinetic energy terms
            TData qL2 = 0, qR2 = 0;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                qL2 += (fwd + (1u + d) * blksize)[i] * uL[d];
                qR2 += (bwd + (1u + d) * blksize)[i] * uR[d];
            }

            // Internal energy per unit mass
            TData eL = (EL[i] - static_cast<TData>(0.5) * qL2) / rhoL[i];
            TData eR = (ER[i] - static_cast<TData>(0.5) * qR2) / rhoR[i];

            // Pressure + enthalpy
            TData pL = GetPressure(rhoL[i], eL);
            TData pR = GetPressure(rhoR[i], eR);
            TData HL = (EL[i] + pL) / rhoL[i];
            TData HR = (ER[i] + pR) / rhoR[i];

            // Roe averages (same as your code)
            TData srL  = std::sqrt(rhoL[i]);
            TData srR  = std::sqrt(rhoR[i]);
            TData srLR = srL + srR;

            TData uRoe[3] = {0, 0, 0};
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                uRoe[d] = (srL * uL[d] + srR * uR[d]) / srLR;
            }

            TData URoe2 = 0;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                URoe2 += uRoe[d] * uRoe[d];
            }

            TData HRoe = (srL * HL + srR * HR) / srLR;

            TData cRoe = GetRoeSoundSpeed(rhoL[i], pL, eL, HL, srL, rhoR[i], pR,
                                          eR, HR, srR, HRoe, URoe2, srLR);

            // max eigenvalue in normal direction
            TData a = std::abs(uRoe[0]) + cRoe;

            // Physical fluxes in normal direction (note: uL[0] is normal
            // velocity)
            const TData Fn_rho_L = (fwd + 1u * blksize)[i];
            const TData Fn_rho_R = (bwd + 1u * blksize)[i];

            // Mass
            rhof[i] = static_cast<TData>(0.5) *
                      (Fn_rho_L + Fn_rho_R - a * (rhoR[i] - rhoL[i]));

            // Normal momentum (d=0): p + m0*u0
            {
                TData Fm0_L = pL + (fwd + 1u * blksize)[i] * uL[0];
                TData Fm0_R = pR + (bwd + 1u * blksize)[i] * uR[0];
                (flux + 1u * blksize)[i] =
                    static_cast<TData>(0.5) *
                    (Fm0_L + Fm0_R -
                     a * ((bwd + 1u * blksize)[i] - (fwd + 1u * blksize)[i]));
            }

            // Tangential momentum(s): m0*u_t
            for (unsigned int d = 1; d < NDIM; ++d)
            {
                TData Fmd_L = (fwd + 1u * blksize)[i] * uL[d];
                TData Fmd_R = (bwd + 1u * blksize)[i] * uR[d];
                (flux + (1u + d) * blksize)[i] =
                    static_cast<TData>(0.5) *
                    (Fmd_L + Fmd_R -
                     a * ((bwd + (1u + d) * blksize)[i] -
                          (fwd + (1u + d) * blksize)[i]));
            }

            // Energy: u0*(E+p)
            {
                TData FE_L = uL[0] * (EL[i] + pL);
                TData FE_R = uR[0] * (ER[i] + pR);
                Ef[i]      = static_cast<TData>(0.5) *
                        (FE_L + FE_R - a * (ER[i] - EL[i]));
            }
        });
}

} // namespace Nektar::Operators::detail
