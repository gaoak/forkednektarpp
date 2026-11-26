///////////////////////////////////////////////////////////////////////////////
//
// File: TestFusedElmtPolicies.hpp
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
// Description: Model Euler and Navier-Stokes flux policies for the unit tests
// of the fused element-local operators.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>

namespace Nektar::UnitTests
{

/**
 * @brief An ideal-gas Euler flux in @p DIM dimensions, as a flux policy of
 * the fused element-local advection operator: the conserved variables
 * (density, momentum, total energy), no auxiliary fields.
 *
 * @tparam DIM  Dimension, one to three.
 */
template <unsigned int DIM> struct EulerModel
{
    static constexpr unsigned int NDIM = DIM;
    static constexpr unsigned int NVAR = DIM + 2;
    static constexpr unsigned int NAUX = 0;
    static constexpr const char *name  = DIM == 1   ? "EulerModel1D"
                                         : DIM == 2 ? "EulerModel2D"
                                                    : "EulerModel3D";

    /// Ratio of specific heats.
    double gamma = 1.4;

    EulerModel() = default;

    /// The ratio of specific heats from the session's `Gamma`.
    explicit EulerModel(const LibUtilities::SessionReaderSharedPtr &session)
    {
        session->LoadParameter("Gamma", gamma, 1.4);
    }

    /// The pressure of @p state.
    template <typename TValue>
    NEK_HOSTDEVICE_INLINE TValue Pressure(const TValue (&state)[NVAR]) const
    {
        const TValue invRho = TValue(1.0) / state[0];
        TValue kinetic      = TValue(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            kinetic = kinetic + state[1 + d] * state[1 + d];
        }
        kinetic = TValue(0.5) * kinetic * invRho;
        return TValue(gamma - 1.0) * (state[NDIM + 1] - kinetic);
    }

    /// The Euler flux: `rho u_d`, `rho u_i u_d + p delta_id`,
    /// `(E + p) u_d` in direction d.
    template <typename TValue>
    NEK_HOSTDEVICE_INLINE void AdvFlux(const TValue (&state)[NVAR],
                                       [[maybe_unused]] const TValue *aux,
                                       TValue (&advFlux)[NDIM][NVAR]) const
    {
        const TValue invRho   = TValue(1.0) / state[0];
        const TValue pressure = Pressure(state);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const TValue velocity = state[1 + d] * invRho;
            advFlux[d][0]         = state[1 + d];
            for (unsigned int i = 0; i < NDIM; ++i)
            {
                advFlux[d][1 + i] = state[1 + i] * velocity;
            }
            advFlux[d][1 + d]    = advFlux[d][1 + d] + pressure;
            advFlux[d][NDIM + 1] = (state[NDIM + 1] + pressure) * velocity;
        }
    }
};

/**
 * @brief The compressible Navier-Stokes flux of an ideal gas in @p DIM
 * dimensions, as a flux policy of the fused element-local advection-diffusion
 * operator: the Euler flux of EulerModel, and a Newtonian viscous flux under
 * the Stokes hypothesis with Fourier heat conduction, the viscosity scaled
 * point by point by the one auxiliary field.
 *
 * @tparam DIM  Dimension, one to three.
 */
template <unsigned int DIM> struct NavierStokesModel
{
    static constexpr unsigned int NDIM = DIM;
    static constexpr unsigned int NVAR = DIM + 2;
    static constexpr unsigned int NAUX = 1;
    static constexpr const char *name  = DIM == 1   ? "NavierStokesModel1D"
                                         : DIM == 2 ? "NavierStokesModel2D"
                                                    : "NavierStokesModel3D";

    /// The inviscid part.
    EulerModel<DIM> euler;
    /// Dynamic viscosity, scaled by the auxiliary field, and Prandtl number.
    double mu      = 0.05;
    double prandtl = 0.72;

    NavierStokesModel() = default;

    /// The parameters from the session's `Gamma`, `mu` and `Pr`.
    explicit NavierStokesModel(
        const LibUtilities::SessionReaderSharedPtr &session)
        : euler(session)
    {
        session->LoadParameter("mu", mu, 0.05);
        session->LoadParameter("Pr", prandtl, 0.72);
    }

    template <typename TValue>
    NEK_HOSTDEVICE_INLINE void AdvFlux(const TValue (&state)[NVAR],
                                       const TValue *aux,
                                       TValue (&advFlux)[NDIM][NVAR]) const
    {
        euler.AdvFlux(state, aux, advFlux);
    }

    /// The viscous flux in direction d: `0`, `tau_id`,
    /// `u_i tau_id + k de/dx_d`, with `tau` the viscous stress, `e` the
    /// specific internal energy and `k = gamma mu / Pr` the conductivity
    /// over the specific heat at constant volume.
    template <typename TValue>
    NEK_HOSTDEVICE_INLINE void ViscFlux(const TValue (&state)[NVAR],
                                        const TValue (&grad)[NDIM][NVAR],
                                        const TValue *aux,
                                        TValue (&viscFlux)[NDIM][NVAR]) const
    {
        const TValue invRho       = TValue(1.0) / state[0];
        const TValue viscosity    = TValue(mu) * aux[0];
        const TValue conductivity = TValue(euler.gamma / prandtl) * viscosity;

        TValue velocity[NDIM], velGrad[NDIM][NDIM];
        for (unsigned int i = 0; i < NDIM; ++i)
        {
            velocity[i] = state[1 + i] * invRho;
        }
        // velGrad[d][i] is the derivative of u_i along direction d.
        TValue divergence = TValue(0.0);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            for (unsigned int i = 0; i < NDIM; ++i)
            {
                velGrad[d][i] =
                    (grad[d][1 + i] - velocity[i] * grad[d][0]) * invRho;
            }
            divergence = divergence + velGrad[d][d];
        }
        const TValue specificEnergy = state[NDIM + 1] * invRho;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            // The derivative of the specific internal energy along d.
            TValue energyGrad =
                (grad[d][NDIM + 1] - specificEnergy * grad[d][0]) * invRho;
            for (unsigned int i = 0; i < NDIM; ++i)
            {
                energyGrad = energyGrad - velocity[i] * velGrad[d][i];
            }

            viscFlux[d][0]  = TValue(0.0);
            TValue workRate = TValue(0.0);
            for (unsigned int i = 0; i < NDIM; ++i)
            {
                TValue stress = viscosity * (velGrad[d][i] + velGrad[i][d]);
                if (i == d)
                {
                    stress =
                        stress - TValue(2.0 / 3.0) * viscosity * divergence;
                }
                viscFlux[d][1 + i] = stress;
                workRate           = workRate + velocity[i] * stress;
            }
            viscFlux[d][NDIM + 1] = workRate + conductivity * energyGrad;
        }
    }
};

} // namespace Nektar::UnitTests
