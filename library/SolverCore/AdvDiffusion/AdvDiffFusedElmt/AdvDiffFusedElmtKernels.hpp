///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffFusedElmtKernels.hpp
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
// Description: The pointwise flux of the fused element-local advection-
// diffusion operator, shared by every implementation.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvDiffFusedElmtKernels.hpp
 * @brief The pointwise flux of the fused element-local advection-diffusion
 * operator, shared by every implementation.
 *
 * @details
 * A flux policy of the advection-diffusion pass is one of the advection pass
 * (AdvectionFusedElmtKernels.hpp: `NDIM`, `NVAR`, `NAUX`, `name`, the
 * constructors and `AdvFlux()`) that also supplies, at one point:
 * - `ViscFlux(state, grad, aux, viscFlux)`: the viscous flux.
 *
 * Fluxes and gradients are direction-major: `viscFlux[d][n]` is the flux of
 * variable n in direction d, and `grad[d][n]` the derivative of variable n
 * along direction d. Both entry points are templates on the value type,
 * called with the SIMD type on the host and with the scalar on the device.
 */

#pragma once

#include <LibUtilities/Backends/Backends.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The flux vector the derivative inner product of the
 * advection-diffusion pass takes at one point, `flux = viscFlux - advFlux`,
 * and the viscous flux the normal viscous flux on the traces is formed from.
 *
 * @tparam Physics  The flux policy (see the file description).
 * @tparam TValue   Value type: the SIMD type or the scalar.
 *
 * @param   physics   The flux policy.
 * @param   state     The state at the point.
 * @param   grad      Its gradient.
 * @param   aux       The auxiliary values at the point, or null.
 * @param   flux      Out: the flux vector.
 * @param   viscFlux  Out: the viscous flux.
 */
template <typename Physics, typename TValue>
NEK_HOSTDEVICE_INLINE void AdvDiffFusedElmtFlux(
    const Physics &physics, const TValue (&state)[Physics::NVAR],
    const TValue (&grad)[Physics::NDIM][Physics::NVAR], const TValue *aux,
    TValue (&flux)[Physics::NDIM][Physics::NVAR],
    TValue (&viscFlux)[Physics::NDIM][Physics::NVAR])
{
    physics.AdvFlux(state, aux, flux);
    physics.ViscFlux(state, grad, aux, viscFlux);
    for (unsigned int d = 0; d < Physics::NDIM; ++d)
    {
        for (unsigned int n = 0; n < Physics::NVAR; ++n)
        {
            flux[d][n] = viscFlux[d][n] - flux[d][n];
        }
    }
}

} // namespace Nektar::SolverCore::detail
