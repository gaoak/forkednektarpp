///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtKernels.hpp
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
// Description: The pointwise flux of the fused element-local advection
// operator, shared by every implementation.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionFusedElmtKernels.hpp
 * @brief The pointwise flux of the fused element-local advection operator,
 * shared by every implementation.
 *
 * @details
 * A flux policy of the advection pass supplies:
 * - `NDIM`, `NVAR` and `NAUX`, and a static `name`;
 * - a default constructor and one from the session, reading the policy's
 *   parameters;
 * - `AdvFlux(state, aux, advFlux)`: the advective flux at one point.
 *
 * Fluxes are direction-major: `advFlux[d][n]` is the flux of variable n in
 * direction d. @c aux holds the policy's `NAUX` auxiliary values at the
 * point, and is null when it has none. The entry point is a template on the
 * value type, called with the SIMD type on the host and with the scalar on
 * the device. A policy of the advection-diffusion pass
 * (AdvDiffFusedElmtKernels.hpp) also supplies the viscous flux.
 */

#pragma once

#include <LibUtilities/Backends/Backends.hpp>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The flux vector the derivative inner product of the advection pass
 * takes at one point, `flux = -advFlux`.
 *
 * @tparam Physics  The flux policy (see the file description).
 * @tparam TValue   Value type: the SIMD type or the scalar.
 *
 * @param   physics  The flux policy.
 * @param   state    The state at the point.
 * @param   aux      The auxiliary values at the point, or null.
 * @param   flux     Out: the flux vector.
 */
template <typename Physics, typename TValue>
NEK_HOSTDEVICE_INLINE void AdvectionFusedElmtFlux(
    const Physics &physics, const TValue (&state)[Physics::NVAR],
    const TValue *aux, TValue (&flux)[Physics::NDIM][Physics::NVAR])
{
    physics.AdvFlux(state, aux, flux);
    for (unsigned int d = 0; d < Physics::NDIM; ++d)
    {
        for (unsigned int n = 0; n < Physics::NVAR; ++n)
        {
            flux[d][n] = -flux[d][n];
        }
    }
}

} // namespace Nektar::SolverCore::detail
