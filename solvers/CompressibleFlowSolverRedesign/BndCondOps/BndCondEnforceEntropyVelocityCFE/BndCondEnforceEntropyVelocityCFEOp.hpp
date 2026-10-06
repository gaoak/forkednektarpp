///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondEnforceEntropyVelocityCFEOp.hpp
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

#include <SolverCore/BndCond/BndCondUpdateOp.hpp>

namespace Nektar
{

/**
 * @brief Subsonic Riemann inflow holding the entropy and the velocity.
 *
 * A subsonic inflow admits one incoming characteristic, so the Riemann
 * invariant carried out of the domain fixes one combination of the state and
 * one degree of freedom is left to prescribe. Which quantity fills it is not
 * free: see Ganlin Lyu, Chao Chen, Xi Du and Spencer J. Sherwin, *Stable,
 * entropy-pressure compatible subsonic Riemann boundary condition for embedded
 * DG compressible flow simulations*, arXiv:2205.14257, which shows by a
 * linearised analysis that enforcing the entropy is what makes the condition
 * stable. This operator is the variant that takes the entropy and the velocity
 * from the boundary data; its siblings take the entropy with the pressure or
 * with the total enthalpy instead.
 *
 * The state is built from two different places, which is the whole idea:
 *
 * - the **entropy** \f$s = p/\rho^\gamma\f$ comes from the prescribed boundary
 *   state, and
 * - the **sound speed** is carried over unchanged from the interior,
 *
 * and the density and pressure follow from the pair,
 * \f$\rho = (c^2/\gamma s)^{1/(\gamma-1)}\f$, \f$p = c^2\rho/\gamma\f$. The
 * velocity vector is the prescribed one with its normal component shifted by
 * the mismatch between the Riemann normal velocity and the prescribed one, so
 * that the tangential components survive intact.
 *
 * Unlike the other compressible boundary conditions here this one is *not*
 * templated on the equation of state, because it cannot be: the entropy
 * relation above is the ideal gas one. Rather than pretend otherwise it reads
 * gamma from the session and refuses to construct for anything else, so the
 * limitation is announced rather than silently wrong.
 *
 * @see BndCondEnforceEntropyVelocityCFEOpImpl for the condition itself.
 */
template <typename TData>
class BndCondEnforceEntropyVelocityCFEOp
    : public SolverCore::BndCondUpdateOp<TData>
{
public:
    static std::shared_ptr<BndCondEnforceEntropyVelocityCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return MultiRegions::Operator<TData>::template Create<
            BndCondEnforceEntropyVelocityCFEOp>(expansionList, components,
                                                execStr);
    }

    static inline const std::string name = "BndCondEnforceEntropyVelocityCFE";

protected:
    BndCondEnforceEntropyVelocityCFEOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondEnforceEntropyVelocityCFEOp() override = default;
};

} // namespace Nektar
