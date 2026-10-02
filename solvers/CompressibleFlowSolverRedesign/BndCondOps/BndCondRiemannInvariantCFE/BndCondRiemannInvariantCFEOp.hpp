///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondRiemannInvariantCFEOp.hpp
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
 * @brief Characteristic farfield boundary from the Riemann invariants.
 *
 * The classical farfield condition. The two invariants
 * \f$R^\pm = u_n \pm 2c/(\gamma-1)\f$ are each taken from whichever side
 * they are carried from - the interior for a characteristic leaving the domain,
 * the freestream for one entering - and the boundary normal velocity and sound
 * speed follow from the pair,
 *
 * \f[ u_n^{BC} = \tfrac12 (R^+ + R^-), \qquad
 *     c^{BC}   = \tfrac{\gamma-1}{4} (R^+ - R^-). \f]
 *
 * The remaining thermodynamic freedom is the entropy, and it comes from
 * whichever side is upstream: the freestream on inflow, the interior on
 * outflow. Density and pressure follow from it and the sound speed. The
 * velocity is that side's velocity vector with its normal component corrected
 * to \f$u_n^{BC}\f$, so the tangential components pass through.
 *
 * Supersonic reduces to the obvious thing without a special case: both
 * invariants come from one side, so the reconstruction returns that side's
 * state unchanged.
 *
 * ### Where the freestream comes from
 *
 * From the session *parameters* `rhoInf`, `pInf`, `uInf`, `vInf`, `wInf`, not
 * from the region's boundary values. That is why this condition does not share
 * the machinery of the entropy Riemann conditions, which take their prescribed
 * state from the boundary values instead. The `VALUE` on such a region is
 * therefore ignored, which is worth knowing before wondering why editing it
 * changes nothing.
 *
 * Ideal gas only, for the same reason as the entropy conditions: the entropy
 * relation \f$s = p/\rho^\gamma\f$ is the ideal gas one.
 */
template <typename TData>
class BndCondRiemannInvariantCFEOp : public SolverCore::BndCondUpdateOp<TData>
{
public:
    static std::shared_ptr<BndCondRiemannInvariantCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operators::Operator<TData>::template Create<
            BndCondRiemannInvariantCFEOp>(expansionList, components, execStr);
    }

    static inline const std::string name = "BndCondRiemannInvariantCFEOp";

protected:
    BndCondRiemannInvariantCFEOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondRiemannInvariantCFEOp() override = default;
};

} // namespace Nektar
