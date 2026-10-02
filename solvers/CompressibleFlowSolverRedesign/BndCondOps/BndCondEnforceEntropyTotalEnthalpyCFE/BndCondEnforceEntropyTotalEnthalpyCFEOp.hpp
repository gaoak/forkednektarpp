///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondEnforceEntropyTotalEnthalpyCFEOp.hpp
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
 * @brief Subsonic Riemann boundary holding the entropy and the total enthalpy.
 *
 * The stagnation variant: the remaining degree of freedom at a subsonic inflow
 * is filled by the total enthalpy rather than by a static quantity, which is
 * the natural choice when the upstream reservoir is what is known. The sound
 * speed is then no longer available directly and comes from a quadratic; see
 * EntropyTotalEnthalpyStar().
 *
 * Its subsonic outflow is the same isentropic pressure branch the
 * entropy-pressure condition uses.
 *
 * @see Ganlin Lyu, Chao Chen, Xi Du and Spencer J. Sherwin, *Stable,
 * entropy-pressure compatible subsonic Riemann boundary condition for embedded
 * DG compressible flow simulations*, arXiv:2205.14257, for the analysis these
 * conditions come from.
 */
template <typename TData>
class BndCondEnforceEntropyTotalEnthalpyCFEOp
    : public SolverCore::BndCondUpdateOp<TData>
{
public:
    static std::shared_ptr<BndCondEnforceEntropyTotalEnthalpyCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operators::Operator<TData>::template Create<
            BndCondEnforceEntropyTotalEnthalpyCFEOp>(expansionList, components,
                                                     execStr);
    }

    static inline const std::string name =
        "BndCondEnforceEntropyTotalEnthalpyCFEOp";

protected:
    BndCondEnforceEntropyTotalEnthalpyCFEOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondEnforceEntropyTotalEnthalpyCFEOp() override = default;
};

} // namespace Nektar
