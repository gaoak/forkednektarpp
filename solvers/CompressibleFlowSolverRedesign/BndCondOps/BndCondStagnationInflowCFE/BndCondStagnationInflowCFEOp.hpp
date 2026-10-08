///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondStagnationInflowCFEOp.hpp
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
 * @brief Subsonic inflow from a stagnation state and a flow direction.
 *
 * The reservoir condition. The session prescribes the stagnation density and
 * energy and a direction; the speed is whatever the interior brings to the
 * boundary, and the static state follows from the stagnation one by a
 * one-dimensional energy balance along an isentrope:
 *
 * \f[ h_0 = \gamma E_0/\rho_0, \qquad h = h_0 - \tfrac12 |u|^2, \qquad
 *     \rho = \rho_0 \left(\frac{\rho_0 h}{\gamma E_0}\right)^{1/(\gamma-1)}.
 * \f]
 *
 * ### The session values are not what they look like
 *
 * `rho` and `E` carry the *stagnation* state, and the momentum entries carry a
 * flow *direction*, not a momentum: only their ratio matters, since the
 * magnitude comes from the interior. They are normalised on being read.
 *
 * A direction of zero is not degenerate but a request: it means normal to the
 * boundary, and the inward normal is used. That is what
 *
 *     <D VAR="rhou" USERDEFINEDTYPE="StagnationInflow" VALUE="0" />
 *
 * asks for in the SquareDomain cases.
 *
 * Ideal gas only, the enthalpy relations above being the ideal gas ones.
 */
template <typename TData>
class BndCondStagnationInflowCFEOp : public SolverCore::BndCondUpdateOp<TData>
{
public:
    static std::shared_ptr<BndCondStagnationInflowCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return MultiRegions::Operator<TData>::template Create<
            BndCondStagnationInflowCFEOp>(expansionList, components, execStr);
    }

    static inline const std::string name = "BndCondStagnationInflowCFEOp";

protected:
    BndCondStagnationInflowCFEOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondStagnationInflowCFEOp() override = default;
};

} // namespace Nektar
