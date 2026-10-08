///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondWallCFEOp.hpp
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

#include <boost/algorithm/string/case_conv.hpp>

#include <SolverCore/BndCond/BndCondUpdateOp.hpp>

namespace Nektar
{

/**
 * @brief Viscous wall states for the compressible flow solver.
 *
 * Handles the boundary regions tagged `WallViscous` or `WallAdiabatic`. Both
 * impose no slip, which on a static mesh is the mirrored state
 *
 *     (rho, -rho u, E)
 *
 * taken from the interior trace: density and energy pass through unchanged and
 * the momentum is reversed, so that the *averaged* wall velocity is zero. The
 * same state serves the advection and the diffusion flux, which is why it is
 * written once into the shared boundary storage rather than imposed twice.
 *
 * Regions with any other tag are left alone, so a session may mix walls with
 * ordinary Dirichlet conditions.
 *
 * A wall also carries a thermal condition, and which one depends on the tag:
 * `WallAdiabatic` insulates, imposed as a zero weight on the energy component
 * of the viscous boundary flux, and `WallViscous` is held at the session
 * temperature `Twall`, imposed on the exterior state the diffusion averages
 * against. Neither reaches the boundary storage, because neither may reach the
 * Riemann solver.
 *
 * Resolves the equation of state from the session, as the volume flux
 * operators do: the isothermal condition needs the internal energy that goes
 * with a prescribed temperature.
 */
template <typename TData>
class BndCondWallCFEOp : public SolverCore::BndCondUpdateOp<TData>
{

public:
    static std::shared_ptr<BndCondWallCFEOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        std::string EoSName;

        if (expansionList->GetSession()->DefinesEquationOfState())
        {
            EoSName = boost::to_upper_copy(
                expansionList->GetSession()->GetEquationOfState().type);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "No EquationOfState section defined in session file");
        }

        // The key carries the equation of state, as the volume flux operators
        // do, so the lookup cannot go through Operator::Create<TOperator>(),
        // which builds the key from the class name alone.
        const std::string execStr0 =
            (execStr == "") ? MultiRegions::Operator<TData>::GetOpExecSpace(
                                  expansionList->GetSession())
                            : execStr;

        const std::string requestedKey = name + EoSName + execStr0;

        MultiRegions::OperatorFactory<TData> &factory =
            MultiRegions::GetOperatorFactory<TData>();

        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<BndCondWallCFEOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    static inline const std::string name = "BndCondWallCFE";

protected:
    BndCondWallCFEOp(const MultiRegions::ExpListSharedPtr &expansionList,
                     const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondWallCFEOp() override = default;
};

} // namespace Nektar
