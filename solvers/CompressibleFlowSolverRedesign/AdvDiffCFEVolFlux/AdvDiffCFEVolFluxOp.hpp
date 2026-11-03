///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffCFEVolFluxOp.hpp
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
// Description: Combined Euler and viscous volume flux for the Compressible
// Flow Equations (CFE)
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SolverCore/AdvDiffusion/AdvDiffVolumeFluxOp.hpp"
#include <LibUtilities/BasicUtils/StringUtils.hpp>

namespace Nektar
{

/**
 * @brief The Navier-Stokes volume flux with the viscous flux on its own
 * (see SolverCore::AdvDiffVolumeFluxOp): the viscous flux and the viscous
 * minus the Euler flux, formed together in one pass over the state and its
 * gradient.
 */
template <typename TData>
class AdvDiffCFEVolFluxOp : public SolverCore::AdvDiffVolumeFluxOp<TData>
{
public:
    static std::shared_ptr<AdvDiffCFEVolFluxOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        std::string EoSName;

        if (expansionList->GetSession()->DefinesEquationOfState())
        {
            EoSName = StringUtils::ToUpperCopy(
                expansionList->GetSession()->GetEquationOfState().type);
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "No EquationOfState section defined in session file");
        }

        return std::dynamic_pointer_cast<AdvDiffCFEVolFluxOp<TData>>(
            SolverCore::AdvDiffVolumeFluxOp<TData>::Create(
                expansionList, components, name + EoSName, execStr));
    }

    static inline const std::string name = "AdvDiffCFEVolFlux";

protected:
    AdvDiffCFEVolFluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
                        const std::vector<std::string> &components)
        : SolverCore::AdvDiffVolumeFluxOp<TData>(expansionList, components)
    {
    }

    ~AdvDiffCFEVolFluxOp() override = default;
};

} // namespace Nektar
