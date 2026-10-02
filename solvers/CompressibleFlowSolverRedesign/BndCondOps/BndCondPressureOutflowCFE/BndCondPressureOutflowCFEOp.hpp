///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondPressureOutflowCFEOp.hpp
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
// Description: Pressure outflow boundary condition base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <boost/algorithm/string/case_conv.hpp>

#include <SolverCore/BndCond/BndCondUpdateOp.hpp>

namespace Nektar
{

/**
 * @brief Subsonic pressure outflow, as a boundary state.
 *
 * Resolves the equation of state from the session, in the same way as the
 * volume flux operators, because the condition needs both a sound speed - to
 * decide whether the outflow is subsonic - and the internal energy that goes
 * with a prescribed pressure.
 *
 * @see BndCondPressureOutflowCFEOpImpl for the condition itself.
 */
template <typename TData>
class BndCondPressureOutflowCFEOp : public SolverCore::BndCondUpdateOp<TData>
{
public:
    static std::shared_ptr<BndCondPressureOutflowCFEOp<TData>> Create(
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
            (execStr == "") ? Operators::Operator<TData>::GetOpExecSpace(
                                  expansionList->GetSession())
                            : execStr;

        const std::string requestedKey = name + EoSName + execStr0;

        Operators::OperatorFactory<TData> &factory =
            Operators::GetOperatorFactory<TData>();

        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<BndCondPressureOutflowCFEOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    static inline const std::string name = "BndCondPressureOutflowCFE";

protected:
    BndCondPressureOutflowCFEOp(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::BndCondUpdateOp<TData>(expansionList, components)
    {
    }

    ~BndCondPressureOutflowCFEOp() override = default;
};

} // namespace Nektar
