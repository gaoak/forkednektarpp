///////////////////////////////////////////////////////////////////////////////
//
// File: PreconOp.hpp
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

#include "Operators/ElmtOps/ElmtOp.hpp"

namespace Nektar::Operators
{

// Precon base class
template <typename TData>
class PreconOp : public ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>
{
public:
    static std::shared_ptr<PreconOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string method0 = method;
        if (method == "" &&
            session->DefinesGlobalSysSolnInfo(components[0], "Preconditioner"))
        {
            method0 =
                session->GetGlobalSysSolnInfo(components[0], "Preconditioner");
        }
        else if (method == "" && session->DefinesSolverInfo("Preconditioner"))
        {
            method0 = session->GetSolverInfo("Preconditioner");
        }

        std::string execStr0 = (execStr == "")
                                   ? Operator<TData>::GetOpExecSpace(session)
                                   : execStr;

        std::string requestedKey = method0 + "Precon" + execStr0;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<PreconOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    void Configure(const std::shared_ptr<
                   ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> &op)
    {
        v_Configure(op);
    }

protected:
    PreconOp(const MultiRegions::ExpListSharedPtr &expansionList,
             const std::vector<std::string> &components)
        : ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>(expansionList,
                                                              components)
    {
    }

    ~PreconOp() override = default;

    virtual void v_Configure(
        const std::shared_ptr<
            ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> &op) = 0;
};

} // namespace Nektar::Operators
