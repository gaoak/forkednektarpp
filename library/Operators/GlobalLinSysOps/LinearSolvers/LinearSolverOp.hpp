///////////////////////////////////////////////////////////////////////////////
//
// File: LinearSolverOp.hpp
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

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondOp.hpp"
#include "Operators/ElmtOps/ElmtOp.hpp"
#include "Operators/PreconOps/PreconOp.hpp"

#include "Operators/Math/Math.hpp"
#include "Operators/Math/MathKernels.hpp"

namespace Nektar::Operators
{

// LinearSolver base class
template <typename TData> class LinearSolverOp : public Operator<TData>
{
public:
    static std::shared_ptr<LinearSolverOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &method = "", const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string method0 = (method == "")
                                  ? session->GetSolverInfo("LinSysIterSolver")
                                  : method;

        std::string execStr0 = (execStr == "")
                                   ? Operator<TData>::GetOpExecSpace(session)
                                   : execStr;

        std::string requestedKey = method0 + execStr0;

        OperatorFactory<TData> &factory = GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<LinearSolverOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList));
    }

    void Apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(Field<TData, FieldState::Coeff> &in,
                    Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void SetLHS(const std::shared_ptr<
                ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> &ptr)
    {
        this->m_lhs = ptr;
    }

    void SetPrecon(const std::shared_ptr<PreconOp<TData>> &ptr)
    {
        this->m_precon = ptr;
    }

protected:
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> m_lhs;
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>>
        m_precon;

    LinearSolverOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {
    }

    ~LinearSolverOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Coeff> &in,
                         Field<TData, FieldState::Coeff> &out) = 0;
};

} // namespace Nektar::Operators
