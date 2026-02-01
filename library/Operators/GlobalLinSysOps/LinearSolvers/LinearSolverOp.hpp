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
#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/AssmbScatr/AssmbScatrZeroDirOp.hpp"
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
        const std::vector<std::string> &components,
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
            factory.CreateInstance(requestedKey, expansionList, components));
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
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> m_lhs;
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>>
        m_precon;
    std::shared_ptr<RobBndCondOp<TData>> m_robBndCondOp;
    std::unique_ptr<AssmbScatrOp<TData>> m_assmbScatrOp;
    std::unique_ptr<AssmbScatrOp<TData>> m_assmbScatrZeroDirOp;
    Math m_math;
    bool m_root;

    TData m_tol                = 0.0;
    unsigned int m_maxIter     = 0;
    bool m_leftPreconditioner  = false;
    bool m_rightPreconditioner = false;

    LinearSolverOp(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~LinearSolverOp() override = default;

    virtual void v_Apply(Field<TData, FieldState::Coeff> &in,
                         Field<TData, FieldState::Coeff> &out) = 0;

    template <typename ExecSpace> void SetLinearSolver(void)
    {
        auto session = this->m_expansionList->GetSession();

        // Set operators.
        this->m_assmbScatrOp =
            std::make_unique<detail::AssmbScatrOpImpl<ExecSpace, TData>>(
                this->m_expansionList, this->m_components);
        this->m_assmbScatrZeroDirOp =
            std::make_unique<detail::AssmbScatrZeroDirOpImpl<ExecSpace, TData>>(
                this->m_expansionList, this->m_components);
        this->m_robBndCondOp = RobBndCondOp<TData>::Create(
            this->m_expansionList, this->m_components, ExecSpace::name);
        this->m_rowComm = session->GetComm()->GetRowComm();
        this->m_root    = this->m_rowComm->GetRank() == 0;

        // Set parameters.
        int leftPreconditioner  = 0;
        int rightPreconditioner = 0;
        session->LoadParameter("NekLinSysMaxIterations", this->m_maxIter, 5000);
        session->LoadParameter("IterativeSolverTolerance", this->m_tol,
                               1.0E-09);
        session->LoadParameter("LinSysLeftPrecon", leftPreconditioner, 0);
        session->LoadParameter("LinSysRightPrecon", rightPreconditioner, 0);
        this->m_leftPreconditioner  = leftPreconditioner;
        this->m_rightPreconditioner = rightPreconditioner;

        this->m_math = Math(ExecSpace::name);
    }

    void DirectSolve(std::vector<std::vector<TData>> &A, std::vector<TData> &b)
    {
        unsigned int n = A.size();

        // Forward Elimination with Partial Pivoting.
        for (unsigned int k = 0; k < n; ++k)
        {
            // --- Partial Pivoting ---
            unsigned int maxRow = k;
            TData maxVal        = std::abs(A[k][k]);
            for (unsigned int i = k + 1; i < n; ++i)
            {
                if (std::abs(A[i][k]) > maxVal)
                {
                    maxVal = std::abs(A[i][k]);
                    maxRow = i;
                }
            }

            std::swap(A[k], A[maxRow]);
            std::swap(b[k], b[maxRow]);

            // --- Elimination Stage ---
            for (unsigned int i = k + 1; i < n; ++i)
            {
                TData factor = A[i][k] / A[k][k];
                b[i] -= factor * b[k];
                for (unsigned int j = k; j < n; ++j)
                {
                    A[i][j] -= factor * A[k][j];
                }
            }
        }

        // Backward Substitution.
        for (int i = n - 1; i >= 0; --i)
        {
            TData sum = 0;
            for (unsigned int j = i + 1; j < n; ++j)
            {
                sum += A[i][j] * b[j];
            }
            b[i] = (b[i] - sum) / A[i][i];
        }
    }
};

} // namespace Nektar::Operators
