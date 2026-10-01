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

#include <optional>
#include <ostream>
#include <sstream>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/BndCondOps/RobBndCond/RobBndCondOp.hpp"
#include "Operators/ElmtOps/ElmtOp.hpp"
#include "SolverCore/PreconOps/PreconOp.hpp"
#include "SolverCore/SolverCore.hpp"

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "LibUtilities/BasicUtils/Math/MathHelper.hpp"

namespace Nektar::SolverCore
{

// LinearSolver base class
template <typename TData>
class LinearSolverOp : public Operators::Operator<TData>
{
public:
    static std::shared_ptr<LinearSolverOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const std::string &execStr = "")
    {
        // Force a genuine cross-library symbol reference into libSolverCore so
        // the linear solver *OpImpl static factory registrations are not
        // dropped by the linker/loader -- see EnsureLinked() in SolverCore.hpp.
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string method0 = method;
        if (method == "" && session->DefinesGlobalSysSolnInfo(
                                components[0], "LinSysIterSolver"))
        {
            method0 = session->GetGlobalSysSolnInfo(components[0],
                                                    "LinSysIterSolver");
        }
        else if (method == "" && session->DefinesSolverInfo("LinSysIterSolver"))
        {
            method0 = session->GetSolverInfo("LinSysIterSolver");
        }

        // TODO fix name of Conjugate Gradient solver for legacy/device support
        // Rename from legacy specification
        if (method0 == "ConjugateGradientLoc" || method0 == "ConjugateGradient")
        {
            method0 = "ConjGrad";
        }

        std::string execStr0 =
            (execStr == "")
                ? Operators::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        std::string requestedKey = method0 + execStr0;

        Operators::OperatorFactory<TData> &factory =
            Operators::GetOperatorFactory<TData>();

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

    void Apply(LibUtilities::Field<TData, FieldState::Coeff> &in,
               LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Coeff> &in,
                    LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        this->v_Apply(in, out);
    }

    void SetLHS(const std::shared_ptr<Operators::ElmtOp<
                    FieldState::Coeff, FieldState::Coeff, TData>> &ptr)
    {
        this->m_lhs = ptr;
    }

    void SetPrecon(const std::shared_ptr<PreconOp<TData>> &ptr)
    {
        this->m_precon = ptr;
    }

    void UpdatePrecon(void)
    {
        this->m_precon->Configure(this->m_lhs);
    }

    unsigned int GetNiterations(void)
    {
        return m_niter;
    }

protected:
    LibUtilities::CommSharedPtr m_rowComm = nullptr;
    std::shared_ptr<
        Operators::ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>>
        m_lhs;
    std::shared_ptr<PreconOp<TData>> m_precon;
    std::shared_ptr<Operators::RobBndCondOp<TData>> m_robBndCondOp;
    std::unique_ptr<Operators::AssmbScatrOp<TData>> m_assmbScatrOp;
    std::unique_ptr<Operators::AssmbScatrOp<TData>> m_assmbScatrZeroDirOp;
    Math::MathHelper m_math;
    bool m_root;

    TData m_tol                = 0.0;
    unsigned int m_maxIter     = 0;
    unsigned int m_niter       = 0;
    bool m_absoluteTolerance   = false;
    bool m_leftPreconditioner  = false;
    bool m_rightPreconditioner = false;
    bool m_verboseOutput       = false;

    LinearSolverOp(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    ~LinearSolverOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Coeff> &in,
        LibUtilities::Field<TData, FieldState::Coeff> &out) = 0;

    std::string GetVerboseName(const std::string &solverName) const
    {
        std::stringstream msg;
        msg << solverName;

        if (!this->m_components.empty())
        {
            msg << " components = [";
            for (unsigned int i = 0; i < this->m_components.size(); ++i)
            {
                if (i > 0)
                {
                    msg << ", ";
                }
                msg << this->m_components[i];
            }
            msg << "]";
        }

        return msg.str();
    }

    bool IsVerboseOutputEnabled(void) const
    {
        return m_verboseOutput;
    }

    template <typename TExtraPrinter>
    void PrintVerboseOutput(const std::string &solverName,
                            const std::string &residualName,
                            const TData residual, const TData rhsMagnitude,
                            TExtraPrinter extraPrinter) const
    {
        if (!m_verboseOutput || !m_root)
        {
            return;
        }

        std::cout << GetVerboseName(solverName)
                  << " iterations made = " << m_niter << " using "
                  << (m_absoluteTolerance ? "absolute" : "relative")
                  << " tolerance of " << m_tol << " " << residualName << " = "
                  << residual << " rhs_mag = " << std::sqrt(rhsMagnitude);
        extraPrinter(std::cout);
        std::cout << std::endl;
    }

    void PrintVerboseOutput(const std::string &solverName,
                            const std::string &residualName,
                            const TData residual,
                            const TData rhsMagnitude) const
    {
        PrintVerboseOutput(solverName, residualName, residual, rhsMagnitude,
                           [](std::ostream &) {});
    }

    /**
     * @brief Squared magnitude the convergence test is relative to.
     *
     * One for an absolute tolerance; otherwise @p inMagnitude, the squared
     * magnitude of the right-hand side passed to Apply(). For the linear
     * systems this is the residual \f$b - Ax_0\f$ of the initial guess. A
     * vanishing magnitude is replaced by one so that a zero right-hand side
     * does not demand an exact solve.
     */
    TData GetRhsMagnitude(const TData inMagnitude) const
    {
        if (m_absoluteTolerance)
        {
            return 1.0;
        }

        return (inMagnitude > 1.0e-6) ? inMagnitude : 1.0;
    }

    template <typename ExecSpace> void SetLinearSolver(void)
    {
        auto session = this->m_expansionList->GetSession();

        // Set operators.
        this->m_assmbScatrOp = std::make_unique<
            Operators::detail::AssmbScatrOpImpl<ExecSpace, TData>>(
            this->m_expansionList, this->m_components);
        this->m_assmbScatrZeroDirOp = std::make_unique<
            Operators::detail::AssmbScatrZeroDirOpImpl<ExecSpace, TData>>(
            this->m_expansionList, this->m_components);
        this->m_robBndCondOp = Operators::RobBndCondOp<TData>::Create(
            this->m_expansionList, this->m_components, ExecSpace::name);
        this->m_rowComm       = session->GetComm()->GetRowComm();
        this->m_root          = this->m_rowComm->GetRank() == 0;
        this->m_verboseOutput = session->DefinesCmdLineArgument("verbose");

        // Set parameters.
        LoadSetting(session, "NekLinSysMaxIterations", this->m_maxIter, 5000u);
        LoadSetting(session, "IterativeSolverTolerance", this->m_tol,
                    static_cast<TData>(1.0E-09));
        auto absoluteTolerance =
            GetGlobalSysSolnInfo(session, "AbsoluteTolerance");
        this->m_absoluteTolerance =
            absoluteTolerance && boost::iequals(*absoluteTolerance, "True");

        // Set math helper function.
        this->m_math = Math::MathHelper(ExecSpace::name);
    }

    /**
     * @brief GLOBALSYSSOLNINFO property of the solved variables.
     *
     * Components solved together share one iteration and therefore one
     * setting: the property of the first component is used, with a warning
     * if another component defines it differently. Empty if the first
     * component does not define it.
     */
    std::optional<std::string> GetGlobalSysSolnInfo(
        const LibUtilities::SessionReaderSharedPtr &session,
        const std::string &property) const
    {
        std::optional<std::string> value;
        const std::string &var0 = this->m_components[0];
        if (session->DefinesGlobalSysSolnInfo(var0, property))
        {
            value = session->GetGlobalSysSolnInfo(var0, property);
        }

        for (size_t i = 1; i < this->m_components.size(); ++i)
        {
            const std::string &var = this->m_components[i];
            if (session->DefinesGlobalSysSolnInfo(var, property) &&
                (!value ||
                 session->GetGlobalSysSolnInfo(var, property) != *value))
            {
                WARNINGL0(false, "GLOBALSYSSOLNINFO " + property +
                                     " of variable " + var +
                                     " is ignored: it is solved together "
                                     "with " +
                                     var0 + ", whose setting is used.");
            }
        }

        return value;
    }

    /**
     * @brief Load a solver setting.
     *
     * Takes the GLOBALSYSSOLNINFO property of the solved variables if
     * defined, else the session parameter of the same name, else
     * @p defaultValue.
     */
    template <typename T>
    void LoadSetting(const LibUtilities::SessionReaderSharedPtr &session,
                     const std::string &name, T &value,
                     const T &defaultValue) const
    {
        if (auto info = GetGlobalSysSolnInfo(session, name))
        {
            value = boost::lexical_cast<T>(*info);
        }
        else
        {
            session->LoadParameter(name, value, defaultValue);
        }
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

    // QR factorization through Givens rotation
    void DoGivensRotation(const unsigned int starttem,
                          const unsigned int endtem, std::vector<TData> &c,
                          std::vector<TData> &s, std::vector<TData> &h,
                          std::vector<TData> &eta)
    {
        TData dbl;
        TData dd;
        TData hh;
        unsigned int idtem = endtem - 1;

        // The starttem and endtem are beginning and ending order of Givens
        // rotation They usually equal to the beginning position and ending
        // position of Hessenburg matrix But sometimes starttem will change,
        // like if it is initial 0 and becomes nonzero because previous Givens
        // rotation See Yu Pan's User Guide
        for (unsigned int i = starttem; i < idtem; ++i)
        {
            dbl      = c[i] * h[i] - s[i] * h[i + 1];
            h[i + 1] = s[i] * h[i] + c[i] * h[i + 1];
            h[i]     = dbl;
        }
        dd = h[idtem];
        hh = h[endtem];
        if (hh == 0.0)
        {
            c[idtem] = 1.0;
            s[idtem] = 0.0;
        }
        else if (std::abs(hh) > std::abs(dd))
        {
            dbl      = -dd / hh;
            s[idtem] = 1.0 / std::sqrt(1.0 + dbl * dbl);
            c[idtem] = dbl * s[idtem];
        }
        else
        {
            dbl      = -hh / dd;
            c[idtem] = 1.0 / std::sqrt(1.0 + dbl * dbl);
            s[idtem] = dbl * c[idtem];
        }

        h[idtem]  = c[idtem] * h[idtem] - s[idtem] * h[endtem];
        h[endtem] = 0.0;

        dbl         = c[idtem] * eta[idtem] - s[idtem] * eta[endtem];
        eta[endtem] = s[idtem] * eta[idtem] + c[idtem] * eta[endtem];
        eta[idtem]  = dbl;
    }

    void DoBackward(const unsigned int n,
                    const std::vector<std::vector<TData>> &A,
                    const std::vector<TData> &b, std::vector<TData> &y)
    {
        TData sum;
        y[n - 1] = b[n - 1] / A[n - 1][n - 1];
        for (unsigned int i = n - 2; i + 1 > 0; --i)
        {
            sum = b[i];
            for (unsigned int j = i + 1; j < n; ++j)
            {
                sum -= y[j] * A[j][i];
            }
            y[i] = sum / A[i][i];
        }
    }
};

} // namespace Nektar::SolverCore
