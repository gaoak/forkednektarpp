///////////////////////////////////////////////////////////////////////////////
//
// File: SessionFunction.h
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
// Description: Session Function
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Math/MathHelper.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SolverCore/SolverCoreDeclspec.h>

#include <map>
#include <memory>

namespace Nektar::MultiRegions
{
template <typename TData> class ExpressionOp;
} // namespace Nektar::MultiRegions

namespace Nektar::SolverCore
{

// Struct for field data import handling
struct ImportedField
{
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDefs;
    std::vector<std::vector<double>> fieldData;
};

class SessionFunction
{
public:
    SOLVER_CORE_EXPORT SessionFunction(
        LibUtilities::SessionReaderSharedPtr session,
        MultiRegions::ExpListSharedPtr expList,
        std::vector<std::string> variables, std::string functionName);

    /// Variables the function is evaluated for, one Field component each.
    const std::vector<std::string> &GetVariables() const
    {
        return m_variables;
    }

    /// Type of the function, the same for each of #m_variables it defines.
    LibUtilities::FunctionType GetType() const
    {
        return m_type;
    }

    /// Whether the function is the same at every evaluation.
    SOLVER_CORE_EXPORT bool IsConstant() const;

    /// Set the factor every evaluation is multiplied by; it persists
    /// across evaluations.
    void SetScale(const double scale)
    {
        m_scale = scale;
    }

    /// Choose between adding to the output and overwriting it; it persists
    /// across evaluations.
    void SetAppend(const bool append)
    {
        m_append = append;
    }

    /// Evaluate the function at the quadrature points into @p phys.
    SOLVER_CORE_EXPORT void Evaluate(
        LibUtilities::Field<double, FieldState::Phys> &phys, double time) const;

    /// Evaluate the function at the quadrature points into @p out, reading
    /// field variables from @p in.
    SOLVER_CORE_EXPORT void Evaluate(
        LibUtilities::Field<double, FieldState::Phys> &in,
        LibUtilities::Field<double, FieldState::Phys> &out, double time) const;

    /// Describe the function of each of #m_variables, in that order.
    SOLVER_CORE_EXPORT std::vector<std::string> Describe() const;

private:
    LibUtilities::SessionReaderSharedPtr m_session;
    MultiRegions::ExpListSharedPtr m_expList;
    /// Variables the function is evaluated for, one Field component each.
    std::vector<std::string> m_variables;
    std::string m_name;
    /// Type of the function, the same for each of #m_variables it defines.
    LibUtilities::FunctionType m_type;
    /// Factor every evaluation is multiplied by.
    double m_scale = 1.0;
    /// Whether an evaluation adds to the output rather than overwriting it.
    bool m_append = false;
    mutable Math::MathHelper m_math;
    /// Field files read so far, by file name.
    mutable std::map<std::string, ImportedField> m_importedFields;
    /// Expression operator on #m_expList, built on first use.
    mutable std::shared_ptr<MultiRegions::ExpressionOp<double>> m_expressionOp;

    /// Evaluate the expressions of #m_variables at the quadrature points
    /// into @p out, reading field variables from @p in.
    void EvaluateExpression(LibUtilities::Field<double, FieldState::Phys> &in,
                            LibUtilities::Field<double, FieldState::Phys> &out,
                            double time) const;

    /// Read the file function of #m_variables as element coefficients.
    void EvaluateFld(
        LibUtilities::Field<double, FieldState::Coeff> &coeffs) const;
};

using SessionFunctionSharedPtr = std::shared_ptr<SessionFunction>;

} // namespace Nektar::SolverCore
