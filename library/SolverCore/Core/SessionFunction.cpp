////////////////////////////////////////////////////////////////////////////////
//
// File: SessionFunction.cpp
//
// For more information, please see: http://www.nektar.info/
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
////////////////////////////////////////////////////////////////////////////////

#include <SolverCore/Core/SessionFunction.h>

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <MultiRegions/Common/Operator.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>

#include <algorithm>
#include <iterator>
#include <map>

namespace Nektar::SolverCore
{

namespace
{
/**
 * @brief Type of function @p name, which must be the same for each of
 * @p variables it defines.
 *
 * A variable the function does not define is zero, so a function that
 * defines none of @p variables is an expression.
 */
LibUtilities::FunctionType GetUniformFunctionType(
    const LibUtilities::SessionReaderSharedPtr &session,
    const std::string &name, const std::vector<std::string> &variables)
{
    std::vector<std::string> definedVariables;
    std::copy_if(variables.begin(), variables.end(),
                 std::back_inserter(definedVariables),
                 [&](const std::string &variable) {
                     return session->DefinesFunction(name, variable);
                 });
    if (definedVariables.empty())
    {
        return LibUtilities::eFunctionTypeExpression;
    }

    const auto type = session->GetFunctionType(name, definedVariables[0]);
    for (std::size_t i = 1; i < definedVariables.size(); ++i)
    {
        ASSERTL0(session->GetFunctionType(name, definedVariables[i]) == type,
                 "Function '" + name +
                     "' must define either all its variables from file <F> "
                     "or all from expression <E>.");
    }

    return type;
}
} // namespace

SessionFunction::SessionFunction(LibUtilities::SessionReaderSharedPtr session,
                                 MultiRegions::ExpListSharedPtr expList,
                                 std::vector<std::string> variables,
                                 std::string functionName)
    : m_session(std::move(session)), m_expList(std::move(expList)),
      m_variables(std::move(variables)), m_name(std::move(functionName)),
      m_math(MultiRegions::Operator<double>::GetOpExecSpace(m_session))
{
    ASSERTL0(m_session->DefinesFunction(m_name),
             "Function '" + m_name + "' does not exist.");
    m_type = GetUniformFunctionType(m_session, m_name, m_variables);
}

/**
 * @brief Whether the function is the same at every evaluation: a file, or
 * expressions that use neither the time nor a field variable of their
 * variable list.
 */
bool SessionFunction::IsConstant() const
{
    return m_type != LibUtilities::eFunctionTypeExpression ||
           std::none_of(
               m_variables.begin(), m_variables.end(),
               [this](const std::string &variable) {
                   if (!m_session->DefinesFunction(m_name, variable))
                   {
                       return false;
                   }
                   const auto equation =
                       m_session->GetFunction(m_name, variable);
                   const auto used = MultiRegions::ExpressionBlockOp<double>::
                       GetExpressionVariables(equation->GetExpression());
                   // Entries 0-3 of the variable list are x, y, z and t.
                   const auto names =
                       MultiRegions::ExpressionBlockOp<double>::GetEvarsNames(
                           equation);
                   return used.count("t") != 0 ||
                          std::any_of(names.begin() + std::min<std::size_t>(
                                                          4, names.size()),
                                      names.end(),
                                      [&used](const std::string &name) {
                                          return used.count(name) != 0;
                                      });
               });
}

/**
 * @brief Evaluate the function at the quadrature points into @p phys,
 * reading field variables from @p phys.
 */
void SessionFunction::Evaluate(
    LibUtilities::Field<double, FieldState::Phys> &phys, double time) const
{
    Evaluate(phys, phys, time);
}

/**
 * @brief Evaluate the function at the quadrature points into @p out, scaled
 * by #m_scale, or add it to @p out when #m_append.
 *
 * An expression is evaluated, and scaled, at the quadrature points, reading
 * field variables from @p in; a file is read as element coefficients
 * (EvaluateFld()), scaled and evaluated at the quadrature points
 * (BwdTransOp).
 */
void SessionFunction::Evaluate(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, double time) const
{
    if (m_type == LibUtilities::eFunctionTypeExpression)
    {
        EvaluateExpression(in, out, time);
    }
    else if (m_type == LibUtilities::eFunctionTypeFile)
    {
        auto coeffs = LibUtilities::Field<double, FieldState::Coeff>(
            "function coefficients",
            MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
                m_expList,
                MultiRegions::Operator<double>::GetDefaultInterleaveWidth(
                    m_session)),
            out.GetNumComponents(), out.GetNumHomoModes());
        auto bwdTransOp =
            MultiRegions::BwdTransOp<double>::Create(m_expList, m_variables);

        EvaluateFld(coeffs);
        if (m_scale != 1.0)
        {
            m_math.mul(m_scale, coeffs, coeffs);
        }
        bwdTransOp->SetAppend(m_append);
        bwdTransOp->Apply(coeffs, out);
    }
    else
    {
        NEKERROR(ErrorUtil::efatal,
                 "SessionFunction supports expression and file functions "
                 "only.");
    }
}

std::vector<std::string> SessionFunction::Describe() const
{
    std::vector<std::string> descriptions;
    descriptions.reserve(m_variables.size());
    for (const auto &variable : m_variables)
    {
        descriptions.push_back(
            !m_session->DefinesFunction(m_name, variable) ? "0"
            : m_type == LibUtilities::eFunctionTypeExpression
                ? m_session->GetFunction(m_name, variable)->GetExpression()
                : "from file " +
                      m_session->GetFunctionFilename(m_name, variable));
    }

    return descriptions;
}

void SessionFunction::EvaluateExpression(
    LibUtilities::Field<double, FieldState::Phys> &in,
    LibUtilities::Field<double, FieldState::Phys> &out, double time) const
{
    ASSERTL0(out.GetNumComponents() == m_variables.size(),
             "SessionFunction::EvaluateExpression mismatch in supplied number "
             "of variables and number of components for Field.");

    // Initialise operator and read the function's expressions.
    if (!m_expressionOp)
    {
        m_expressionOp =
            MultiRegions::ExpressionOp<double>::Create(m_expList, m_variables);

        std::vector<LibUtilities::EquationSharedPtr> expressions;
        for (const auto &var : m_variables)
        {
            expressions.push_back(
                m_session->DefinesFunction(m_name, var)
                    ? m_session->GetFunction(m_name, var)
                    : MemoryManager<LibUtilities::Equation>::AllocateSharedPtr(
                          m_session->GetInterpreter(), "0"));
        }
        m_expressionOp->SetExpressions(expressions);
    }
    m_expressionOp->SetTime(time);
    m_expressionOp->SetScale(m_scale);
    m_expressionOp->SetAppend(m_append);

    // Evaluate the expressions.
    m_expressionOp->Apply(in, out);
}

/**
 * @brief Read the file function of #m_variables into @p coeffs.
 *
 * Each element's modes are copied from the file, truncated or padded to the
 * expansion's order; a variable the function does not define is zero. A
 * file is read once and kept for later calls.
 */
void SessionFunction::EvaluateFld(
    LibUtilities::Field<double, FieldState::Coeff> &coeffs) const
{
    ASSERTL0(coeffs.GetNumComponents() == m_variables.size(),
             "SessionFunction::EvaluateFld mismatch in supplied number of "
             "variables and number of components for Field.");

    const std::size_t ncoeffs = m_expList->GetNcoeffs();
    std::vector<double> allCoeffs(ncoeffs * m_variables.size(), 0.0);

    // Loop all variables.
    for (std::size_t n = 0; n < m_variables.size(); ++n)
    {
        const std::string &variable = m_variables[n];
        if (!m_session->DefinesFunction(m_name, variable))
        {
            continue;
        }
        const auto type = m_session->GetFunctionType(m_name, variable);
        ASSERTL0(type == LibUtilities::eFunctionTypeFile,
                 "SessionFunction::EvaluateFld only supports file functions.");

        const std::string filename =
            m_session->GetFunctionFilename(m_name, variable);
        const auto ext = fs::path(filename).extension().string();
        ASSERTL0(ext != ".pts" && ext != ".csv",
                 "SolverCore SessionFunction only supports .fld initial "
                 "conditions; .pts/.csv interpolation is not implemented.");

        // Check if field was already imported, and import if not. Only the
        // parts of a file written in parallel that hold elements of this
        // process are read.
        if (m_importedFields.find(filename) == m_importedFields.end())
        {
            // An empty array would compare equal to NullInt1DArray and read
            // every part, so a process without elements asks for an id that
            // no element has. It still takes part in the import, which is
            // collective for HDF5 files.
            const int nElmts = m_expList->GetExpSize();
            Array<OneD, int> elementIDs(std::max(nElmts, 1), -1);
            for (int i = 0; i < nElmts; ++i)
            {
                elementIDs[i] = m_expList->GetExp(i)->GetGeom()->GetGlobalID();
            }

            ImportedField imported;
            auto fldIO =
                LibUtilities::FieldIO::CreateForFile(m_session, filename);
            fldIO->Import(filename, imported.fieldDefs, imported.fieldData,
                          LibUtilities::NullFieldMetaDataMap, elementIDs);
            m_importedFields.emplace(filename, std::move(imported));
        }

        std::string fileVar =
            m_session->GetFunctionFilenameVariable(m_name, variable);
        if (fileVar.empty())
        {
            fileVar = variable;
        }

        std::vector<double> compCoeffs(ncoeffs, 0.0);
        auto &imported = m_importedFields.at(filename);
        // Re-order FieldData into m_expList order. A file may split its
        // fields over several definitions, as one written with fields of
        // different orders does, so only those holding the variable are read.
        bool found = false;
        for (std::size_t i = 0; i < imported.fieldDefs.size(); ++i)
        {
            const auto &defFields = imported.fieldDefs[i]->m_fields;
            if (std::find(defFields.begin(), defFields.end(), fileVar) ==
                defFields.end())
            {
                continue;
            }
            found = true;
            // ExpList handles element-id reordering and modal interpolation.
            m_expList->ExtractDataToCoeffs(imported.fieldDefs[i],
                                           imported.fieldData[i], fileVar,
                                           compCoeffs);
        }
        // A process without elements reads no part of the file.
        ASSERTL0(found || m_expList->GetExpSize() == 0,
                 "Field (" + fileVar + ") not found in file " + filename + ".");

        std::copy(compCoeffs.begin(), compCoeffs.end(),
                  allCoeffs.begin() + n * ncoeffs);
    }

    coeffs.CopyVector<NektarSpaces::HostSpace>(allCoeffs);
}

} // namespace Nektar::SolverCore
