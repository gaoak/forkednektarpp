////////////////////////////////////////////////////////////////////////////////
//
// File: SessionFunction.cpp
//
// For more information, please see: http://www.nektar.info/
//
// The MIT License
//
// Copyright (c) 2017 Kilian Lackhove
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

#include <map>

namespace Nektar::SolverCore
{

SessionFunction::SessionFunction(LibUtilities::SessionReaderSharedPtr session,
                                 MultiRegions::ExpListSharedPtr expList,
                                 std::string functionName)
    : m_session(std::move(session)), m_expList(std::move(expList)),
      m_name(std::move(functionName))
{
    ASSERTL0(m_session->DefinesFunction(m_name),
             "Function '" + m_name + "' does not exist.");
}

void SessionFunction::EvaluateFld(
    const std::vector<std::string> &variables,
    MultiRegions::Field<double, FieldState::Coeff> &coeffs,
    [[maybe_unused]] double time, [[maybe_unused]] unsigned int domain) const
{
    ASSERTL0(coeffs.GetNumComponents() == variables.size(),
             "SessionFunction::EvaluateFld mismatch in supplied number of "
             "variables and number of components for Field.");

    const std::size_t ncoeffs = m_expList->GetNcoeffs();
    std::vector<double> allCoeffs(ncoeffs * variables.size(), 0.0);
    std::map<std::string, ImportedField> importedFields;

    // Loop all variables.
    for (std::size_t n = 0; n < variables.size(); ++n)
    {
        const std::string &variable = variables[n];
        const auto type = m_session->GetFunctionType(m_name, variable, domain);
        ASSERTL0(type == LibUtilities::eFunctionTypeFile,
                 "SessionFunction::EvaluateFld only supports file functions.");

        const std::string filename =
            m_session->GetFunctionFilename(m_name, variable, domain);
        const auto ext = fs::path(filename).extension().string();
        ASSERTL0(ext != ".pts" && ext != ".csv",
                 "SolverCore SessionFunction only supports .fld initial "
                 "conditions; .pts/.csv interpolation is not implemented.");

        // Check if field was already imported, and import if not
        if (importedFields.find(filename) == importedFields.end())
        {
            ImportedField imported;
            auto fldIO =
                LibUtilities::FieldIO::CreateForFile(m_session, filename);
            fldIO->Import(filename, imported.fieldDefs, imported.fieldData);
            importedFields.emplace(filename, std::move(imported));
        }

        std::string fileVar =
            m_session->GetFunctionFilenameVariable(m_name, variable, domain);
        if (fileVar.empty())
        {
            fileVar = variable;
        }

        std::vector<double> compCoeffs(ncoeffs, 0.0);
        auto &imported = importedFields.at(filename);
        // Re-order FieldData into expList order
        for (std::size_t i = 0; i < imported.fieldDefs.size(); ++i)
        {
            // ExpList handles element-id reordering and modal interpolation.
            m_expList->ExtractDataToCoeffs(imported.fieldDefs[i],
                                           imported.fieldData[i], fileVar,
                                           compCoeffs);
        }

        std::copy(compCoeffs.begin(), compCoeffs.end(),
                  allCoeffs.begin() + n * ncoeffs);
    }

    coeffs.CopyVector<NektarSpaces::HostSpace>(allCoeffs);
}

void SessionFunction::EvaluateExpression(
    const std::vector<std::string> &variables,
    MultiRegions::Field<double, FieldState::Phys> &phys, double time) const
{
    ASSERTL0(phys.GetNumComponents() == variables.size(),
             "SessionFunction::EvaluateExpression mismatch in supplied number "
             "of variables and number of components for Field.");

    // Initialise operators
    auto initialOp =
        Operators::ExpressionOp<double>::Create(m_expList, variables);

    // Read initial conditions and configure operator
    std::vector<LibUtilities::EquationSharedPtr> initialConditons;
    for (const auto &var : variables)
    {
        initialConditons.push_back(m_session->GetFunction(m_name, var));
    }
    initialOp->SetExpressions(initialConditons);
    initialOp->SetTime(time);

    // Evaluate initial conditions from expression
    initialOp->Apply(phys, phys);
}

std::string SessionFunction::Describe(const std::string &variable,
                                      int domain) const
{
    const auto type = m_session->GetFunctionType(m_name, variable, domain);
    if (type == LibUtilities::eFunctionTypeExpression)
    {
        return m_session->GetFunction(m_name, variable, domain)
            ->GetExpression();
    }

    if (type == LibUtilities::eFunctionTypeFile)
    {
        return "from file " +
               m_session->GetFunctionFilename(m_name, variable, domain);
    }

    return "0 (default)";
}

} // namespace Nektar::SolverCore
