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
#include <MultiRegions/Common/Operator.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>

#include <algorithm>
#include <map>

namespace Nektar::SolverCore
{

SessionFunction::SessionFunction(LibUtilities::SessionReaderSharedPtr session,
                                 MultiRegions::ExpListSharedPtr expList,
                                 std::string functionName)
    : m_session(std::move(session)), m_expList(std::move(expList)),
      m_name(std::move(functionName)),
      m_importedFields(std::make_shared<std::map<std::string, ImportedField>>())
{
    ASSERTL0(m_session->DefinesFunction(m_name),
             "Function '" + m_name + "' does not exist.");
}

/**
 * @brief Construct the function of @p function on another expansion list.
 *
 * The two share the field files read so far and those either reads later,
 * so a file read through one is not read again through the other. What is
 * read is kept until the last of them is destroyed. Since a file is read for
 * the elements of the list that reads it, @p expList must hold the elements
 * of the expansion list of @p function, in the same order.
 *
 * Where @p expList has other bases on the same quadrature points, as a
 * lower-order pressure has, a file is read through the expansion list of
 * @p function and projected element by element onto @p expList (see
 * EvaluateFld()). Otherwise it is read into @p expList.
 *
 * @param   function    Function to evaluate on @p expList.
 * @param   expList     Expansion list to evaluate it on.
 */
SessionFunction::SessionFunction(const SessionFunction &function,
                                 MultiRegions::ExpListSharedPtr expList)
    : m_session(function.m_session), m_expList(std::move(expList)),
      m_name(function.m_name), m_importedFields(function.m_importedFields)
{
    bool sameElements =
        m_expList->GetExpSize() == function.m_expList->GetExpSize();
    bool samePoints = true;
    bool otherBases = false;
    for (int i = 0; sameElements && i < m_expList->GetExpSize(); ++i)
    {
        const auto &exp     = m_expList->GetExp(i);
        const auto &fileExp = function.m_expList->GetExp(i);
        sameElements =
            exp->GetGeom()->GetGlobalID() == fileExp->GetGeom()->GetGlobalID();
        samePoints =
            samePoints && exp->GetPointsKeys() == fileExp->GetPointsKeys();
        for (int d = 0; d < exp->GetNumBases(); ++d)
        {
            otherBases = otherBases || exp->GetBasis(d)->GetBasisKey() !=
                                           fileExp->GetBasis(d)->GetBasisKey();
        }
    }
    ASSERTL0(sameElements, "SessionFunction can only be shared between "
                           "expansion lists of the same elements.");

    if (otherBases && samePoints)
    {
        m_fileExpList = function.m_expList;
    }
}

/**
 * @brief Read the coefficients of @p variables from the file function into
 * the element-wise coefficient order of @p expList, which holds the elements
 * of #m_expList.
 *
 * Each element's modes are copied from the file, truncated or padded to the
 * expansion's order. The result holds all components one after the other.
 * A file is read once and kept for later calls, here and on the copies of
 * this function made for other expansion lists.
 */
std::vector<double> SessionFunction::ReadFldCoeffs(
    const MultiRegions::ExpListSharedPtr &expList,
    const std::vector<std::string> &variables, unsigned int domain) const
{
    const std::size_t ncoeffs = expList->GetNcoeffs();
    std::vector<double> allCoeffs(ncoeffs * variables.size(), 0.0);
    auto &importedFields = *m_importedFields;

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

        // Check if field was already imported, and import if not. Only the
        // parts of a file written in parallel that hold elements of this
        // process are read.
        if (importedFields.find(filename) == importedFields.end())
        {
            // An empty array would compare equal to NullInt1DArray and read
            // every part, so a process without elements asks for an id that
            // no element has. It still takes part in the import, which is
            // collective for HDF5 files.
            const int nElmts = expList->GetExpSize();
            Array<OneD, int> elementIDs(std::max(nElmts, 1), -1);
            for (int i = 0; i < nElmts; ++i)
            {
                elementIDs[i] = expList->GetExp(i)->GetGeom()->GetGlobalID();
            }

            ImportedField imported;
            auto fldIO =
                LibUtilities::FieldIO::CreateForFile(m_session, filename);
            fldIO->Import(filename, imported.fieldDefs, imported.fieldData,
                          LibUtilities::NullFieldMetaDataMap, elementIDs);
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
        // Re-order FieldData into expList order. A file may split its
        // fields over several definitions, as one written with a pressure of
        // another order does, so only those holding the variable are read.
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
            expList->ExtractDataToCoeffs(imported.fieldDefs[i],
                                         imported.fieldData[i], fileVar,
                                         compCoeffs);
        }
        // A process without elements reads no part of the file.
        ASSERTL0(found || expList->GetExpSize() == 0,
                 "Field (" + fileVar + ") not found in file " + filename + ".");

        std::copy(compCoeffs.begin(), compCoeffs.end(),
                  allCoeffs.begin() + n * ncoeffs);
    }

    return allCoeffs;
}

/**
 * @brief Read the file function of @p variables into @p coeffs.
 *
 * The file is read into #m_expList, each element's modes truncated or
 * padded to the expansion's order. A copy of the function made for an
 * expansion list with other bases on the same quadrature points instead
 * reads it into the expansion list of the function copied, evaluates it at
 * the quadrature points (BwdTransOp) and projects it element by element
 * onto #m_expList (IProductWRTBaseOp, then MultiplyByElmtInvMassOp). For a
 * lower order, that differs from the truncation: with a modified basis,
 * truncation to linear order keeps the vertex values only.
 */
void SessionFunction::EvaluateFld(
    const std::vector<std::string> &variables,
    LibUtilities::Field<double, FieldState::Coeff> &coeffs,
    [[maybe_unused]] double time, unsigned int domain) const
{
    ASSERTL0(coeffs.GetNumComponents() == variables.size(),
             "SessionFunction::EvaluateFld mismatch in supplied number of "
             "variables and number of components for Field.");

    if (!m_fileExpList)
    {
        coeffs.CopyVector<NektarSpaces::HostSpace>(
            ReadFldCoeffs(m_expList, variables, domain));
        return;
    }

    // Both lists hold the same elements in the same order, on the same
    // quadrature points, so the values read are moved from one list's
    // storage to the other's through their element-wise order.
    const auto interleaveWidth =
        MultiRegions::Operator<double>::GetDefaultInterleaveWidth(m_session);
    const auto ncomps = coeffs.GetNumComponents();
    const auto nhomo  = coeffs.GetNumHomoModes();
    auto fileCoeffs   = LibUtilities::Field<double, FieldState::Coeff>(
        "file coefficients",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_fileExpList, interleaveWidth),
        ncomps, nhomo);
    auto filePhys = LibUtilities::Field<double, FieldState::Phys>(
        "file values",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            m_fileExpList, interleaveWidth),
        ncomps, nhomo);
    auto phys = LibUtilities::Field<double, FieldState::Phys>(
        "file values projected",
        MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
            m_expList, interleaveWidth),
        ncomps, nhomo);
    auto iprod = LibUtilities::Field<double, FieldState::Coeff>(
        "file inner product",
        MultiRegions::GetBlockAttributes<double, FieldState::Coeff>(
            m_expList, interleaveWidth),
        ncomps, nhomo);

    fileCoeffs.CopyVector<NektarSpaces::HostSpace>(
        ReadFldCoeffs(m_fileExpList, variables, domain));
    MultiRegions::BwdTransOp<double>::Create(m_fileExpList, variables)
        ->Apply(fileCoeffs, filePhys);
    phys.CopyVector<NektarSpaces::HostSpace>(filePhys.ToVector<double>());
    MultiRegions::IProductWRTBaseOp<double>::Create(m_expList, variables)
        ->Apply(phys, iprod);
    MultiRegions::MultiplyByElmtInvMassOp<double>::Create(m_expList, variables)
        ->Apply(iprod, coeffs);
}

void SessionFunction::EvaluateExpression(
    const std::vector<std::string> &variables,
    LibUtilities::Field<double, FieldState::Phys> &phys, double time) const
{
    ASSERTL0(phys.GetNumComponents() == variables.size(),
             "SessionFunction::EvaluateExpression mismatch in supplied number "
             "of variables and number of components for Field.");

    // Initialise operators
    auto initialOp =
        MultiRegions::ExpressionOp<double>::Create(m_expList, variables);

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
