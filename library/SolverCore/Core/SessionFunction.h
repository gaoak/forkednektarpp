///////////////////////////////////////////////////////////////////////////////
//
// File: SessionFunction.h
//
// For more information, please see: http://www.nektar.info
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
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <MultiRegions/ElmtOps/Expression/ExpressionOp.hpp>

#include <SolverCore/SolverCoreDeclspec.h>

#include <map>
#include <memory>

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
        MultiRegions::ExpListSharedPtr expList, std::string functionName);

    /// The same function on another expansion list, sharing the files read.
    SOLVER_CORE_EXPORT SessionFunction(const SessionFunction &function,
                                       MultiRegions::ExpListSharedPtr expList);

    SOLVER_CORE_EXPORT void EvaluateFld(
        const std::vector<std::string> &variables,
        LibUtilities::Field<double, FieldState::Coeff> &coeffs,
        [[maybe_unused]] double time, unsigned int domain = 0) const;

    SOLVER_CORE_EXPORT void EvaluateExpression(
        const std::vector<std::string> &variables,
        LibUtilities::Field<double, FieldState::Phys> &phys, double time) const;

    SOLVER_CORE_EXPORT std::string Describe(const std::string &variable,
                                            int domain = 0) const;

private:
    LibUtilities::SessionReaderSharedPtr m_session;
    MultiRegions::ExpListSharedPtr m_expList;
    std::string m_name;
    /// Field files read so far, by file name; shared by the copies of this
    /// function made for other expansion lists.
    std::shared_ptr<std::map<std::string, ImportedField>> m_importedFields;
    /// Expansion list a file is read through before it is projected onto
    /// #m_expList; null to read it into #m_expList.
    MultiRegions::ExpListSharedPtr m_fileExpList;

    /// Read file coefficients in the element order of @p expList.
    std::vector<double> ReadFldCoeffs(
        const MultiRegions::ExpListSharedPtr &expList,
        const std::vector<std::string> &variables, unsigned int domain) const;
};

using SessionFunctionSharedPtr = std::shared_ptr<SessionFunction>;

} // namespace Nektar::SolverCore
