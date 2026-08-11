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

#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"
#include <MultiRegions/Field/Field.hpp>

#include <SolverCore/SolverCoreDeclspec.h>

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

    SOLVER_CORE_EXPORT void EvaluateFld(
        const std::vector<std::string> &variables,
        MultiRegions::Field<double, FieldState::Coeff> &coeffs,
        [[maybe_unused]] double time, unsigned int domain = 0) const;

    SOLVER_CORE_EXPORT void EvaluateExpression(
        const std::vector<std::string> &variables,
        MultiRegions::Field<double, FieldState::Phys> &phys, double time) const;

    SOLVER_CORE_EXPORT std::string Describe(const std::string &variable,
                                            int domain = 0) const;

private:
    LibUtilities::SessionReaderSharedPtr m_session;
    MultiRegions::ExpListSharedPtr m_expList;
    std::string m_name;
};

using SessionFunctionSharedPtr = std::shared_ptr<SessionFunction>;

} // namespace Nektar::SolverCore
