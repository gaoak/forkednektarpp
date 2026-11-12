///////////////////////////////////////////////////////////////////////////////
//
// File: ForcingBody.cpp
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
// Description: Body forcing
//
///////////////////////////////////////////////////////////////////////////////

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/Forcing/ForcingBody.h>

namespace Nektar::SolverCore
{

std::string ForcingBody::className =
    GetForcingFactory().RegisterCreatorFunction("Body", ForcingBody::create,
                                                "Body Forcing");

ForcingBody::ForcingBody(const LibUtilities::SessionReaderSharedPtr &session,
                         const MultiRegions::ExpListSharedPtr &expList,
                         const std::vector<std::string> &variables)
    : Forcing(session, expList, variables)
{
}

void ForcingBody::v_InitObject(const LibUtilities::SessionNode &force)
{
    // Read the function name from BODYFORCE, preserving the legacy
    // session-file syntax.
    m_funcName = force.GetValue("BODYFORCE");
    ASSERTL0(!m_funcName.empty(), "BODYFORCE must specify a function name.");

    m_function = std::make_shared<SessionFunction>(m_session, m_expList,
                                                   m_variables, m_funcName);

    m_constant = m_function->IsConstant();
}

void ForcingBody::v_Apply(LibUtilities::Field<double, FieldState::Phys> &in,
                          LibUtilities::Field<double, FieldState::Phys> &out,
                          const double time, const double factor)
{
    if (!m_constant)
    {
        m_function->SetScale(factor);
        m_function->SetAppend(m_append);
        m_function->Evaluate(in, out, time);
    }
    else
    {
        // A constant forcing is evaluated once, on first use.
        if (!m_constantForcing)
        {
            m_constantForcing = LibUtilities::Field<double, FieldState::Phys>(
                "ForcingBody constant forcing",
                MultiRegions::GetBlockAttributes<double, FieldState::Phys>(
                    m_expList,
                    MultiRegions::Operator<double>::GetDefaultInterleaveWidth(
                        m_session)),
                m_variables, out.GetNumHomoModes());
            m_function->Evaluate(m_constantForcing, 0.0);
        }

        if (m_append)
        {
            m_math.daxpy(factor, m_constantForcing, out, out);
        }
        else
        {
            m_math.mul(factor, m_constantForcing, out);
        }
    }
}

void ForcingBody::v_SetAppend(const bool append)
{
    m_append = append;
}

} // namespace Nektar::SolverCore
