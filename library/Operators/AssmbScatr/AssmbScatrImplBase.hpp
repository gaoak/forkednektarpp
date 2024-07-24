///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrImplBase.hpp
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

#include "Operators/OperatorAssmbScatr.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

// Base implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorAssmbScatrImplBase : public OperatorAssmbScatr<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorAssmbScatrImplBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        m_solnType = m_assmbMap->GetGlobalSysSolnType();
        m_nLocal   = m_assmbMap->GetNumLocalCoeffs();
        m_nGlobal  = (m_solnType == eIterativeFull)
                         ? m_assmbMap->GetNumGlobalCoeffs()
                         : m_assmbMap->GetNumGlobalBndCoeffs();
        m_nDir     = m_assmbMap->GetNumGlobalDirBndCoeffs();

        m_global = MemoryRegion<TData>::template create<MemSpace>(
            "AssmbScatr global", m_nGlobal);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false) override
    {
        this->Assemble(in, m_global);

        // Zeroing Dirichlet BC
        if (zeroDir)
        {
            m_global.initialize(0, m_nDir);
        }

        this->GlobalToLocal(m_global, out);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
    GlobalSysSolnType m_solnType;

    MemoryRegion<TData> m_global;

    size_t m_nLocal;
    size_t m_nGlobal;
    size_t m_nDir;
};

} // namespace Nektar::Operators::detail
