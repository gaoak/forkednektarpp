///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzImplBase.hpp
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

#include "ElmtOps/OperatorHelmholtz.hpp"

#include "ElmtOps/OperatorBwdTrans.hpp"
#include "ElmtOps/OperatorIProductWRTBase.hpp"
#include "ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "ElmtOps/OperatorPhysDeriv.hpp"

#include "Common/OperatorHelper.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorHelmholtzImplBase : public OperatorHelmholtz<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorHelmholtzImplBase(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmholtz<TData>(expansionList)
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_diffCoeff = MemoryRegion<TData>::template create<MemSpace>(
            "Helmholtz diffCoeff", nCoord * nCoord);

        m_diffCoeff.initialize(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        for (size_t d = 0; d < nCoord; d++)
        {
            diffCoeff[d * nCoord + d] = 1.0; // temporary solution
        }
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    MemoryRegion<TData> m_diffCoeff;
};

} // namespace Nektar::Operators::detail
