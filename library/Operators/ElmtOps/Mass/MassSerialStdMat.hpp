///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialStdMat.hpp
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

#include "Operators/ElmtOps/OperatorBwdTrans.hpp"
#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "Operators/ElmtOps/OperatorMass.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorMassImpl : public OperatorMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMassImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMass<TData>(expansionList)
    {
        m_BwdTransOp =
            BwdTrans<TData>::template Create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::template Create<ExecSpace, Implementation>(
                this->m_expansionList);
        m_PhysBlockAttributes =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        int CompSize = in.GetNumComponents();

        // initialise bwd storage space if not for correct number of components
        if (m_bwd.GetNumComponents() != CompSize)
        {
            m_bwd = Field<TData, FieldState::Phys>::template Create<MemSpace>(
                "Mass tmp", m_PhysBlockAttributes, CompSize,
                ExecSpace::alignment);
        }

        // Step 1: BwdTrans
        m_BwdTransOp->apply(in, m_bwd);

        // Step 2: Inner product for mass matrix operation
        m_IProductWRTBaseOp->apply(m_bwd, out);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorMassImpl<ExecSpace, Implementation, TData>>(expansionList);
    }

protected:
    Field<TData, FieldState::Phys> m_bwd;

    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;

    std::vector<BlockAttributes> m_PhysBlockAttributes;
};

} // namespace Nektar::Operators::detail
