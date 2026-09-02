///////////////////////////////////////////////////////////////////////////////
//
// File: MeanRemovalOpImpl.hpp
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

#include "SolverCore/MeanRemoval/MeanRemovalOp.hpp"

#include <LibUtilities/BasicUtils/Math/Math.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class MeanRemovalOpImpl : public MeanRemovalOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MeanRemovalOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : MeanRemovalOp<TData>(expansionList, components)
    {
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory.
    static std::unique_ptr<Operators::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<MeanRemovalOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &inout) override
    {
        ASSERTL1(inout.GetNumHomoModes() == 1,
                 "MeanRemoval is not implemented for homogeneous expansions.");

        const auto nComp = inout.GetNumComponents();

        // m_data layout: [ integral_c0, ..., integral_c{n-1}, volume ].
        if (this->m_data.size() != nComp + 1)
        {
            this->m_data = LibUtilities::MemoryRegion<TData>(
                "MeanRemoval", nComp + 1, eHostPinned);
        }
        this->m_data.template Initialize<MemSpace>(TData{0});

        for (unsigned int blk = 0; blk < inout.GetBlocks().size(); ++blk)
        {
            auto &inblock = inout.GetBlocks()[blk];
            this->m_blockOp[blk]->Apply(inblock, this->m_data);
        }

        auto rowComm = this->m_expansionList->GetComm()->GetRowComm();
        rowComm->template AllReduce<MemSpace>(this->m_data,
                                              LibUtilities::ReduceSum);

        auto dataPtr =
            this->m_data.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        ASSERTL0(dataPtr[nComp] > 0.0,
                 "MeanRemovalOp encountered a non-positive volume.");

        // Negated per-component volume-averaged mean, ready for addScalar.
        std::vector<TData> negMean(nComp);
        for (unsigned int nc = 0; nc < nComp; ++nc)
        {
            negMean[nc] = -(dataPtr[nc] / dataPtr[nComp]);
        }

        Math::addScalar<ExecSpace>(negMean, inout, inout);
    }
};

} // namespace Nektar::SolverCore::detail
