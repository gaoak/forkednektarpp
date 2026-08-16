///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2OpImpl.hpp
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

#include "Operators/Norm/NormL2/NormL2Op.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class NormL2OpImpl : public NormL2Op<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormL2OpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : NormL2Op<TData>(expansionList, components)
    {
    }

    static std::string className;

    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<NormL2OpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) override
    {
        ASSERTL1(in.GetNumHomoModes() == 1,
                 "The NormL2 is not implemented for homogeneous expansions.");

        auto numComp = in.GetNumComponents();
        if (this->m_data.size() != numComp + 1)
        {
            this->m_data = LibUtilities::MemoryRegion<TData>(
                "NormL2", numComp + 1, eHostPinned);
        }
        this->m_data.template Initialize<MemSpace>(TData{0});

        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblock = in.GetBlocks()[blk];
            this->m_blockOp[blk]->Apply(inblock, this->m_data);
        }

        auto rowComm = this->m_expansionList->GetComm()->GetRowComm();
        rowComm->template AllReduce<MemSpace>(this->m_data,
                                              LibUtilities::ReduceSum);

        auto dataPtr =
            this->m_data.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

        if (this->m_normalised)
        {
            ASSERTL1(dataPtr[numComp] > 0.0,
                     "NormL2Op encountered a non-positive volume.");
        }
        else
        {
            dataPtr[numComp] = 1.0;
        }

        for (unsigned int nc = 0; nc < numComp; ++nc)
        {
            dataPtr[nc] = std::sqrt(dataPtr[nc] / dataPtr[numComp]);
        }
    }
};

} // namespace Nektar::Operators::detail
