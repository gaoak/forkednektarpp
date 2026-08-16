///////////////////////////////////////////////////////////////////////////////
//
// File: NormLinfOpImpl.hpp
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

#include "Operators/Norm/NormLinf/NormLinfOp.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class NormLinfOpImpl : public NormLinfOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormLinfOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                   const std::vector<std::string> &components)
        : NormLinfOp<TData>(expansionList, components)
    {
    }

    static std::string className;

    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<NormLinfOpImpl<ExecSpace, TData>>(expansionList,
                                                                  components);
    }

protected:
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) override
    {
        ASSERTL1(in.GetNumHomoModes() == 1,
                 "The NormLinf is not implemented for homogeneous expansions.");

        // Initialise norm memory region
        if (this->m_data.size() != in.GetNumComponents())
        {
            this->m_data = LibUtilities::MemoryRegion<TData>(
                "NormLinfDeviceReduce", in.GetNumComponents(), eHostPinned);
        }

        // Reset norms
        this->m_data.template Initialize<MemSpace>(TData{0});

        // Loop all blocks
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblock = in.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, this->m_data);
        }

        // Communicate norms
        auto rowComm = this->m_expansionList->GetComm()->GetRowComm();
        rowComm->template AllReduce<MemSpace>(this->m_data,
                                              Nektar::LibUtilities::ReduceMax);
    }
};

} // namespace Nektar::Operators::detail
