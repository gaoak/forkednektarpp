///////////////////////////////////////////////////////////////////////////////
//
// File: NormLinfOp.hpp
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

#include <MultiRegions/ContField.h>

#include "Operators/Common/Operator.hpp"
#include "Operators/Math/Math.hpp"
#include "Operators/Norm/NormLinf/NormLinfBlockOp.hpp"

namespace Nektar::Operators
{

template <typename TData> class NormLinfOp : public Operator<TData>
{
public:
    ~NormLinfOp() override = default;

    static std::shared_ptr<NormLinfOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? session->GetCmdLineArgument<std::string>("opExecSpace")
                : execStr;

        auto op = Operator<TData>::template Create<NormLinfOp>(
            expansionList, components, execStr0);

        auto blocks =
            GetBlockAttributes<TData, FieldState::Phys>(expansionList);

        for (unsigned int block_idx = 0; block_idx < blocks.size(); block_idx++)
        {
            const auto exp_idx = GetCollection(expansionList, block_idx)
                                     .GetExpVector()[0]
                                     ->GetElmtId();
            const auto exp = expansionList->GetExp(exp_idx);

            op->m_blockOp.push_back(NormLinfBlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));
        }

        return op;
    }

    static inline const std::string name = "NormLinf";

    void Apply(Field<TData, FieldState::Phys> &in)
    {
        ASSERTL1(in.GetNumHomoModes() == 1,
                 "The NormLinf is not implemented for homogeneous expansions.");

        // Initialise norm memory region
        if (m_data.size() != in.GetNumComponents())
        {
            m_data = MemoryRegion<TData>("NormLinfDeviceReduce",
                                         in.GetNumComponents(), eHostPinned);
        }

        // Reset norms
        m_math.fill(0.0, m_data);

        // Loop all blocks
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &inblock = in.GetBlocks()[blk];

            this->m_blockOp[blk]->Apply(inblock, m_data);
        }

        // Communicate norms
        auto rowComm = this->m_expansionList->GetComm()->GetRowComm();
        rowComm->template AllReduce<NektarSpaces::DeviceSpace>(
            m_data, Nektar::LibUtilities::ReduceMax);
    }

    void operator()(Field<TData, FieldState::Phys> &in)
    {
        Apply(in);
    }

    std::vector<TData> GetNorms()
    {
        auto ptr = m_data.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        return std::vector<TData>(ptr, ptr + m_data.size());
    }

protected:
    Math m_math;
    MemoryRegion<TData> m_data;
    std::vector<std::shared_ptr<NormLinfBlockOp<TData>>> m_blockOp;

    NormLinfOp(const MultiRegions::ExpListSharedPtr &expansionList,
               const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }
};
} // namespace Nektar::Operators
