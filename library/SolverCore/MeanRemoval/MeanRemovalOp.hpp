///////////////////////////////////////////////////////////////////////////////
//
// File: MeanRemovalOp.hpp
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

#include "SolverCore/MeanRemoval/MeanRemovalBlockOp.hpp"
#include "SolverCore/SolverCore.hpp"
#include <MultiRegions/Common/Operator.hpp>

namespace Nektar::SolverCore
{

// Subtracts the volume-averaged mean of a physical-space field from itself,
// in place. A PDE variable governed by a singular system (e.g. pressure
// under Neumann/periodic-only boundaries) is only determined up to an
// arbitrary additive constant; this operator removes that constant so that
// norms/errors computed from the field reflect genuine discretisation error
// rather than the arbitrary offset.
//
// Dispatch across execution spaces (Serial/AVX/Device) is resolved entirely
// by Create(), via the same OperatorFactory used by every Operators-library
// operator -- callers just do MeanRemovalOp<TData>::Create(...)->Apply(field),
// with no execution-space conditionals of their own.
template <typename TData>
class MeanRemovalOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<MeanRemovalOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        // Force a genuine cross-library symbol reference into libSolverCore so
        // the MeanRemoval*OpImpl static factory registrations are not dropped
        // by the linker/loader -- see EnsureLinked() in SolverCore.hpp.
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        auto op = MultiRegions::Operator<TData>::template Create<MeanRemovalOp>(
            expansionList, components, execStr0);

        auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);

        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];

            op->m_blockOp.push_back(MeanRemovalBlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));
        }

        return op;
    }

    static inline const std::string name = "MeanRemoval";

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        v_Apply(inout);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &inout)
    {
        v_Apply(inout);
    }

protected:
    LibUtilities::MemoryRegion<TData> m_data;
    std::vector<std::shared_ptr<MeanRemovalBlockOp<TData>>> m_blockOp;

    MeanRemovalOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~MeanRemovalOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &inout) = 0;
};
} // namespace Nektar::SolverCore
