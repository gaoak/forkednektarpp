///////////////////////////////////////////////////////////////////////////////
//
// File: NormLinfBlockOpSerialAVX.hpp
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

#include "SolverCore/NormOps/NormLinf/NormLinfBlockOp.hpp"
#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class NormLinfBlockOpImpl : public NormLinfBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormLinfBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : NormLinfBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
    }

    static std::string className;

    static std::unique_ptr<NormLinfBlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<NormLinfBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                     TData>::type::width;

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::MemoryRegion<TData> &data) override
    {
        const auto numComp = inblock.GetNumComponents();

        // Loop all components.
        const auto maskptr =
            Math::internalMathKernelMask<MemSpace>::GetInstance(inblock);
        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>();
        for (unsigned int nc = 0; nc < numComp * inblock.GetNumHomoModes();
             ++nc)
        {
            Math::linfnormKernel<ExecSpace, false>(inblock.CompSize(), maskptr,
                                                   inptr, dataptr + nc);
            inptr += inblock.CompSize();
        }
    }
};

} // namespace Nektar::SolverCore::detail
