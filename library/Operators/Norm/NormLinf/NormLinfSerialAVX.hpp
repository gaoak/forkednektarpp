///////////////////////////////////////////////////////////////////////////////
//
// File: NormLinfSerialAVX.hpp
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

#include <algorithm>
#include <cmath>

#include "Operators/Norm/NormLinf/NormLinfBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class NormLinfBlockOpImpl : public NormLinfBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormLinfBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        NekDataWarehouseSharedPtr dataWarehouse)
        : NormLinfBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
    }

    static std::string className;

    static std::unique_ptr<NormLinfBlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<NormLinfBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth =
        simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                     TData>::type::width;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 MemoryRegion<TData> &data) override
    {
        const auto numComp = inblock.GetNumComponents();

        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto nelmt           = inblock.GetNumElements();
        const auto ndata           = inblock.GetNumData();

        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>();

        for (unsigned int nc = 0; nc < numComp * inblock.GetNumHomoModes();
             ++nc)
        {
            TData acc = 0.0;
            for (size_t e = 0; e < nelmt; ++e)
            {
                const size_t lane   = e % interleaveWidth;
                const size_t group  = e / interleaveWidth;
                const size_t offset = group * interleaveWidth * ndata + lane;
                for (unsigned int q = 0; q < ndata; ++q)
                {
                    acc = std::max(
                        acc, std::abs(inptr[offset + q * interleaveWidth]));
                }
            }
            dataptr[nc] = std::max(acc, dataptr[nc]);

            inptr += inblock.CompSize();
        }
    }
};

} // namespace Nektar::Operators::detail
