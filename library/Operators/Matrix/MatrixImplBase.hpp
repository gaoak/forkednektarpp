///////////////////////////////////////////////////////////////////////////////
//
// File: MatrixImplBase.hpp
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

#include "Operators/OperatorMatrix.hpp"

#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/Field/MemoryRegion.hpp"

namespace Nektar::Operators::detail
{

// Matrix implementation
template <typename ExecSpace, typename Implementation, FieldState TFieldState,
          typename TData>
class OperatorMatrixImplBase : public OperatorMatrix<TFieldState, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMatrixImplBase(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMatrix<TFieldState, TData>(expansionList)
    {
        // get size of the matrix from the blocks given by state and
        // expansionlist
        auto blocks = GetBlockAttributes(TFieldState, expansionList);
        m_size      = std::accumulate(blocks.begin(), blocks.end(), 0,
                                 [](size_t acc, const BlockAttributes &block) {
                                     return acc + block.block_size;
                                 });

        // create memory for square matrix of given size
        // allocate memory device
        m_matrix = MemoryRegion<TData>::template create<MemSpace>(
            "Matrix matrix", m_size * m_size);
    }

    size_t size() override
    {
        return m_size;
    }

    void fill(std::vector<TData> src) override
    {
        // Copy the host memory to the device memory
        m_matrix =
            MemoryRegion<TData>::template fromVector<MemSpace, TData>(src);
    }

    std::string toString() override
    {
        // Copy the device memory to the host memory for printing
        const TData *pMat =
            m_matrix.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        std::string str;
        for (size_t i = 0; i < m_size; ++i)
        {
            for (size_t j = 0; j < m_size; ++j)
            {
                if (j > 0)
                {
                    str += "\t";
                }

                str += std::to_string(*(pMat++));
            }

            str += "\n";
        }

        return str;
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    size_t m_size;

    MemoryRegion<TData> m_matrix;
};

} // namespace Nektar::Operators::detail
