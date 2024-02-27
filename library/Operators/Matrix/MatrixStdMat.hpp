///////////////////////////////////////////////////////////////////////////////
//
// File: MatrixStdMat.hpp
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
#include <numeric>

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
class OperatorMatrixImpl<TData, TFieldState, ImplStdMat>
    : public OperatorMatrix<TData, TFieldState>
{
public:
    OperatorMatrixImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMatrix<TData, TFieldState>(expansionList)
    {
        // get size of the matrix from the blocks given by state and
        // expansionlist
        auto blocks = GetBlockAttributes(TFieldState, expansionList);
        m_size      = std::accumulate(blocks.begin(), blocks.end(), 0,
                                 [](size_t acc, const BlockAttributes &block) {
                                     return acc + block.block_size;
                                 });

        // create memory for square matrix of given size
        m_matrix = std::vector<TData>(m_size * m_size);
    }

    size_t size() override
    {
        return m_size;
    }

    void fill(std::vector<TData> src) override
    {
        std::copy(src.begin(), src.end(), m_matrix.begin());
    }

    std::string toString() override
    {
        auto pMat = m_matrix.cbegin();
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

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        auto *pIn    = in.GetStorage().GetCPUPtr();
        auto *pOut   = out.GetStorage().GetCPUPtr();
        auto pMatrix = m_matrix.cbegin();

        for (size_t i = 0; i < m_size; ++i)
        {
            *(pOut) = 0.;
            for (size_t j = 0; j < m_size; ++j)
            {
                *(pOut) += *(pIn) * *(pMatrix);
                pIn++;
                pMatrix++;
            }
            pOut++;
            pIn = in.GetStorage().GetCPUPtr();
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorMatrixImpl<TData, TFieldState, ImplStdMat>>(expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    size_t m_size;
    std::vector<TData> m_matrix;
};

} // namespace Nektar::Operators::detail
