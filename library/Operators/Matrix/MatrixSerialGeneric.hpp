///////////////////////////////////////////////////////////////////////////////
//
// File: MatrixSerialGeneric.hpp
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

#include "Operators/Matrix/MatrixImplBase.hpp"

#include <numeric>

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, FieldState TFieldState,
          typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value>::type>
class OperatorMatrixImpl
    : public OperatorMatrixImplBase<ExecSpace, Implementation, TFieldState,
                                    TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMatrixImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMatrixImplBase<ExecSpace, Implementation, TFieldState, TData>(
              expansionList)
    {
    }

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();
        const auto *matrixPtr =
            this->m_matrix.template GetPtr<MemSpace, ReadOnly>();

        auto *pIn     = inPtr;
        auto *pOut    = outPtr;
        auto *pMatrix = matrixPtr;

        for (size_t i = 0; i < this->m_size; ++i)
        {
            *(pOut) = 0.;

            for (size_t j = 0; j < this->m_size; ++j)
            {
                *(pOut) += *(pIn) * *(pMatrix);
                pIn++;
                pMatrix++;
            }

            pOut++;
            pIn = inPtr;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorMatrixImpl<ExecSpace, Implementation, TFieldState, TData>>(
            expansionList);
    }
};

} // namespace Nektar::Operators::detail
