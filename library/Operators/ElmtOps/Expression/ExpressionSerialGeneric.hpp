///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionSerialGeneric.hpp
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
// Description: Implementation of the math expression evaluator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/Expression/OperatorExpression.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorExpressionImpl : public BlockOperatorExpression<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorExpressionImpl(const LocalRegions::ExpansionSharedPtr &exp,
                                NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorExpression<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nqTot     = exp->GetTotPoints();
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BlockOperatorExpressionImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;

    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;

    std::vector<LibUtilities::EquationSharedPtr> m_expressions;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        auto coordptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            CoordKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                            inblock.GetNumElements(), false));

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Evaluate expression.
            auto coordptr = coordptr_init;
            for (size_t e = 0, cnt = 0; e < nelmt; e++)
            {
                // Kernel operation.
                for (unsigned int pt = 0; pt < m_nqTot; ++pt, ++cnt)
                {
                    // Evaluate the function assuming fixed input of x, y and z
                    // coordinate.
                    auto fce = m_expressions[nc]->Evaluate(
                        *(coordptr), *(coordptr + 1), *(coordptr + 2));

                    // Add fce to outptr.
                    *(outptr + cnt) = *(inptr + cnt) + fce;

                    coordptr += m_dimension;
                }
            }

            // Increment pointer.
            inptr += inblock.size();
            outptr += outblock.size();
        }
    }

    void v_SetExpressions(
        std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        this->m_expressions = exprs;
    }
};

} // namespace Nektar::Operators::detail
