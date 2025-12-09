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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "Operators/ElmtOps/Expression/ExpressionBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class ExpressionBlockOpImpl : public ExpressionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    ExpressionBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : ExpressionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nqTot     = exp->GetTotPoints();

        m_coordptr = this->m_dataWarehouse->template GetData<MemSpace>(
            CoordKey<TData>(block_idx, m_implInterleaveWidth, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            ExpressionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;

    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    std::vector<unsigned int> m_numEvars;
    const TData *m_coordptr;

    TData m_time;
    TData m_scale;

    std::vector<LibUtilities::EquationSharedPtr> m_expressions;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        const auto nelmt = inblock.GetNumElements();
        const auto compSize =
            inblock.GetNumData() * inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        unsigned int nc;
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Get component index
            nc = n / inblock.GetNumHomoModes();

            // Pre-allocate vector for point-wise fielddata
            std::vector<double> fielddata(m_numEvars[nc]);

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      outblock.GetNumElementsWithPadding(),
                                      outblock.GetNumData(), (TData *)outptr);

            // Evaluate expression.
            auto coordptr = m_coordptr;
            for (size_t e = 0, cnt = 0; e < nelmt; e++)
            {
                // Kernel operation.
                unsigned int nev;
                TData fce = 0.0;
                for (unsigned int pt = 0; pt < m_nqTot; ++pt, ++cnt)
                {
                    // Gather fielddata
                    fielddata[0] = *(coordptr);
                    fielddata[1] = *(coordptr + 1);
                    fielddata[2] = *(coordptr + 2);
                    fielddata[3] = m_time;

                    // Add EVARS, if required
                    // Note we assume that inblock holds all fields as
                    // components
                    nev = 0;
                    for (unsigned i = 4; i < m_numEvars[nc]; i++, nev++)
                    {
                        fielddata[i] = *(inptr + nev * compSize + cnt);
                    }

                    // Evaluate the function assuming fixed input of x, y and z
                    // coordinate.
                    fce = m_expressions[n]->Evaluate(fielddata);

                    // Add fce to outptr.
                    *(outptr + cnt) += m_scale * fce;

                    coordptr += m_dimension;
                }
            }

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      inblock.GetNumElementsWithPadding(),
                                      inblock.GetNumData(), (TData *)inptr);

            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      outblock.GetNumElementsWithPadding(),
                                      outblock.GetNumData(), (TData *)outptr);

            // Increment pointer.
            outptr += outblock.size();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        this->m_expressions = exprs;
    }

    void v_SetTime(const TData &time) override
    {
        this->m_time = time;
    }

    void v_SetScale(const TData &scale) override
    {
        this->m_scale = scale;
    }

    void v_SetNumEvars(const std::vector<unsigned int> &numEvars) override
    {
        this->m_numEvars = numEvars;
    }
};

} // namespace Nektar::Operators::detail
