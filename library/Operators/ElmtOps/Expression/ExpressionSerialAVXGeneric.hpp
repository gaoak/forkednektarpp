///////////////////////////////////////////////////////////////////////////////
//
// File: ExpressionSerialAVXGeneric.hpp
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
        this->m_dimension = exp->GetShapeDimension();
        this->m_coordDim  = exp->GetCoordim();
        this->m_nqTot     = exp->GetTotPoints();

        this->m_coordptr = this->m_dataWarehouse->template GetData<MemSpace>(
            CoordKey<TData>(block_idx, this->m_implInterleaveWidth, false));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
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
    const TData *m_coordptr;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_expressions.size() == inblock.GetNumComponents() &&
                     this->m_expressions.size() == outblock.GetNumComponents(),
                 "Number of expressions must match number of components in "
                 "input and output Field when calling Apply().")

        ASSERTL1(this->m_expressions.size() == this->m_cmask.size(),
                 "Number of expressions must match size of component mask when "
                 "calling Apply().")

        const auto nelmt    = inblock.GetNumElements();
        const auto compSize = inblock.CompSize();

        // Initialize pointers.
        auto inptr = (&inblock != &outblock)
                         ? inblock.template GetPtr<MemSpace, ReadOnly>()
                         : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr =
            (&inblock != &outblock)
                ? (this->m_append)
                      ? outblock.template GetPtr<MemSpace, ReadWrite>()
                      : outblock.template GetPtr<MemSpace, WriteOnly>()
                : (TData *)inptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int n = 0; n < inblock.GetNumHomoModes(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                ReshapeStorage<ExecSpace>(
                    this->m_implInterleaveWidth, interleaveWidth,
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr +
                        (n + nc * inblock.GetNumHomoModes()) * compSize);
            }

            // Synchronization barrier to allow SYCL-CPU back-end to reuse
            // Serial/AVX code.
            if constexpr (std::is_same_v<ExecSpace, NektarSpaces::Device>)
            {
                nekStreamSynchronize(nullptr);
            }

            for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
            {
                // Check component mask
                if (this->m_cmask[nc])
                {
                    if (&inblock != &outblock && this->m_append)
                    {
                        ReshapeStorage<ExecSpace>(
                            this->m_implInterleaveWidth, interleaveWidth,
                            outblock.GetNumElementsWithPadding(),
                            outblock.GetNumData(),
                            (TData *)outptr +
                                (n + nc * outblock.GetNumHomoModes()) *
                                    compSize);
                    }

                    // Pre-allocate vector for point-wise fielddata
                    std::vector<double> fielddata(this->m_numEvars[nc]);

                    // Evaluate expression.
                    auto coordptr = this->m_coordptr;
                    for (size_t e = 0, cnt = 0; e < nelmt; e++)
                    {
                        // Kernel operation.
                        for (unsigned int pt = 0; pt < this->m_nqTot;
                             ++pt, ++cnt)
                        {
                            // Gather fielddata
                            // Note we set y and z coordinate only if the
                            // coordinate dimension is large enough otherwise
                            // they default to zero This is relevant for example
                            // for boundary elements where the domain uses one
                            // more dimension than the boundary
                            fielddata[0] = coordptr[0];
                            fielddata[1] =
                                this->m_coordDim > 1 ? coordptr[1] : 0.0;
                            fielddata[2] =
                                this->m_coordDim > 2 ? coordptr[2] : 0.0;
                            fielddata[3] = this->m_time;

                            // Add EVARS, if required
                            // Note we assume that inblock holds all fields as
                            // components
                            for (unsigned i = 4, nev = 0;
                                 i < this->m_numEvars[nc]; i++, nev++)
                            {
                                fielddata[i] =
                                    inptr[(n +
                                           nev * inblock.GetNumHomoModes()) *
                                              compSize +
                                          cnt];
                            }

                            // Evaluate the function assuming fixed input of x,
                            // y and z coordinate.
                            auto fce =
                                this->m_expressions[nc]->Evaluate(fielddata);

                            // Add fce to outptr.
                            const size_t ind =
                                (n + nc * outblock.GetNumHomoModes()) *
                                    compSize +
                                cnt;
                            outptr[ind] =
                                (this->m_append)
                                    ? outptr[ind] + this->m_scale * fce
                                    : this->m_scale * fce;

                            coordptr += this->m_coordDim;
                        }
                    }

                    // Synchronization barrier to allow SYCL-CPU back-end to
                    // reuse Serial/AVX code.
                    if constexpr (std::is_same_v<ExecSpace,
                                                 NektarSpaces::Device>)
                    {
                        nekStreamSynchronize(nullptr);
                    }

                    // Reshape, if necessary.
                    if (&inblock != &outblock)
                    {
                        ReshapeStorage<ExecSpace>(
                            interleaveWidth, this->m_implInterleaveWidth,
                            outblock.GetNumElementsWithPadding(),
                            outblock.GetNumData(),
                            (TData *)outptr +
                                (n + nc * outblock.GetNumHomoModes()) *
                                    compSize);
                    }
                }
            }

            // Reshape, if necessary.
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, this->m_implInterleaveWidth,
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr +
                        (n + nc * inblock.GetNumHomoModes()) * compSize);
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetExpressions(
        const std::vector<LibUtilities::EquationSharedPtr> &exprs) override
    {
        this->m_expressions = exprs;
    }
};

} // namespace Nektar::Operators::detail
