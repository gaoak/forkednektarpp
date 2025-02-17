///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialStdMat.hpp
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

#include <StdRegions/StdExpansion.h>

#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorPhysDerivImpl : public BlockOperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorPhysDerivImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorPhysDeriv<TData>(exp, dataWarehouse)
    {
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Determine shape and type of the element.
        const auto deformed = this->m_exp->GetMetricInfo()->GetGtype() ==
                              SpatialDomains::eDeformed;
        const auto shapeType = this->m_exp->DetShapeType();
        const auto dimension = this->m_exp->GetShapeDimension();
        const auto nqTot     = this->m_exp->GetTotPoints();
        const auto nCoord    = this->m_exp->GetCoordim();
        const auto ndf       = nCoord * dimension;

        auto nElmts = inblock.GetNumElements();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < dimension; d++)
        {
            basisKeys[d] = this->m_exp->GetBasis(d)->GetBasisKey();
        }
        auto matptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, shapeType, ePhysDerivStdMat));

        // Fetch derivative factor.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Allocate storate.
        if (m_deriv.size() == 0)
        {
            m_deriv = std::vector<TData>(dimension * nqTot * nElmts);
        }

        // Get workspace pointer.
        auto derivPtr = m_deriv.data();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            auto dfptr = dfptr_init;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            for (unsigned int d = 0; d < dimension; ++d)
            {
                // Perform matrix-matrix multiply.
                Blas::Gemm('N', 'N', nqTot, nElmts, nqTot, 1.0,
                           matptr + d * nqTot * nqTot, nqTot, inptr, nqTot, 0.0,
                           derivPtr + d * nqTot * nElmts, nqTot);
            }

            if (deformed)
            {
                for (unsigned int k = 0; k < nCoord; k++)
                {
                    TData *ptr = outptr + k * outblock.size();

                    Nektar::parallel_for<ExecSpace>(
                        0, nElmts * nqTot, [&](const unsigned int i) {
                            ptr[i] =
                                dfptr[ndf * i + k * dimension] * derivPtr[i];
                        });
                    for (unsigned int d = 1; d < dimension; d++)
                    {
                        Nektar::parallel_for<ExecSpace>(
                            0, nElmts * nqTot, [&](const unsigned int i) {
                                ptr[i] += dfptr[ndf * i + k * dimension + d] *
                                          derivPtr[i + d * nqTot * nElmts];
                            });
                    }
                }
            }
            else
            {
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts, [&](const unsigned int e) {
                        for (unsigned int k = 0; k < nCoord; k++)
                        {
                            TData *ptr = outptr + k * outblock.size();
                            for (unsigned int i = 0; i < nqTot; i++)
                            {
                                ptr[nqTot * e + i] =
                                    dfptr[ndf * e + k * dimension] *
                                    derivPtr[nqTot * e + i];
                            }
                            for (unsigned int d = 1; d < dimension; d++)
                            {
                                for (unsigned int i = 0; i < nqTot; i++)
                                {
                                    ptr[nqTot * e + i] +=
                                        dfptr[ndf * e + k * dimension + d] *
                                        derivPtr[nqTot * e + i +
                                                 d * nqTot * nElmts];
                                }
                            }
                        }
                    });
            }

            inptr += inblock.size();
            outptr += nCoord * outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BlockOperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    std::vector<TData> m_deriv;

    static constexpr unsigned int m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
