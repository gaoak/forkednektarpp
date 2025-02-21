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
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }
        m_matptr = dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType, ePhysDerivStdMat));
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        const auto nElmts = inblock.GetNumElements();
        const auto ndf    = m_coordDim * m_dimension;

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch derivative factor.
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Allocate storate.
        if (m_deriv.size() == 0)
        {
            m_deriv = std::vector<TData>(m_dimension * m_nqTot * nElmts);
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

            for (unsigned int d = 0; d < m_dimension; ++d)
            {
                // Perform matrix-matrix multiply.
                Blas::Gemm('N', 'N', m_nqTot, nElmts, m_nqTot, 1.0,
                           m_matptr + d * m_nqTot * m_nqTot, m_nqTot, inptr,
                           m_nqTot, 0.0, derivPtr + d * m_nqTot * nElmts,
                           m_nqTot);
            }

            if (m_isDeformed)
            {
                for (unsigned int k = 0; k < m_coordDim; k++)
                {
                    TData *ptr = outptr + k * outblock.size();

                    Nektar::parallel_for<ExecSpace>(
                        0, nElmts * m_nqTot, [&](const unsigned int i) {
                            ptr[i] =
                                dfptr[ndf * i + k * m_dimension] * derivPtr[i];
                        });
                    for (unsigned int d = 1; d < m_dimension; d++)
                    {
                        Nektar::parallel_for<ExecSpace>(
                            0, nElmts * m_nqTot, [&](const unsigned int i) {
                                ptr[i] += dfptr[ndf * i + k * m_dimension + d] *
                                          derivPtr[i + d * m_nqTot * nElmts];
                            });
                    }
                }
            }
            else
            {
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts, [&](const unsigned int e) {
                        for (unsigned int k = 0; k < m_coordDim; k++)
                        {
                            TData *ptr = outptr + k * outblock.size();
                            for (unsigned int i = 0; i < m_nqTot; i++)
                            {
                                ptr[m_nqTot * e + i] =
                                    dfptr[ndf * e + k * m_dimension] *
                                    derivPtr[m_nqTot * e + i];
                            }
                            for (unsigned int d = 1; d < m_dimension; d++)
                            {
                                for (unsigned int i = 0; i < m_nqTot; i++)
                                {
                                    ptr[m_nqTot * e + i] +=
                                        dfptr[ndf * e + k * m_dimension + d] *
                                        derivPtr[m_nqTot * e + i +
                                                 d * m_nqTot * nElmts];
                                }
                            }
                        }
                    });
            }

            // Increment pointer.
            inptr += inblock.size();
            outptr += m_coordDim * outblock.size();
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
    static constexpr unsigned int m_implInterleaveWidth = 1;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    std::vector<TData> m_deriv;
};

} // namespace Nektar::Operators::detail
