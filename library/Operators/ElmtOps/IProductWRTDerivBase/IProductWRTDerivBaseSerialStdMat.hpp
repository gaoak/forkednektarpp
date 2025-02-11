///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialStdMat.hpp
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

#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_nComps = out.GetNumComponents();
        ASSERTL1(m_nComps == in.GetNumComponents() /
                                 this->m_expansionList->GetCoordim(0),
                 "Number of input and output components differ");

        // Initialize index.
        m_exp_idx = 0;

        // Loop over the blocks.
        for (m_blk = 0; m_blk < in.GetBlocks().size(); ++m_blk)
        {
            m_expPtr = this->m_expansionList->GetExp(m_exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[m_blk];
            auto &outblock = out.GetBlocks()[m_blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            m_exp_idx += inblock.GetNumElements();
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = this->m_append
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Determine shape and type of the element.
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        const auto shapeType = m_expPtr->DetShapeType();
        const auto dimension = m_expPtr->GetShapeDimension();
        const auto nCoord    = m_expPtr->GetCoordim();
        const auto nqTot     = m_expPtr->GetTotPoints();
        const auto nmTot     = m_expPtr->GetNcoeffs();
        const auto ndf       = dimension * nCoord;

        // Fetch Jacobian and deriv factors.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(m_exp_idx, m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(m_exp_idx, m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        auto nElmts = inblock.GetNumElements();

        // Allocate storate.
        if (m_wsp.size() <= m_blk)
        {
            m_wsp.push_back(std::vector<TData>(dimension * nElmts * nqTot));
        }

        // Get workspace pointer.
        auto wspptr = m_wsp[m_blk].data();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);
            if (nCoord > 1)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr + inblock.size());
            }
            if (nCoord > 2)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr + 2 * inblock.size());
            }
            if (this->m_append)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    outblock.GetInterleaveWidth(),
                    outblock.GetNumElementsWithPadding(), outblock.GetNumData(),
                    outptr);
            }

            // Calculate dx/dxi in[0] + dy/dxi in[1] + dz/dxi in[2].
            if (deformed)
            {
                for (unsigned int d = 0; d < dimension; d++)
                {
                    TData *ptr = wspptr + d * nElmts * nqTot;

                    Nektar::parallel_for<ExecSpace>(
                        0, nElmts * nqTot, [&](const unsigned int i) {
                            ptr[i] = dfptr[ndf * i + d] * inptr[i];
                            for (unsigned int k = 1; k < nCoord; ++k)
                            {
                                ptr[i] += dfptr[ndf * i + k * dimension + d] *
                                          inptr[i + k * inblock.size()];
                            }
                        });
                }
            }
            else
            {
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts, [&](const unsigned int e) {
                        for (unsigned int d = 0; d < dimension; d++)
                        {
                            TData *ptr = wspptr + d * nElmts * nqTot;
                            for (unsigned int i = 0; i < nqTot; i++)
                            {
                                ptr[nqTot * e + i] =
                                    dfptr[ndf * e + d] * inptr[nqTot * e + i];
                            }
                            for (unsigned int k = 1; k < nCoord; ++k)
                            {
                                for (unsigned int i = 0; i < nqTot; i++)
                                {
                                    ptr[nqTot * e + i] +=
                                        dfptr[ndf * e + k * dimension + d] *
                                        inptr[nqTot * e + i +
                                              k * inblock.size()];
                                }
                            }
                        }
                    });
            }

            // Multiply by jacobian.
            if (deformed)
            {
                for (unsigned int d = 0; d < dimension; d++)
                {
                    TData *ptr = wspptr + d * nElmts * nqTot;
                    Nektar::parallel_for<ExecSpace>(
                        0, nElmts * nqTot,
                        [&](const unsigned int i) { ptr[i] *= jacptr[i]; });
                }
            }
            else
            {
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts, [&](const unsigned int e) {
                        for (unsigned int d = 0; d < dimension; d++)
                        {
                            TData *ptr = wspptr + d * nElmts * nqTot;
                            for (unsigned int i = 0; i < nqTot; i++)
                            {
                                ptr[nqTot * e + i] *= jacptr[e];
                            }
                        }
                    });
            }

            // Fetch matrix.
            std::vector<LibUtilities::BasisKey> basisKeys(
                dimension, LibUtilities::NullBasisKey);
            for (unsigned int d = 0; d < dimension; d++)
            {
                basisKeys[d] = m_expPtr->GetBasis(d)->GetBasisKey();
            }
            auto matptr = this->m_dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, shapeType,
                                 eIProductWRTDerivBaseStdMat));

            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < dimension; d++)
            {
                TData *ptr = wspptr + d * nElmts * nqTot;

                TData alpha = (d != 0 || this->m_append);
                Blas::Gemm('N', 'N', nmTot, nElmts, nqTot, 1.0,
                           matptr + d * nqTot * nmTot, nmTot, ptr, nqTot, alpha,
                           outptr, nmTot);
            }

            inptr += nCoord * inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

private:
    unsigned int m_exp_idx;
    unsigned int m_blk;
    unsigned int m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    std::vector<std::vector<TData>> m_wsp;

    static constexpr size_t m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
