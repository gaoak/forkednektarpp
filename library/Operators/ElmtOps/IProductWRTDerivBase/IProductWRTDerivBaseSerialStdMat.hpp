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
class BlockOperatorIProductWRTDerivBaseImpl
    : public BlockOperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorIProductWRTDerivBaseImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorIProductWRTDerivBase<TData>(exp, dataWarehouse)
    {
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = this->m_append
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Determine shape and type of the element.
        const auto deformed = this->m_exp->GetMetricInfo()->GetGtype() ==
                              SpatialDomains::eDeformed;
        const auto shapeType = this->m_exp->DetShapeType();
        const auto dimension = this->m_exp->GetShapeDimension();
        const auto nmTot     = this->m_exp->GetNcoeffs();
        const auto nqTot     = this->m_exp->GetTotPoints();
        const auto nCoord    = this->m_exp->GetCoordim();
        const auto ndf       = dimension * nCoord;

        auto nElmts = inblock.GetNumElements();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < dimension; d++)
        {
            basisKeys[d] = this->m_exp->GetBasis(d)->GetBasisKey();
        }
        auto matptr =
            this->m_dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                basisKeys, shapeType, eIProductWRTDerivBaseStdMat));

        // Fetch Jacobian and deriv factors.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));

        // Allocate storate.
        if (m_wsp.size() == 0)
        {
            m_wsp = std::vector<TData>(dimension * nElmts * nqTot);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.data();

        // Loop over components.
        for (unsigned int nc = 0; nc < outblock.GetNumComponents(); ++nc)
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

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<BlockOperatorIProductWRTDerivBaseImpl<
            ExecSpace, Implementation, TData>>(exp, dataWarehouse);
    }

protected:
    std::vector<TData> m_wsp;

    static constexpr unsigned int m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
