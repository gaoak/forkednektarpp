///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialStdMat.hpp
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

#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorIProductWRTBaseImpl
    : public BlockOperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorIProductWRTBaseImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorIProductWRTBase<TData>(exp, dataWarehouse)
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
        const auto nmTot     = this->m_exp->GetNcoeffs();
        const auto nqTot     = this->m_exp->GetTotPoints();

        auto nElmts = inblock.GetNumElements();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < dimension; d++)
        {
            basisKeys[d] = this->m_exp->GetBasis(d)->GetBasisKey();
        }
        auto matptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, shapeType, eIProductWRTBaseStdMat));

        // Fetch Jacobian.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));

        // Allocate storate.
        if (m_wsp.size() == 0)
        {
            m_wsp = std::vector<TData>(dimension * nElmts * nqTot);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.data();

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Multiply by jacobian.
            if (deformed)
            {
                for (unsigned int i = 0; i < nElmts * nqTot; ++i)
                {
                    wspptr[i] = jacptr[i] * inptr[i];
                }
            }
            else
            {
                for (unsigned int e = 0; e < nElmts; ++e)
                {
                    for (unsigned int i = 0; i < nqTot; ++i)
                    {
                        wspptr[e * nqTot + i] =
                            jacptr[e] * inptr[e * nqTot + i];
                    }
                }
            }

            // Perform matrix-matrix multiply.
            Blas::Gemm('N', 'N', nmTot, nElmts, nqTot, this->m_scale, matptr,
                       nmTot, wspptr, nqTot, 0.0, outptr, nmTot);

            inptr += inblock.size();
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
        return std::make_unique<
            BlockOperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    std::vector<TData> m_wsp;

    static constexpr unsigned int m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
