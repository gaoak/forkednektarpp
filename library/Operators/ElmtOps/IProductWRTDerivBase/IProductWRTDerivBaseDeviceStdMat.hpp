///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceStdMat.hpp
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

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TData>(exp, dataWarehouse)
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

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_matptr = dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eIProductWRTDerivBaseStdMat, nodalType));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTDerivBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    MemoryRegion<TData> m_wsp;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance();

        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nelmtTot =
            inblock.GetNumElementsWithPadding() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = this->m_append
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), true));

        // Allocate storage.
        if (m_wsp.size() == 0)
        {
            m_wsp = MemoryRegion<TData>::Create(
                m_dimension * nelmtTot * m_nqTot, ExecSpace::alignment);
        }

        // Get workspace pointer.
        auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

        // Loop over components.
        const auto inoffset  = inblock.size() * inblock.GetNumHomoModes();
        const auto wspoffset = m_nqTot * nelmtTot;
        for (unsigned int n = 0; n < outblock.GetNumComponents(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                          inblock.GetInterleaveWidth(),
                                          nelmtTot, inblock.GetNumData(),
                                          (TData *)inptr + d * inblock.size());
            }
            if (this->m_append)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, outblock.GetInterleaveWidth(),
                    nelmtTot, outblock.GetNumData(), outptr);
            }

            // Multiply by derivative factor and Jacobian.
            if (m_isDeformed)
            {
                MultiplyByJacobianAndDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, inoffset,
                    wspoffset, jacptr, dfptr, inptr, wspptr);
            }
            else
            {
                MultiplyByJacobianAndDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, inoffset,
                    wspoffset, jacptr, dfptr, inptr, wspptr);
            }

            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                TData alpha = 1.0;
                TData beta  = (d != 0 || this->m_append);

                NekGemm(handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, alpha,
                        m_matptr + d * m_nqTot * m_nmTot, m_nmTot,
                        wspptr + d * nelmtTot * m_nqTot, m_nqTot, beta, outptr,
                        m_nmTot);
            }

            // Increment pointers.
            inptr += m_coordDim * inblock.size() * inblock.GetNumHomoModes();
            outptr += outblock.size() * outblock.GetNumHomoModes();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
