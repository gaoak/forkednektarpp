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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp"

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(const unsigned int block_idx,
                                    const LocalRegions::ExpansionSharedPtr &exp,
                                    NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TFieldOut, TData>(block_idx, exp,
                                                        dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
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

        if constexpr (TFieldOut == FieldState::Coeff)
        {
            m_outTot = exp->GetNcoeffs();
            m_matptr = dataWarehouse->template GetData<MemSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType,
                                 eIProductWRTDerivBaseStdMat, nodalType));
        }
        else
        {
            m_outTot = exp->GetTotPoints();
            m_matptr =
                dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, ePhysDerivStdMatTranspose));

            m_weights = this->m_dataWarehouse->template GetData<MemSpace>(
                WeightsKey<TData>(block_idx, m_implInterleaveWidth));
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, true));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<ElmtBlockOp<FieldState::Phys, TFieldOut, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<IProductWRTDerivBaseBlockOpImpl<
            ExecSpace, Implementation, TFieldOut, TData>>(block_idx, exp,
                                                          dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_outTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    const TData *m_jacptr;
    const TData *m_dfptr;
    const TData *m_weights;

    void v_Apply(
        MultiRegions::BlockAccessor<TData, FieldState::Phys> &inblock,
        MultiRegions::BlockAccessor<TData, TFieldOut> &outblock) override
    {
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nelmtTot =
            inblock.GetNumElementsWithPadding() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            this->m_append
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * nelmtTot * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Loop over components.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        const auto wspoffset = m_nqTot * nelmtTot;
        for (unsigned int n = 0; n < outblock.GetNumComponents(); ++n)
        {
            // Reshape, if necessary.
            for (unsigned int k = 0; k < m_coordDim; ++k)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, inInterleaveWidth, nelmtTot,
                    inblock.GetNumData(), (TData *)inptr + k * inoffset,
                    m_streamID);
            }
            if (this->m_append)
            {
                ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, outInterleaveWidth, nelmtTot,
                    outblock.GetNumData(), (TData *)outptr, m_streamID);
            }

            if constexpr (TFieldOut == FieldState::Coeff)
            {
                // Multiply by derivative factor and Jacobian.
                if (m_isDeformed)
                {
                    JacobianDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, nhomo,
                        inoffset, wspoffset, m_jacptr, m_dfptr, inptr, wspptr,
                        m_streamID);
                }
                else
                {
                    JacobianDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, nhomo,
                        inoffset, wspoffset, m_jacptr, m_dfptr, inptr, wspptr,
                        m_streamID);
                }
            }
            else
            {
                // Multiply by derivative factor and Jacobian.
                if (m_isDeformed)
                {
                    JacobianDerivFactorWeightsKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, nhomo,
                        inoffset, wspoffset, m_jacptr, m_dfptr, m_weights,
                        inptr, wspptr, m_streamID);
                }
                else
                {
                    JacobianDerivFactorWeightsKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, nhomo,
                        inoffset, wspoffset, m_jacptr, m_dfptr, m_weights,
                        inptr, wspptr, m_streamID);
                }
            }

            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                TData alpha = this->m_scale;
                TData beta  = (d != 0 || this->m_append);

                NekBlas::Gemm(handle, "N", "N", m_outTot, nelmtTot, m_nqTot,
                              alpha, m_matptr + d * m_nqTot * m_outTot,
                              m_outTot, wspptr + d * nelmtTot * m_nqTot,
                              m_nqTot, beta, outptr, m_outTot);
            }

            // Reshape back, if necessary.
            for (unsigned int k = 0; k < m_coordDim; ++k)
            {
                ReshapeStorage<ExecSpace>(
                    inInterleaveWidth, m_implInterleaveWidth, nelmtTot,
                    inblock.GetNumData(), (TData *)inptr + k * inoffset,
                    m_streamID);
            }
            ReshapeStorage<ExecSpace>(inInterleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, outblock.GetNumData(), outptr,
                                      m_streamID);

            // Increment pointers.
            inptr += m_coordDim * inoffset;
            outptr += outoffset;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
