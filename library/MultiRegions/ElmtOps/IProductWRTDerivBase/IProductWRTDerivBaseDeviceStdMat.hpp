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
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    IProductWRTDerivBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
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
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::eIProductWRTDerivBaseStdMat, nodalType));
        }
        else
        {
            m_outTot = exp->GetTotPoints();
            m_matptr = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::ePhysDerivStdMatTranspose));

            m_weights = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::WeightsKey<TData>(block_idx,
                                                m_implInterleaveWidth));
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<ElmtBlockOp<FieldState::Phys, TFieldOut, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
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
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock) override
    {
        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto ncomp    = outblock.GetNumComponents() * nhomo;
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * nhomo;

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * nelmt * ncomp * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Directions held per variable in the input block. In 3DH1
        // (nhomo > 1) there are always three (x, y, z) regardless of the base
        // mesh coordDim, the z slot being the one DerivZOp reads; the xy pass
        // still consumes only m_coordDim of them.
        const auto inDim = (nhomo > 1) ? 3u : m_coordDim;

        // Offsets between the components of a block. The metric of every
        // component is held at once, so the offset between two directions of
        // the workspace spans all of them.
        const auto inoffset  = inblock.CompSize() * nhomo;
        const auto outoffset = outblock.CompSize() * nhomo;
        const auto wspoffset = m_nqTot * nelmt * ncomp;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp * inDim,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);
        }

        // Multiply by derivative factor and Jacobian, the planes and
        // components on the second and third grid dimensions. The coordinate
        // directions of a component are held in inDim slots of the input
        // block.
        if constexpr (TFieldOut == FieldState::Coeff)
        {
            if (m_isDeformed)
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorKernel<true>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, inptr, wspptr);
            }
            else
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorKernel<false>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, inptr, wspptr);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorWeightsKernel<true>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, m_weights, inptr, wspptr);
            }
            else
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorWeightsKernel<false>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, m_weights, inptr, wspptr);
            }
        }

        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            TData alpha = this->m_scale;
            TData beta  = (d != 0 || this->m_append);

            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_outTot, nelmtTot, m_nqTot, alpha,
                m_matptr + d * m_nqTot * m_outTot, m_outTot, 0,
                wspptr + d * wspoffset, m_nqTot, m_nqTot * nelmtTot, beta,
                outptr, m_outTot, outoffset, outblock.GetNumComponents());
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp * inDim,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
