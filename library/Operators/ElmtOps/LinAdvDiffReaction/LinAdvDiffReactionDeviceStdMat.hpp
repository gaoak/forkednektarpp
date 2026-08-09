///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceStdMat.hpp
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

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class LinAdvDiffReactionBlockOpImpl : public LinAdvDiffReactionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    LinAdvDiffReactionBlockOpImpl(const unsigned int block_idx,
                                  const LocalRegions::ExpansionSharedPtr &exp,
                                  NekDataWarehouseSharedPtr dataWarehouse)
        : LinAdvDiffReactionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
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

        m_bwdmat   = dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eBwdTransStdMat, nodalType));
        m_ipbmat   = dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eIProductWRTBaseStdMat, nodalType));
        m_derivmat = dataWarehouse->template GetData<MemSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType, eDerivStdMat, nodalType));
        m_ipdmat = dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eIProductWRTDerivBaseStdMat, nodalType));

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, true));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            LinAdvDiffReactionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_bwdmat;
    const TData *m_ipbmat;
    const TData *m_ipdmat;
    const TData *m_derivmat;
    const TData *m_jacptr;
    const TData *m_dfptr;
    TData *m_advVel;

    void v_Apply(BlockAccessor<TData, FieldState::Coeff> &inblock,
                 BlockAccessor<TData, FieldState::Coeff> &outblock) override
    {
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nelmtTot =
            inblock.GetNumElementsWithPadding() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // Get static workspace pointer.
        auto bwdptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                nelmtTot * m_nqTot + m_dimension * nelmtTot * m_nqTot,
                m_streamID);
        auto derivptr = bwdptr + nelmtTot * m_nqTot;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot * nhomo + blockSize - 1u) / blockSize;

        // Loop over components.
        const auto adveloffset = m_nqTot * nelmt;
        const auto derivoffset = m_nqTot * nelmtTot;
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto advptr = this->m_advVel;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            // Step 1: BwdTrans
            // Perform matrix-matrix multiply.
            if (this->m_lambda != 0.0)
            {
                NekBlas::Gemm(handle, "N", "N", m_nqTot, nelmtTot, m_nmTot,
                              (TData)1.0, m_bwdmat, m_nqTot, inptr, m_nmTot,
                              (TData)0.0, bwdptr, m_nqTot);
            }

            // Step 2: Deriv
            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekBlas::Gemm(handle, "N", "N", m_nqTot, nelmtTot, m_nmTot,
                              (TData)1.0, m_derivmat + d * m_nqTot * m_nmTot,
                              m_nqTot, inptr, m_nmTot, (TData)0.0,
                              derivptr + d * m_nqTot * nelmtTot, m_nqTot);
            }

            // Step 3: Multiply by diffusion coefficient, derivative
            // factor and Jacobian and add advection.
            if (m_isDeformed)
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (ApplyMetricKernel<true>), gridSize, blockSize, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, adveloffset, diffCoeffPtr, m_jacptr, m_dfptr,
                    advptr, derivptr, derivptr, bwdptr, this->m_lambda);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (ApplyMetricKernel<false>), gridSize, blockSize, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, adveloffset, diffCoeffPtr, m_jacptr, m_dfptr,
                    advptr, derivptr, derivptr, bwdptr, this->m_lambda);
            }

            // Step 4: IProduct
            // Perform matrix-matrix multiply.
            NekBlas::Gemm(handle, "N", "N", m_nmTot, nelmtTot, m_nqTot,
                          (TData)1.0, m_ipbmat, m_nmTot, bwdptr, m_nqTot,
                          (TData)0.0, outptr, m_nmTot);

            // Step 5: IProductWRTDerivBase
            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekBlas::Gemm(handle, "N", "N", m_nmTot, nelmtTot, m_nqTot,
                              (TData)1.0, m_ipdmat + d * m_nqTot * m_nmTot,
                              m_nmTot, derivptr + d * nelmtTot * m_nqTot,
                              m_nqTot, (TData)1.0, outptr, m_nmTot);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, outblock.GetNumData(), outptr,
                                      m_streamID);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetAdvVel(BlockAccessor<TData, FieldState::Phys> &advVel) override
    {
        const auto interleaveWidth = advVel.GetInterleaveWidth();
        this->m_advVel =
            advVel.template GetPtr<MemSpace, ReadWrite>(m_streamID);
        ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth,
            advVel.GetNumElementsWithPadding() * this->m_exp->GetCoordim(),
            advVel.GetNumData(), this->m_advVel, m_streamID);
        advVel.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
