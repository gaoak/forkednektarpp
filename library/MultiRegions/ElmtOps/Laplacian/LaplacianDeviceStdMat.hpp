///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianDeviceStdMat.hpp
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
// Description: Device standard-matrix (StdMat) implementation of the
// per-block weak Laplacian operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LaplacianDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the
 * per-block weak Laplacian operator.
 *
 * @details
 * This header provides the definition of detail::LaplacianBlockOpImpl
 * for the Device execution space with the StdMat implementation tag;
 * see LaplacianBlockOp.hpp for the interface it implements and
 * LaplacianOp.hpp for the registration scheme.
 *
 * The operator is decomposed into three stages against standard-region
 * matrices in device memory:
 * 1. Deriv on the input coefficients to form the reference derivatives;
 * 2. ApplyMetricKernel to contract the reference derivatives with the
 *    geometry and diffusion tensor;
 * 3. IProductWRTDerivBase on the contracted reference derivatives to
 *    form the weak Laplacian term, summing the directional
 *    contributions into the output, which is overwritten.
 *
 * All matrix products are issued through NekBlas::GemmStridedBatched
 * on the block's stream.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through LaplacianOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/Laplacian/LaplacianBlockOp.hpp>

#include <MultiRegions/ElmtOps/Laplacian/LaplacianDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat block implementation of the weak Laplacian
 * operator: the three-stage matrix-matrix decomposition issued through
 * strided-batched GEMMs on the block's device stream.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class LaplacianBlockOpImpl : public LaplacianBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator, capture element metadata, and
     * fetch the standard-region matrices, Jacobians and derivative
     * factors from the data warehouse.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; the device stream used
     *                          for all operations is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    LaplacianBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : LaplacianBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        m_derivmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eDerivStdMat, nodalType));
        m_ipdmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::eIProductWRTDerivBaseStdMat,
                nodalType));

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            LaplacianBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Vector width this implementation operates at (1 for Device).
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream ID assigned to this block (block_idx + 1).
    unsigned int m_streamID;
    /// Shape type of the element block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Coefficients-to-reference-derivative matrices (nqTot x nmTot).
    const TData *m_derivmat;
    /// Inner-product-with-basis-derivative matrices, weights included.
    const TData *m_ipdmat;
    /// Jacobian of the block: one value per element, or one per
    /// quadrature point and element on a deformed block.
    const TData *m_jacptr;
    /// Derivative factors of the block, in the transposed
    /// (point-major-per-element) device layout.
    const TData *m_dfptr;

    /**
     * @brief Apply the weak Laplacian operator to the whole block through
     * strided-batched GEMMs.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // Get static workspace pointer.
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * nelmt * ncomp * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block. The derivatives of every
        // component are held at once, so the offset between two directions
        // spans all of them.
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        const auto derivoffset = m_nqTot * nelmt * ncomp;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Step 1: Deriv
        // Perform batched matrix-matrix multiply, one multiply per direction,
        // with the components and homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmt * ncomp, m_nmTot, (TData)1.0,
            m_derivmat, m_nqTot, m_nqTot * m_nmTot, inptr, m_nmTot, 0,
            (TData)0.0, derivptr, m_nqTot, derivoffset, m_dimension);

        // Step 2: Multiply by diffusion coefficient, derivative
        // factor and Jacobian.
        if (m_isDeformed)
        {

            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (ApplyMetricKernel<true>), gridSize, ncomp, blockSize, 1,
                m_streamID, m_nqTot, m_coordDim, m_dimension, nelmt,
                derivoffset, derivoffset, diffCoeffPtr, m_jacptr, m_dfptr,
                derivptr, derivptr);
        }
        else
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (ApplyMetricKernel<false>), gridSize, ncomp, blockSize, 1,
                m_streamID, m_nqTot, m_coordDim, m_dimension, nelmt,
                derivoffset, derivoffset, diffCoeffPtr, m_jacptr, m_dfptr,
                derivptr, derivptr);
        }

        // Step 3: IProductWRTDerivBase
        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            TData alpha = 1.0;
            TData beta  = (d != 0);

            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, alpha,
                m_ipdmat + d * m_nqTot * m_nmTot, m_nmTot, 0,
                derivptr + d * derivoffset, m_nqTot, m_nqTot * nelmtTot, beta,
                outptr, m_nmTot, outoffset, inblock.GetNumComponents());
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
