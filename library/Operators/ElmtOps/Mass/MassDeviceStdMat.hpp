///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceStdMat.hpp
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
// per-block mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the per-block
 * mass operator.
 *
 * @details
 * The device counterpart of MassSerialAVXStdMat.hpp: the same one or two
 * dense multiplies with standard-region matrices and the same Jacobian
 * stage between or after them, but with each multiply as a single
 * strided-batched GEMM covering every component at once instead of a GEMM
 * per element group, and the Jacobian stage as one device launch over every
 * point, element, component and homogeneous mode of the block.
 *
 * This header provides the definition of detail::MassBlockOpImpl for the
 * Device execution space with the StdMat implementation tag; the
 * sum-factorised Device definition lives in MassDeviceSumFac.hpp, and each
 * CMake-generated registration translation unit includes exactly one
 * implementation header.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through MassOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/Mass/MassBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Device StdMat mass operator: the whole block's operation as one or
 * two strided-batched dense matrix products on the back-end BLAS, with the
 * Jacobian applied by a device kernel between or after them.
 *
 * @details
 * As in the Serial/AVX StdMat implementation, the block's geometry type
 * decides which standard-region matrices the constructor fetches from the
 * data warehouse, because it decides where the Jacobian can be applied --
 * but here the matrices are the untransposed variants (eMassStdMat,
 * eBwdTransStdMat, eIProductWRTBaseStdMat), since the GEMM has them as its
 * left factor. Every element shares them, so each stage is one GEMM per
 * component,
 *
 *        out (nmTot x nelmtTot) = M * in (nmTot x nelmtTot)
 *
 * for a regular block, with M the standard-region mass matrix, which
 * carries the quadrature weights, followed by MultiplyByJacobian on the
 * coefficient-space result -- valid because a regular element's Jacobian
 * is a single constant; and for a deformed block the backward transform
 * into the static per-stream workspace,
 *
 *        wsp (nqTot x nelmtTot) = B * in  (nmTot x nelmtTot),
 *
 * the point-by-point multiplication of the workspace by the element
 * Jacobian, and the integration back,
 *
 *        out (nmTot x nelmtTot) = W * wsp (nqTot x nelmtTot),
 *
 * W being the inner-product matrix, weights included. Each GEMM is issued
 * for all components at once as a strided-batched GEMM with one batch
 * entry per component; the columns of the operands are the elements
 * (padding elements included, their results being ignored) and the
 * homogeneous modes, nelmtTot = padded elements * homogeneous modes, and
 * the batch stride is the distance between consecutive components of the
 * block. NekBlas::GemmStridedBatched forwards to the BLAS of the enabled
 * device back-end. The GEMMs need each element's data contiguous, so the
 * implementation works at interleave width 1 (#m_implInterleaveWidth) and
 * reshapes the block storage to that width and back around the calls.
 *
 * Each block operator owns device stream block_idx + 1, so different
 * blocks' work may overlap on the device.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 *
 * @see MassDeviceSumFac.hpp for the sum-factorised device paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class MassBlockOpImpl : public MassBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the standard-region matrices of the
     * block's geometry type and the block's Jacobians (see the class
     * description).
     *
     * The matrix keys carry the element's basis keys and shape and, for a
     * nodal expansion, the nodal points type. The Jacobians are fetched at
     * interleave width 1.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    MassBlockOpImpl(const unsigned int block_idx,
                    const LocalRegions::ExpansionSharedPtr &exp,
                    LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MassBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        if (m_isDeformed)
        {
            m_bwdmat = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eBwdTransStdMat,
                                             nodalType));
            m_ipbmat = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eIProductWRTBaseStdMat,
                                             nodalType));
        }
        else
        {
            m_massmat = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eMassStdMat,
                                             nodalType));
        }

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            MassBlockOpImpl<ExecSpace, Implementation, TData>>(block_idx, exp,
                                                               dataWarehouse);
    }

protected:
    /// Element interleave width the kernel and GEMMs operate at: contiguous
    /// per-element data, since the elements are the GEMMs' columns.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream all work of this block operator is issued on, one per
    /// block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobians); selects
    /// both the matrices fetched by the constructor and the branch taken in
    /// v_Apply().
    bool m_isDeformed;
    /// Reference (shape) dimension of the elements (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements; not read
    /// by this implementation.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Backward-transform matrix (nqTot x nmTot, column-major) in device
    /// memory; set, and used, only for a deformed block.
    const TData *m_bwdmat;
    /// Inner-product matrix (nmTot x nqTot, column-major), quadrature
    /// weights included; set, and used, only for a deformed block.
    const TData *m_ipbmat;
    /// Standard-region mass matrix (nmTot x nmTot, column-major),
    /// quadrature weights included; set, and used, only for a regular
    /// block.
    const TData *m_massmat;
    /// Jacobians of the block in device memory, at interleave width 1: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;

    /**
     * @brief Apply the mass operator to the whole block by strided-batched
     * dense matrix multiplication.
     *
     * The input -- all components and homogeneous modes at once -- is
     * de-interleaved to #m_implInterleaveWidth. For a deformed block one
     * strided-batched GEMM against #m_bwdmat takes the coefficients to the
     * static per-stream workspace (sized nelmt * ncomp * nqTot),
     * MultiplyByJacobian multiplies every point of every component and
     * homogeneous mode by its Jacobian in place, and a second
     * strided-batched GEMM against #m_ipbmat takes the workspace to the
     * output; for a regular block a single strided-batched GEMM against
     * #m_massmat runs straight from the input and MultiplyByJacobian then
     * scales each element's coefficients by its one Jacobian. In both the
     * homogeneous modes are folded into the columns and the components into
     * the batch, so there is no loop over components or modes here. Both
     * fields are interleaved back to the input's width, which the output
     * block is also set to, and everything is issued on #m_streamID.
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

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                nelmt * ncomp * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Offsets between the components of a block.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        const auto wspoffset = m_nqTot * nelmtTot;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (m_isDeformed)
        {
            // Step 1: BwdTrans
            // Perform batched matrix-matrix multiply, one multiply per
            // component, with the homogeneous modes held in the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nqTot, nelmtTot, m_nmTot, (TData)1.0,
                m_bwdmat, m_nqTot, 0, inptr, m_nmTot, inoffset, (TData)0.0,
                wspptr, m_nqTot, wspoffset, inblock.GetNumComponents());

            // Multiply by jacobian.
            MultiplyByJacobian<ExecSpace, true>(nelmt, m_nqTot, ncomp, m_jacptr,
                                                wspptr, wspptr, (TData)1.0,
                                                m_streamID);

            // Step 2: IProduct
            // Perform batched matrix-matrix multiply, one multiply per
            // component, with the homogeneous modes held in the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, (TData)1.0,
                m_ipbmat, m_nmTot, 0, wspptr, m_nqTot, wspoffset, (TData)0.0,
                outptr, m_nmTot, outoffset, inblock.GetNumComponents());
        }
        else
        {
            // Perform batched matrix-matrix multiply, one multiply per
            // component, with the homogeneous modes held in the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nmTot, (TData)1.0,
                m_massmat, m_nmTot, 0, inptr, m_nmTot, inoffset, (TData)0.0,
                outptr, m_nmTot, outoffset, inblock.GetNumComponents());

            // Multiply by jacobian.
            MultiplyByJacobian<ExecSpace, false>(nelmt, m_nmTot, ncomp,
                                                 m_jacptr, outptr, outptr,
                                                 (TData)1.0, m_streamID);
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

} // namespace Nektar::Operators::detail
