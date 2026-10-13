///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassBlockOpDeviceGeneric.hpp
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
// Description: Device Generic implementation of the elemental inverse
// mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MultiplyByElmtInvMassBlockOpDeviceGeneric.hpp
 * @brief Device implementation of the per-block element-wise inverse mass
 * operator, registered under the Generic implementation tag.
 *
 * @details
 * The whole block goes through one strided-batched GEMM on the back-end BLAS --
 * batched over the components against the shared standard-region inverse and
 * followed by one device launch dividing by the element Jacobians for a
 * regular block, batched over the elements against their own inverse mass
 * matrices for a deformed one.
 *
 * This header provides the definition of
 * detail::MultiplyByElmtInvMassBlockOpImpl for the Device execution space;
 * the Serial/AVX definition lives in
 * MultiplyByElmtInvMassBlockOpSerialAVXGeneric.hpp, and each CMake-generated
 * registration translation unit includes exactly one of the two headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through MultiplyByElmtInvMassOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LocalRegions/Expansion.h>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device Generic inverse mass operator: the whole block's operation
 * as one strided-batched dense matrix product on the back-end BLAS, with
 * the Jacobian divided out by a device kernel on a regular block.
 *
 * @details
 * The constructor fetches the standard-region inverse mass matrix
 * (#m_invmassptr, the StdMatKey with eInvMassStdMat: nmTot x nmTot,
 * column-major, the untransposed variant since the GEMM has it as its left
 * factor) and the block's Jacobians (#m_jacptr, the JacobianKey at
 * #m_implInterleaveWidth: one constant per element for a regular block)
 * from the data warehouse, so both are built once and shared between
 * operators. Both are fetched for deformed blocks too, whose path uses
 * neither.
 *
 * The GEMMs need each element's coefficients contiguous, so the
 * implementation works at interleave width 1 (#m_implInterleaveWidth) and
 * reshapes the block storage to that width and back around the calls. The
 * whole block -- all components and homogeneous modes -- goes through one
 * NekBlas::GemmStridedBatched call:
 * - regular block: one batch entry per component, every entry applying
 *   the same standard-region inverse (batch stride 0 on the matrix),
 *
 *        out (nmTot x nelmtTot) = (M^std)^{-1} * in (nmTot x nelmtTot),
 *
 *   the columns being the padded elements (padding included, their
 *   results ignored) times the homogeneous modes, nelmtTot, and the batch
 *   stride the distance between consecutive components of the block.
 *   DivideByJacobian then divides each element's coefficients by its one
 *   Jacobian in every component and homogeneous mode, completing
 *   \f$(M^e)^{-1} = (M^{\mathrm{std}})^{-1} / J^e\f$;
 * - deformed block: one batch entry per padded element, each applying
 *   that element's own inverse mass matrix from #m_dinvmass (batch stride
 *   nmTot x nmTot) to a right-hand side whose columns are the element's
 *   coefficients in each component and homogeneous mode (leading
 *   dimension the block's CompSize(), batch stride nmTot),
 *
 *        out^e (nmTot x ncomp) = (M^e)^{-1} * in^e (nmTot x ncomp),
 *
 *   ncomp being the component count times the number of homogeneous
 *   modes. The geometry is already contained in the matrices, so no
 *   Jacobian is applied.
 *
 * Each block operator owns device stream block_idx + 1, so different
 * blocks' work may overlap on the device.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (Generic).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class MultiplyByElmtInvMassBlockOpImpl
    : public MultiplyByElmtInvMassBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the standard-region inverse mass matrix
     * and the block's Jacobians (see the class description).
     *
     * The matrix key carries the element's basis keys and shape and, for a
     * nodal expansion, the nodal points type. The Jacobians are fetched at
     * interleave width 1.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    MultiplyByElmtInvMassBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MultiplyByElmtInvMassBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        m_invmassptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::eInvMassStdMat, nodalType));

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
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
            MultiplyByElmtInvMassBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the GEMMs operate at: contiguous
    /// per-element data, since the elements are the GEMMs' columns.
    static constexpr unsigned int m_implInterleaveWidth = 1;

    /// Device stream all work of this block operator is issued on, one per
    /// block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed; selects the branch taken in
    /// v_Apply().
    bool m_isDeformed;
    /// Reference (shape) dimension of the elements (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements; not read
    /// by this implementation.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element; not read by this implementation,
    /// which acts in coefficient space.
    unsigned int m_nqTot;
    /// Standard-region inverse mass matrix (nmTot x nmTot, column-major)
    /// in device memory, cached in the data warehouse; used by the regular
    /// path only.
    const TData *m_invmassptr;
    /// Jacobians of the block in device memory at interleave width 1, one
    /// per element, cached in the data warehouse; used by the regular path
    /// only.
    const TData *m_jacptr;
    /// Per-element inverse mass matrices of a deformed block in device
    /// memory, as set through v_SetInvMassMatrix(); empty for a regular
    /// block.
    LibUtilities::MemoryRegion<TData> m_dinvmass;

    /**
     * @brief Apply the inverse mass matrices to the whole block by one
     * strided-batched dense matrix multiplication.
     *
     * The input -- all components and homogeneous modes at once -- is
     * de-interleaved to #m_implInterleaveWidth. For a deformed block one
     * strided-batched GEMM against the per-element matrices in #m_dinvmass
     * runs straight from the input to the output; for a regular block one
     * strided-batched GEMM against #m_invmassptr does so and
     * DivideByJacobian then divides each element's coefficients by its one
     * Jacobian. There is no loop over components or modes here: the
     * regular call folds the homogeneous modes into the columns and the
     * components into the batch, the deformed call the components and
     * homogeneous modes into the columns and the elements into the batch.
     * Both fields are interleaved back to the input's width, which the
     * output block is also set to, and everything is issued on
     * #m_streamID.
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

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Offsets between the components of a block.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        if (m_isDeformed)
        {
            // Fetch deformed mass matrix.
            auto dmatptr = this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>(
                m_streamID);

            // Perform batched matrix-matrix multiply, one multiply per
            // element, with the variables and the homogeneous modes held in
            // the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, ncomp, m_nmTot, (TData)1.0, dmatptr,
                m_nmTot, m_nmTot * m_nmTot, inptr, inblock.CompSize(), m_nmTot,
                (TData)0.0, outptr, outblock.CompSize(), m_nmTot, nelmt);
        }
        else
        {
            // Perform batched matrix-matrix multiply, one multiply per
            // component, with the homogeneous modes held in the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nmTot, (TData)1.0,
                m_invmassptr, m_nmTot, 0, inptr, m_nmTot, inoffset, (TData)0.0,
                outptr, m_nmTot, outoffset, inblock.GetNumComponents());

            // Divide by Jacobian.
            DivideByJacobian<ExecSpace, false>(nelmt, m_nmTot, ncomp, m_jacptr,
                                               outptr, outptr, m_streamID);
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

    /**
     * @brief Copy the block's per-element inverse mass matrices into a
     * memory region of the device memory space.
     *
     * @param   invmass     Per-element inverse mass matrices; empty for a
     *                      regular block, in which case #m_dinvmass stays
     *                      empty and unused (see
     *                      MultiplyByElmtInvMassBlockOp::SetInvMassMatrix).
     */
    void v_SetInvMassMatrix(std::vector<TData> &invmass) override
    {
        this->m_dinvmass = LibUtilities::MemoryRegion<
            TData>::template FromVector<MemSpace, TData>(invmass);
    }
};

} // namespace Nektar::MultiRegions::detail
