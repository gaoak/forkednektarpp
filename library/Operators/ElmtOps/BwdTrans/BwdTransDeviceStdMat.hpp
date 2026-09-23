///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceStdMat.hpp
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
// Description: Device standard-matrix implementation of the per-block
// backward transform
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the per-block
 * backward transform: one strided batched BLAS GEMM per block.
 *
 * Included by the translation units CMake generates for the "StdMat"
 * implementation on the Device execution space; see
 * BwdTransSerialAVXStdMat.hpp for why the implementation headers may
 * each define the same primary template.
 *
 * @see BwdTransDeviceSumFac.hpp for the sum-factorised device paths.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through BwdTransOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Device StdMat block implementation of the backward transform:
 * the whole block's transform as a single strided batched dense matrix
 * product on the back-end BLAS.
 *
 * Every element shares the standard-region backward-transform matrix
 * \f$B\f$ -- a StdRegions::StdMatKey with StdRegions::eBwdTransStdMat,
 * nqTot x nmTot and column-major -- so the transform of the whole block
 * is the one product
 *
 *     out (nqTot x nelmtTot) = B * in (nmTot x nelmtTot)
 *
 * repeated once per component. The columns are the elements of the
 * block, padding elements included, with the homogeneous modes folded
 * into the column count; the components are the batch dimension, so
 * NekBlas::GemmStridedBatched issues all of them in one call. The
 * matrix stride is zero, since the same \f$B\f$ serves every batch
 * entry, while the input and output strides step one component apart.
 * NekBlas forwards to the BLAS of the enabled device back-end. The GEMM
 * needs each element's data contiguous, so the implementation works at
 * interleave width 1 (#m_implInterleaveWidth) and reshapes the block
 * storage to that width and back around the call. In append mode the
 * GEMM runs with beta = 1 and accumulates onto the existing output.
 *
 * Each block operator owns device stream `block_idx + 1`, so different
 * blocks' transforms may overlap on the device.
 *
 * The element geometry never enters: the backward transform is a
 * standard-region operation, so #m_isDeformed and #m_coordDim are
 * recorded only for uniformity with the other element operators.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (Operators::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see BwdTransSerialAVXStdMat.hpp for the Serial/AVX StdMat path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class BwdTransBlockOpImpl : public BwdTransBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the block's backward-transform matrix, record the
     * shape sizes the GEMM is issued with, and claim the block's device
     * stream.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; also sets
     *                          #m_streamID.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse the matrix is cached in.
     */
    BwdTransBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BwdTransBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eBwdTransStdMat,
                                         nodalType));
    }

    /// Block-operator factory registration key,
    /// `"BlockBwdTrans" + execution space + implementation`. Defined by
    /// the CMake-generated declaration file, whose initialiser performs
    /// the registration.
    static std::string className;

    /**
     * @brief Creator function registered with the block-operator factory
     * under #className.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     *
     * @return The newly created block operator.
     */
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BwdTransBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the GEMM operates at: the batched product
    /// requires contiguous per-element columns, so storage is
    /// non-interleaved.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; unused here (see the
    /// class notes).
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; unused
    /// here.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Backward-transform matrix (nqTot x nmTot, column-major) in device
    /// memory, cached in the data warehouse.
    const TData *m_matptr;

    /**
     * @brief Backward-transform the block with one strided batched GEMM,
     * batched over the components.
     *
     * The storage is reshaped to interleave width 1 -- a no-op for
     * non-interleaved fields, and done for the output only in append
     * mode, where the accumulation reads it -- the batched GEMM
     * described in the class notes is issued on this block's stream, and
     * the storage is reshaped back to the input's width, which the
     * output block is also set to.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated into in append mode.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Offsets between the components of a block.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);
        }

        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmtTot, m_nmTot, (TData)1.0, m_matptr,
            m_nqTot, 0, inptr, m_nmTot, inoffset, (TData)this->m_append, outptr,
            m_nqTot, outoffset, inblock.GetNumComponents());

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
