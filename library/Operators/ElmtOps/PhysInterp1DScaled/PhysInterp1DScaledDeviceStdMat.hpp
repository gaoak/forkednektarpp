///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledDeviceStdMat.hpp
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

/**
 * @file PhysInterp1DScaledDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the per-block
 * scaled physical-space interpolation: one strided-batched BLAS GEMM per
 * block.
 *
 * @details
 * Included by the registration translation units CMake generates for the
 * "StdMat" implementation on the Device execution space (see
 * PhysInterp1DScaledSerialAVXStdMat.hpp for why the implementation
 * headers may each define the same primary template).
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through PhysInterp1DScaledOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Device StdMat block implementation of the scaled interpolation:
 * the whole block's interpolation as a single strided-batched dense
 * matrix product on the back-end BLAS.
 *
 * @details
 * As in the Serial/AVX StdMat implementation the matrix depends on the
 * scale factor, so the constructor records only the block's shape, basis
 * keys, nodal points type and input point counts #m_nm;
 * v_SetScaleFactor() derives the output counts #m_nq and fetches the
 * dense interpolation matrix I (nqTot x nmTot, column-major,
 * StdRegions::StdMatKey with StdRegions::ePhysInterpStdMat -- the
 * untransposed counterpart of the Serial/AVX matrix) into device memory.
 * Every element then shares it, and the block's interpolation is one GEMM
 * per component,
 *
 *     out (nqTot x nelmtTot) = I * in (nmTot x nelmtTot),
 *
 * where the columns of `in`/`out` are the elements (padding elements
 * included, their results being ignored) and the homogeneous modes,
 * nelmtTot = padded elements * homogeneous modes. All components are
 * issued at once as a strided-batched GEMM with one batch entry per
 * component, the batch stride being the distance between consecutive
 * components of the block and the matrix stride zero.
 * NekBlas::GemmStridedBatched forwards to the BLAS of the enabled device
 * back-end. The GEMM needs each element's data contiguous, so the
 * implementation works at interleave width 1 (#m_implInterleaveWidth) and
 * reshapes the block storage to that width and back around the call.
 *
 * Each block operator owns device stream `block_idx + 1`, so different
 * blocks' interpolations may overlap on the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (Operators::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see PhysInterp1DScaledDeviceSumFac.hpp for the sum-factorised device
 * paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysInterp1DScaledBlockOpImpl : public PhysInterp1DScaledBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream and capture
     * the block's element metadata and everything the matrix warehouse
     * key needs; the matrix itself is fetched by v_SetScaleFactor().
     *
     * Records the shape, geometry type, reference and coordinate
     * dimension, the input point count of every direction, the
     * per-direction basis keys and, for a nodal expansion, its nodal
     * points type.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysInterp1DScaledBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysInterp1DScaledBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        // Fetch basis key.
        m_basisKeys = std::vector<LibUtilities::BasisKey>(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_nodalType = (exp->IsNodalNonTensorialExp())
                          ? exp->GetNodalPointsKey().GetPointsType()
                          : LibUtilities::eNoPointsType;
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysInterp1DScaledBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// The GEMM requires contiguous per-element columns, i.e.
    /// non-interleaved storage.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Basis key of each direction, kept for the matrix warehouse key.
    std::vector<LibUtilities::BasisKey> m_basisKeys;
    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Nodal points type for the nodal shapes, eNoPointsType otherwise;
    /// part of the matrix warehouse key.
    LibUtilities::PointsType m_nodalType;
    /// Whether the block's geometry is deformed; unused here (the
    /// interpolation is geometry independent).
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; unused
    /// here.
    unsigned int m_coordDim;
    /// Input points per element, the product of #m_nm; set by
    /// v_SetScaleFactor.
    unsigned int m_nmTot;
    /// Output points per element, the product of #m_nq; set by
    /// v_SetScaleFactor.
    unsigned int m_nqTot;
    /// Input (unscaled) quadrature points per direction. Named after the
    /// modal dimension of the matrix it feeds, which for this operator is
    /// a point count.
    std::vector<unsigned int> m_nm;
    /// Output (scaled) quadrature points per direction; set by
    /// v_SetScaleFactor.
    std::vector<unsigned int> m_nq;
    /// Dense interpolation matrix (nqTot x nmTot, column-major) in device
    /// memory, cached in the data warehouse; only valid once
    /// v_SetScaleFactor has run.
    const TData *m_matptr;

    /**
     * @brief Interpolate the block with one strided-batched GEMM covering
     * every component and homogeneous mode.
     *
     * The storage of all components is reshaped to interleave width 1 (a
     * no-op for non-interleaved fields) in one pass, the GEMM described
     * in the class notes is issued on this block's stream with the
     * homogeneous modes folded into the columns and the components into
     * the batch, and both fields are reshaped back to the input's width,
     * which the output block is also set to.
     *
     * The scale factor must have been set first: #m_matptr, #m_nmTot and
     * #m_nqTot are read here but only written by v_SetScaleFactor, and
     * nothing in this path checks that it has run.
     *
     * @param   inblock     Physical-space input block on the elements'
     *                      own quadrature grids.
     * @param   outblock    Physical-space output block on the scaled
     *                      grids; overwritten.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
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

        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmtTot, m_nmTot, (TData)1.0, m_matptr,
            m_nqTot, 0, inptr, m_nmTot, inoffset, (TData)0.0, outptr, m_nqTot,
            outoffset, inblock.GetNumComponents());

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
     * @brief Fix the output point counts for @p scale and fetch the
     * matching dense interpolation matrix into device memory.
     *
     * The point-count rule and the caching behaviour are those of the
     * Serial/AVX StdMat implementation -- direction 0 scaled and
     * truncated, a direction one point below direction 0 keeping that
     * offset, the totals taken as products, and the output counts part
     * of the StdMatKey -- the only difference being that the
     * untransposed matrix (ePhysInterpStdMat) is fetched here, as the
     * GEMM of v_Apply wants it.
     *
     * Everything is recomputed from the stored input counts, so a block
     * operator may be re-targeted at a new scale factor by calling this
     * again.
     *
     * @param   scale   Multiplier applied to the per-direction
     *                  quadrature-point counts.
     */
    void v_SetScaleFactor(const TData &scale) override
    {
        this->m_scale = scale;
        m_nq.clear();
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            if (d == 0)
            {
                m_nq.push_back(this->m_scale * m_nm[0]);
            }
            else if (d == 1)
            {
                // if delta between nm0 and nm1 is 1 then keep this delta
                // for new points to capitalise on switch templating
                const auto nq1 =
                    (m_nm[0] - m_nm[1] == 1)
                        ? (unsigned int)(this->m_scale * m_nm[0]) - 1
                        : (unsigned int)(this->m_scale * m_nm[1]);
                m_nq.push_back(nq1);
            }
            else if (d == 2)
            {
                const auto nq2 =
                    (m_nm[0] - m_nm[2] == 1)
                        ? (unsigned int)(this->m_scale * m_nm[0]) - 1
                        : (unsigned int)(this->m_scale * m_nm[2]);
                m_nq.push_back(nq2);
            }
        }

        m_nmTot =
            std::accumulate(m_nm.begin(), m_nm.end(), 1, std::multiplies());
        m_nqTot =
            std::accumulate(m_nq.begin(), m_nq.end(), 1, std::multiplies());

        m_matptr = this->m_dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(m_basisKeys, m_shapeType,
                                         StdRegions::ePhysInterpStdMat,
                                         m_nodalType, m_nq));
    }
};

} // namespace Nektar::Operators::detail
