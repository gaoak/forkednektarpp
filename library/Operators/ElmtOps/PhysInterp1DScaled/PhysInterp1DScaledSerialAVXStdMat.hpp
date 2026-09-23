///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledSerialAVXStdMat.hpp
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
 * @file PhysInterp1DScaledSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block scaled physical-space interpolation.
 *
 * @details
 * All elements of a block share the same points distribution, so one
 * dense interpolation matrix over all directions serves the whole block,
 * applied by one libxsmm GEMM per group of SIMD lanes. This header
 * defines the detail::PhysInterp1DScaledBlockOpImpl used when the
 * "StdMat" implementation is requested on the Serial or AVX execution
 * space; CMake includes it into the registration translation units
 * generated for exactly those factory entries, which is why this header,
 * the SumFac header and the device headers may each define the same
 * primary template without ever colliding.
 *
 * @see PhysInterp1DScaledSerialAVXSumFac.hpp for the sum-factorised
 * alternative and PhysInterp1DScaledDeviceStdMat.hpp for the device
 * StdMat path.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief StdMat block implementation of the scaled interpolation for the
 * Serial and AVX execution spaces: one small dense matrix multiply per
 * group of SIMD lanes.
 *
 * @details
 * The interpolation matrix is not known at construction -- it depends on
 * the scale factor -- so the constructor only records what the warehouse
 * key needs: the block's shape, its per-direction basis keys, its nodal
 * points type and the input point counts #m_nm. v_SetScaleFactor() then
 * derives the output counts #m_nq and fetches the transposed dense
 * interpolation matrix (StdRegions::StdMatKey with
 * StdRegions::ePhysInterpStdMatTranspose: an nmTot x nqTot column-major
 * array \f$I^T\f$, built one row per input point by interpolating each
 * unit physical-space vector through LibUtilities::Interp1D/2D/3D) from
 * the data warehouse, so it is built once per basis and scale factor and
 * shared between operators.
 *
 * One libxsmm GEMM per group of simd_t::width elements then computes
 *
 *     out-panel (width x nqTot) = in-panel (width x nmTot) * I^T,
 *
 * the panels being the group's interleaved storage: lane j of a panel
 * column holds element j's value at that point, so the vectorisation is
 * across the elements of a group. On the Serial execution space simd_t is
 * scalar (width 1) and a panel degenerates to a single element's values.
 *
 * The element geometry never enters: the interpolation acts on the
 * reference element's points, so #m_isDeformed and #m_coordDim are
 * recorded only for uniformity with the other element operators.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if (scalar for
 *                         Serial, tinysimd::simd for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (Operators::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysInterp1DScaledBlockOpImpl : public PhysInterp1DScaledBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata and everything the
     * matrix warehouse key needs; the matrix itself is fetched by
     * v_SetScaleFactor().
     *
     * Records the shape, geometry type, reference and coordinate
     * dimension, the input point count of every direction, the
     * per-direction basis keys and, for a nodal expansion, its nodal
     * points type.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
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
    /// Interleave width the GEMM operates at: the SIMD vector width (1 on
    /// Serial), one element per lane.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// Transposed dense interpolation matrix (nmTot x nqTot,
    /// column-major), cached in the data warehouse; only valid once
    /// v_SetScaleFactor has run.
    const TData *m_matptr;

    /**
     * @brief Interpolate every element of the block by dense matrix
     * multiplication.
     *
     * Follows the same chunked reshape pattern as the other Serial/AVX
     * StdMat implementations: the field may be interleaved at a width
     * other than the SIMD width, so work proceeds in chunks of
     * `chunkSize = max(simd width, input width)` elements per component
     * (times homogeneous mode); each chunk is reshaped in place to the
     * SIMD width, its `width_ratio` lane groups are multiplied through
     * the GEMM kernel, and the chunk is reshaped back once its last group
     * is done. On return the output block's interleave width is set to
     * the input's.
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
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, 0.0);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        m_nmTot, (TData *)inptr);
                }

                // Perform matrix-matrix multiply.
                gemm_kernel(inptr, m_matptr, outptr);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nmTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /**
     * @brief Fix the output point counts for @p scale and fetch the
     * matching dense interpolation matrix.
     *
     * Direction 0 gets `nq0 = (unsigned int)(scale * nm0)`; a direction
     * whose input point count is exactly one below direction 0's keeps
     * that offset, `nqd = (unsigned int)(scale * nm0) - 1`, and any other
     * direction gets `nqd = (unsigned int)(scale * nmd)`. That is the
     * rule of MultiRegions::ExpList::Get1DScaledTotPoints, so the output
     * sizes agree with a field sized by it. #m_nmTot and #m_nqTot are the
     * products over the directions, and the matrix is fetched under a
     * StdMatKey carrying the output counts, so each set of output
     * counts gets its own cached entry.
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
                                         StdRegions::ePhysInterpStdMatTranspose,
                                         m_nodalType, m_nq));
    }
};

} // namespace Nektar::Operators::detail
