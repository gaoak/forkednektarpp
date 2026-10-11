///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceSerialAVXStdMat.hpp
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
 * @file DivergenceSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix implementation of the block
 * divergence.
 *
 * @details
 * Instead of sum-factorised 1D contractions, this variant multiplies
 * each SIMD group of elements by dense standard-region derivative
 * matrices (one `nqTot x nqTot` matrix per reference direction, with
 * any collapsed-coordinate correction baked in) via libxsmm GEMM
 * calls, then contracts the resulting reference-direction derivatives
 * with the derivative factors. See DivergenceSerialAVXSumFac.hpp for
 * the sum-factorised alternative.
 *
 * This header provides the definition of detail::DivergenceBlockOpImpl
 * for the Serial and AVX execution spaces with the StdMat
 * implementation tag; the class template name is shared with the other
 * implementation headers, and each CMake-generated registration
 * translation unit includes exactly one of them.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through DivergenceOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX standard-matrix block divergence: per plane, SIMD
 * group of elements and vector component, dense GEMMs produce all
 * reference-direction derivatives, which are then contracted with the
 * component's row of derivative factors and accumulated into the
 * output.
 *
 * @details
 * ### What one application does
 * For each homogeneous plane, each group of `simd_t::width` elements
 * and each component c of the vector field:
 * -# one libxsmm GEMM per reference direction d multiplies the
 *    group's interleaved data for component c by the transposed
 *    standard-region derivative matrix (ePhysDerivStdMatTranspose;
 *    transposed because the group data is stored lane-innermost),
 *    filling workspace slice d with \f$\partial u_c/\partial\xi_d\f$
 *    at every point;
 * -# MultiplyByDerivDirFactorKernel<ExecSpace, APPEND, DEFORMED>
 *    contracts the slices with row c of the derivative factors,
 *    \f$\partial u_c/\partial x_c = \sum_d
 *    (\partial\xi_d/\partial x_c)\,\partial u_c/\partial\xi_d\f$,
 *    overwriting the output for c = 0 and accumulating (APPEND) for
 *    the remaining components -- so the output ends up holding
 *    \f$\sum_c \partial u_c/\partial x_c\f$.
 *
 * The chunk-wise re-interleaving between the block's stored width and
 * the SIMD width follows the same scheme as the sum-factorised
 * variant (see the DivergenceSerialAVXSumFac class notes); the
 * reference-direction derivatives are staged in the static per-stream
 * workspace shared by all block operators. The planes are processed in
 * turn over the same derivative factors, each giving the xy divergence
 * of that plane.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation Implementation tag (MultiRegions::StdMat).
 * @tparam TData          Floating-point type of the field data.
 *
 * @see DivergenceDeviceStdMat.hpp for the device counterpart.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the stacked transposed derivative matrices
     * (ePhysDerivStdMatTranspose, one `nqTot x nqTot` block per
     * reference direction, keyed on the block's basis keys) and the
     * block's derivative factors interleaved at the SIMD width.
     *
     * Asserts (debug builds) that the shape dimension equals the
     * coordinate dimension, i.e. embedded expansions are not
     * supported.
     *
     * @param   block_idx       Index of the block within the
     *                          expansion list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        ASSERTL1(m_dimension == m_coordDim,
                 "Shape dimension and coordinate dimension are not the same.");

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::ePhysDerivStdMatTranspose));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
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
            DivergenceBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the GEMMs and the derivative-factor
    /// contraction operate at: the SIMD vector width (1 on Serial).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed (derivative factors
    /// then vary per quadrature point rather than per element).
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the element; equal to #m_dimension
    /// here, as asserted in the constructor.
    unsigned int m_coordDim;
    /// Number of expansion modes per element; unused by this
    /// implementation.
    unsigned int m_nmTot;
    /// Number of quadrature points per element.
    unsigned int m_nqTot;
    /// Stacked transposed standard-region derivative matrices
    /// (ePhysDerivStdMatTranspose), one `nqTot x nqTot` block per
    /// reference direction, keyed on the block's basis keys.
    const TData *m_matptr;
    /// Derivative factors of the block, interleaved at
    /// #m_implInterleaveWidth: per element group, all factors of one
    /// point (deformed) or element (regular) are adjacent, lanes
    /// innermost.
    const TData *m_dfptr;

    /**
     * @brief Differentiate every vector component of the block by
     * dense GEMM and accumulate the contracted derivatives into the
     * output.
     *
     * The two steps described in the class notes are run for each
     * component c of the vector field, working through each plane of
     * the block in groups of #m_implInterleaveWidth elements, with the
     * derivative factors restarting at every plane: one GEMM per
     * reference direction fills the shared static workspace with
     * component c's reference-direction derivatives, one slice of
     * #m_nqTot vectors per direction, then
     * MultiplyByDerivDirFactorKernel -- handed that slice stride --
     * contracts them with row c of the derivative factors, overwriting
     * the output for c = 0 and accumulating (APPEND) for the rest.
     *
     * As in the other Serial/AVX implementations the field may be
     * interleaved at a width other than #m_implInterleaveWidth, so
     * each component's data is reshaped to #m_implInterleaveWidth at
     * the start of the chunk of groups it belongs to and reshaped
     * back, together with the output, once that chunk's last group is
     * done; the output block's interleave width is set to the
     * input's on return.
     *
     * @param   inblock     Physical-space input block; its first
     *                      #m_dimension components are the vector
     *                      field, each spanning all the planes.
     * @param   outblock    Physical-space output block receiving the
     *                      accumulated divergence of every plane in
     *                      its first component.
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

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto inoffset = inblock.CompSize() * nhomo;

        // Loop over the planes. Each holds one xy divergence, over the same
        // geometry, so the derivative factors restart with every plane while
        // the field pointers run on through the block.
        for (unsigned int p = 0; p < nhomo; ++p)
        {
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        m_nqTot, (TData *)inptr);
                }

                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    gemm_kernel(inptr, m_matptr + d * m_nqTot * m_nqTot,
                                wspptr + d * m_nqTot * simd_t::width);
                }

                // Multiply by derivative factor.
                if (m_isDeformed)
                {
                    MultiplyByDerivDirFactorKernel<ExecSpace, false, true>(
                        0, m_nqTot, m_coordDim, m_dimension, 1, m_nqTot,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(wspptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
                else
                {
                    MultiplyByDerivDirFactorKernel<ExecSpace, false, false>(
                        0, m_nqTot, m_coordDim, m_dimension, 1, m_nqTot,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(wspptr),
                        reinterpret_cast<simd_t *>(outptr));
                }

                // Calculate dv/dy & dw/dz
                for (unsigned c = 1; c < m_dimension; ++c)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, interleaveWidth, chunkSize,
                            m_nqTot, (TData *)inptr + c * inoffset);
                    }

                    // Perform matrix-matrix multiply.
                    for (unsigned int d = 0; d < m_dimension; d++)
                    {
                        gemm_kernel(inptr + c * inoffset,
                                    m_matptr + d * m_nqTot * m_nqTot,
                                    wspptr + d * m_nqTot * simd_t::width);
                    }

                    // Multiply by derivative factor.
                    if (m_isDeformed)
                    {
                        MultiplyByDerivDirFactorKernel<ExecSpace, true, true>(
                            c, m_nqTot, m_coordDim, m_dimension, 1, m_nqTot,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(wspptr),
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        MultiplyByDerivDirFactorKernel<ExecSpace, true, false>(
                            c, m_nqTot, m_coordDim, m_dimension, 1, m_nqTot,
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(wspptr),
                            reinterpret_cast<simd_t *>(outptr));
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned c = 0; c < m_dimension; ++c)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            (TData *)inptr + c * inoffset -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                    }

                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers.
                dfptr +=
                    (m_isDeformed)
                        ? m_coordDim * m_dimension * m_nqTot * simd_t::width
                        : m_coordDim * m_dimension * simd_t::width;
                inptr += m_nqTot * simd_t::width;
                outptr += m_nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
