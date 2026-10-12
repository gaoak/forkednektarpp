///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassSerialAVXGeneric.hpp
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
// Description: Serial/AVX Generic implementation of the elemental
// inverse mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MultiplyByElmtInvMassSerialAVXGeneric.hpp
 * @brief Serial/AVX implementation of the per-block element-wise inverse
 * mass operator, registered under the Generic implementation tag.
 *
 * @details
 * One small dense multiply per group of SIMD lanes: a libxsmm GEMM against
 * the shared standard-region inverse followed by a division by the element
 * Jacobian for a regular block, a strided-batched GEMV against the
 * element's own inverse mass matrix for a deformed one. On the AVX
 * execution space each SIMD lane owns one element of a group, so the
 * operands are the lane-interleaved batch of element data; on Serial the
 * SIMD type is a scalar and the group size is one.
 *
 * This header provides the definition of
 * detail::MultiplyByElmtInvMassBlockOpImpl for the Serial and AVX
 * execution spaces; the Device definition lives in
 * MultiplyByElmtInvMassDeviceGeneric.hpp. The class template name is
 * shared, and each CMake-generated registration translation unit includes
 * exactly one of the two headers.
 *
 * @see MultiplyByElmtInvMassBlockOp.hpp for the regular/deformed case
 * distinction realised here.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <LocalRegions/Expansion.h>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX Generic inverse mass operator: one small dense multiply
 * per group of SIMD lanes, the element Jacobian divided out afterwards on
 * a regular block.
 *
 * @details
 * The constructor fetches two pieces of shared data from the data
 * warehouse, so that both are built once and reused by every operator on
 * the expansion list:
 * - #m_invmassptr, the StdMatKey with eInvMassStdMatTranspose: the
 *   transposed standard-region inverse mass matrix (nmTot x nmTot,
 *   column-major), the layout the libxsmm kernel below expects;
 * - #m_jacptr, the block's Jacobians (JacobianKey at
 *   #m_implInterleaveWidth), one constant per element for a regular block,
 *   interleaved like the field data.
 *
 * Both are fetched for deformed blocks too, whose path uses neither.
 *
 * Both geometry cases work one element group -- simd_t::width consecutive
 * elements, one per vector lane -- at a time:
 * - regular block: one libxsmm GEMM per group, dispatched through
 *   LibxsmmDispatchWrapper with alpha = 1 and beta = 0, computes
 *
 *        out (width x nmTot) = in (width x nmTot) * (M^std)^{-T},
 *
 *   where lane j of a panel column holds element j's value of that mode,
 *   so the vectorisation is across the elements of the group;
 *   DivideByJacobian then divides each lane by its element's Jacobian,
 *   completing \f$(M^e)^{-1} = (M^{\mathrm{std}})^{-1} / J^e\f$;
 * - deformed block: one NekBlas::GemvStridedBatched per group performs
 *   simd_t::width matrix-vector products, lane j using element j's own
 *   inverse mass matrix from #m_dinvmass (batch stride nmTot x nmTot); the
 *   geometry is already contained in those matrices, so no Jacobian is
 *   applied. On AVX the input and output vectors are addressed with
 *   increment simd_t::width and batch stride 1, i.e. the interleaved
 *   lanes; on Serial with unit increment and batch stride nmTot, the batch
 *   being a single element.
 *
 * The element geometry enters only through the Jacobian (regular) or the
 * per-element matrices (deformed).
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (Generic).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class MultiplyByElmtInvMassBlockOpImpl
    : public MultiplyByElmtInvMassBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;

    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata and fetch the transposed
     * standard-region inverse mass matrix and the block's Jacobians (see
     * the class description).
     *
     * The matrix key carries the element's basis keys and shape and, for a
     * nodal expansion, the nodal points type. The Jacobians are fetched at
     * the SIMD width.
     *
     * @param   block_idx       Index of the block.
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
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eInvMassStdMatTranspose,
                                         nodalType));

        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    /// Block-operator class name; defined by generated factory code.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            MultiplyByElmtInvMassBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the kernels operate at: the SIMD vector
    /// width (1 on Serial).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// Transposed standard-region inverse mass matrix (nmTot x nmTot,
    /// column-major), cached in the data warehouse; used by the regular
    /// path only.
    const TData *m_invmassptr;
    /// Jacobians of the block, one per element, interleaved at
    /// #m_implInterleaveWidth and cached in the data warehouse; used by
    /// the regular path only.
    const TData *m_jacptr;
    /// Per-element inverse mass matrices of a deformed block, as set
    /// through v_SetInvMassMatrix(); empty for a regular block.
    LibUtilities::MemoryRegion<TData> m_dinvmass;

    /**
     * @brief Apply the inverse mass matrices to every element of the
     * block, one element group at a time.
     *
     * The field may be interleaved at a width other than simd_t::width: a
     * field is interleaved at the width of the execution space it was
     * prepared for, which need not equal simd_t::width here but is taken
     * to be 1 or a multiple of it. Work therefore proceeds in chunks of
     * `chunkSize = max(simd_t::width, input width)` elements: each chunk
     * is reshaped in place to simd_t::width interleaving, its
     * `width_ratio` lane groups are processed by the geometry-dependent
     * kernel described in the class description, and both input and
     * output chunks are reshaped back once the last group of the chunk is
     * done. On return the output block's interleave width is set to the
     * input's.
     *
     * The outer loop runs over components times homogeneous modes; the
     * data pointers advance continuously through the block, while the
     * per-element pointer walked by the active branch -- the deformed
     * inverse mass matrices, or the Jacobians -- is reset at the start of
     * every pass, so each pass sees the same per-element data. The regular
     * branch's matrix needs no reset: it is the single standard-region
     * inverse, applied to every group from #m_invmassptr.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Get BLAS handle. Always BLAS: the libxsmm Gemv does not support
        // strided vectors, which the AVX layout needs.
        auto handle = NekBlas::Handle<NektarSpaces::Serial>::GetInstance(0);

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            if (m_isDeformed)
            {
                // Fetch deformed mass matrix.
                auto dmatptr =
                    this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>();

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

                    // Perform batched matrix-vector multiply.
                    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                    {
                        NekBlas::GemvStridedBatched(
                            handle, "N", m_nmTot, m_nmTot, (TData)1.0, dmatptr,
                            m_nmTot, m_nmTot * m_nmTot, inptr, simd_t::width, 1,
                            (TData)0.0, outptr, simd_t::width, 1,
                            simd_t::width);
                    }
                    else
                    {
                        NekBlas::GemvStridedBatched(
                            handle, "N", m_nmTot, m_nmTot, (TData)1.0, dmatptr,
                            m_nmTot, m_nmTot * m_nmTot, inptr, 1, m_nmTot,
                            (TData)0.0, outptr, 1, m_nmTot, simd_t::width);
                    }

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
                            m_nmTot,
                            (TData *)outptr -
                                (width_ratio - 1) * m_nmTot * simd_t::width);
                    }

                    // Increment pointers.
                    inptr += m_nmTot * simd_t::width;
                    outptr += m_nmTot * simd_t::width;
                    dmatptr += m_nmTot * m_nmTot * simd_t::width;
                }
            }
            else
            {
                auto jacptr = m_jacptr;

                // Dispatch kernel.
                auto invmass_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    simd_t::width, m_nmTot, m_nmTot, (TData)1.0, (TData)0.0);

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
                    invmass_kernel(inptr, m_invmassptr, outptr);

                    // Divide by Jacobian.
                    DivideByJacobian<ExecSpace, false>(
                        1, m_nmTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr));

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
                            m_nmTot,
                            (TData *)outptr -
                                (width_ratio - 1) * m_nmTot * simd_t::width);
                    }

                    // Increment pointers.
                    inptr += m_nmTot * simd_t::width;
                    outptr += m_nmTot * simd_t::width;
                    jacptr += simd_t::width;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /**
     * @brief Take a copy of the block's per-element inverse mass matrices
     * into host memory.
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
