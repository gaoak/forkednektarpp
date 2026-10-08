///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialAVXStdMat.hpp
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
// Description: Serial/AVX standard-matrix (StdMat) implementation of
// the per-block inner product with the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block inner product with the basis.
 *
 * @details
 * All elements of a block share the same basis and quadrature, so one
 * dense standard-region matrix serves the whole block; only the element
 * Jacobian, which the matrix does not carry, has to be applied per
 * element. One libxsmm GEMM per element group then contracts the
 * Jacobian-weighted values (or, with the quadrature metric off, the raw
 * values) against the matrix. On the AVX execution space each SIMD lane
 * owns one element of a group, so the GEMM's left operand is the
 * lane-interleaved batch of element data; on Serial the SIMD type is a
 * scalar and the group size is one.
 *
 * This header provides the definition of detail::IProductWRTBaseBlockOpImpl
 * for the Serial and AVX execution spaces with the StdMat implementation
 * tag. The sum-factorised Serial/AVX definition lives in
 * IProductWRTBaseSerialAVXSumFac.hpp and the Device definitions in
 * IProductWRTBaseDeviceStdMat.hpp and IProductWRTBaseDeviceSumFac.hpp; the
 * class template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTBaseOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX StdMat inner product with the basis: one small dense
 * matrix multiply per group of SIMD lanes, preceded by a Jacobian kernel
 * when the quadrature metric is in use.
 *
 * @details
 * The constructor fetches two standard-region matrices from the data
 * warehouse, so they are built once and shared between operators:
 * - #m_matptr, the StdMatKey with eIProductWRTBaseStdMatTranspose: an
 *   nqTot x nmTot column-major array whose (q, n) entry is
 *   \f$w_q\,\phi_n(\xi_q)\f$, the basis transpose with the standard-region
 *   quadrature weights already folded in (the warehouse builds it by
 *   applying StdExpansion::IProductWRTBase to the physical unit vectors);
 * - #m_BT_matptr, the StdMatKey with eBwdTransStdMat: the plain
 *   backward-transform matrix \f$B\f$ (nqTot x nmTot column-major), which
 *   used as the right-hand factor of the same GEMM applies
 *   \f$B^{\mathsf T}\f$ without any weights.
 * It also fetches the block's Jacobians, interleaved at
 * #m_implInterleaveWidth like the field data: one value per element for a
 * regular geometry, nqTot values per element for a deformed one.
 *
 * One small dense GEMM per group of simd_t::width elements -- dispatched
 * through LibxsmmDispatchWrapper with alpha = 1 and beta = 0 -- then
 * computes
 *
 *        out (width x nmTot) = in (width x nqTot) * M,
 *
 * with M either the weighted matrix or \f$B\f$, the panels being the
 * group's interleaved storage: lane j of a panel column holds element j's
 * value of that point or mode, so the vectorisation is across the elements
 * of the group. On the Serial execution space simd_t is scalar (width 1)
 * and the panel degenerates to a single element's values.
 *
 * The Device StdMat implementation fetches the same two matrices in the
 * opposite orientation (eIProductWRTBaseStdMat and
 * eBwdTransStdMatTranspose), its GEMM having the matrix on the left; both
 * realise the same operator.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata and fetch the two
     * standard-region matrices and the block's Jacobians (see the class
     * description).
     *
     * The matrix keys carry the element's basis keys and shape and, for a
     * nodal expansion, the nodal points type. The Jacobians are fetched at
     * the SIMD width.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    IProductWRTBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType,
                StdRegions::eIProductWRTBaseStdMatTranspose, nodalType));

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));

        m_BT_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eBwdTransStdMat,
                                         nodalType));
    }

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the kernel and GEMM operate at: the SIMD
    /// width (1 on the Serial execution space).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobians); selects the
    /// DEFORMED branch of the Jacobian kernel and its stride.
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
    /// Weighted transposed inner-product matrix (nqTot x nmTot,
    /// column-major); used when the quadrature metric is on.
    const TData *m_matptr;
    /// Backward-transform matrix (nqTot x nmTot, column-major), used as
    /// the unweighted basis transpose when the quadrature metric is off.
    const TData *m_BT_matptr;
    /// Jacobians of the block, interleaved at #m_implInterleaveWidth: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;

    /**
     * @brief Apply the operator to every element of the block by dense
     * matrix multiplication.
     *
     * Per component and homogeneous mode, and per group of simd_t::width
     * elements: the input is reshaped to #m_implInterleaveWidth, the
     * Jacobian kernel and the GEMM run, and the storage is reshaped back.
     * Fields stored at a larger interleave width are handled in chunks of
     * chunkSize = max(simd width, input width) elements, reshaped once per
     * width_ratio groups. On return the output block's interleave width is
     * set to the input's.
     *
     * With the quadrature metric on, MultiplyByJacobian writes
     * scale * J * u for the group into the static workspace -- the only
     * place the Jacobian and the scale factor enter -- and the GEMM against
     * #m_matptr, which carries the quadrature weights, follows. With the
     * metric off the GEMM runs straight from the input against
     * #m_BT_matptr, and the result is scaled in place afterwards if the
     * scale factor is not 1. The Jacobian pointer advances by one vector
     * per group for a regular geometry and by nqTot vectors for a deformed
     * one, and is rewound for every component.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
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
            (this->m_integration)
                ? BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                      simd_t::width * m_nqTot)
                : nullptr;

        // Dispatch kernel.
        auto gemm_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 0.0);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;

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

                if (this->m_integration) // integration with weights
                {
                    // Multiply by jacobian.
                    if (m_isDeformed)
                    {
                        MultiplyByJacobian<ExecSpace, true>(
                            1, m_nqTot,
                            reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(wspptr), this->m_scale);
                        jacptr += m_nqTot * simd_t::width;
                    }
                    else
                    {
                        MultiplyByJacobian<ExecSpace, false>(
                            1, m_nqTot,
                            reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(wspptr), this->m_scale);
                        jacptr += simd_t::width;
                    }

                    // Perform matrix-matrix multiply.
                    gemm_kernel(wspptr, m_matptr, outptr);
                }
                else
                {
                    // Just perform matrix-matrix multiply of B^T
                    gemm_kernel(inptr, m_BT_matptr, outptr);

                    if (this->m_scale != 1.0)
                    {
                        for (unsigned q = 0; q < m_nmTot; ++q)
                        {
                            reinterpret_cast<simd_t *>(outptr)[q] *=
                                this->m_scale;
                        }
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
