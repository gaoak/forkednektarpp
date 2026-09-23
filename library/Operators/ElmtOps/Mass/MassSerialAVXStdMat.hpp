///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialAVXStdMat.hpp
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
// the per-block mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block mass operator.
 *
 * @details
 * All elements of a block share the same basis and quadrature, so one dense
 * standard-region matrix serves the whole block; only the element Jacobian,
 * which the matrices do not carry, has to be applied per element. One or
 * two libxsmm GEMMs per element group then apply the matrices. On the AVX
 * execution space each SIMD lane owns one element of a group, so the GEMM's
 * left operand is the lane-interleaved batch of element data; on Serial the
 * SIMD type is a scalar and the group size is one.
 *
 * This header provides the definition of detail::MassBlockOpImpl for the
 * Serial and AVX execution spaces with the StdMat implementation tag. The
 * sum-factorised Serial/AVX definition lives in MassSerialAVXSumFac.hpp and
 * the Device definitions in MassDeviceStdMat.hpp and MassDeviceSumFac.hpp;
 * the class template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/Mass/MassBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Serial/AVX StdMat mass operator: one or two small dense matrix
 * multiplies per group of SIMD lanes, the element Jacobian applied between
 * or after them.
 *
 * @details
 * The constructor picks its standard-region matrices from the data
 * warehouse -- so they are built once and shared between operators --
 * according to the block's geometry type, because that decides where the
 * Jacobian can be applied:
 * - a regular block takes #m_massmat, the StdMatKey with
 *   eMassStdMatTranspose: the transposed standard-region mass matrix
 *   (nmTot x nmTot, column-major), whose entries are the reference-element
 *   inner products \f$\sum_i w_i\,\phi_n(\xi_i)\,\phi_m(\xi_i)\f$ and which
 *   therefore already carries the quadrature weights. The element Jacobian
 *   is a single constant, so it is applied afterwards to the
 *   coefficient-space result;
 * - a deformed block takes the two matrices of the composition instead --
 *   #m_bwdmat (eBwdTransStdMatTranspose) and #m_ipbmat
 *   (eIProductWRTBaseStdMatTranspose, which carries the weights) -- because
 *   the per-point Jacobian has to multiply the intermediate physical values
 *   between them.
 * The pointer of the branch not taken is left unset. The constructor also
 * fetches the block's Jacobians, interleaved at #m_implInterleaveWidth like
 * the field data: one value per element for a regular geometry, nqTot
 * values per element for a deformed one.
 *
 * The multiplies are small dense GEMMs, one per group of simd_t::width
 * elements, dispatched through LibxsmmDispatchWrapper with alpha = 1 and
 * beta = 0. For a regular block that is the single
 *
 *        out (width x nmTot) = in (width x nmTot) * M^T,
 *
 * for a deformed one the pair
 *
 *        wsp (width x nqTot) = in  (width x nmTot) * B^T,
 *        out (width x nmTot) = wsp (width x nqTot) * W^T,
 *
 * with the Jacobian multiplication in between. The panels are the group's
 * interleaved storage: lane j of a panel column holds element j's value of
 * that mode or point, so the vectorisation is across the elements of the
 * group. On the Serial execution space simd_t is scalar (width 1) and a
 * panel degenerates to a single element's values.
 *
 * The Device StdMat implementation fetches the same matrices in the
 * opposite orientation (eMassStdMat, eBwdTransStdMat and
 * eIProductWRTBaseStdMat), its GEMMs having the matrix on the left; both
 * realise the same operator.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class MassBlockOpImpl : public MassBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata and fetch the
     * standard-region matrices of the block's geometry type and the
     * block's Jacobians (see the class description).
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
    MassBlockOpImpl(const unsigned int block_idx,
                    const LocalRegions::ExpansionSharedPtr &exp,
                    LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MassBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        if (m_isDeformed)
        {
            m_bwdmat = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::eBwdTransStdMatTranspose, nodalType));
            m_ipbmat = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::eIProductWRTBaseStdMatTranspose, nodalType));
        }
        else
        {
            m_massmat = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eMassStdMatTranspose,
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
    /// Element interleave width the GEMMs operate at: the SIMD width (1 on
    /// the Serial execution space). The Jacobians are fetched at this width
    /// too.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// Transposed standard-region mass matrix (nmTot x nmTot,
    /// column-major), quadrature weights included; set, and used, only for
    /// a regular block.
    const TData *m_massmat;
    /// Transposed backward-transform matrix (nmTot x nqTot, column-major);
    /// set, and used, only for a deformed block.
    const TData *m_bwdmat;
    /// Transposed inner-product matrix (nqTot x nmTot, column-major),
    /// quadrature weights included; set, and used, only for a deformed
    /// block.
    const TData *m_ipbmat;
    /// Jacobians of the block, interleaved at #m_implInterleaveWidth: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;

    /**
     * @brief Apply the element mass matrix to every element of the block by
     * dense matrix multiplication.
     *
     * Per component and homogeneous mode, and per group of simd_t::width
     * elements: the input is reshaped to #m_implInterleaveWidth, the GEMMs
     * and the Jacobian kernel run, and the storage is reshaped back. Fields
     * stored at a larger interleave width are handled in chunks of
     * chunkSize = max(simd width, input width) elements, reshaped once per
     * width_ratio groups. On return the output block's interleave width is
     * set to the input's.
     *
     * Per group the deformed branch runs the backward-transform GEMM into
     * the static workspace (sized for one group's physical values),
     * multiplies that workspace by the group's nqTot interleaved Jacobians
     * through MultiplyByJacobian, and feeds it to the inner-product GEMM;
     * the regular branch runs the single mass GEMM and multiplies the
     * resulting coefficients by the group's one Jacobian per element. The
     * Jacobian pointer advances accordingly -- nqTot vectors per group when
     * deformed, one otherwise -- and is rewound for every component.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Dispatch kernel.
        auto bwd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, 0.0);
        auto ipb_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 0.0);
        auto mass_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nmTot, 1.0, 0.0);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                simd_t::width * m_nqTot);

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
                        m_nmTot, (TData *)inptr);
                }

                if (m_isDeformed)
                {
                    // Step 1: BwdTrans
                    // Perform matrix-matrix multiply.
                    bwd_kernel(inptr, m_bwdmat, wspptr);

                    // Multiply by jacobian.
                    MultiplyByJacobian<ExecSpace, true>(
                        1, m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(wspptr),
                        reinterpret_cast<simd_t *>(wspptr), 1.0);

                    // Step 2: IProduct
                    // Perform matrix-matrix multiply.
                    ipb_kernel(wspptr, m_ipbmat, outptr);

                    // Increment pointers.
                    jacptr += m_nqTot * simd_t::width;
                }
                else
                {
                    // Perform matrix-matrix multiply.
                    mass_kernel(inptr, m_massmat, outptr);

                    // Multiply by jacobian.
                    MultiplyByJacobian<ExecSpace, false>(
                        1, m_nmTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr), 1.0);

                    // Increment pointers.
                    jacptr += simd_t::width;
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
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
