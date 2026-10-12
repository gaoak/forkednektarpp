///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXStdMat.hpp
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
// the per-block inner product with the gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseSerialAVXStdMat.hpp
 * @brief Serial/AVX standard-matrix (StdMat) implementation of the
 * per-block inner product with the gradient of the basis.
 *
 * @details
 * All elements of a block share the same basis, so one dense
 * standard-region matrix per reference direction serves the whole block:
 * a pointwise kernel first contracts the input's coordDim components
 * against the derivative factors and applies the Jacobian (and, for a Phys
 * output, the quadrature weights), see
 * IProductWRTDerivBaseSerialAVXStdMatKernels.hpp; one libxsmm GEMM per
 * direction and element group then contracts the result against that
 * direction's matrix. On the AVX execution space each SIMD lane owns one
 * element of a group, so the GEMM's left operand is the lane-interleaved
 * batch of element data; on Serial the SIMD type is a scalar and the group
 * size is one.
 *
 * This header provides the definition of
 * detail::IProductWRTDerivBaseBlockOpImpl for the Serial and AVX execution
 * spaces with the StdMat implementation tag. The sum-factorised Serial/AVX
 * definition lives in IProductWRTDerivBaseSerialAVXSumFac.hpp and the
 * Device definitions in IProductWRTDerivBaseDeviceStdMat.hpp and
 * IProductWRTDerivBaseDeviceSumFac.hpp; the class template name is shared,
 * and each CMake-generated registration translation unit includes exactly
 * one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTDerivBaseOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX StdMat inner product with the gradient of the basis:
 * a geometric kernel followed by one dense matrix multiplication per
 * reference direction, one element per SIMD lane.
 *
 * @details
 * The work is done in two stages per group of simd_t::width elements.
 *
 * -# JacobianDerivFactorKernel or JacobianDerivFactorWeightsKernel
 *    (IProductWRTDerivBaseSerialAVXStdMatKernels.hpp) contracts the
 *    input's coordDim components against the block's derivative factors
 *    and multiplies by the Jacobian, producing one physical-space field
 *    \f$g_d\f$ per reference direction in the static workspace (see
 *    IProductWRTDerivBaseOp.hpp for the definition of \f$g_d\f$). The
 *    scale factor #m_scale is folded in here, the GEMM kernels being
 *    dispatched with alpha = 1.
 * -# One libxsmm GEMM per reference direction contracts \f$g_d\f$ against
 *    that direction's standard-region matrix, all directions landing on
 *    the same output:
 *
 *        out (width x outTot) = g_d (width x nqTot) * M_d + beta * out,
 *
 *    with beta = 1 for every direction after the first, and for the first
 *    as well in append mode, so that outside append mode the first
 *    direction overwrites. The panels are the group's interleaved storage
 *    -- lane j of a column holds element j's value -- so the vectorisation
 *    is across the elements of the group.
 *
 * The matrices \f$M_d\f$ come from the data warehouse and differ with the
 * output state:
 * - Coeff: eIProductWRTDerivBaseStdMatTranspose, the transpose of the
 *   standard-region IProductWRTDerivBase matrix (nqTot x nmTot per
 *   direction). It already contains the standard region's quadrature
 *   weights, which is why the first stage applies only the derivative
 *   factors and the Jacobian.
 * - Phys: ePhysDerivStdMat, the standard-region derivative matrix
 *   (nqTot x nqTot per direction), in the orientation this GEMM needs for
 *   the transposed derivative of the weak form. The weights are not in
 *   this matrix, so the first stage applies them explicitly from the
 *   block's quadrature weights #m_weights.
 *
 * The collapsed-coordinate corrections of the simplex shapes need no
 * special treatment: they are built into the standard-region matrices.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TFieldOut      Output field state (Coeff or Phys).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata and fetch the
     * standard-region matrices, Jacobians and derivative factors (see the
     * class description).
     *
     * The matrix key depends on the output state: for a Coeff output the
     * transposed IProductWRTDerivBase matrix (keyed with the nodal points
     * type for a nodal expansion) and outTot = ncoeffs; for a Phys output
     * the derivative matrix, outTot = nqTot, and in addition the block's
     * quadrature weights.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    IProductWRTDerivBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TFieldOut, TData>(block_idx, exp,
                                                        dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        if constexpr (TFieldOut == FieldState::Coeff)
        {
            LibUtilities::PointsType nodalType =
                (exp->IsNodalNonTensorialExp())
                    ? exp->GetNodalPointsKey().GetPointsType()
                    : LibUtilities::eNoPointsType;

            m_outTot = exp->GetNcoeffs();
            m_matptr = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::eIProductWRTDerivBaseStdMatTranspose,
                    nodalType));
        }
        else
        {
            m_outTot = exp->GetTotPoints();
            m_matptr = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::ePhysDerivStdMat));

            m_weights = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::WeightsKey<TData>(block_idx,
                                                m_implInterleaveWidth));
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
    }

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<ElmtBlockOp<FieldState::Phys, TFieldOut, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<IProductWRTDerivBaseBlockOpImpl<
            ExecSpace, Implementation, TFieldOut, TData>>(block_idx, exp,
                                                          dataWarehouse);
    }

protected:
    /// Element interleave width the kernels and GEMMs operate at: the SIMD
    /// width (1 on the Serial execution space).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobian and derivative
    /// factors); selects the DEFORMED branch of the kernels and their
    /// stride.
    bool m_isDeformed;
    /// Reference (shape) dimension of the elements (1, 2 or 3): the number
    /// of reference directions summed over.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements: the number
    /// of input components consumed per output component.
    unsigned int m_coordDim;
    /// Output values per element: coefficients for a Coeff output,
    /// quadrature points for a Phys output.
    unsigned int m_outTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Per-direction standard-region matrices, laid out one after another
    /// (direction d starts at m_matptr + d * nqTot * outTot); which matrix
    /// this is depends on the output state, see the class description.
    const TData *m_matptr;
    /// Jacobians of the block, interleaved at #m_implInterleaveWidth: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;
    /// Derivative factors of the block, interleaved at
    /// #m_implInterleaveWidth and fetched untransposed: the
    /// coordDim * dimension factors of a point are stored together, indexed
    /// k * dimension + d, point by point on a deformed block and once per
    /// element otherwise (see DerivFactorKey).
    const TData *m_dfptr;
    /// Quadrature weights of the element, interleaved at
    /// #m_implInterleaveWidth; fetched and used for a Phys output only (for
    /// a Coeff output the weights are inside the matrix).
    const TData *m_weights;

    /**
     * @brief Apply the operator to every element of the block by dense
     * matrix multiplication.
     *
     * Per output component and homogeneous mode, and per group of
     * simd_t::width elements: the input's coordDim components and, in
     * append mode, the output are reshaped to #m_implInterleaveWidth, the
     * geometric kernel fills the workspace with the dimension fields
     * \f$g_d\f$, the GEMMs accumulate them onto the output, and the storage
     * is reshaped back. Fields stored at a larger interleave width are
     * handled in chunks of chunkSize = max(simd width, input width)
     * elements, reshaped once per width_ratio groups.
     *
     * The two GEMM kernels differ only in beta: the first direction
     * overwrites the output (or accumulates when append mode is set), the
     * remaining directions always accumulate.
     *
     * After the element loop of the last homogeneous mode of an output
     * component the input pointer skips the remaining input components of
     * that output component (three in all on a multi-plane 3DH1 expansion,
     * coordDim otherwise), so that the next component starts at its own
     * direction-0 data. The output block's recorded interleave width is set to
     * the input's on return.
     *
     * @param   inblock     Physical-space input block holding
     *                      \f${\bf f}\f$.
     * @param   outblock    Output block in state @p TFieldOut;
     *                      overwritten, or accumulated onto in append mode.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = this->m_append
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * simd_t::width * m_nqTot);

        // Dispatch kernel.
        auto gemm_kernel0 = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_outTot, m_nqTot, 1.0, (TData)this->m_append);
        auto gemm_kernel1 = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_outTot, m_nqTot, 1.0, 1.0);

        // Directions held per variable in the input block. In 3DH1
        // (nhomo > 1) there are always three (x, y, z) regardless of the base
        // mesh coordDim, the z slot being the one DerivZOp reads; the xy pass
        // still consumes only m_coordDim of them.
        const auto inDim = (inblock.GetNumHomoModes() > 1) ? 3u : m_coordDim;

        // Offsets between the components of a block.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto inoffset_vec = inoffset / simd_t::width;

        // Loop over components.
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            m_nqTot, (TData *)inptr + k * inoffset);
                    }

                    if (this->m_append)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, m_outTot, (TData *)outptr);
                    }
                }

                if constexpr (TFieldOut == FieldState::Coeff)
                {
                    // Multiply by derivative factor and Jacobian.
                    if (m_isDeformed)
                    {
                        JacobianDerivFactorKernel<ExecSpace, true>(
                            m_nqTot, m_coordDim, m_dimension, 1, inoffset_vec,
                            m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(wspptr),
                            (const simd_t)this->m_scale);
                        jacptr += m_nqTot * simd_t::width;
                        dfptr +=
                            m_coordDim * m_dimension * m_nqTot * simd_t::width;
                    }
                    else
                    {
                        JacobianDerivFactorKernel<ExecSpace, false>(
                            m_nqTot, m_coordDim, m_dimension, 1, inoffset_vec,
                            m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(wspptr),
                            (const simd_t)this->m_scale);
                        jacptr += simd_t::width;
                        dfptr += m_coordDim * m_dimension * simd_t::width;
                    }
                }
                else
                {
                    if (m_isDeformed)
                    {
                        JacobianDerivFactorWeightsKernel<ExecSpace, true>(
                            m_nqTot, m_coordDim, m_dimension, 1, inoffset_vec,
                            m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(m_weights),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(wspptr),
                            (const simd_t)this->m_scale);
                        jacptr += m_nqTot * simd_t::width;
                        dfptr +=
                            m_coordDim * m_dimension * m_nqTot * simd_t::width;
                    }
                    else
                    {
                        JacobianDerivFactorWeightsKernel<ExecSpace, false>(
                            m_nqTot, m_coordDim, m_dimension, 1, inoffset_vec,
                            m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(m_weights),
                            reinterpret_cast<const simd_t *>(inptr),
                            reinterpret_cast<simd_t *>(wspptr),
                            (const simd_t)this->m_scale);
                        jacptr += simd_t::width;
                        dfptr += m_coordDim * m_dimension * simd_t::width;
                    }
                }

                // Perform matrix-matrix multiply.
                gemm_kernel0(wspptr, m_matptr, outptr);
                for (unsigned int d = 1; d < m_dimension; d++)
                {
                    gemm_kernel1(wspptr + d * simd_t::width * m_nqTot,
                                 m_matptr + d * m_nqTot * m_outTot, outptr);
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nqTot,
                            (TData *)inptr + k * inoffset -
                                (width_ratio - 1) * m_nqTot * simd_t::width);
                    }
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_outTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_outTot * simd_t::width);
                }

                // Increment pointers.
                inptr += m_nqTot * simd_t::width;
                outptr += m_outTot * simd_t::width;
            }

            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (inDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
