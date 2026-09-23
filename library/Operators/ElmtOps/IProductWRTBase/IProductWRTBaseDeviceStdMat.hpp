///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceStdMat.hpp
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
// per-block inner product with the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the per-block
 * inner product with the basis.
 *
 * @details
 * The device counterpart of IProductWRTBaseSerialAVXStdMat.hpp: the same
 * two stages -- a Jacobian kernel, when the quadrature metric is in use,
 * followed by one dense multiply with a standard-region matrix -- but with
 * the Jacobian kernel as one device launch over every point, element,
 * component and homogeneous mode of the block, and the multiply as a
 * single strided-batched GEMM covering every component at once instead of
 * a GEMM per element group.
 *
 * This header provides the definition of detail::IProductWRTBaseBlockOpImpl
 * for the Device execution space with the StdMat implementation tag; the
 * sum-factorised Device definition lives in IProductWRTBaseDeviceSumFac.hpp,
 * and each CMake-generated registration translation unit includes exactly
 * one implementation header.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTBaseOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief Device StdMat inner product with the basis: the whole block's
 * inner product as one strided-batched dense matrix product on the
 * back-end BLAS, preceded by a Jacobian kernel when the quadrature metric
 * is in use.
 *
 * @details
 * Every element shares the standard-region matrices, so the block's inner
 * product is one GEMM per component,
 *
 *        out (nmTot x nelmtTot) = alpha * M (nmTot x nqTot)
 *                                       * in (nqTot x nelmtTot),
 *
 * issued for all components at once as a strided-batched GEMM with one
 * batch entry per component. The columns of in and out are the elements
 * (padding elements included, their results being ignored) and the
 * homogeneous modes, nelmtTot = padded elements * homogeneous modes; this
 * is why the implementation works at interleave width 1
 * (#m_implInterleaveWidth) and reshapes the block storage to that width
 * and back around the call. M is one of the two matrices the constructor
 * fetches from the data warehouse:
 * - #m_matptr, the StdMatKey with eIProductWRTBaseStdMat: an nmTot x nqTot
 *   column-major array whose (n, q) entry is \f$w_q\,\phi_n(\xi_q)\f$, the
 *   basis transpose with the standard-region quadrature weights folded in.
 *   Used when the quadrature metric is on, applied with alpha = 1 to the
 *   values MultiplyByJacobian has written to the static per-stream
 *   workspace, scale * J * u for every point of every component;
 * - #m_BT_matptr, the StdMatKey with eBwdTransStdMatTranspose:
 *   \f$B^{\mathsf T}\f$ (nmTot x nqTot column-major), the unweighted basis
 *   transpose, applied straight to the input when the metric is off, with
 *   the scale factor as alpha.
 *
 * The Serial/AVX StdMat implementation fetches the same two matrices in
 * the opposite orientation (eIProductWRTBaseStdMatTranspose and
 * eBwdTransStdMat), its GEMM having the element panel on the left; both
 * realise the same operator. Each block operator owns device stream
 * block_idx + 1, so different blocks may overlap on the device.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TData          Floating-point type of the field data.
 *
 * @see IProductWRTBaseDeviceSumFac.hpp for the sum-factorised device paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the two standard-region matrices and the
     * block's Jacobians (see the class description).
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
    IProductWRTBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
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
                                         StdRegions::eIProductWRTBaseStdMat,
                                         nodalType));

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));

        m_BT_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eBwdTransStdMatTranspose,
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
    /// Element interleave width the kernel and GEMM operate at: contiguous
    /// per-element data, since the elements are the GEMM's columns.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream all work of this block operator is issued on, one
    /// per block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobians); selects the
    /// DEFORMED branch of the Jacobian kernel.
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
    /// Weighted inner-product matrix (nmTot x nqTot, column-major) in
    /// device memory; used when the quadrature metric is on.
    const TData *m_matptr;
    /// Jacobians of the block in device memory, at interleave width 1: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;
    /// Transposed backward-transform matrix (nmTot x nqTot, column-major),
    /// the unweighted basis transpose used when the quadrature metric is
    /// off.
    const TData *m_BT_matptr;

    /**
     * @brief Apply the operator to the whole block by strided-batched dense
     * matrix multiplication.
     *
     * The input -- all components and homogeneous modes at once -- is
     * de-interleaved to #m_implInterleaveWidth. With the quadrature metric
     * on, MultiplyByJacobian writes scale * J * u for every point of every
     * component and homogeneous mode into the static per-stream workspace,
     * and one strided-batched GEMM against #m_matptr, with alpha = 1 and
     * one batch entry per component, takes the workspace to the output;
     * with the metric off a single strided-batched GEMM against
     * #m_BT_matptr runs straight from the input, with the scale factor as
     * its alpha. In both the homogeneous modes are folded into the columns,
     * so there is no loop over components or modes here. Both fields are
     * interleaved back to the input's width, which the output block is
     * also set to, and everything is issued on #m_streamID.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
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
            (this->m_integration)
                ? BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                      nelmt * ncomp * m_nqTot, m_streamID)
                : nullptr;

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
        if (this->m_integration) // integration with weights
        {
            // Multiply by jacobian.
            if (m_isDeformed)
            {
                MultiplyByJacobian<ExecSpace, true>(nelmt, m_nqTot, ncomp,
                                                    m_jacptr, inptr, wspptr,
                                                    this->m_scale, m_streamID);
            }
            else
            {
                MultiplyByJacobian<ExecSpace, false>(nelmt, m_nqTot, ncomp,
                                                     m_jacptr, inptr, wspptr,
                                                     this->m_scale, m_streamID);
            }

            // Perform batched matrix-matrix multiply, one multiply per
            // component, with the homogeneous modes held in the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, (TData)1.0,
                m_matptr, m_nmTot, 0, wspptr, m_nqTot, wspoffset, (TData)0.0,
                outptr, m_nmTot, outoffset, inblock.GetNumComponents());
        }
        else
        {
            // Perform batched matrix-matrix multiply of B^T, one multiply
            // per component, with the homogeneous modes held in the columns.
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, this->m_scale,
                m_BT_matptr, m_nmTot, 0, inptr, m_nqTot, inoffset, (TData)0.0,
                outptr, m_nmTot, outoffset, inblock.GetNumComponents());
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
