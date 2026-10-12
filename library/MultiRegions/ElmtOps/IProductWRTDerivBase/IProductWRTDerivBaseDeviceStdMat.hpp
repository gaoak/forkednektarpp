///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseDeviceStdMat.hpp
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
// per-block inner product with the gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the per-block
 * inner product with the gradient of the basis.
 *
 * @details
 * Two stages -- a geometric kernel followed by one dense multiply per
 * reference direction -- with the geometric stage as one device kernel
 * launch over every point, element, homogeneous mode and output component
 * of the block, and the multiplies as one strided-batched GEMM per
 * direction covering every component at once.
 *
 * This header provides the definition of
 * detail::IProductWRTDerivBaseBlockOpImpl for the Device execution space
 * with the StdMat implementation tag; the sum-factorised Device definition
 * lives in IProductWRTDerivBaseDeviceSumFac.hpp, and each CMake-generated
 * registration translation unit includes exactly one implementation
 * header.
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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat inner product with the gradient of the basis: a
 * geometric kernel followed by one strided-batched dense matrix
 * multiplication per reference direction.
 *
 * @details
 * The block is processed in two stages, both issued on the block's
 * stream:
 *
 * -# JacobianDerivFactorKernel or JacobianDerivFactorWeightsKernel
 *    (IProductWRTDerivBaseDeviceStdMatKernels.hpp) contracts each output
 *    component's coordDim input components against the block's
 *    derivative factors and multiplies by the Jacobian, writing one
 *    physical-space field \f$g_d\f$ per reference direction into the
 *    workspace; a single launch covers every element and quadrature point
 *    on its first grid dimension and the homogeneous modes and output
 *    components on the second and third.
 * -# One strided-batched GEMM per reference direction, with one batch
 *    entry per output component, contracts \f$g_d\f$ against that
 *    direction's standard-region matrix,
 *
 *        out (outTot x nelmtTot) = alpha * M_d (outTot x nqTot)
 *                                        * g_d (nqTot x nelmtTot)
 *                                + beta * out,
 *
 *    with alpha = #m_scale and beta = 1 for every direction after the
 *    first (and for the first as well in append mode), so the directions
 *    accumulate. The elements and homogeneous modes of a component are
 *    the columns (nelmtTot = padded elements * homogeneous modes), which
 *    is why the data is de-interleaved (#m_implInterleaveWidth is 1)
 *    around the two stages.
 *
 * The matrices come from the data warehouse and differ with the output
 * state:
 * - Coeff: eIProductWRTDerivBaseStdMat (nmTot x nqTot per direction),
 *   which already contains the standard region's quadrature weights, so
 *   stage one applies only the derivative factors and the Jacobian.
 * - Phys: ePhysDerivStdMatTranspose, the transposed standard-region
 *   derivative matrix (nqTot x nqTot per direction) the weak form needs.
 *   The weights are not in it, so stage one applies them explicitly from
 *   #m_weights.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Implementation tag (StdMat).
 * @tparam TFieldOut      Output field state (Coeff or Phys).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the standard-region matrices, Jacobians
     * and derivative factors (see the class description).
     *
     * The matrix key depends on the output state: for a Coeff output the
     * IProductWRTDerivBase matrix (keyed with the nodal points type for a
     * nodal expansion) and outTot = ncoeffs; for a Phys output the
     * transposed derivative matrix, outTot = nqTot, and in addition the
     * block's quadrature weights. The derivative factors are fetched in
     * the transposed layout the kernels index.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
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
        m_streamID = block_idx + 1;

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

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        if constexpr (TFieldOut == FieldState::Coeff)
        {
            m_outTot = exp->GetNcoeffs();
            m_matptr = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::eIProductWRTDerivBaseStdMat, nodalType));
        }
        else
        {
            m_outTot = exp->GetTotPoints();
            m_matptr = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    StdRegions::ePhysDerivStdMatTranspose));

            m_weights = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::WeightsKey<TData>(block_idx,
                                                m_implInterleaveWidth));
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
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
    /// Element interleave width the kernels and GEMMs operate at:
    /// contiguous per-element data, since the elements are the GEMM's
    /// columns.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream all work of this block operator is issued on, one
    /// per block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobian and derivative
    /// factors); selects the DEFORMED branch of the geometric kernel.
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
    /// Per-direction standard-region matrices in device memory, laid out
    /// one after another (direction d starts at
    /// m_matptr + d * nqTot * outTot); which matrix this is depends on the
    /// output state, see the class description.
    const TData *m_matptr;
    /// Jacobians of the block: one value per element, or one per
    /// quadrature point on a deformed block.
    const TData *m_jacptr;
    /// Derivative factors of the block, fetched transposed: per element
    /// each of the coordDim * dimension factors occupies a contiguous run
    /// of nqTot values on a deformed block, a single value on a regular one
    /// (see DerivFactorKey).
    const TData *m_dfptr;
    /// Quadrature weights of the element; fetched and used for a Phys
    /// output only (for a Coeff output the weights are inside the matrix).
    const TData *m_weights;

    /**
     * @brief Apply the operator to the whole block by batched dense matrix
     * multiplication.
     *
     * The input -- all components, homogeneous modes and coordinate
     * directions at once, three directions per output component on a
     * multi-plane 3DH1 expansion -- and, in append mode, the output are
     * de-interleaved to #m_implInterleaveWidth. A single launch of the
     * geometric kernel then fills the workspace, reading each output
     * component's coordDim input components and writing its dimension
     * fields \f$g_d\f$ into the component's slot of every direction
     * plane. One strided-batched GEMM per direction accumulates
     * the planes onto the output, the batch running over the components
     * and the homogeneous modes folded into the columns, and both fields
     * are interleaved back. Everything is issued on #m_streamID, so
     * different blocks may overlap on the device.
     *
     * There is no loop over element groups or homogeneous modes here: the
     * modes are extra columns of the GEMM and extra work items of the
     * geometric kernel.
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
        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto ncomp    = outblock.GetNumComponents() * nhomo;
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * nhomo;

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * nelmt * ncomp * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Directions held per variable in the input block. In 3DH1
        // (nhomo > 1) there are always three (x, y, z) regardless of the base
        // mesh coordDim, the z slot being the one DerivZOp reads; the xy pass
        // still consumes only m_coordDim of them.
        const auto inDim = (nhomo > 1) ? 3u : m_coordDim;

        // Offsets between the components of a block. The metric of every
        // component is held at once, so the offset between two directions of
        // the workspace spans all of them.
        const auto inoffset  = inblock.CompSize() * nhomo;
        const auto outoffset = outblock.CompSize() * nhomo;
        const auto wspoffset = m_nqTot * nelmt * ncomp;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp * inDim,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);
        }

        // Multiply by derivative factor and Jacobian, the planes and
        // components on the second and third grid dimensions. The coordinate
        // directions of a component are held in inDim slots of the input
        // block.
        if constexpr (TFieldOut == FieldState::Coeff)
        {
            if (m_isDeformed)
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorKernel<true>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, inptr, wspptr);
            }
            else
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorKernel<false>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, inptr, wspptr);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorWeightsKernel<true>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, m_weights, inptr, wspptr);
            }
            else
            {
                DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                    (JacobianDerivFactorWeightsKernel<false>), gridSize, nhomo,
                    outblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                    m_nqTot, m_coordDim, m_dimension, nelmt, inoffset,
                    wspoffset, m_jacptr, m_dfptr, m_weights, inptr, wspptr);
            }
        }

        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            TData alpha = this->m_scale;
            TData beta  = (d != 0 || this->m_append);

            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_outTot, nelmtTot, m_nqTot, alpha,
                m_matptr + d * m_nqTot * m_outTot, m_outTot, 0,
                wspptr + d * wspoffset, m_nqTot, m_nqTot * nelmtTot, beta,
                outptr, m_outTot, outoffset, outblock.GetNumComponents());
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp * inDim,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
