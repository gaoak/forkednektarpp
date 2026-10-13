///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionBlockOpDeviceStdMat.hpp
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
// Description: Device StdMat implementation of the linearised
// advection-diffusion-reaction block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionBlockOpDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the per-block
 * linearised advection-diffusion-reaction operator: strided-batched BLAS
 * GEMMs around one pointwise device kernel.
 *
 * @details
 * The operator is a backward transform, reference derivatives, a metric
 * kernel, then the two inner products. Each matrix stage is issued as a
 * single strided-batched GEMM through NekBlas -- the vendor BLAS behind
 * it depends on the build -- covering every element, component and
 * homogeneous mode of the block at once, so the matrices are fetched
 * untransposed (eBwdTransStdMat and its siblings) and the block
 * data is read as column-major matrices instead of lane-interleaved: the
 * implementation interleave width is one. The metric kernel is a device
 * kernel launched once over all quadrature points of the block
 * (ApplyMetricKernel, in LinAdvDiffReactionBlockOpDeviceStdMatKernels.hpp), and
 * all device work is issued on the block's own stream, so different
 * blocks may overlap.
 *
 * This header provides the definition of
 * detail::LinAdvDiffReactionBlockOpImpl for the Device execution space
 * with the StdMat implementation tag; the sum-factorised Device
 * definitions live in LinAdvDiffReactionBlockOpDeviceSumFac.hpp. The class
 * template name is shared, and each CMake-generated registration
 * translation unit includes exactly one of these headers.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through LinAdvDiffReactionOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp>

#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOpDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat block implementation of the linearised
 * advection-diffusion-reaction operator: dense standard-region matrices
 * through the vendor BLAS, around a pointwise metric kernel.
 *
 * @details
 * Every element of the block shares the standard-region matrices, so
 * each stage of the operator is a strided-batched matrix product over
 * the whole block, the columns of the operands being the elements
 * (padding elements included, their results ignored). One v_Apply()
 * pass issues
 * -# `bwd = B * in`, the nqTot x nmTot backward-transform matrix against
 *    each component's nmTot x (elements times homogeneous modes) block
 *    of coefficients, one batch entry per component -- skipped when
 *    lambda is zero, in which case the metric kernel does not read it;
 * -# `deriv_d = D_d * in`, one batch entry per reference direction,
 *    against the block's data read as one nmTot x (elements times
 *    components times homogeneous modes) matrix, writing one plane per
 *    direction into the static workspace;
 * -# one ApplyMetricKernel launch over every quadrature point of every
 *    element, with the components and homogeneous modes on the second
 *    grid dimension, which applies the chain
 *    rule, the velocity, lambda, the diffusivity tensor and the
 *    Jacobian;
 * -# `out = IPB * bwd`, one batch entry per component, overwriting the
 *    output;
 * -# `out += IPD_d * deriv_d`, one GEMM per reference direction and one
 *    batch entry per component (beta = 1).
 *
 * The GEMMs need each element's data contiguous, so the implementation
 * works at interleave width 1 (#m_implInterleaveWidth) and reshapes the
 * block storage to that width and back around the sequence. The
 * derivative factors are fetched with the transpose flag set (the last
 * argument of DerivFactorKey): on deformed blocks that makes each factor
 * component contiguous over the element's quadrature points, which is
 * the layout the kernel reads; for regular geometry the warehouse
 * ignores the flag and stores one value per factor component per
 * element. The velocity registered by SetAdvVel() is reshaped to width
 * one the first time v_Apply() finds it at another width, and is then
 * read in place; the components share the velocity of their plane.
 * Each block operator owns device stream `block_idx + 1`.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see LinAdvDiffReactionBlockOpDeviceSumFac.hpp for the sum-factorised Device
 * paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class LinAdvDiffReactionBlockOpImpl : public LinAdvDiffReactionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the four standard-region matrices, the
     * Jacobians and the derivative factors of the block from the data
     * warehouse.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    LinAdvDiffReactionBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : LinAdvDiffReactionBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        m_bwdmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eBwdTransStdMat,
                                         nodalType));
        m_ipbmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eIProductWRTBaseStdMat,
                                         nodalType));
        m_derivmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eDerivStdMat, nodalType));
        m_ipdmat = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::eIProductWRTDerivBaseStdMat,
                nodalType));

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
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            LinAdvDiffReactionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// The GEMMs require contiguous per-element columns, i.e.
    /// non-interleaved storage.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; selects the
    /// ApplyMetricKernel instantiation and how the geometric data is
    /// indexed.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of advection-velocity components and the rank of the
    /// diffusivity tensor.
    unsigned int m_coordDim;
    /// Coefficients per element.
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// Backward-transform matrix (nqTot x nmTot, column-major) in
    /// device memory, cached in the data warehouse.
    const TData *m_bwdmat;
    /// Inner-product-with-basis matrix (nmTot x nqTot).
    const TData *m_ipbmat;
    /// Reference-derivative matrices, one nqTot x nmTot block per
    /// reference direction.
    const TData *m_derivmat;
    /// Inner-product-with-basis-gradient matrices, one nmTot x nqTot
    /// block per reference direction.
    const TData *m_ipdmat;
    /// Jacobians of the block's elements: one value per element when
    /// regular, one per quadrature point when deformed.
    const TData *m_jacptr;
    /// Derivative factors \f$\partial\xi_d/\partial x_k\f$ of the
    /// block's elements, not interleaved and fetched with the transpose
    /// flag set: for a deformed element the nqTot values of one factor
    /// are contiguous (the layout the device kernel reads), while a
    /// regular block ignores the flag and stores one value per factor
    /// component per element (see DerivFactorKey).
    const TData *m_dfptr;

    /**
     * @brief Apply the operator to the whole block with one
     * strided-batched GEMM per stage and one metric-kernel launch.
     *
     * The velocity is reshaped to width one if it is still held at
     * another width. The block storage is then reshaped to width one --
     * the input over all its components and homogeneous modes at once,
     * a no-op for non-interleaved fields -- the five stages described in
     * the class notes are issued on #m_streamID, and both the input and
     * the output storage are reshaped back to the input's width, which
     * the output block is also set to. The two quadrature-space
     * workspaces -- nqTot values per element, component and homogeneous
     * mode for the backward transform and #m_dimension times that for
     * the derivatives -- come from the static per-stream workspace shared
     * by all block operators. The batch strides are the component strides
     * of the blocks (CompSize() times the number of homogeneous modes)
     * on the coefficient side and nqTot times the number of elements and
     * homogeneous modes on the quadrature side; the derivative planes
     * lie nqTot times the number of elements, components and modes
     * apart. The metric kernel is launched with the device's default
     * block size over a two-dimensional grid, the first dimension covering
     * the quadrature points of every element and the second the components
     * and homogeneous modes; the velocity is passed once, with the stride
     * nqTot times the number of elements and homogeneous modes between its
     * components, and the kernel picks the plane.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block, overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>(
                    m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr, m_streamID);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

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
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto advVelPtr =
            this->m_advVel->template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // Get static workspace pointer.
        auto bwdptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                nelmt * ncomp * m_nqTot + m_dimension * nelmt * ncomp * m_nqTot,
                m_streamID);
        auto derivptr = bwdptr + nelmt * ncomp * m_nqTot;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block. The derivatives of every
        // component are held at once, so the offset between two directions
        // spans all of them, while the advection velocity is shared by the
        // variables and spans all of the planes.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        const auto adveloffset = m_nqTot * nelmtTot;
        const auto derivoffset = m_nqTot * nelmt * ncomp;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Step 1: BwdTrans
        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        if (this->m_lambda != 0.0)
        {
            NekBlas::GemmStridedBatched(handle, "N", "N", m_nqTot, nelmtTot,
                                        m_nmTot, (TData)1.0, m_bwdmat, m_nqTot,
                                        0, inptr, m_nmTot, inoffset, (TData)0.0,
                                        bwdptr, m_nqTot, m_nqTot * nelmtTot,
                                        inblock.GetNumComponents());
        }

        // Step 2: Deriv
        // Perform batched matrix-matrix multiply, one multiply per direction,
        // with the components and homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmt * ncomp, m_nmTot, (TData)1.0,
            m_derivmat, m_nqTot, m_nqTot * m_nmTot, inptr, m_nmTot, 0,
            (TData)0.0, derivptr, m_nqTot, derivoffset, m_dimension);

        // Step 3: Multiply by diffusion coefficient, derivative
        // factor and Jacobian and add advection.
        if (m_isDeformed)
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (ApplyMetricKernel<true>), gridSize, ncomp, blockSize, 1,
                m_streamID, m_nqTot, m_coordDim, m_dimension, nelmt,
                inblock.GetNumHomoModes(), derivoffset, derivoffset,
                adveloffset, diffCoeffPtr, m_jacptr, m_dfptr, advVelPtr,
                derivptr, derivptr, bwdptr, this->m_lambda);
        }
        else
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (ApplyMetricKernel<false>), gridSize, ncomp, blockSize, 1,
                m_streamID, m_nqTot, m_coordDim, m_dimension, nelmt,
                inblock.GetNumHomoModes(), derivoffset, derivoffset,
                adveloffset, diffCoeffPtr, m_jacptr, m_dfptr, advVelPtr,
                derivptr, derivptr, bwdptr, this->m_lambda);
        }

        // Step 4: IProduct
        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, (TData)1.0, m_ipbmat,
            m_nmTot, 0, bwdptr, m_nqTot, m_nqTot * nelmtTot, (TData)0.0, outptr,
            m_nmTot, outoffset, inblock.GetNumComponents());

        // Step 5: IProductWRTDerivBase
        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, (TData)1.0,
                m_ipdmat + d * m_nqTot * m_nmTot, m_nmTot, 0,
                derivptr + d * derivoffset, m_nqTot, m_nqTot * nelmtTot,
                (TData)1.0, outptr, m_nmTot, outoffset,
                inblock.GetNumComponents());
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

} // namespace Nektar::MultiRegions::detail
