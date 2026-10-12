///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceDeviceStdMat.hpp
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
 * @file DivergenceDeviceStdMat.hpp
 * @brief Device standard-matrix implementation of the Divergence block
 * operator: dense standard-region derivative matrices applied through
 * the back-end BLAS.
 *
 * @details
 * Included by the translation unit CMake generates for the "StdMat"
 * implementation on the Device execution space. See DivergenceOp.hpp
 * for what the operator computes. The whole block -- every element,
 * plane and vector component -- is covered by a single
 * strided-batched GEMM through NekBlas (the vendor BLAS behind it
 * depends on the build), so the matrices are fetched untransposed
 * (ePhysDerivStdMat) and the block data is read as one column-major
 * matrix: the implementation interleave width is one.
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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat implementation of DivergenceBlockOp: one
 * strided-batched BLAS GEMM forming every reference-direction
 * derivative of every component of the block, followed by one
 * derivative-factor contraction per component accumulating into a
 * single output component.
 *
 * @details
 * ### What one application does
 * Every element of the block shares the standard-region derivative
 * matrices (StdMatKey with ePhysDerivStdMat: one `nqTot x nqTot`
 * matrix per reference direction, with any collapsed-coordinate
 * correction baked in), so one GEMM per reference direction
 * differentiates the whole block at once. The `dimension` products
 * are issued as a single NekBlas::GemmStridedBatched call whose batch
 * index is the reference direction d,
 *
 *     wsp_d (nqTot x dimension*nelmtTot)
 *         = D_d * in (nqTot x dimension*nelmtTot),
 *
 * with `nelmtTot = nelmt * nhomo`, the right-hand operand being the
 * block's first `dimension` components read as one matrix -- the
 * columns are the elements of every plane of component 0, then those
 * of component 1, and so on, padding elements included (their results
 * are ignored) -- with a batch stride of zero, so every derivative
 * matrix is applied to all components. Workspace slice d thus holds
 * \f$\partial u_c/\partial\xi_d\f$ for every component c, one
 * component run after another, and the static per-stream workspace is
 * sized for `dimension * dimension` such runs.
 * MultiplyByDerivDirFactorKernel (with the APPEND and DEFORMED flags)
 * is then launched once per component c: starting from that
 * component's run in slice 0 and stepping between the direction slices
 * with a stride of `dimension` runs, it contracts
 * \f$\partial u_c/\partial\xi_d\f$ over d with row c of the
 * derivative factors, overwriting the output for c = 0 and
 * accumulating (APPEND) for the remaining components, so the block's
 * single output component ends up holding
 * \f[ \sum_c \sum_d \frac{\partial\xi_d}{\partial x_c}
 *     \frac{\partial u_c}{\partial\xi_d}. \f]
 *
 * ### Layout and launch
 * The GEMM needs each element's data contiguous, so the
 * implementation works at interleave width 1
 * (#m_implInterleaveWidth) and reshapes the block storage -- all
 * input components and planes in one call, the output separately --
 * to that width and back around the call. Reading the components as
 * consecutive column blocks relies on a component's planes following
 * one another, so that consecutive components stay adjacent. The
 * contraction kernel runs the block's points on the first grid
 * dimension and the planes on the second, all planes sharing the
 * derivative factors, which are fetched in the factor-major
 * ("transposed") layout the kernel indexes. Each block operator owns
 * device stream `block_idx + 1`, so different blocks may overlap on
 * the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  Implementation tag the including
 *                         translation unit registers
 *                         (MultiRegions::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see DivergenceDeviceSumFac.hpp for the sum-factorised device
 * paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the stacked standard-region derivative matrices
     * and the block's derivative factors, and bind the block's device
     * stream.
     *
     * The matrices are keyed on the block's basis keys and shape
     * (ePhysDerivStdMat, one `nqTot x nqTot` block per reference
     * direction). The derivative factors are fetched in transposed
     * form at interleave width 1: per element the
     * `coordDim * dimension` factors follow one another, each
     * carrying one value per quadrature point for a deformed block
     * and a single value for a regular one.
     *
     * Asserts (debug builds) that the shape dimension equals the
     * coordinate dimension, i.e. embedded expansions are not
     * supported.
     *
     * @param   block_idx       Index of the block within the
     *                          expansion list's Collections; the
     *                          stream used for all device work is
     *                          block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::ePhysDerivStdMat));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
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
    /// Interleave width the GEMM path operates at: 1, i.e. contiguous
    /// per-element data.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
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
    /// Stacked standard-region derivative matrices, one
    /// `nqTot x nqTot` block per reference direction, in device
    /// memory.
    const TData *m_matptr;
    /// Derivative factors of the whole block, transposed layout (see
    /// the constructor notes).
    const TData *m_dfptr;

    /**
     * @brief Differentiate the components of the block with one
     * strided-batched GEMM and accumulate the contracted derivatives
     * into the output.
     *
     * Reshapes the input components to interleave width 1 in a
     * single call, issues the strided-batched GEMM into the static
     * per-stream workspace, contracts the workspace with the
     * derivative factors component by component, every plane in one
     * launch (deformed and regular geometries use separate kernel
     * instantiations, and the first component overwrites where the
     * others append), then reshapes inputs and output back and sets
     * the output block's interleave width to the input's. All work is
     * issued on this block operator's stream.
     *
     * @param   inblock     Physical-space input block; its first
     *                      shape-dimension components are the vector
     *                      field, each spanning all the planes.
     * @param   outblock    Physical-space output block receiving the
     *                      divergence of every plane in its first
     *                      component.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto ncomp    = inblock.GetNumComponents() * nhomo;
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * nhomo;

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer. The standard derivatives of every
        // component are held at once so that the chain rule is applied to a
        // contiguous workspace.
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_dimension * m_dimension * nelmtTot * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block. The derivatives of every
        // component are held at once, so the offset between two directions of
        // the workspace spans all of them, planes included: a component's
        // planes follow one another, so consecutive components stay adjacent
        // and the whole input is still one contiguous run of columns.
        const auto derivoffset = m_nqTot * nelmtTot;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Standard derivatives of all the components.
        // Perform batched matrix-matrix multiply, one multiply per direction,
        // with the components and their planes held in the columns.
        NekBlas::GemmStridedBatched(handle, "N", "N", m_nqTot,
                                    m_dimension * nelmtTot, m_nqTot, (TData)1.0,
                                    m_matptr, m_nqTot, m_nqTot * m_nqTot, inptr,
                                    m_nqTot, 0, (TData)0.0, derivptr, m_nqTot,
                                    m_dimension * derivoffset, m_dimension);

        // Multiply by derivative factor. Each component contributes one
        // direction of the divergence, so they are accumulated one at a time,
        // every plane within the one launch.
        for (unsigned int c = 0; c < m_dimension; c++)
        {
            if (m_isDeformed)
            {
                if (c == 0)
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (MultiplyByDerivDirFactorKernel<false, true>), gridSize,
                        nhomo, blockSize, 1, m_streamID, c, m_nqTot, m_coordDim,
                        m_dimension, nelmt, m_dimension * derivoffset, m_dfptr,
                        derivptr + c * derivoffset, outptr);
                }
                else
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (MultiplyByDerivDirFactorKernel<true, true>), gridSize,
                        nhomo, blockSize, 1, m_streamID, c, m_nqTot, m_coordDim,
                        m_dimension, nelmt, m_dimension * derivoffset, m_dfptr,
                        derivptr + c * derivoffset, outptr);
                }
            }
            else
            {
                if (c == 0)
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (MultiplyByDerivDirFactorKernel<false, false>),
                        gridSize, nhomo, blockSize, 1, m_streamID, c, m_nqTot,
                        m_coordDim, m_dimension, nelmt,
                        m_dimension * derivoffset, m_dfptr,
                        derivptr + c * derivoffset, outptr);
                }
                else
                {
                    DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(
                        (MultiplyByDerivDirFactorKernel<true, false>), gridSize,
                        nhomo, blockSize, 1, m_streamID, c, m_nqTot, m_coordDim,
                        m_dimension, nelmt, m_dimension * derivoffset, m_dfptr,
                        derivptr + c * derivoffset, outptr);
                }
            }
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * nhomo,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
