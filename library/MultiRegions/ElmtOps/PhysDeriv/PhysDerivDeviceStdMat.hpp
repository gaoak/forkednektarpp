///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceStdMat.hpp
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
// Description: Device standard-matrix implementation of the per-block
// physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivDeviceStdMat.hpp
 * @brief Device standard-matrix (StdMat) implementation of the
 * per-block physical derivative: one strided-batched BLAS GEMM per
 * standard direction, then a single-launch chain rule.
 *
 * @details
 * Included by the translation units CMake generates for the "StdMat"
 * implementation on the Device execution space.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through PhysDerivOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device StdMat block implementation of the physical derivative:
 * the whole block's standard derivatives as dense matrix products on
 * the back-end BLAS, followed by the geometric chain rule.
 *
 * @details
 * With every element sharing the standard-region derivative matrices
 * \f$D_d\f$ (`dimension` consecutive nqTot x nqTot column-major arrays,
 * StdMatKey with ePhysDerivStdMat), the differentiation of the whole
 * block is one strided-batched GEMM per standard direction @em d
 * (NekBlas::GemmStridedBatched), whose batch index is the input
 * component @em c:
 *
 *     out_{c,d} (nqTot x nelmt*nhomo) = D_d * in_c (nqTot x nelmt*nhomo),
 *
 * where the columns are the elements times the homogeneous modes -- one
 * column per (element, homogeneous mode) pair, padding elements
 * included, whose results are ignored -- `in_c` is component @em c of
 * the input (batch stride one input component) and `out_{c,d}` is
 * output plane @em d of component @em c (batch stride `outDim` output
 * planes, `outDim` being `coordDim`, or three for a 3DH1 field with
 * more than one homogeneous plane). The same matrix serves every
 * component, so its batch stride is zero. MultiplyByDerivFactorKernel
 * then overwrites planes 0 .. coordDim-1 of every component in place
 * with their contraction against the block's derivative factors.
 * The matrices come from StdExpansion::PhysDeriv, so the
 * collapsed-coordinate chain rule of the simplex shapes is already
 * contained in them.
 *
 * The GEMMs need each element's data contiguous, so the implementation
 * works at interleave width 1 (#m_implInterleaveWidth); the derivative
 * factors are fetched transposed to match, each factor forming a
 * contiguous plane per element -- of nqTot values when the geometry is
 * deformed, of one value when it is regular.
 *
 * Each block operator owns device stream `block_idx + 1`, so different
 * blocks' derivatives may overlap on the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::StdMat here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see PhysDerivDeviceSumFac.hpp for the sum-factorised device paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the stacked standard-region derivative matrices and
     * the block's derivative factors, and bind the block's device
     * stream.
     *
     * The matrices are keyed on the block's basis keys and shape
     * (ePhysDerivStdMat, one nqTot x nqTot block per standard
     * direction). The derivative factors are fetched in transposed
     * form at interleave width 1: per element the
     * `coordDim * dimension` factors follow one another, each carrying
     * one value per quadrature point for a deformed block and a single
     * value for a regular one.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysDerivBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysDerivBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            PhysDerivBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// The GEMMs require contiguous per-element columns, i.e.
    /// non-interleaved storage; the derivative factors are fetched at
    /// this width too.
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed: selects the DEFORMED
    /// instantiation of MultiplyByDerivFactorKernel, i.e. one set of
    /// derivative factors per quadrature point rather than per element.
    bool m_isDeformed;
    /// Dimension of the reference element (1, 2 or 3); also the number
    /// of strided-batched GEMMs issued per application.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of Cartesian derivative components produced.
    unsigned int m_coordDim;
    /// Coefficients per element; unused here (this operator never
    /// enters coefficient space).
    unsigned int m_nmTot;
    /// Quadrature points per element.
    unsigned int m_nqTot;
    /// The `dimension` standard-region derivative matrices (each
    /// nqTot x nqTot, column-major, stored back to back) in device
    /// memory, cached in the data warehouse.
    const TData *m_matptr;
    /// Derivative factors of the block in transposed layout: per
    /// element, `coordDim * dimension` planes of one value (regular) or
    /// of nqTot values (deformed).
    const TData *m_dfptr;

    /**
     * @brief Differentiate the block with `dimension` strided-batched
     * GEMMs, then apply the chain rule in one kernel launch.
     *
     * The whole input block -- every component and homogeneous mode --
     * is reshaped to interleave width 1 in one call (a no-op for
     * non-interleaved fields), the `dimension` strided-batched GEMMs
     * described in the class notes are issued on this block's stream,
     * each covering all components at once with the elements and
     * homogeneous modes of a component as its columns, and
     * MultiplyByDerivFactorKernel rewrites planes 0 .. coordDim-1 of
     * every component in place in one launch, the elements' points on
     * the first grid dimension, the homogeneous modes on the second and
     * the components on the third. Input and output are then reshaped
     * back to the input's width, the output being `outDim` times larger
     * since each component owns that many planes, and the output
     * block's interleave width is set to the input's.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block, holding
     *                      coordDim (or, for 3DH1, three) components
     *                      per input component; overwritten.
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

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
        const unsigned int gridSize =
            (nelmt * m_nqTot + blockSize - 1u) / blockSize;

        // Offsets between the components of a block. In 3DH1 (nhomo > 1) the
        // output has 3 slots per component (x, y, z) regardless of base mesh
        // coordDim, leaving slot 2 free for the z-derivative written by
        // DerivZOpImpl.
        const auto outDim    = (nhomo > 1) ? 3u : m_coordDim;
        const auto inoffset  = inblock.CompSize() * nhomo;
        const auto outoffset = outblock.CompSize() * nhomo;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            NekBlas::GemmStridedBatched(
                handle, "N", "N", m_nqTot, nelmtTot, m_nqTot, (TData)1.0,
                m_matptr + d * m_nqTot * m_nqTot, m_nqTot, 0, inptr, m_nqTot,
                inoffset, (TData)0.0, outptr + d * outoffset, m_nqTot,
                outDim * outoffset, inblock.GetNumComponents());
        }

        // Multiply by derivative factor, the planes and components on the
        // second and third grid dimensions. The directions of a component are
        // held in outDim slots of the output block.
        if (m_isDeformed)
        {
            DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (MultiplyByDerivFactorKernel<true>), gridSize, nhomo,
                inblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                m_nqTot, m_coordDim, m_dimension, outDim, nelmt, outoffset,
                outoffset, m_dfptr, outptr, outptr);
        }
        else
        {
            DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(
                (MultiplyByDerivFactorKernel<false>), gridSize, nhomo,
                inblock.GetNumComponents(), blockSize, 1, 1, m_streamID,
                m_nqTot, m_coordDim, m_dimension, outDim, nelmt, outoffset,
                outoffset, m_dfptr, outptr, outptr);
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp * outDim,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
