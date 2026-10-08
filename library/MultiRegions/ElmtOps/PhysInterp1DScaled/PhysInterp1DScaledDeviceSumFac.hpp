///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledDeviceSumFac.hpp
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
 * @file PhysInterp1DScaledDeviceSumFac.hpp
 * @brief Device sum-factorised implementations of the per-block scaled
 * physical-space interpolation, serving both the SumFac and the SumFacTOP
 * strategy.
 *
 * @details
 * This one header is included by the registration translation units
 * CMake generates for either sum-factorisation strategy on the Device
 * execution space; the Implementation tag selects between the two kernel
 * families at compile time, when the enable_if-constrained
 * BwdTransKernelLauncher overloads and the workspace and shared-memory
 * size helpers resolve:
 * - SumFac: one element per thread, the lanes of a warp processing
 *   warpSize elements in lock-step on warp-interleaved data, with the
 *   inter-stage intermediates in a global-memory workspace.
 * - SumFacTOP: one element per thread block, the block's threads indexed
 *   over the element's per-stage output entries, with the tables and
 *   intermediates staged in shared memory.
 *
 * As on the host, the kernels are the backward transform's -- an
 * interpolation matrix has the layout of a 1D basis table, so
 * interpolating point values is a backward transform whose modes are the
 * input points. They are reached through
 * PhysInterp1DScaledDeviceSumFacKernels.hpp and
 * PhysInterp1DScaledDeviceSumFacTOPKernels.hpp, which do nothing but
 * include the BwdTrans kernel headers. Only the tensor-product launchers
 * are instantiated -- for LibUtilities::Seg in 1D, LibUtilities::Quad in
 * 2D and LibUtilities::Hex in 3D -- whatever the block's shape, since the
 * physical points of the collapsed shapes form a tensor-product grid in
 * the collapsed coordinates; the generated size switch makes that choice
 * exactly as in the Serial/AVX implementation. Their modal machinery is
 * switched off accordingly: `isModified` is passed as false, no
 * nodal-to-modal matrix is fetched (the `nodToMod` argument is null) and
 * the two mode-index tables the 3D launchers take for the tetrahedra are
 * null too. The APPEND template argument is always false; this family
 * has no append mode.
 *
 * The shape dispatch of v_Apply() and the generated ShapeBlock()
 * specialisations mirror the Serial/AVX SumFac implementation -- the same
 * BlockOpShapeBlock.cpp.in template and PhysInterp1D switch generate them
 * for both -- so see PhysInterp1DScaledSerialAVXSumFac.hpp for that
 * description.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through PhysInterp1DScaledOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSINTERP1D

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised block implementation of the scaled
 * interpolation (SumFac or SumFacTOP, selected by the Implementation
 * tag).
 *
 * @details
 * The constructor records only the block's shape and its input point
 * counts #m_nm; the per-direction interpolation matrices depend on the
 * scale factor and are fetched into device memory by v_SetScaleFactor().
 *
 * The two strategies want different data layouts, captured by
 * #m_implInterleaveWidth: SumFac reads warp-interleaved data (one element
 * per lane), SumFacTOP contiguous per-element data (width 1).
 * OperatorNDImpl() reshapes the block storage to that width and back
 * around its single launch, which covers every component and homogeneous
 * mode of the block. Each block operator owns device stream
 * `block_idx + 1`, so different blocks may overlap on the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see PhysInterp1DScaledDeviceStdMat.hpp for the dense-matrix device
 * path and PhysInterp1DScaledSerialAVXSumFac.hpp for the host
 * counterpart.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysInterp1DScaledBlockOpImpl : public PhysInterp1DScaledBlockOp<TData>
{
    using BlockOpBase = PhysInterp1DScaledBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream and capture
     * the block's element metadata; the interpolation matrices are
     * fetched by v_SetScaleFactor().
     *
     * Records the shape, geometry type, reference and coordinate
     * dimension, whether the first-direction basis is of eModified_A
     * type and the input point count of every direction, and sets both
     * mode-index slots of #m_index to null.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysInterp1DScaledBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysInterp1DScaledBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        m_index = {nullptr, nullptr};
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
            PhysInterp1DScaledBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Interleave width the kernels operate at: the warp size for SumFac
    /// (one element per lane), 1 for SumFacTOP (contiguous per-element
    /// data).
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, MultiRegions::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; unused here (the
    /// interpolation is geometry independent).
    bool m_isDeformed;
    /// First-direction basis is eModified_A; unused here -- the launches
    /// pass `isModified` as false, the operator acting on point values
    /// rather than modes.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; unused
    /// here.
    unsigned int m_coordDim;
    /// Input (unscaled) quadrature points per direction, passed to the
    /// kernels as their mode count.
    std::vector<unsigned int> m_nm;
    /// Output (scaled) quadrature points per direction; set by
    /// v_SetScaleFactor.
    std::vector<unsigned int> m_nq;
    /// 1D interpolation matrix per direction in device memory:
    /// `m_B[d][p * m_nq[d] + i]` is the interpolant of input point p of
    /// direction d evaluated at its output point i. Cached in the data
    /// warehouse; set by v_SetScaleFactor.
    std::vector<const TData *> m_B;
    /// The two mode-index table slots the 3D BwdTrans launchers take for
    /// the tetrahedra (see NumIndex()); always null here, the
    /// tensor-product kernels needing none.
    std::vector<const unsigned int *> m_index;

    /**
     * @brief Check that a scale factor has been set and dispatch on the
     * block's shape to the generated ShapeBlock() specialisation.
     *
     * @param   inblock     Physical-space input block on the elements'
     *                      own quadrature grids.
     * @param   outblock    Physical-space output block on the scaled
     *                      grids; overwritten.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_scale != -1.0,
                 "Scale factor has not been initialised");

        switch (m_shapeType)
        {
            // Segment
            case LibUtilities::Seg:
            {
                ShapeBlock<LibUtilities::Seg>(inblock, outblock);
                break;
            }
            // Quadrilateral
            case LibUtilities::Quad:
            {
                ShapeBlock<LibUtilities::Quad>(inblock, outblock);
                break;
            }
            // Triangle
            case LibUtilities::Tri:
            {
                ShapeBlock<LibUtilities::Tri>(inblock, outblock);
                break;
            }
            // Nodal triangle
            case LibUtilities::NodalTri:
            {
                ShapeBlock<LibUtilities::NodalTri>(inblock, outblock);
                break;
            }
            // Hexahedron
            case LibUtilities::Hex:
            {
                ShapeBlock<LibUtilities::Hex>(inblock, outblock);
                break;
            }
            // Tetrahedron
            case LibUtilities::Tet:
            {
                ShapeBlock<LibUtilities::Tet>(inblock, outblock);
                break;
            }
            // Nodal tetrahedron
            case LibUtilities::NodalTet:
            {
                ShapeBlock<LibUtilities::NodalTet>(inblock, outblock);
                break;
            }
            // Pyramid
            case LibUtilities::Pyr:
            {
                ShapeBlock<LibUtilities::Pyr>(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                ShapeBlock<LibUtilities::Prism>(inblock, outblock);
                break;
            }
            // Nodal prism
            case LibUtilities::NodalPrism:
            {
                ShapeBlock<LibUtilities::NodalPrism>(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    /**
     * @brief Fix the output point counts for @p scale and fetch the
     * per-direction interpolation matrices into device memory.
     *
     * The point-count rule and the warehouse keys are those of the
     * Serial/AVX SumFac implementation -- direction 0 scaled and
     * truncated, a direction one point below direction 0 keeping that
     * offset, and one cached BasisDataKey(eInterp) entry per direction
     * and target size, the target points keeping the basis's own points
     * type. No workspace is sized here: the SumFac kernels take the
     * static per-stream buffer at launch time and the SumFacTOP kernels
     * stage everything in shared memory.
     *
     * The counts and matrices are cleared and rebuilt from the stored
     * input counts, so a block operator may be re-targeted at a new
     * scale factor by calling this again.
     *
     * @param   scale   Multiplier applied to the per-direction
     *                  quadrature-point counts.
     */
    void v_SetScaleFactor(const TData &scale) override
    {
        this->m_scale = scale;
        m_nq          = this->GetScaledNumPoints(m_nm, this->m_scale);
        m_B.clear();
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eInterp, m_nq[d])));
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of precomputed index arrays used by the kernels in dim dimensions.
    static constexpr unsigned int NumIndex(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 0 : 2;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysInterp1D.h.in.
        static_assert((DIM == 1 && IsSizeParameter1D_v<TSizeParameter>) ||
                          (DIM == 2 && IsSizeParameter2D_v<TSizeParameter>) ||
                          (DIM == 3 && IsSizeParameter3D_v<TSizeParameter>),
                      "OperatorND expects a size parameter matching the "
                      "dimension of the shape.");

        OperatorNDImpl<SHAPE_TYPE>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumIndex(DIM)>());
    }

    /**
     * @brief Worker for every dimension: interpolate all elements of the
     * block with a single fused kernel launch.
     *
     * @details
     * Sizes the launch first. The grid has two dimensions. Its x extent
     * covers the block's elements, padding included: with SumFac one
     * thread per element, warpSize threads per thread block and as many
     * thread blocks as that takes; with SumFacTOP one thread block per
     * element at a time, its thread count the element's input point
     * count rounded up to a warp multiple and capped at the default block
     * size, and an occupancy-based number of thread blocks that stride
     * over the remaining elements (GetDeviceBlockSize() and
     * GetDeviceGridSize()). Its y extent is ncomp, the component count
     * times the homogeneous mode count, so that one launch covers every
     * component and mode and each kernel recovers its component from its
     * block index. The dynamic shared memory request is
     * BwdTransSharedMemorySize for the shape and strategy at hand, in
     * bytes -- zero under SumFac; under SumFacTOP room for the
     * interpolation matrices, the element's input values and the
     * inter-stage intermediates -- and the global workspace request is
     * BwdTransWorkSpaceSize scaled by ncomp, taken from the static
     * per-stream buffer: under SumFac nm1 values per element in 2D and
     * nm1 * nm2 + nm2 in 3D for the stage intermediates, none in 1D and
     * none under SumFacTOP.
     *
     * The launch goes through DEVICE_2DGRID_KERNEL_LAUNCHER to the
     * BwdTransKernelLauncher overload of this dimension and strategy,
     * instantiated for the segment, quadrilateral or hexahedron (see the
     * file notes) with APPEND false, so the output is overwritten. The
     * index sequences expand to the null mode-index slots #m_index (3D
     * only) and the interpolation matrices #m_B, which the kernels read
     * as their basis tables; `isModified` is passed as false and the
     * nodal-to-modal matrix as null; the two fields and the workspace
     * follow.
     *
     * The storage for all ncomp components is reshaped to
     * #m_implInterleaveWidth in a single pass before the launch, and both
     * input and output are reshaped back to the input's width once the
     * kernel has been queued; the output block's recorded width is set to
     * the input's on return. Everything is issued on #m_streamID.
     *
     * @tparam SHAPE_TYPE      LibUtilities::Seg, LibUtilities::Quad or
     *                         LibUtilities::Hex, chosen by the generated
     *                         switch from the block's dimension.
     * @tparam TSizeParameter  Size parameter of this dimension, templated
     *                         or not; its mode counts are the input point
     *                         counts and its quadrature counts the output
     *                         point counts.
     * @tparam ind0            Reference directions, 0 to ndim - 1;
     *                         selects the matrices #m_B.
     * @tparam ind1            Mode-index slots #m_index handed to the
     *                         launcher: none in 1D and 2D, two in 3D (see
     *                         NumIndex()).
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten.
     * @param   sizeParam   Element sizes, in runtime or compile-time
     *                      form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TSizeParameter,
              unsigned int... ind0, unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const size_t wspSize =
            BwdTransWorkSpaceSize<SHAPE_TYPE, Implementation>(nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Neither side of this operator is modal, so there is no nodal to
        // modal transform to apply and no collapsed coordinate correction
        // to make.
        const TData *nodToMod = nullptr;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            BwdTransSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // BwdTrans kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (BwdTransKernelLauncher<SHAPE_TYPE, Implementation, false>),
            gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
            nelmt, false, m_index[ind1]..., m_B[ind0]..., nodToMod, inptr,
            outptr, wspptr);

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
