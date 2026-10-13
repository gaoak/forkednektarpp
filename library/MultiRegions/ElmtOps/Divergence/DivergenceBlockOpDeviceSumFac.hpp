///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceBlockOpDeviceSumFac.hpp
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
 * @file DivergenceBlockOpDeviceSumFac.hpp
 * @brief Device sum-factorised implementations of the per-block
 * divergence, serving both the SumFac and the SumFacTOP strategy.
 *
 * @details
 * This one header is included by the translation units CMake
 * generates for either sum-factorisation strategy on the Device
 * execution space; the Implementation tag selects between the two
 * kernel families at compile time when the (enable_if-constrained)
 * DivergenceKernelLauncher overloads resolve:
 * - SumFac (DivergenceBlockOpDeviceSumFacKernels.hpp): one element per
 *   thread. The lanes of a warp process warpSize elements in
 *   lock-step on warp-interleaved data.
 * - SumFacTOP (DivergenceBlockOpDeviceSumFacTOPKernels.hpp): one element per
 *   thread block. The block's threads are indexed over the element's
 *   quadrature points, with the component being differentiated and
 *   the running sum staged in shared memory, exposing parallelism
 *   within an element rather than across elements.
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations CMake generates from BlockOpShapeBlock.cpp.in; see
 * DivergenceOp.hpp for what the operator computes.
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
#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOp.hpp>

#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOpDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised block divergence (SumFac or SumFacTOP,
 * selected by the Implementation tag): each component of the vector
 * field is differentiated in the reference directions and contracted
 * with its row of derivative factors, the results accumulating into
 * the single output component.
 *
 * @details
 * The constructor caches the per-direction 1D derivative matrices
 * and, for 2D and 3D blocks, the collapsed-coordinate factor tables --
 * all in device memory -- and fetches the block's derivative factors.
 * The per-component chaining that turns the directional derivatives
 * into the divergence lives inside the kernels: the DIR = 0 call
 * overwrites the output and the DIR = 1, 2 calls accumulate onto it,
 * so the output holds \f$\sum_c \partial u_c/\partial x_c\f$ (see
 * the DivergenceKernelLauncher overloads for SumFac and SumFacTOP). In
 * one dimension the single launch differentiates component 0 alone.
 *
 * The two strategies want different data layouts, captured by
 * #m_implInterleaveWidth: SumFac reads warp-interleaved data (one
 * element per lane) and the point-major derivative factors, SumFacTOP
 * contiguous per-element data (width 1) and the factor-major
 * ("transposed") ones. OperatorNDImpl() reshapes the block storage to
 * that width and back around each launch. Each block operator owns
 * device stream `block_idx + 1`, so different blocks may overlap on
 * the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see DivergenceBlockOpDeviceStdMat.hpp for the dense-matrix device path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using BlockOpBase = DivergenceBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Gather everything the kernels need from the block's
     * representative expansion and the data warehouse, in device
     * memory.
     *
     * Per reference direction this fetches the mode and point counts
     * and the 1D derivative matrix (eDerivative) evaluated at the
     * direction's quadrature points. For 2D and 3D shapes it also
     * fetches the collapsed-coordinate factor tables consumed by the
     * PhysDerivDir kernels for the simplex-type shapes: per
     * direction, eHalfMultOnePlusZero holds \f$(1+z_i)/2\f$ and
     * eTwoOverOneMinusZero holds \f$2/(1-z_i)\f$ at the quadrature
     * points. The tables are fetched for every 2D block (two of them)
     * and every 3D block (four of them) whatever the shape, since the
     * condition is on the element dimension alone. Finally it fetches
     * the block's derivative factors in the layout the selected
     * kernels index them in -- per group of
     * #m_implInterleaveWidth elements, point-major for SumFac (all
     * factors of one quadrature point adjacent, lanes innermost) and,
     * with the transpose flag set, factor-major for SumFacTOP (each
     * factor's point values contiguous).
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

        ASSERTL1(m_dimension == m_coordDim,
                 "Shape dimension and coordinate dimension are not the same.");

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Fetch basis data.
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        constexpr bool transpose =
            std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(
                block_idx, m_implInterleaveWidth, transpose));
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
    /// Interleave width the kernels operate at: the warp size for
    /// SumFac (one element per lane), 1 for SumFacTOP (contiguous
    /// per-element data).
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, MultiRegions::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; selects the DEFORMED
    /// kernel instantiations, in which the derivative factors vary
    /// per quadrature point rather than per element.
    bool m_isDeformed;
    /// Whether the first basis is of eModified_A type (collapsed-
    /// coordinate correction flag); unused by this implementation,
    /// which switches on the shape type instead.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the element; equal to #m_dimension
    /// here, as asserted in the constructor.
    unsigned int m_coordDim;
    /// Modes per direction; recorded at construction but not used by
    /// this operator, which works purely in physical space.
    std::vector<unsigned int> m_nm;
    /// Number of quadrature points per reference direction.
    std::vector<unsigned int> m_nq;
    /// 1D derivative matrix per reference direction (eDerivative), in
    /// device memory.
    std::vector<const TData *> m_D;
    /// Collapsed-coordinate factor tables for the simplex-type shapes
    /// (see the constructor notes); empty in 1D.
    std::vector<const TData *> m_f;
    /// Derivative factors of the whole block, in the layout matching
    /// the selected implementation (see the constructor notes).
    const TData *m_dfptr;

    /// @brief Dispatch to the generated ShapeBlock() specialisation
    /// matching #m_shapeType.
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
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

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of collapsed coordinate factors used by the kernels in dim
    // dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitch1DCoordsCode.h.in.
        static_assert(
            (DIM == 1 && IsPhysSizeParameter1D_v<TPhysSizeParameter>) ||
                (DIM == 2 && IsPhysSizeParameter2D_v<TPhysSizeParameter>) ||
                (DIM == 3 && IsPhysSizeParameter3D_v<TPhysSizeParameter>),
            "OperatorND expects a size parameter matching the dimension of the "
            "shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    /**
     * @brief Worker for every dimension: one kernel launch computing
     * the divergence of the block's first ndim components.
     *
     * @details
     * The launch configuration comes from the shared helpers: dynamic
     * shared memory from DivergenceSharedMemorySize for the shape and
     * strategy at hand, in bytes (nothing for segments; for SumFac the
     * collapsed-coordinate tables of the simplex-type shapes and
     * nothing for quadrilaterals and hexahedra; for SumFacTOP two
     * per-element workspaces of nqTot entries), threads per block from
     * GetDeviceBlockSize (a warp for SumFac; for SumFacTOP the
     * element's point count rounded up to whole warps and capped at
     * the device default block size) and the grid from
     * GetDeviceGridSize (for SumFac enough thread blocks to cover the
     * block's elements, padding included; for SumFacTOP an
     * occupancy-based number of persistent thread blocks that stride
     * over the elements). All input components and planes are
     * reshaped together to #m_implInterleaveWidth, in a single call,
     * the DivergenceKernelLauncher overload of this dimension and
     * strategy is launched on #m_streamID through
     * DEVICE_2DGRID_KERNEL_LAUNCHER -- the index sequences expanding to
     * the derivative matrices #m_D and factor tables #m_f it needs, the
     * component stride (one component across all its planes) telling
     * it where components 1 and 2 start, and the planes riding the
     * second grid dimension over the same derivative factors -- and
     * the inputs are reshaped back together the same way, the output
     * separately. On return the output block's interleave width is set
     * to the input's.
     *
     * @note In one dimension only component 0 is differentiated.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by
     *                             point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the derivative matrices #m_D.
     * @tparam ind1                Collapsed-coordinate factor tables
     *                             #m_f the kernels use: none in 1D, two
     *                             in 2D, four in 3D (see NumFactor()).
     *
     * @param   inblock     Physical-space input block with at least
     *                      ndim components.
     * @param   outblock    Physical-space output block receiving the
     *                      divergence of every plane in its first
     *                      component.
     * @param   sizeParam   Element sizes, in runtime or compile-time
     *                      form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        // Get block sizes.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto ncomp = inblock.GetNumComponents() * nhomo;
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            DivergenceSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto inoffset = outblock.CompSize() * nhomo;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, ncomp * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Calculate derivative du/dx. Each plane holds one xy divergence over
        // the same geometry, so they ride the second grid dimension of a
        // single launch and the kernel picks its plane from the block index.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (DivergenceKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, nhomo, blocksize, 1, shmemsize, m_streamID, sizeParam,
            nelmt, inoffset, m_D[ind0]..., m_f[ind1]..., m_dfptr, inptr,
            outptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, ncomp * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * nhomo,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
