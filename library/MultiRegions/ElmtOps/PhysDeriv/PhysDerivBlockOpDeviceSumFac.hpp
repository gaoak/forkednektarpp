///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivBlockOpDeviceSumFac.hpp
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
// Description: Device sum-factorised implementations of the
// per-block physical derivative.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysDerivBlockOpDeviceSumFac.hpp
 * @brief Device sum-factorised implementations of the per-block
 * physical derivative, serving both the SumFac and the SumFacTOP
 * strategy.
 *
 * @details
 * This one header is included by the translation units CMake generates
 * for either sum-factorisation strategy on the Device execution space;
 * the Implementation tag selects between the two kernel families at
 * compile time when the (enable_if-constrained) PhysDerivKernelLauncher
 * overloads resolve:
 * - SumFac (PhysDerivBlockOpDeviceSumFacKernels.hpp): one element per thread.
 *   The lanes of a warp process warpSize elements in lock-step on
 *   warp-interleaved data, each thread walking the element's points in
 *   turn.
 * - SumFacTOP (PhysDerivBlockOpDeviceSumFacTOPKernels.hpp): one element per
 *   thread block. The block's threads stride over the element's
 *   quadrature points, each owning the full chain of sums for every
 *   output point it visits; in two and three dimensions the element's
 *   input values are staged in shared memory first. This exposes
 *   parallelism within an element rather than across elements.
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in; see
 * PhysDerivOp.hpp for what the operator computes.
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
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp>

#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised block implementation of the physical
 * derivative (SumFac or SumFacTOP, selected by the Implementation tag).
 *
 * @details
 * The constructor caches the per-direction 1D differentiation matrices
 * and, for the two- and three-dimensional shapes, the
 * collapsed-coordinate factor tables, in device memory and as plain
 * scalars, one thread owning one element or one point.
 *
 * The two strategies want different data layouts, captured by
 * #m_implInterleaveWidth: SumFac reads warp-interleaved data (one
 * element per lane), SumFacTOP contiguous per-element data (width 1).
 * The derivative factors are fetched to match: untransposed for SumFac,
 * so that the `ndf` factors of a point are adjacent and warp
 * interleaved, and transposed for SumFacTOP, so that each factor forms
 * a contiguous plane over the element's points. OperatorNDImpl()
 * reshapes the block storage to #m_implInterleaveWidth and back around
 * its single launch, which covers every component and homogeneous mode
 * of the block. Each block operator owns device stream `block_idx + 1`,
 * so different blocks may overlap on the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see PhysDerivBlockOpDeviceStdMat.hpp for the dense-matrix device path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using BlockOpBase = PhysDerivBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Gather everything the kernels need from the block's
     * representative expansion and the data warehouse, in device
     * memory, and bind the block's device stream.
     *
     * Per reference direction this fetches the mode and point counts
     * and the 1D differentiation matrix (eDerivative) evaluated at the
     * direction's quadrature points. For 2D and 3D shapes it also
     * fetches the collapsed-coordinate factor tables consumed by the
     * PhysDeriv kernels for the simplex-type shapes: per direction,
     * eHalfMultOnePlusZero holds \f$(1+z_i)/2\f$ and
     * eTwoOverOneMinusZero holds \f$2/(1-z_i)\f$ at the quadrature
     * points. Finally it fetches the block's derivative factors in the
     * layout the selected kernels index them in -- per group of
     * #m_implInterleaveWidth elements, point-major for SumFac (all
     * factors of one quadrature point adjacent, lanes innermost) and,
     * with the transpose flag set, factor-major for SumFacTOP (each
     * factor's point values contiguous).
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
            PhysDerivBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    /// kernel instantiations, in which the derivative factors vary per
    /// quadrature point rather than per element.
    bool m_isDeformed;
    /// First-direction basis is eModified_A; recorded for uniformity
    /// with the other families but not used by this operator, whose
    /// kernels branch on the shape alone.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3); the number of
    /// tensorial derivatives taken per element.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of Cartesian derivative components produced.
    unsigned int m_coordDim;
    /// Modes per direction; recorded at construction but not used by
    /// this operator, which works purely in physical space.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per direction.
    std::vector<unsigned int> m_nq;
    /// 1D differentiation matrix per direction, in device memory.
    std::vector<const TData *> m_D;
    /// Collapsed-coordinate factor tables: in 2D `(f0, f1)`, in 3D
    /// `(f0, f1, f1m, f2)` as named in the kernels; empty in 1D.
    std::vector<const TData *> m_f;
    /// Derivative factors of the block, in the layout matching
    /// #m_implInterleaveWidth (untransposed and warp interleaved for
    /// SumFac, transposed and contiguous for SumFacTOP).
    const TData *m_dfptr;

    /// @brief Dispatch to the generated ShapeBlock() specialisation
    /// matching #m_shapeType. An unsupported shape leaves the output
    /// block unwritten and reports on std::cout.
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
     * @brief Worker for every dimension: one kernel launch
     * differentiating every component and homogeneous mode of the
     * block.
     *
     * @details
     * The launch configuration comes from the shared helpers: dynamic
     * shared memory from PhysDerivSharedMemorySize for the shape and
     * strategy at hand, in bytes (nothing for segments; for SumFac the
     * collapsed-coordinate tables of the simplex-type shapes and
     * nothing for quadrilaterals and hexahedra; for SumFacTOP one
     * element's nqTot input values), threads per block from
     * GetDeviceBlockSize (a warp for SumFac; for SumFacTOP the
     * element's point count rounded up to whole warps and capped at
     * the device default block size) and the grid's X extent from
     * GetDeviceGridSize (for SumFac enough thread blocks to cover the
     * block's elements, padding included; for SumFacTOP an
     * occupancy-based number of persistent thread blocks that stride
     * over the elements). The whole input storage --
     * `ncomp = GetNumComponents() * GetNumHomoModes()` slabs of
     * `nelmt` elements -- is reshaped to #m_implInterleaveWidth in one
     * call, and the PhysDerivKernelLauncher overload of this dimension
     * and strategy is launched on #m_streamID through
     * DEVICE_3DGRID_KERNEL_LAUNCHER with a three-dimensional grid whose
     * X axis covers the elements while the Y and Z extents are the
     * homogeneous-mode and component counts, so the launcher itself
     * recovers which slab a thread block belongs to from its block
     * index; the index sequences expand to the differentiation
     * matrices #m_D and factor tables #m_f it needs, and `outoffset`
     * (one output component including all its homogeneous modes) tells
     * it where the Cartesian planes of a slab start. Finally the input
     * and the output, the latter `outDim` times larger since each slab
     * owns that many Cartesian planes (`outDim` is `coordDim`, or
     * three for a 3DH1 field with more than one homogeneous plane),
     * are reshaped back in one call each and the output block's
     * interleave width is set to the input's.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by
     *                             point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the differentiation matrices
     *                             #m_D.
     * @tparam ind1                Collapsed-coordinate factor tables
     *                             #m_f the kernels use: none in 1D, two
     *                             in 2D, four in 3D (see NumFactor()).
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten.
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
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const unsigned int shmemsize =
            sizeof(TData) *
            PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        const auto outDim = (outblock.GetNumHomoModes() > 1) ? 3u : m_coordDim;
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Calculate derivative.
        DEVICE_3DGRID_KERNEL_LAUNCHER(
            (PhysDerivKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, inblock.GetNumHomoModes(), inblock.GetNumComponents(),
            blocksize, 1, 1, shmemsize, m_streamID, sizeParam, nelmt, outoffset,
            m_D[ind0]..., m_f[ind1]..., m_dfptr, inptr, outptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nelmt * ncomp * outDim,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
