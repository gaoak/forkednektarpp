///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDeviceSumFac.hpp
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
// Description: Device sum-factorised implementation of the advection block
// operator, SumFac and SumFacTOP.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionDeviceSumFac.hpp
 * @brief Device sum-factorised implementation of the advection block
 * operator, serving both the SumFac and the SumFacTOP strategy.
 *
 * @details
 * As in the Serial/AVX sum-factorised implementation
 * (AdvectionSerialAVXSumFac.hpp), the derivatives along the reference
 * tensor directions are formed from the one-dimensional derivative
 * matrices and then finished by applying the collapsed-to-standard
 * coordinate correction for simplex-type shapes, the geometric derivative
 * factors and the dot product with the advection velocity. On the device
 * both stages happen inside a single kernel per element.
 *
 * This one header is included by the translation units CMake generates for
 * either sum-factorisation strategy on the Device execution space; the
 * Implementation tag selects between the two kernel families at compile
 * time, when the enable_if-constrained AdvectionKernelLauncher overloads
 * resolve:
 * - SumFac (AdvectionDeviceSumFacKernels.hpp): one element per thread. The
 *   lanes of a warp advance through warpSize elements in lock-step on
 *   warp-interleaved data, which is what the Serial/AVX implementation
 *   does with SIMD lanes.
 * - SumFacTOP (AdvectionDeviceSumFacTOPKernels.hpp): one element per
 *   thread block. The block's threads are indexed over the element's
 *   quadrature points and its input is staged in shared memory, exposing
 *   parallelism within an element rather than across elements.
 *
 * The two strategies want different storage and derivative-factor
 * layouts; see the class description. The shape dispatch of v_Apply() and
 * the generated ShapeBlock() specialisations mirror the Serial/AVX SumFac
 * implementation -- the same BlockOpShapeBlock.cpp.in template generates
 * them for both -- so see AdvectionSerialAVXSumFac.hpp for that
 * description.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through AdvectionOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Advection/AdvectionBlockOp.hpp>

#include <MultiRegions/ElmtOps/Advection/AdvectionDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/Advection/AdvectionDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised advection block operator (SumFac or
 * SumFacTOP, selected by the Implementation tag).
 *
 * @details
 * The constructor collects, per reference direction, the element sizes and
 * one-dimensional derivative matrices, plus the pointwise chain-rule
 * factors (1 + eta)/2 and 2/(1 - eta) evaluated at the quadrature points
 * (data-warehouse keys eHalfMultOnePlusZero and eTwoOverOneMinusZero), and
 * the block's derivative factors. The factor tables are fetched for every
 * 2D block (two of them) and every 3D block (four of them) whatever the
 * shape, since the condition in the constructor is on the element
 * dimension alone; only the shapes with collapsed coordinates use them,
 * the kernels discarding the tables of quadrilaterals and hexahedra
 * through a compile-time shape test. See Advection2DSumFacKernel and
 * Advection3DSumFacKernel for which factor serves which direction. All of
 * this data lives in device memory.
 *
 * The strategy shows up in two layout choices:
 * - #m_implInterleaveWidth is the warp size for SumFac (one element per
 *   lane) and one for SumFacTOP (contiguous per-element data).
 *   OperatorNDImpl() reshapes the block storage to that width around each
 *   launch and back afterwards.
 * - The derivative factors are fetched untransposed for SumFac, which
 *   wants the factors of a point adjacent, and transposed for SumFacTOP,
 *   which wants each factor component contiguous over the element's
 *   points because its threads are indexed over points (the last argument
 *   of DerivFactorKey). The distinction only affects deformed blocks; for
 *   regular geometry the warehouse ignores the flag and stores one value
 *   per factor component per element.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that issues a single kernel launch per
 * block covering every element, component and homogeneous mode. Each block
 * operator owns device stream block_idx + 1, so different blocks may
 * overlap on the device.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData          Floating-point type of the field data.
 *
 * @see AdvectionDeviceStdMat.hpp for the dense-matrix Device path;
 * AdvectionSerialAVXSumFac.hpp for the host sum-factorised sibling.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionBlockOpImpl : public AdvectionBlockOp<TData>
{
    using BlockOpBase = AdvectionBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the 1D derivative matrices,
     * collapsed-coordinate factors and derivative factors (see the class
     * description).
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionBlockOp<TData>(block_idx, exp, dataWarehouse)
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

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            AdvectionBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the kernels expect: the warp size for
    /// SumFac (one element per lane), one for SumFacTOP.
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, MultiRegions::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    /// Device stream all work of this block operator is issued on, one
    /// per block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements; drives the dispatch in v_Apply().
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point derivative factors).
    bool m_isDeformed;
    /// Whether the direction-0 basis is of type eModified_A.
    bool m_isModified;
    /// Reference (shape) dimension of the elements.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements.
    unsigned int m_coordDim;
    /// Modes per reference direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per reference direction.
    std::vector<unsigned int> m_nq;
    /// One-dimensional derivative matrices, one per direction.
    std::vector<const TData *> m_D;
    /// Collapsed-coordinate chain-rule factors (see the class
    /// description); empty for 1D blocks.
    std::vector<const TData *> m_f;
    /// Derivative factors of the whole block, interleaved at
    /// #m_implInterleaveWidth and transposed for SumFacTOP; see
    /// DerivFactorKey.
    const TData *m_dfptr;

    /**
     * @brief Dispatch on the block's shape to the generated ShapeBlock()
     * specialisation. An unsupported shape type is reported on
     * std::cout and leaves the output block untouched.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated onto when append is set.
     */
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
     * @brief Worker for every dimension: advect all elements of the block
     * with a single kernel launch covering every component and
     * homogeneous mode.
     *
     * @details
     * Reshapes the advection velocity to #m_implInterleaveWidth if it is
     * still held at another width, then sizes the launch. The grid has
     * two dimensions. The first covers the block's elements, padding
     * included: with SumFac one thread per element, warpSize threads per
     * thread block and as many thread blocks as that takes; with
     * SumFacTOP one thread block per element at a time, its thread count
     * the element's number of quadrature points rounded up to a warp
     * multiple and capped at the default block size, and an
     * occupancy-based number of thread blocks that stride over the
     * remaining elements (GetDeviceBlockSize() and GetDeviceGridSize()).
     * The second dimension is ncomp = GetNumComponents() *
     * GetNumHomoModes(), one row per component or homogeneous mode. The
     * dynamic shared memory request is sizeof(TData) times
     * AdvectionSharedMemorySize for the shape and strategy at hand: the
     * collapsed-coordinate
     * factor tables for SumFac (nothing for segments, quadrilaterals and
     * hexahedra) and one element's input for SumFacTOP (nothing for
     * segments).
     *
     * The launch goes through DEVICE_2DGRID_KERNEL_LAUNCHER to the
     * AdvectionKernelLauncher overload of this dimension and strategy,
     * with the index sequences expanding to the derivative matrices #m_D
     * and factor tables #m_f the kernel needs. The velocity is passed once
     * together with the stride between its components,
     * nelmt * nqTot * nhomo, and the plane count nhomo: the kernel reads
     * the velocity of each homogeneous mode's own plane, and every
     * component is advected with the same velocity.
     *
     * Block storage held at another interleave width is reshaped in place
     * around the launch, in one pass over all nelmt * ncomp entries, and
     * not symmetrically: the input is converted to the implementation
     * width and back again, whereas the output is converted forward only
     * when it is accumulated onto (append set) -- there is nothing to
     * preserve otherwise -- and is in both cases converted from the
     * implementation width to the input's width once the kernel has been
     * queued. The output block's recorded width is set to the input's on
     * return. Everything is issued on #m_streamID.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the derivative matrices #m_D.
     * @tparam ind1                Collapsed-coordinate factor tables #m_f
     *                             the kernels use: none in 1D, two in 2D,
     *                             four in 3D (see NumFactor()).
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated onto when append is set.
     * @param   sizeParam   Element sizes, in runtime or compile-time form.
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

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        // Get block sizes.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Initialize advVel pointers. The advection velocity carries the same
        // planes as the input, so one of its components spans all of them.
        auto advVelPtr =
            this->m_advVel->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        const auto advVelOffset = nelmt * nqTot * nhomo;

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const unsigned int shmemsize =
            sizeof(TData) *
            AdvectionSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Calculate derivative.
        if (this->m_append)
        {
            // Reshape, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);

            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (AdvectionKernelLauncher<SHAPE_TYPE, Implementation, true,
                                         DEFORMED>),
                gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
                nelmt, m_D[ind0]..., m_f[ind1]..., m_dfptr, advVelPtr,
                advVelOffset, nhomo, inptr, outptr, this->m_scale);
        }
        else
        {
            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (AdvectionKernelLauncher<SHAPE_TYPE, Implementation, false,
                                         DEFORMED>),
                gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
                nelmt, m_D[ind0]..., m_f[ind1]..., m_dfptr, advVelPtr,
                advVelOffset, nhomo, inptr, outptr, this->m_scale);
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
