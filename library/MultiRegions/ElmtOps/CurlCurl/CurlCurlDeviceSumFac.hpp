///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceSumFac.hpp
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
 * @file CurlCurlDeviceSumFac.hpp
 * @brief Device sum-factorised implementation of the curl-curl block
 * operator, serving both the SumFac and the SumFacTOP strategy.
 *
 * @details
 * As in the Serial/AVX sum-factorised implementation
 * (CurlCurlSerialAVXSumFac.hpp), the derivatives along the reference
 * tensor directions are formed from the one-dimensional derivative
 * matrices and then finished by applying the collapsed-to-standard
 * coordinate correction for simplex-type shapes and the geometric
 * derivative factors and taking the curl; the vorticity omega = curl u so
 * obtained is differentiated the same way to give out = curl omega. On the
 * device both curls happen inside a single fused kernel launch per block,
 * the intermediate omega never leaving the kernel's workspace. A
 * multi-plane 3DH1 block instead gets the plane part of a single curl on
 * every plane, from Curl3DH1KernelLauncher with the planes on the second
 * grid dimension, which CurlCurlOpImpl completes with the z parts.
 *
 * This one header is included by the translation units CMake generates for
 * either sum-factorisation strategy on the Device execution space; the
 * Implementation tag selects between the two kernel families at compile
 * time, when the enable_if-constrained CurlCurlKernelLauncher overloads
 * resolve:
 * - SumFac (CurlCurlDeviceSumFacKernels.hpp): one element per thread. The
 *   lanes of a warp advance through warpSize elements in lock-step on
 *   warp-interleaved data, which is what the Serial/AVX implementation
 *   does with SIMD lanes; omega is held in a global workspace, each lane
 *   reading back what it wrote.
 * - SumFacTOP (CurlCurlDeviceSumFacTOPKernels.hpp): one element per
 *   thread block. The block's threads are indexed over the element's
 *   quadrature points and both the input components and omega are staged
 *   in shared memory, exposing parallelism within an element rather than
 *   across elements.
 *
 * The two strategies want different storage and derivative-factor
 * layouts; see the class description. The shape dispatch of v_Apply() and
 * the generated ShapeBlock() specialisations mirror the Serial/AVX SumFac
 * implementation -- the same BlockOpShapeBlock.cpp.in template generates
 * them for both -- so see CurlCurlSerialAVXSumFac.hpp for that
 * description.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through CurlCurlOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp>

#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised curl-curl block operator (SumFac or
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
 * through a compile-time shape test. See the kernel documentation for
 * which factor serves which direction. All of this data lives in device
 * memory.
 *
 * The strategy shows up in two layout choices:
 * - #m_implInterleaveWidth is the warp size for SumFac (one element per
 *   lane) and one for SumFacTOP (contiguous per-element data).
 *   OperatorNDImpl() reshapes the block storage to that width around the
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
 * block covering every element and the ndim components of the field, or
 * hands a multi-plane 3DH1 block to OperatorCurl3DH1(). Each block
 * operator owns device stream block_idx + 1, so different blocks may
 * overlap on the device.
 *
 * @note The segment path (ShapeBlock<Seg>, the 1D CurlCurlKernelLauncher)
 * evaluates only the first derivative of a single component, the
 * curl-curl being undefined in one dimension; it exists so that the
 * generated per-shape translation units compile. The constructor asserts
 * that the block is two- or three-dimensional, but only with ASSERTL1,
 * which is compiled out of non-debug builds.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData          Floating-point type of the field data.
 *
 * @see CurlCurlDeviceStdMat.hpp for the dense-matrix Device path;
 * CurlCurlSerialAVXSumFac.hpp for the host sum-factorised sibling.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using BlockOpBase = CurlCurlBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the 1D derivative matrices,
     * collapsed-coordinate factors and derivative factors (see the class
     * description).
     *
     * The curl-curl operator is only defined for two- and
     * three-dimensional expansions whose coordinate dimension equals the
     * shape dimension; both are asserted here.
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        ASSERTL1(m_coordDim == 2 || m_coordDim == 3,
                 "CurlCurl operator only defined for 2D and 3D.");

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
            CurlCurlBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    /// Reference (shape) dimension of the elements (2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the elements; asserted equal to
    /// #m_dimension.
    unsigned int m_coordDim;
    /// Modes per reference direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per reference direction.
    std::vector<unsigned int> m_nq;
    /// One-dimensional derivative matrices, one per direction.
    std::vector<const TData *> m_D;
    /// Collapsed-coordinate chain-rule factors (see the class
    /// description).
    std::vector<const TData *> m_f;
    /// Derivative factors of the whole block, interleaved at
    /// #m_implInterleaveWidth and transposed for SumFacTOP; see
    /// DerivFactorKey.
    const TData *m_dfptr;

    /**
     * @brief Dispatch on the block's shape to the generated ShapeBlock()
     * specialisation. An unlisted shape prints a message and returns,
     * leaving the output block unwritten.
     *
     * @param   inblock     Physical-space input block holding at least
     *                      the shape dimension's number of components of
     *                      u, or the three components on a multi-plane
     *                      3DH1 block.
     * @param   outblock    Physical-space output block receiving the
     *                      components of curl(curl u), or the plane part
     *                      of one curl on a multi-plane 3DH1 block.
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
     * @brief Worker for every dimension: evaluate curl(curl u) on all
     * elements of the block with a single fused kernel launch.
     *
     * @details
     * Sizes the launch first. The grid has one dimension and covers the
     * block's elements, padding included: with SumFac one thread per
     * element, warpSize threads per thread block and as many thread blocks
     * as that takes; with SumFacTOP one thread block per element at a
     * time, its thread count the element's number of quadrature points
     * rounded up to a warp multiple and capped at the default block size,
     * and an occupancy-based number of thread blocks that stride over the
     * remaining elements (GetDeviceBlockSize() and GetDeviceGridSize()).
     * The dynamic shared memory request is CurlCurlSharedMemorySize for
     * the shape and strategy at hand, in bytes: the collapsed-coordinate
     * factor tables for SumFac (nothing for segments, quadrilaterals and
     * hexahedra) and one element's input components plus its vorticity
     * for SumFacTOP (nothing for segments). The global workspace request
     * is CurlCurlWorkSpaceSize: the block's vorticity planes for SumFac,
     * nothing for SumFacTOP, whose kernels keep omega in shared memory.
     *
     * The launch goes through DEVICE_1DGRID_KERNEL_LAUNCHER to the
     * CurlCurlKernelLauncher overload of this dimension and strategy, with
     * the index sequences expanding to the derivative matrices #m_D and
     * factor tables #m_f the kernel needs. The kernel is given the
     * component stride of the output block, CompSize(), and applies it to
     * the input pointer as well: the first ndim components of the input
     * are read and ndim output components written, in one launch. A
     * multi-plane 3DH1 block is handed to OperatorCurl3DH1() before any of
     * this when ndim is two, and raises a fatal error otherwise.
     *
     * Block storage held at another interleave width is reshaped in place
     * around the launch, in one pass over the ndim * nelmt element
     * components: the input is converted to the implementation width and
     * back again, and the output is converted from the implementation
     * width to the input's width once the kernel has been queued. The
     * output block's recorded width is set to the input's on return.
     * Everything is issued on #m_streamID.
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
     * @param   outblock    Physical-space output block.
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
        constexpr unsigned int ndim = sizeof...(ind0);

        // A multi-plane 3DH1 field is a stack of plane curls on a 2D shape,
        // which is a different kernel over the same sweep.
        if (inblock.GetNumHomoModes() > 1)
        {
            if constexpr (ndim == 2)
            {
                OperatorCurl3DH1<SHAPE_TYPE, DEFORMED>(
                    inblock, outblock, sizeParam,
                    std::integer_sequence<unsigned int, ind0...>(),
                    std::integer_sequence<unsigned int, ind1...>());
                return;
            }
            else
            {
                // 3DH1 planes are two-dimensional and 3DH2 is not wired at
                // all. The guard is L0: without it the launch below would
                // take the extra planes for extra variables and come back
                // silently wrong in a release build.
                NEKERROR(ErrorUtil::efatal,
                         "CurlCurlBlockOp supports homogeneous expansions "
                         "only on two-dimensional planes.");
            }
        }

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer. The fused kernel only needs storage
        // for the intermediate omega = curl(u).
        const size_t wspSize =
            CurlCurlWorkSpaceSize<SHAPE_TYPE, Implementation>(nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            CurlCurlSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        const auto inoffset = outblock.CompSize();

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, ndim * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Curl-curl kernel.
        DEVICE_1DGRID_KERNEL_LAUNCHER(
            (CurlCurlKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, blocksize, shmemsize, m_streamID, sizeParam, nelmt,
            inoffset, m_D[ind0]..., m_f[ind1]..., m_dfptr, inptr, wspptr,
            outptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, ndim * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, ndim * nelmt,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /// \brief One curl's plane part on each plane of a 3DH1 block: three
    /// components in, three out.
    ///
    /// The z part is the operator's z-op and the double curl is the pair
    /// taken twice with the output carrying omega in between, so this is one
    /// curl's worth of plane derivatives and nothing more. Each plane holds the
    /// same geometry, so they ride the second grid dimension of a single launch
    /// rather than a loop on the host.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorCurl3DH1(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        // A curl needs all three components whichever way round it is taken.
        constexpr unsigned int nComp = 3u;

        ASSERTL1(inblock.GetNumComponents() == nComp &&
                     outblock.GetNumComponents() == nComp,
                 "The homogeneous curl needs all three components");

        // Get block sizes.
        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            CurlCurlSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nqTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto inoffset  = inblock.CompSize() * nhomo;
        const auto outoffset = outblock.CompSize() * nhomo;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nComp * nhomo * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Plane part of one curl.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (Curl3DH1KernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, nhomo, blocksize, 1, shmemsize, m_streamID, sizeParam,
            nelmt, inoffset, outoffset, m_D[ind0]..., m_f[ind1]..., m_dfptr,
            inptr, outptr);

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nComp * nhomo * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, nComp * nhomo * nelmt,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
