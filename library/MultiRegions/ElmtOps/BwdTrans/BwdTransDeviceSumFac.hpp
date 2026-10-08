///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFac.hpp
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
// Description: Device sum-factorised implementations of the per-block
// backward transform, serving both the SumFac and the SumFacTOP strategy
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransDeviceSumFac.hpp
 * @brief Device sum-factorised implementations of the per-block backward
 * transform, serving both the SumFac and the SumFacTOP strategy.
 *
 * Sum factorisation replaces the dense nqTot x nmTot product of the
 * StdMat path by one contraction per direction, so a tensor-product
 * element costs \f$O(n^{d+1})\f$ rather than \f$O(n^{2d})\f$. This one
 * header is included by the translation units CMake generates for either
 * sum-factorisation strategy on the Device execution space; the
 * Implementation tag selects between two kernel families at compile
 * time:
 * - SumFac (BwdTransDeviceSumFacKernels.hpp): one element per thread.
 *   The lanes of a warp process warpSize elements in lock-step on
 *   warp-interleaved data, with the inter-stage intermediates in a
 *   global-memory workspace.
 * - SumFacTOP (BwdTransDeviceSumFacTOPKernels.hpp, threaded by output
 *   point): one element per thread block. The block's threads are
 *   indexed over the element's per-stage output entries, with the basis
 *   tables, coefficients and intermediates staged in shared memory,
 *   exposing parallelism within an element rather than across elements.
 *
 * The shape dispatch mirrors the Serial/AVX SumFac implementation:
 * v_Apply() switches on the runtime shape into the ShapeBlock()
 * specialisations that CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape. NEKTAR_BLOCKOP_SWITCH_CODE, defined below,
 * selects the switch body those definitions expand.
 *
 * @see BwdTransSerialAVXSumFac.hpp for the Serial/AVX sum-factorised
 * path and BwdTransDeviceStdMat.hpp for the dense-matrix device path.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through BwdTransOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOp.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised block implementation of the backward
 * transform, SumFac or SumFacTOP as the Implementation tag selects.
 *
 * As in the Serial/AVX SumFac implementation, the constructor caches the
 * per-direction 1D basis tables and, for the nodal shapes, the
 * nodal-to-modal matrix from the data warehouse -- here in device
 * memory. For tetrahedra under SumFacTOP it additionally fetches the
 * precomputed mode-index tables (#m_index) that map a flat triangular
 * mode index back to its p and q, so a thread can locate its mode pair
 * without looping.
 *
 * The two strategies want different data layouts, captured by
 * #m_implInterleaveWidth: SumFac reads warp-interleaved data, one
 * element per lane, SumFacTOP contiguous per-element data (width 1).
 * OperatorNDImpl() reshapes the block storage to that width and back
 * around each launch. Each block operator owns device stream
 * `block_idx + 1`, so different blocks may overlap on the device.
 *
 * The element geometry never enters: the backward transform is a
 * standard-region operation, so #m_coordDim is recorded only for
 * uniformity with the other element operators, and the two DEFORMED
 * instantiations the generated switch selects between compute the same
 * result.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see BwdTransSerialAVXSumFac.hpp for the Serial/AVX sum-factorised
 * path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class BwdTransBlockOpImpl : public BwdTransBlockOp<TData>
{
    using BlockOpBase = BwdTransBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's basis tables, nodal-to-modal matrix and,
     * for tetrahedra under SumFacTOP, the mode-index tables, and claim
     * the block's device stream.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections; also sets
     *                          #m_streamID.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse the tables are cached in.
     */
    BwdTransBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BwdTransBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eBasis)));
        }

        if ((m_shapeType == LibUtilities::NodalTri) ||
            (m_shapeType == LibUtilities::NodalPrism) ||
            (m_shapeType == LibUtilities::NodalTet))
        {
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

            // Fetch NodalToModal Matrix if required.
            m_nodToMod = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eNodalToModal,
                                             nodalType));
        }
        else
        {
            m_nodToMod = (const TData *)nullptr;
        }

        if (m_dimension == 3)
        {
            // Precompute index, if necessary.
            const bool indexing =
                ((m_shapeType == LibUtilities::Tet ||
                  m_shapeType == LibUtilities::NodalTet) &&
                 std::is_same_v<Implementation, MultiRegions::SumFacTOP>);

            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<MemSpace>(
                               LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                          m_nm[1], m_nm[2], 0))
                         : nullptr);
            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<MemSpace>(
                               LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                          m_nm[1], m_nm[2], 3))
                         : nullptr);
        }
    }

    /// Block-operator factory registration key,
    /// `"BlockBwdTrans" + execution space + implementation`. Defined by
    /// the CMake-generated declaration file, whose initialiser performs
    /// the registration.
    static std::string className;

    /**
     * @brief Creator function registered with the block-operator factory
     * under #className.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     *
     * @return The newly created block operator.
     */
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            BwdTransBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    /// Whether the block's geometry is deformed: read by the generated
    /// dispatch to select the DEFORMED branch, whose two instantiations
    /// compute the same result here, the backward transform being
    /// geometry independent.
    bool m_isDeformed;
    /// First-direction basis is eModified_A: apply the collapsed mode
    /// corrections in the kernels.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; unused
    /// here.
    unsigned int m_coordDim;
    /// Modes per direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per direction.
    std::vector<unsigned int> m_nq;
    /// 1D basis table per direction, in device memory.
    std::vector<const TData *> m_B;
    /// Tetrahedral p and q mode-index tables (SumFacTOP only; see the
    /// class notes), null for the other three-dimensional shapes and
    /// for SumFac.
    std::vector<const unsigned int *> m_index;
    /// Nodal-to-modal matrix for the nodal shapes; null otherwise.
    const TData *m_nodToMod;

    /**
     * @brief Dispatch on the runtime shape to the generated per-shape
     * entry point.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated into in append mode.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
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

    /**
     * @brief Per-shape entry point: selects the element sizes for
     * @p SHAPE_TYPE and calls OperatorND() with them.
     *
     * Specialised for each shape by the definitions CMake generates from
     * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
     * translation unit per shape, so that the templated kernels are
     * instantiated for one shape at a time.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /**
     * @brief Number of precomputed mode-index tables the kernels are
     * handed in @p dim dimensions: the p and q tables in three, none
     * below.
     *
     * @param   dim     Dimension of the reference element.
     *
     * @return Number of entries of #m_index the kernels are handed.
     */
    static constexpr unsigned int NumIndex(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 0 : 2;
    }

    /**
     * @brief Entry point for every shape and dimension: builds the index
     * sequences the implementation expands and forwards to
     * OperatorNDImpl().
     *
     * @p sizeParam carries the element's extents, and carries them
     * either as runtime values or as template parameters: the generated
     * switch hands over a NonTemplatedSizeParameter for the general case
     * and a TemplatedSizeParameter for the sizes it has a compile-time
     * case for, which lets the kernels fix their loop bounds and the
     * launch its bounds at compile time. Either form is accepted, and
     * the static assertion below is what ties whichever arrives to the
     * dimension of @p SHAPE_TYPE.
     *
     * @tparam SHAPE_TYPE      Shape the kernels are instantiated for.
     * @tparam DEFORMED        Deformed-geometry branch selected by the
     *                         generated switch.
     * @tparam TSizeParameter  Size-parameter type, runtime or templated.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block.
     * @param   sizeParam   Element extents in either form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitchCode.h.in.
        static_assert((DIM == 1 && IsSizeParameter1D_v<TSizeParameter>) ||
                          (DIM == 2 && IsSizeParameter2D_v<TSizeParameter>) ||
                          (DIM == 3 && IsSizeParameter3D_v<TSizeParameter>),
                      "OperatorND expects a size parameter matching the "
                      "dimension of the shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumIndex(DIM)>());
    }

    /**
     * @brief Configure and launch the backward-transform kernel for the
     * whole block.
     *
     * The two index packs are what let one definition serve all three
     * dimensions: @p ind0 expands to one basis table per direction and
     * @p ind1 to the precomputed mode-index tables, so the launch names
     * exactly the arguments that dimension needs.
     *
     * The launch configuration comes from the shared helpers. Dynamic
     * shared memory is BwdTransSharedMemorySize() for the shape and
     * strategy -- zero under SumFac, which stages nothing in shared
     * memory. Threads per block come from GetDeviceBlockSize(): a warp
     * under SumFac, and under SumFacTOP the element's total mode count
     * rounded up to whole warps and capped at the device's default block
     * size. The grid is two-dimensional, its second dimension being the
     * component count ncomp (components times homogeneous modes) and its
     * first coming from GetDeviceGridSize(): one block per group of
     * elements under SumFac, and under SumFacTOP a grid sized to the
     * device's occupancy, since there a block serves one element at a
     * time. The global-memory workspace is the static per-stream buffer
     * shared by all block operators, sized by BwdTransWorkSpaceSize()
     * times ncomp -- the stage intermediates under SumFac, and zero
     * under SumFacTOP, which stages them in shared memory instead.
     *
     * The storage for all nelmt * ncomp elements is reshaped once to
     * #m_implInterleaveWidth, a single launch covers every component on
     * this block's stream, and the storage is reshaped back. In append
     * mode the output is also reshaped forward first, so the kernels
     * accumulate in the layout they write. On return the output block's
     * interleave width is set to the input's.
     *
     * @tparam SHAPE_TYPE      Shape the kernels are instantiated for.
     * @tparam DEFORMED        Deformed-geometry branch.
     * @tparam TSizeParameter  Size-parameter type, runtime or templated.
     * @tparam ind0            One index per direction.
     * @tparam ind1            One index per mode-index table.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated into in append mode.
     * @param   sizeParam   Element extents in either form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const size_t wspSize =
            BwdTransWorkSpaceSize<SHAPE_TYPE, Implementation>(nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

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
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);

            // BwdTrans kernel.
            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (BwdTransKernelLauncher<SHAPE_TYPE, Implementation, true>),
                gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
                nelmt, m_isModified, m_index[ind1]..., m_B[ind0]..., m_nodToMod,
                inptr, outptr, wspptr);
        }
        else
        {
            // BwdTrans kernel.
            DEVICE_2DGRID_KERNEL_LAUNCHER(
                (BwdTransKernelLauncher<SHAPE_TYPE, Implementation, false>),
                gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
                nelmt, m_isModified, m_index[ind1]..., m_B[ind0]..., m_nodToMod,
                inptr, outptr, wspptr);
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
