///////////////////////////////////////////////////////////////////////////////
//
// File: MassDeviceSumFac.hpp
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
// Description: Device sum-factorised implementation of the per-block
// mass operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file MassDeviceSumFac.hpp
 * @brief Device sum-factorised implementation of the per-block mass
 * operator, serving both the SumFac and the SumFacTOP strategy.
 *
 * @details
 * As in the Serial/AVX sum-factorised implementation
 * (MassSerialAVXSumFac.hpp), the backward transform and the inner product
 * with the basis are applied one coordinate direction at a time through
 * the one-dimensional basis tables, back to back; on the device both
 * stages happen inside a single fused kernel launch per block.
 *
 * This one header is included by the translation units CMake generates for
 * either sum-factorisation strategy on the Device execution space; the
 * Implementation tag selects between the two kernel families at compile
 * time, when the enable_if-constrained kernel launchers and the workspace
 * and shared-memory size helpers resolve:
 * - SumFac (MassDeviceSumFacKernels.hpp): one element per thread. The
 *   lanes of a warp advance through warpSize elements in lock-step on
 *   warp-interleaved data, which is what the Serial/AVX implementation
 *   does with SIMD lanes; the inter-stage intermediates live in a
 *   global-memory workspace and the quadrature metric is applied inside
 *   the inner-product shape kernel.
 * - SumFacTOP (MassDeviceSumFacTOPKernels.hpp): one element per thread
 *   block. The block's threads are indexed over the element's per-stage
 *   output entries, with the basis tables and the intermediates staged in
 *   shared memory; the quadrature metric is applied to the staged physical
 *   values between the two stages rather than inside the contractions.
 *
 * The two strategies want different storage layouts; see the class
 * description. The shape dispatch of v_Apply() and the generated
 * ShapeBlock() specialisations mirror the Serial/AVX SumFac implementation
 * -- the same BlockOpShapeBlock.cpp.in template generates them for both --
 * so see MassSerialAVXSumFac.hpp for that description.
 *
 * @see MassDeviceStdMat.hpp for the dense-matrix device path.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through MassOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/Mass/MassBlockOp.hpp"

#include "Operators/ElmtOps/Mass/MassDeviceSumFacKernels.hpp"
#include "Operators/ElmtOps/Mass/MassDeviceSumFacTOPKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::Operators::detail
{

/**
 * @brief Device sum-factorised mass operator (SumFac or SumFacTOP, selected
 * by the Implementation tag).
 *
 * @details
 * As in the Serial/AVX SumFac implementation, the constructor caches, per
 * reference direction, the element sizes, the one-dimensional basis tables
 * (#m_B) and quadrature weights (#m_W, collapsed-coordinate factors
 * included), the nodal-to-modal matrix #m_nodToMod of the nodal shapes and
 * the block's Jacobians #m_jacptr, all in device memory. Only one nodal
 * matrix is fetched here: the device kernels obtain the reverse mapping by
 * running MatVecKernel with its TRANSPOSE parameter set, where the host
 * implementation fetches a second, transposed matrix. Under SumFacTOP it
 * additionally fetches, for the shapes with a collapsed mode ordering
 * (triangle, tetrahedron, prism, pyramid and their nodal variants), the
 * precomputed mode-index tables #m_index (see ModeIndexDataWarehouse.hpp),
 * because a thread there is responsible for an arbitrary flat output index
 * rather than for a loop nest and must recover the mode indices from it
 * without looping. How many of the slots a shape needs varies -- the
 * tetrahedra use all four, the prisms three, the pyramid two, the
 * triangles one -- and the rest, along with every slot under SumFac and
 * for the tensor-product shapes, stay null.
 *
 * The strategy shows up in #m_implInterleaveWidth: the warp size for SumFac
 * (one element per lane), one for SumFacTOP (contiguous per-element data).
 * OperatorNDImpl() reshapes the block storage to that width around the
 * launch and back afterwards.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that issues a single kernel launch per
 * block covering every element, component and homogeneous mode. Each block
 * operator owns device stream block_idx + 1, so different blocks may
 * overlap on the device.
 *
 * @tparam ExecSpace      NektarSpaces::Device.
 * @tparam Implementation Operators::SumFac or Operators::SumFacTOP.
 * @tparam TData          Floating-point type of the field data.
 *
 * @see MassDeviceStdMat.hpp for the dense-matrix Device path;
 * MassSerialAVXSumFac.hpp for the host sum-factorised sibling.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class MassBlockOpImpl : public MassBlockOp<TData>
{
    using BlockOpBase = MassBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the one-dimensional tables, nodal-to-modal
     * matrix, mode-index tables and Jacobians (see the class description).
     *
     * @param   block_idx       Index of the block; the stream used for
     *                          all device work is block_idx + 1.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    MassBlockOpImpl(const unsigned int block_idx,
                    const LocalRegions::ExpansionSharedPtr &exp,
                    LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : MassBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eWeights)));
        }

        if ((m_shapeType == LibUtilities::eNodalTri) ||
            (m_shapeType == LibUtilities::eNodalPrism) ||
            (m_shapeType == LibUtilities::eNodalTet))
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

        if (m_dimension == 2)
        {
            // Precompute index, if necessary.
            const bool indexing =
                (m_shapeType == LibUtilities::Tri ||
                 m_shapeType == LibUtilities::NodalTri) &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;
            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<MemSpace>(
                               LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                          m_nm[1], 0))
                         : nullptr);
        }
        else if (m_dimension == 3)
        {
            // Precompute index, if necessary.
            const bool indexingTet =
                (m_shapeType == LibUtilities::Tet ||
                 m_shapeType == LibUtilities::NodalTet) &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;
            const bool indexingPrism =
                (m_shapeType == LibUtilities::Prism ||
                 m_shapeType == LibUtilities::NodalPrism) &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;
            const bool indexingPyr =
                m_shapeType == LibUtilities::Pyr &&
                std::is_same_v<Implementation, Operators::SumFacTOP>;
            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 0))
                    : nullptr);
            m_index.push_back(
                (indexingTet || indexingPrism || indexingPyr)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 1))
                    : nullptr);
            m_index.push_back(
                (indexingTet || indexingPrism)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 2))
                    : nullptr);
            m_index.push_back(
                (indexingTet)
                    ? this->m_dataWarehouse->template GetData<MemSpace>(
                          LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                     m_nm[1], m_nm[2], 3))
                    : nullptr);
        }

        // Fetch Jacobian data.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
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
            MassBlockOpImpl<ExecSpace, Implementation, TData>>(block_idx, exp,
                                                               dataWarehouse);
    }

protected:
    /// Element interleave width the kernels expect: the warp size for
    /// SumFac (one element per lane), one for SumFacTOP. The Jacobians are
    /// fetched at this width.
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, Operators::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    /// Device stream all work of this block operator is issued on, one
    /// per block (block index + 1).
    unsigned int m_streamID;
    /// Shape of the block's elements; drives the dispatch in v_Apply().
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobians); selects the
    /// DEFORMED branch of the generated dispatch and hence of the kernels.
    bool m_isDeformed;
    /// Whether the direction-0 basis is of type eModified_A; enables the
    /// collapsed vertex- and edge-mode corrections in the kernels.
    bool m_isModified;
    /// Reference (shape) dimension of the elements (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements; not read
    /// by this implementation.
    unsigned int m_coordDim;
    /// Modes per reference direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per reference direction.
    std::vector<unsigned int> m_nq;
    /// One-dimensional basis tables, one per direction (eBasis), in device
    /// memory.
    std::vector<const TData *> m_B;
    /// One-dimensional quadrature weights, one per direction (eWeights),
    /// collapsed-coordinate factors included, in device memory; read by
    /// the inner-product stage.
    std::vector<const TData *> m_W;
    /// Mode-index tables of the collapsed mode orderings: one slot in 2D,
    /// four in 3D (see NumIndex()), fetched under SumFacTOP for the shapes
    /// that use them and null otherwise (see the class description).
    std::vector<const unsigned int *> m_index;
    /// Nodal-to-modal matrix of the nodal shapes; null otherwise. Its
    /// transpose is not stored: the kernels transpose on the fly.
    const TData *m_nodToMod;
    /// Jacobians of the block, interleaved at #m_implInterleaveWidth: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;

    /**
     * @brief Dispatch on the block's shape to the generated ShapeBlock()
     * specialisation.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
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

    // Number of precomputed index arrays used by the kernels in dim dimensions.
    static constexpr unsigned int NumIndex(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 4;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
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
     * @brief Worker for every dimension: apply the operator to all elements
     * of the block with a single fused kernel launch.
     *
     * @details
     * Sizes the launch first. The grid has two dimensions. Its x extent
     * covers the block's elements, padding included: with SumFac one thread
     * per element, warpSize threads per thread block and as many thread
     * blocks as that takes; with SumFacTOP one thread block per element at
     * a time, its thread count the element's mode count rounded up to a
     * warp multiple and capped at the default block size, and an
     * occupancy-based number of thread blocks that stride over the
     * remaining elements (GetDeviceBlockSize() and GetDeviceGridSize()).
     * Its y extent is ncomp, the component count times the homogeneous mode
     * count, so that one launch covers every component and mode and each
     * kernel recovers its component from its block index. The dynamic
     * shared memory request is MassSharedMemorySize for the shape and
     * strategy at hand, in bytes (zero under SumFac); the global workspace
     * request is MassWorkSpaceSize scaled by ncomp, taken from the static
     * per-stream buffer. Both helpers have one overload per strategy and
     * dimension, documented in the kernel headers.
     *
     * The launch goes through DEVICE_2DGRID_KERNEL_LAUNCHER to the
     * MassKernelLauncher overload of this dimension and strategy, the index
     * sequences expanding to the mode-index tables #m_index and the
     * per-direction tables #m_B and #m_W; the nodal-to-modal matrix, the
     * Jacobians, the two fields and the workspace follow. The kernels run
     * the transform with APPEND false and the inner product with SCALE and
     * APPEND false, so the output is overwritten.
     *
     * The storage for all ncomp components is reshaped to
     * #m_implInterleaveWidth in a single pass before the launch, and both
     * input and output are reshaped back to the input's width once the
     * kernel has been queued; the output block's recorded width is set to
     * the input's on return. Everything is issued on #m_streamID.
     *
     * @tparam SHAPE_TYPE      Shape of the block, the nodal enumerators
     *                         included.
     * @tparam DEFORMED        Jacobians vary point by point.
     * @tparam TSizeParameter  Size parameter of this dimension, templated
     *                         or not.
     * @tparam ind0            Reference directions, 0 to ndim - 1; selects
     *                         the tables #m_B and #m_W.
     * @tparam ind1            Mode-index tables #m_index the SumFacTOP
     *                         kernels use: none in 1D, one in 2D, four in
     *                         3D (see NumIndex()).
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block; overwritten.
     * @param   sizeParam   Element sizes, in runtime or compile-time form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
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
        auto wspSize =
            MassWorkSpaceSize<SHAPE_TYPE, Implementation>(nelmt, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            MassSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Mass kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (MassKernelLauncher<SHAPE_TYPE, Implementation, DEFORMED>),
            gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
            nelmt, m_isModified, m_index[ind1]..., m_B[ind0]..., m_W[ind0]...,
            m_nodToMod, m_jacptr, inptr, outptr, wspptr);

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

} // namespace Nektar::Operators::detail
