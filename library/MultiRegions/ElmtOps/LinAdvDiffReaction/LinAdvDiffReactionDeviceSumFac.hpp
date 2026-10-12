///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionDeviceSumFac.hpp
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
// Description: Device sum-factorised implementation of the linearised
// advection-diffusion-reaction block operator, SumFac and SumFacTOP.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionDeviceSumFac.hpp
 * @brief Device sum-factorised implementation of the per-block
 * linearised advection-diffusion-reaction operator, serving both the
 * SumFac and the SumFacTOP strategy.
 *
 * @details
 * This one header is included by the translation units CMake generates
 * for either sum-factorisation strategy on the Device execution space;
 * the Implementation tag selects between the two kernel families at
 * compile time, when the enable_if-constrained
 * LinAdvDiffReactionKernelLauncher overloads resolve:
 * - SumFac (LinAdvDiffReactionDeviceSumFacKernels.hpp): one element per
 *   thread. The lanes of a warp advance through warpSize elements in
 *   lock-step on warp-interleaved data, with the stage intermediates in
 *   a global-memory workspace.
 * - SumFacTOP (LinAdvDiffReactionDeviceSumFacTOPKernels.hpp): one
 *   element per thread block. The block's threads are indexed over the
 *   work items of each stage, with the basis tables, the coefficients
 *   and every intermediate staged in shared memory, exposing parallelism
 *   within an element rather than across elements.
 *
 * Both families are the corresponding Helmholtz kernels with this
 * family's advection kernel spliced in after the derivative stage, and
 * both reuse the Helmholtz workspace and shared-memory sizing functions
 * (HelmholtzWorkSpaceSize, HelmholtzSharedMemorySize) unchanged, the
 * intermediates being the same.
 *
 * The two strategies want different storage and derivative-factor
 * layouts; see the class description. v_Apply() dispatches on the
 * block's shape type to the ShapeBlock() specialisations, which CMake
 * generates from LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
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
#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionBlockOp.hpp>

#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionDeviceSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionDeviceSumFacTOPKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Device sum-factorised block implementation of the linearised
 * advection-diffusion-reaction operator (SumFac or SumFacTOP, selected
 * by the Implementation tag).
 *
 * @details
 * The constructor caches the per-direction 1D basis,
 * collocation-derivative and quadrature-weight tables, the
 * collapsed-coordinate factors of the 2D and 3D blocks and the
 * nodal-to-modal matrix of the nodal shapes -- here in device memory --
 * along with the block's Jacobians and derivative factors. Two things
 * depend on the strategy:
 * - under SumFacTOP the precomputed mode-index tables #m_index are
 *   fetched for the shapes with collapsed coordinates, so that a thread
 *   can map its flat mode index back to the individual mode numbers
 *   without looping; the entries stay null otherwise;
 * - the derivative factors are fetched transposed for SumFacTOP, which
 *   wants each factor component contiguous over the element's points
 *   because its threads are indexed over points, and untransposed for
 *   SumFac, which wants the factors of a point adjacent (the last
 *   argument of DerivFactorKey). The distinction only affects deformed
 *   blocks; for regular geometry the warehouse ignores the flag and
 *   stores one value per factor component per element.
 *
 * The two strategies also want different field layouts, captured by
 * #m_implInterleaveWidth: SumFac reads warp-interleaved data (one
 * element per lane), SumFacTOP contiguous per-element data (width 1).
 * OperatorNDImpl() reshapes the block storage to that width around each
 * launch and back afterwards, and issues a single kernel launch per
 * block covering every element, component and homogeneous mode. Each
 * block operator owns device stream `block_idx + 1`, so different blocks
 * may overlap on the device.
 *
 * @tparam ExecSpace       NektarSpaces::Device.
 * @tparam Implementation  MultiRegions::SumFac or MultiRegions::SumFacTOP.
 * @tparam TData           Floating-point type of the field data.
 *
 * @see LinAdvDiffReactionDeviceStdMat.hpp for the dense-matrix Device
 * path.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class LinAdvDiffReactionBlockOpImpl : public LinAdvDiffReactionBlockOp<TData>
{
    using BlockOpBase = LinAdvDiffReactionBlockOp<TData>;
    using MemSpace    = typename ExecSpace::memory_space;

public:
    /**
     * @brief Bind the operator to the block's device stream, capture the
     * element metadata and fetch the 1D tables, collapsed-coordinate
     * factors, nodal-to-modal matrix, mode-index tables, Jacobians and
     * derivative factors (see the class description).
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
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eWeights)));
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

            // Precompute index, if necessary.
            const bool indexing =
                (m_shapeType == LibUtilities::Tri ||
                 m_shapeType == LibUtilities::NodalTri) &&
                std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<MemSpace>(
                               LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                          m_nm[1], 0))
                         : nullptr);
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

            // Precompute index, if necessary.
            const bool indexingTet =
                (m_shapeType == LibUtilities::Tet ||
                 m_shapeType == LibUtilities::NodalTet) &&
                std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
            const bool indexingPrism =
                (m_shapeType == LibUtilities::Prism ||
                 m_shapeType == LibUtilities::NodalPrism) &&
                std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
            const bool indexingPyr =
                m_shapeType == LibUtilities::Pyr &&
                std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
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

        // Fetch Jacobian and deriv factors.
        constexpr bool transpose =
            std::is_same_v<Implementation, MultiRegions::SumFacTOP>;
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
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
    /// Interleave width the kernels operate at: the warp size for
    /// SumFac (one element per lane), 1 for SumFacTOP (contiguous
    /// per-element data).
    static constexpr unsigned int m_implInterleaveWidth =
        std::is_same_v<Implementation, MultiRegions::SumFac>
            ? NektarSpaces::Device::warpSize
            : 1u;

    /// Device stream this block's work is issued on (block index + 1).
    unsigned int m_streamID;
    /// Shape shared by every element of the block; drives the dispatch
    /// in v_Apply().
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; read by the generated
    /// dispatch to pick the DEFORMED instantiation, which changes how
    /// the Jacobian and derivative factors are indexed.
    bool m_isDeformed;
    /// First-direction basis is eModified_A: apply the collapsed
    /// vertex-mode corrections in the shape kernels.
    bool m_isModified;
    /// Dimension of the reference element (1, 2 or 3).
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) element; the
    /// number of advection-velocity components and the rank of the
    /// diffusivity tensor. Passed to the 1D and 2D kernels; the 3D
    /// kernels take it as three.
    unsigned int m_coordDim;
    /// Modes per direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per direction.
    std::vector<unsigned int> m_nq;
    /// 1D basis table per direction, in device memory.
    std::vector<const TData *> m_B;
    /// 1D collocation derivative matrix per direction.
    std::vector<const TData *> m_D;
    /// 1D quadrature weights per direction.
    std::vector<const TData *> m_W;
    /// Collapsed-coordinate factors of the non-tensor shapes, in the
    /// order the kernels expect them: in 2D (1 + eta)/2 of direction 0
    /// and 2/(1 - eta) of direction 1; in 3D (1 + eta)/2 of directions
    /// 0 and 1, then 2/(1 - eta) of directions 1 and 2. Empty for 1D
    /// blocks.
    std::vector<const TData *> m_f;
    /// Mode-index tables (see NumIndex()), fetched under SumFacTOP for
    /// the shapes with collapsed coordinates and null otherwise: they
    /// map a flat mode index back to the individual mode numbers of the
    /// shape's combined ordering.
    std::vector<const unsigned int *> m_index;
    /// Nodal-to-modal matrix for the nodal shapes; null otherwise. The
    /// inner-product stage reuses it transposed, so no separate
    /// transpose is cached.
    const TData *m_nodToMod;
    /// Jacobians of the block's elements, interleaved at
    /// #m_implInterleaveWidth: one value per element when regular, one
    /// per quadrature point when deformed.
    const TData *m_jacptr;
    /// Derivative factors \f$\partial\xi_d/\partial x_k\f$ of the
    /// block's elements, interleaved at #m_implInterleaveWidth and
    /// transposed for SumFacTOP (see the class notes and
    /// DerivFactorKey).
    const TData *m_dfptr;

    /**
     * @brief Dispatch on the block's shape to the generated ShapeBlock()
     * specialisation. An unsupported shape prints a message and returns
     * with @p outblock untouched.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block, overwritten.
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

    // Number of collapsed coordinate factors used by the kernels in dim
    // dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
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
            std::make_integer_sequence<unsigned int, NumIndex(DIM)>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    /**
     * @brief Worker for every dimension: apply the operator to all
     * elements of the block with a single kernel launch covering every
     * component and homogeneous mode.
     *
     * @details
     * Reshapes the advection velocity to #m_implInterleaveWidth if it is
     * still held at another width, then sizes the launch from the shared
     * helpers, with the Helmholtz sizing functions supplying this
     * family's per-element requirements unchanged. The global-memory
     * workspace is HelmholtzWorkSpaceSize for the shape and strategy --
     * zero under SumFacTOP, which stages everything in shared memory --
     * multiplied by ncomp = GetNumComponents() * GetNumHomoModes(), since
     * it must hold every component's intermediates at once; the dynamic
     * shared-memory request is HelmholtzSharedMemorySize, in bytes, and
     * is not scaled by ncomp. Threads per block come from
     * GetDeviceBlockSize: a warp for SumFac; for SumFacTOP the element's
     * total mode count rounded up to whole warps and capped at the
     * default block size. The grid's first dimension comes from
     * GetDeviceGridSize: enough thread blocks for every element under
     * SumFac; under SumFacTOP a persistent, occupancy-based count of
     * thread blocks that stride over the elements, after the
     * shared-memory request has been checked against the device limit.
     * The second grid dimension is ncomp, one row per component or
     * homogeneous mode.
     *
     * The whole block's storage, covering every component and
     * homogeneous mode at once, is reshaped to #m_implInterleaveWidth,
     * the LinAdvDiffReactionKernelLauncher overload matching the
     * Implementation tag and the dimension is launched once on
     * #m_streamID through DEVICE_2DGRID_KERNEL_LAUNCHER, with the index
     * sequences expanding to the mode-index tables #m_index, the tables
     * #m_B, #m_D, #m_W and the factor tables #m_f the kernel needs, and
     * both input and output storage are reshaped back to the input's
     * width, which the output block is also set to. The three velocity
     * component pointers are formed once, by offsetting the block's
     * velocity by whole components (nelmt * nqTot * nhomo values each)
     * whatever the coordinate dimension, and are passed with nhomo so
     * that the kernel reads, for every field component the launch covers,
     * the velocity of that component's plane.
     *
     * @tparam SHAPE_TYPE      Shape of the block, the nodal enumerators
     *                         included.
     * @tparam DEFORMED        Jacobian and derivative factors vary point
     *                         by point.
     * @tparam TSizeParameter  Size parameter of this dimension,
     *                         templated or not.
     * @tparam ind0            Reference directions, 0 to ndim - 1;
     *                         selects the tables #m_B, #m_D and #m_W.
     * @tparam ind1            Mode-index tables #m_index the kernels
     *                         take: none in 1D, one in 2D, four in 3D
     *                         (see NumIndex()).
     * @tparam ind2            Collapsed-coordinate factor tables #m_f
     *                         the kernels use: none in 1D, two in 2D,
     *                         four in 3D (see NumFactor()).
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block, overwritten.
     * @param   sizeParam   Element sizes, in runtime or compile-time form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>)
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

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // Initialize advVel pointers. One velocity component spans every
        // plane, and the kernel picks the plane its slice sits on.
        const auto nhomo = inblock.GetNumHomoModes();
        auto advVelPtr =
            this->m_advVel->template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto advVelSize = nelmt * sizeParam.nqTot() * nhomo;

        // Get static workspace pointer.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const size_t wspSize =
            HelmholtzWorkSpaceSize<SHAPE_TYPE, Implementation>(
                nelmt, m_coordDim, sizeParam);
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                wspSize * ncomp, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) *
            HelmholtzSharedMemorySize<SHAPE_TYPE, Implementation>(sizeParam);
        const unsigned int blocksize =
            GetDeviceBlockSize<Implementation>(sizeParam.nmTot());
        const unsigned int gridsize =
            GetDeviceGridSize<Implementation>(nelmt, blocksize, shmemsize);

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // LinAdvDiffReaction kernel.
        DEVICE_2DGRID_KERNEL_LAUNCHER(
            (LinAdvDiffReactionKernelLauncher<SHAPE_TYPE, Implementation,
                                              DEFORMED>),
            gridsize, ncomp, blocksize, 1, shmemsize, m_streamID, sizeParam,
            m_coordDim, nelmt, nhomo, m_isModified, m_index[ind1]...,
            m_B[ind0]..., m_D[ind0]..., m_W[ind0]..., m_f[ind2]..., m_nodToMod,
            m_dfptr, m_jacptr, diffCoeffPtr, advVelPtr, advVelPtr + advVelSize,
            advVelPtr + 2 * advVelSize, inptr, outptr, wspptr, this->m_lambda);

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
