///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionSerialAVXSumFac.hpp
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
// Description: Serial/AVX sum-factorised implementation of the advection
// block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised implementation of the advection block
 * operator.
 *
 * @details
 * Instead of a dense standard-region matrix, the SumFac strategy
 * differentiates through the tensor-product structure of the basis: the
 * PhysDerivTensor{1,2,3}DKernel routines
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp) produce the
 * derivatives along the reference tensor directions from the
 * one-dimensional derivative matrices, and the Advection{1,2,3}DKernel
 * routines (AdvectionSerialAVXSumFacKernels.hpp) apply the
 * collapsed-to-standard coordinate correction for simplex-type shapes, the
 * geometric derivative factors and the dot product with the advection
 * velocity. Each SIMD lane owns one element, so a group of simd_t::width
 * elements advances in lock-step (width one on the Serial execution
 * space).
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one translation
 * unit per shape: each wraps OperatorND() in a switch over the element
 * sizes so that compile-time constants reach the kernels, falling back to
 * a runtime-sized call for other sizes.
 *
 * This header provides the definition of detail::AdvectionBlockOpImpl for
 * the Serial and AVX execution spaces with the SumFac implementation tag;
 * see AdvectionSerialAVXStdMat.hpp for the StdMat sibling and the note on
 * the one-definition-per-translation-unit scheme.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Advection/AdvectionBlockOp.hpp>

#include <MultiRegions/ElmtOps/Advection/AdvectionSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised advection block operator: tensor-product
 * differentiation followed by the per-shape advection kernels, one element
 * per SIMD lane.
 *
 * @details
 * The constructor collects, per reference direction, the element sizes and
 * one-dimensional derivative matrices, plus the pointwise chain-rule
 * factors (1 + eta)/2 and 2/(1 - eta) evaluated at the quadrature points
 * (data-warehouse keys eHalfMultOnePlusZero and eTwoOverOneMinusZero). The
 * factor tables are fetched for every 2D block (two of them) and every 3D
 * block (four of them) whatever the shape, since the condition in the
 * constructor is on the element dimension alone; only the shapes with
 * collapsed coordinates use them, the kernels discarding the tables of
 * quadrilaterals and hexahedra through a compile-time shape test. See the
 * Advection2DKernel and Advection3DKernel documentation for which factor
 * serves which direction. The constructor also fetches the block's
 * derivative factors and allocates per-direction workspace for the
 * intermediate reference-space derivatives.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that loops over components and homogeneous
 * modes and, within each, over element groups, handing every group to
 * AdvectionKernelLauncher (AdvectionSerialAVXSumFacKernels.hpp), which
 * differentiates it with PhysDerivTensor*DKernel into #m_deriv and
 * finishes it with Advection*DKernel. Fields stored at a larger interleave
 * width are reshaped in place around the group loop, as in the StdMat
 * implementation.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (SumFac).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionBlockOpImpl : public AdvectionBlockOp<TData>
{
    using BlockOpBase = AdvectionBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata, fetch the 1D
     * derivative matrices, collapsed-coordinate factors and derivative
     * factors, and allocate the workspace (see the class description).
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
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
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    this->m_exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch deriv factors data.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Allocate workspace.
        unsigned int nqTot = 1;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            nqTot *= m_nq[d];
        }
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_deriv.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot));
        }
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
    /// Element interleave width the kernels expect: the SIMD width (1 on
    /// the Serial execution space).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    std::vector<const simd_t *> m_f;
    /// Workspace: the reference derivative of one element group, one
    /// vector of nqTot SIMD entries per direction.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_deriv;
    /// Derivative factors of the whole block, interleaved at the SIMD
    /// width and fetched untransposed: a deformed block stores the
    /// coordDim * dimension factors of a point together, point by point,
    /// while a regular block stores a single set per element group and
    /// has no point index at all (see DerivFactorKey).
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief In debug builds, warn once if the block storage is not
     * aligned for simd_t; then dispatch on the block's shape to the
     * generated ShapeBlock() specialisation. An unsupported shape type
     * is reported on std::cout and leaves the output block untouched.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated onto when append is set.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() % simd_t::alignment == 0 &&
                           outblock.GetAlignment() % simd_t::alignment == 0),
            "Input or output Field are not aligned to the required alignment "
            "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

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
     * @brief Worker for every dimension: advect all element groups of the
     * block, component by component.
     *
     * @details
     * Reshapes the advection velocity to #m_implInterleaveWidth if it is
     * still held at another width, then loops over the components and
     * homogeneous modes of the block and, within each, over the element
     * groups. Every group is handed to AdvectionKernelLauncher, whose
     * one-, two- and three-dimensional overloads this one body calls;
     * overload resolution picks the arm from the number of arguments the
     * index sequences expand to. The launcher forms the reference
     * derivatives of the group in #m_deriv and finishes them with the
     * derivative factors and the velocity.
     *
     * The velocity pointer is reset at the start of every component and
     * homogeneous mode to the velocity of that mode's plane, so the same
     * velocity serves every component. Its first three components are
     * passed separately, offset from each other by one component's worth
     * of the block (nelmt * nqTot * nhomo); a 1D or 2D kernel reads only
     * the components its coordinate dimension has. The
     * derivative-factor pointer advances by one group's factors:
     * coordDim * ndim of them, times nqTot for deformed geometry.
     *
     * Fields stored at an interleave width larger than the implementation
     * width are handled by reshaping width_ratio consecutive groups in
     * place. The two sides are not treated symmetrically: the input is
     * converted to the implementation width before the kernel and
     * converted back once the groups have been processed, whereas the
     * output is converted forward only when it is accumulated onto
     * (append set) -- there is nothing to preserve otherwise -- and is in
     * both cases converted from the implementation width to the input's
     * width once the kernel has written it. The output block's recorded
     * interleave width is set to the input's on return.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the derivative matrices #m_D and
     *                             the workspaces #m_deriv.
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
        constexpr unsigned int ndim = sizeof...(ind0);

        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>();
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        unsigned int dfsize = ndim * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Loop over components. The advection velocity carries the same
        // planes as the input, so one of its components spans all of them.
        const auto nhomo  = inblock.GetNumHomoModes();
        auto advVelOffset = nelmt * nqTot * nhomo;

        for (unsigned int n = 0; n < inblock.GetNumComponents() * nhomo; ++n)
        {
            // n runs over the components of the input with their planes
            // innermost, so this iteration takes the velocity of plane
            // n % nhomo, and advVelOffset steps from there to the next
            // velocity component on the same plane.
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadOnly>() +
                (n % nhomo) * this->m_advVel->CompSize();
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        nqTot, (TData *)inptr);
                }

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, nqTot, (TData *)outptr);
                    }

                    // Advection kernel.
                    AdvectionKernelLauncher<SHAPE_TYPE, true, DEFORMED>(
                        sizeParam, m_D[ind0]..., m_f[ind1]...,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(advVelPtr),
                        reinterpret_cast<const simd_t *>(advVelPtr +
                                                         advVelOffset),
                        reinterpret_cast<const simd_t *>(advVelPtr +
                                                         2 * advVelOffset),
                        m_deriv[ind0].data()...,
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr), this->m_scale);
                }
                else
                {
                    // Advection kernel.
                    AdvectionKernelLauncher<SHAPE_TYPE, false, DEFORMED>(
                        sizeParam, m_D[ind0]..., m_f[ind1]...,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(advVelPtr),
                        reinterpret_cast<const simd_t *>(advVelPtr +
                                                         advVelOffset),
                        reinterpret_cast<const simd_t *>(advVelPtr +
                                                         2 * advVelOffset),
                        m_deriv[ind0].data()...,
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr), this->m_scale);
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                advVelPtr += nqTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
