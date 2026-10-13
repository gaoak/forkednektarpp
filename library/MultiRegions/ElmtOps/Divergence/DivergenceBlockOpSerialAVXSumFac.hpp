///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceBlockOpSerialAVXSumFac.hpp
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
 * @file DivergenceBlockOpSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised implementation of the block
 * divergence.
 *
 * @details
 * One template serves both CPU execution spaces: `simd_t` resolves to
 * tinysimd::scalarT<TData> for Serial and tinysimd::simd<TData> for
 * AVX (see simd_type_if), so the same code runs one element at a time
 * or one SIMD group of elements in lock-step, with each vector lane
 * owning one element. Instead of the dense standard-region matrices
 * of the StdMat path, each vector component is differentiated through
 * the tensor-product structure of the basis by the PhysDeriv kernels,
 * which the DivergenceKernelLauncher overloads in
 * DivergenceBlockOpSerialAVXSumFacKernels.hpp chain component by component
 * for one element group.
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape: each wraps OperatorND() in a switch over
 * the element sizes so that compile-time constants reach the kernels,
 * falling back to a runtime-sized call for other sizes.
 *
 * This header provides the definition of detail::DivergenceBlockOpImpl
 * for the Serial and AVX execution spaces with the SumFac
 * implementation tag; see DivergenceBlockOpSerialAVXStdMat.hpp for the StdMat
 * sibling.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOp.hpp>

#include <MultiRegions/ElmtOps/Divergence/DivergenceBlockOpSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised block divergence: per SIMD group of
 * elements, differentiates each vector component in the reference
 * directions by 1D matrix contractions and accumulates the physical
 * derivatives \f$\partial u_c/\partial x_c\f$ into the output.
 *
 * @details
 * ### What one application does
 * For every group of `simd_t::width` elements, DivergenceKernelLauncher
 * runs the following for each component c of the vector field in turn:
 * -# PhysDerivTensor{1,2,3}DKernel applies the 1D derivative matrices
 *    (#m_D) direction by direction, producing the reference-direction
 *    derivatives of \f$u_c\f$ in the workspaces #m_wsp, one per
 *    direction;
 * -# PhysDerivDir{1,2,3}DKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR=c>
 *    corrects collapsed-coordinate derivatives to reference-direction
 *    ones where the shape requires it (using the factor tables #m_f)
 *    and contracts with row c of the derivative factors (#m_dfptr),
 *    i.e. the metric terms \f$\partial\xi_j/\partial x_c\f$.
 *    The c = 0 call overwrites the output; later calls set APPEND and
 *    accumulate, so the block's single output component ends up
 *    holding \f$\sum_c \partial u_c/\partial x_c\f$.
 *
 * The constructor caches the per-direction 1D derivative matrices and,
 * for 2D and 3D blocks, the factor tables, fetches the block's
 * derivative factors and allocates the workspaces. The generated
 * ShapeBlock() specialisations land in OperatorND(), which forwards to
 * OperatorNDImpl(): that loops over the homogeneous planes and, within
 * each, over the element groups, handing every group to the launcher.
 *
 * ### Interleaving
 * The kernels require element data interleaved at `simd_t::width`.
 * The block's storage may be interleaved at a different width;
 * OperatorNDImpl() therefore re-interleaves each chunk of `chunkSize`
 * elements in place -- casting away const on the input pointer --
 * before the kernels run, restores both sides afterwards, and resets
 * the output block's interleave width to the input's.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation Implementation tag (MultiRegions::SumFac).
 * @tparam TData          Floating-point type of the field data.
 *
 * @see DivergenceBlockOpSerialAVXStdMat.hpp for the matrix-based alternative.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using BlockOpBase = DivergenceBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Gather everything the kernels need from the block's
     * representative expansion and the data warehouse.
     *
     * Per reference direction this fetches the mode/point counts and
     * the 1D derivative matrix (eDerivative) evaluated at the
     * direction's quadrature points. For 2D and 3D shapes it also
     * fetches the collapsed-coordinate factor tables consumed by the
     * PhysDerivDir kernels for simplex-type shapes: per direction,
     * eHalfMultOnePlusZero holds \f$(1+z_i)/2\f$ and
     * eTwoOverOneMinusZero holds \f$2/(1-z_i)\f$ at the quadrature
     * points, stored as `simd_t` entries so one reference-space factor
     * applies across all lanes. The tables are fetched for every 2D
     * block (two of them) and every 3D block (four of them) whatever
     * the shape, since the condition is on the element dimension
     * alone. Finally it fetches the block's derivative factors
     * interleaved at the SIMD width and allocates one workspace of
     * `nqTot` vectors per reference direction.
     *
     * Asserts (debug builds) that the shape dimension equals the
     * coordinate dimension, i.e. embedded expansions are not
     * supported.
     *
     * @param   block_idx       Index of the block within the
     *                          expansion list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
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

        // Allocate workspace
        unsigned int nqTot = 1;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            nqTot *= m_nq[d];
        }
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot));
        }
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
    /// Interleave width the kernels operate at: the SIMD vector width
    /// (1 for the Serial space).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape type shared by every element of the block.
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
    /// 1D derivative matrix per reference direction (eDerivative).
    std::vector<const TData *> m_D;
    /// Collapsed-coordinate factor tables for simplex-type shapes
    /// (see the constructor notes); empty in 1D.
    std::vector<const simd_t *> m_f;
    /// Workspaces for the reference-direction derivatives of one
    /// element group: one vector of nqTot entries per reference
    /// direction.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// Derivative factors of the whole block, interleaved at the SIMD
    /// width: per element group, all factors of one point (deformed)
    /// or element (regular) are adjacent, lanes innermost.
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief Warn (once, in debug builds) if the fields are not
     * aligned for the SIMD type, then dispatch to the generated
     * ShapeBlock() specialisation matching #m_shapeType.
     *
     * @param   inblock     Physical-space input block; must supply at
     *                      least one component per shape dimension.
     * @param   outblock    Physical-space output block receiving the
     *                      divergence.
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
     * @brief Worker for every dimension: differentiate all element
     * groups of the block, one homogeneous plane at a time.
     *
     * @details
     * Asserts (debug builds) that the input carries at least ndim
     * components, then sets up one input pointer per component --
     * component d starting `CompSize() * GetNumHomoModes()` entries
     * after component d - 1 -- and runs one pass over the element
     * groups per homogeneous plane. Every group is handed to
     * DivergenceKernelLauncher, whose one-, two- and three-dimensional
     * overloads this one body calls; overload resolution picks the arm
     * from the number of arguments the index sequences expand to. The
     * launcher differentiates each of the ndim components into #m_wsp
     * and contracts it with its row of the derivative factors, the
     * first overwriting the output and the others accumulating. After
     * a group the derivative-factor pointer advances by one group's
     * factors -- ndim * coordDim of them, times nqTot for deformed
     * geometry -- and every data pointer by one group's values. The
     * derivative factors restart with every plane while the data
     * pointers run on, so pass p differentiates plane p of the first
     * ndim components into plane p of the output; further input
     * components are not read.
     *
     * Fields stored at an interleave width larger than the
     * implementation width are handled by reshaping width_ratio
     * consecutive groups in place: the ndim input components are
     * converted to the implementation width before the first group of
     * a chunk, and they and the output are converted back to the
     * input's width after its last group. The output block's recorded
     * interleave width is set to the input's on return.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by
     *                             point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the derivative matrices #m_D,
     *                             the workspaces #m_wsp and the input
     *                             components.
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
        constexpr unsigned int ndim = sizeof...(ind0);

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        ASSERTL1(inblock.GetNumComponents() >= ndim,
                 "Input block does not have enough components");

        unsigned int dfsize = ndim * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto nhomo      = inblock.GetNumHomoModes();
        const auto compOffset = inblock.CompSize() * nhomo;

        // Initialize pointers.
        auto inbase = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();
        const TData *inptr[ndim];
        for (unsigned int d = 0; d < ndim; ++d)
        {
            inptr[d] = inbase + d * compOffset;
        }

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over the planes. Each holds one xy divergence, over the same
        // geometry, so the derivative factors restart with every plane while
        // the field pointers run on through the block.
        for (unsigned int p = 0; p < nhomo; ++p)
        {
            auto dfptr = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int d = 0; d < ndim; ++d)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, interleaveWidth, chunkSize,
                            nqTot, (TData *)inptr[d]);
                    }
                }

                // Divergence kernel.
                DivergenceKernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_D[ind0]..., m_f[ind1]...,
                    reinterpret_cast<const simd_t *>(dfptr),
                    m_wsp[ind0].data()...,
                    reinterpret_cast<const simd_t *>(inptr[ind0])...,
                    reinterpret_cast<simd_t *>(outptr));

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int d = 0; d < ndim; ++d)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)inptr[d] -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                for (unsigned int d = 0; d < ndim; ++d)
                {
                    inptr[d] += nqTot * simd_t::width;
                }
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
