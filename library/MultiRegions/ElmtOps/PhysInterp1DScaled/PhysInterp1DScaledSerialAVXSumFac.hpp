///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledSerialAVXSumFac.hpp
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
// Description: interp in physical space by a scaled number of points
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysInterp1DScaledSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block scaled physical-space interpolation.
 *
 * @details
 * Sum factorisation exploits the tensor-product structure of the point
 * grid: instead of multiplying by the dense nqTot x nmTot matrix, the
 * interpolation is applied one coordinate direction at a time through the
 * 1D interpolation matrices, which reduces both the operation count and
 * the matrix data that must be streamed.
 *
 * ### Reuse of the backward-transform kernels
 * A 1D interpolation matrix has exactly the layout of a 1D basis table,
 * `I[p * nq + i]` being the Lagrange interpolant of input point p
 * evaluated at output point i, so the interpolation is a backward
 * transform in which the input points play the role of the modes. This
 * header therefore includes BwdTransSerialAVXSumFacKernels.hpp and calls
 * its BwdTransKernelLauncher overloads directly; the family has no
 * Serial/AVX kernels header of its own.
 *
 * Only the tensor-product kernels are used -- BwdTransSegKernel in 1D,
 * BwdTransQuadKernel in 2D and BwdTransHexKernel in 3D -- whatever the
 * block's shape, because the physical points of a triangle, tetrahedron,
 * prism or pyramid form a tensor-product grid in the collapsed
 * coordinates. The generated size switch (below) makes that choice: it
 * instantiates OperatorND() for LibUtilities::Seg, LibUtilities::Quad or
 * LibUtilities::Hex according to the block's dimension, never for the
 * block's own shape, so the launcher never selects a collapsed-coordinate
 * kernel, no nodal-to-modal matrix is fetched (the launcher receives a
 * null one) and the m_isModified flag it is handed is not read by the
 * kernels it selects.
 *
 * ### How the pieces fit together
 * v_Apply() dispatches on the block's shape to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape: each wraps OperatorND() in the switch of
 * LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysInterp1D.h.in, selected
 * by the NEKTAR_BLOCKOP_SWITCH_PHYSINTERP1D macro this header defines.
 * That switch takes the size-templated overload -- the sizes become
 * compile-time constants and the kernel loops unroll -- only for blocks
 * whose input and output point counts follow the shape's standard
 * relation (equal in every direction for segment, quadrilateral and
 * hexahedron; one fewer in each collapsed direction for the others) and
 * whose direction-0 counts fall inside the generated ranges: the input
 * count within NEKTAR_SWITCH_MIN to NEKTAR_SWITCH_MAX and the output
 * count from one above the input count up to NEKTAR_SWITCH_QMAX_MUL times
 * it plus NEKTAR_SWITCH_QMAX_ADD. An output count equal to the input
 * count, i.e. a scale factor of one, is not among the generated cases.
 * Everything else falls back to the runtime-sized overload, with a
 * one-off warning.
 *
 * @note The switch tests the output counts it recomputes itself, as the
 * truncation of `scale * nm_d` in every direction, rather than reading
 * m_nq, whose offset rule gives a direction one point shorter than
 * direction 0 the count `nq0 - 1` instead. For such a direction the two
 * agree only when the truncated products happen to differ by exactly
 * one, so a collapsed-shape block whose m_nq would satisfy the standard
 * relation may still take the runtime-sized overload. The sizes actually
 * interpolated onto are m_nq in either case: the templated overloads
 * encode the offset relation directly and the runtime-sized one is
 * handed m_nq.
 *
 * Vectorisation is across elements: on the AVX execution space the
 * element data is interleaved in groups of simd_t::width elements, and
 * every arithmetic operation in the kernels acts on one SIMD vector
 * holding the same point of `width` different elements. On Serial simd_t
 * is scalar, giving the one-element-at-a-time reference path from the
 * same source.
 *
 * PhysGalerkinProject1DScaledOp, the projection back from the scaled
 * grid, shares this switch and these kernels with the roles of the two
 * grids exchanged.
 *
 * @see PhysInterp1DScaledSerialAVXStdMat.hpp for the dense-matrix
 * alternative and PhysInterp1DScaledDeviceSumFac.hpp for the device
 * counterparts.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/PhysInterp1DScaled/PhysInterp1DScaledBlockOp.hpp>

// interpolation is just a bwd trans from a nodal basis so using these kernels
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSINTERP1D

namespace Nektar::MultiRegions::detail
{

/**
 * @brief SumFac block implementation of the scaled interpolation for the
 * Serial and AVX execution spaces.
 *
 * @details
 * The constructor records only the block's shape and its input point
 * counts #m_nm; the interpolation data depends on the scale factor and so
 * is fetched by v_SetScaleFactor(), which caches one 1D interpolation
 * matrix per direction (LibUtilities::BasisDataKey with
 * LibUtilities::eInterp and the direction's output point count) and
 * allocates the persistent SIMD workspaces #m_wsp the kernels stage their
 * directional intermediate sums in.
 *
 * As in the StdMat implementation, the element geometry never enters.
 * #m_isDeformed is nevertheless recorded, for uniformity with the other
 * element-operator families; unlike them this family's generated dispatch
 * does not branch on it, the OperatorND() overloads here carrying no
 * DEFORMED template parameter.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that makes one pass over the element
 * groups per component and homogeneous mode and hands every group to
 * BwdTransKernelLauncher.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if (scalar for
 *                         Serial, tinysimd::simd for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::SumFac here).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysInterp1DScaledBlockOpImpl : public PhysInterp1DScaledBlockOp<TData>
{
    using BlockOpBase = PhysInterp1DScaledBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata; the interpolation
     * matrices and workspaces are fetched and sized by
     * v_SetScaleFactor().
     *
     * Records the shape, geometry type, reference and coordinate
     * dimension, whether the first-direction basis is of eModified_A
     * type and the input point count of every direction.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
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
    /// Interleave width the kernels operate at: the SIMD vector width (1
    /// on Serial), one element per lane.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape shared by every element of the block.
    LibUtilities::ShapeType m_shapeType;
    /// Whether the block's geometry is deformed; unused here (see the
    /// class notes).
    bool m_isDeformed;
    /// First-direction basis is eModified_A; handed to the kernel
    /// launcher, which does not read it for the tensor-product kernels
    /// this operator selects.
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
    /// 1D interpolation matrix per direction: `m_B[d][p * m_nq[d] + i]`
    /// is the interpolant of input point p of direction d evaluated at
    /// its output point i. Cached in the data warehouse; set by
    /// v_SetScaleFactor.
    std::vector<const TData *> m_B;
    /// Kernel workspaces in SIMD vectors (one value per lane): none in
    /// 1D, one in 2D, two in 3D (see NumWorkspace()), sized by the
    /// BwdTrans workspace queries for the quadrilateral and hexahedron
    /// with the input point counts as the mode counts; set by
    /// v_SetScaleFactor.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief Check the preconditions and dispatch to the generated
     * per-shape entry point.
     *
     * Warns (once per block operator in debug builds) if either block's
     * storage is not aligned for the SIMD vector type, asserts that a
     * scale factor has been set, and then switches on the block's shape
     * to the ShapeBlock() specialisation.
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
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() % simd_t::alignment == 0 &&
                           outblock.GetAlignment() % simd_t::alignment == 0),
            "Input or output Field are not aligned to the required alignment "
            "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

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
     * @brief Fix the output point counts for @p scale, fetch the
     * per-direction interpolation matrices and size the workspaces.
     *
     * Direction 0 gets `nq0 = (unsigned int)(scale * nm0)`; a direction
     * whose input point count is exactly one below direction 0's keeps
     * that offset, `nqd = (unsigned int)(scale * nm0) - 1`, and any other
     * direction gets `nqd = (unsigned int)(scale * nmd)`. That is the
     * rule of MultiRegions::ExpList::Get1DScaledTotPoints, so the output
     * sizes agree with a field sized by it. Each direction's
     * interpolation matrix is then fetched under a BasisDataKey carrying
     * that direction's output count, so one cached entry is built per
     * basis and target size; the target points keep the basis's own
     * points type.
     *
     * The workspaces are sized for the kernels actually used -- those of
     * the quadrilateral in two directions and of the hexahedron in three
     * -- with the input point counts standing in for the mode counts:
     * `nm1 * nq0` vectors in 2D, `nq0 * nm1 * nm2` and `nq0 * nq1 * nm2`
     * in 3D. The 1D call has no workspace to size: BwdTrans1DWorkspace is
     * a no-op kept for uniformity with the 2D/3D forms.
     *
     * The counts, matrices and workspaces are all cleared and rebuilt
     * from the stored input counts, so a block operator may be
     * re-targeted at a new scale factor by calling this again.
     *
     * @param   scale   Multiplier applied to the per-direction
     *                  quadrature-point counts.
     */
    void v_SetScaleFactor(const TData &scale) override
    {
        this->m_scale = scale;
        m_nq          = this->GetScaledNumPoints(m_nm, this->m_scale);
        m_B.clear();
        m_wsp.clear();
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    this->m_exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eInterp, m_nq[d])));
        }

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 1)
        {
            BwdTrans1DWorkspace(LibUtilities::Seg, m_nm[0], m_nq[0]);
        }
        else if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            BwdTrans2DWorkspace(LibUtilities::Quad, m_nm[0], m_nm[1], m_nq[0],
                                m_nq[1], wsp0Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0;
            BwdTrans3DWorkspace(LibUtilities::Hex, m_nm[0], m_nm[1], m_nm[2],
                                m_nq[0], m_nq[1], m_nq[2], wsp0Size, wsp1Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of workspaces used by the kernels in dim dimensions.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 2;
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
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>());
    }

    /**
     * @brief Worker for every dimension: interpolate all element groups
     * of the block.
     *
     * @details
     * Makes one pass over the element groups per component and
     * homogeneous mode. Every group is handed to BwdTransKernelLauncher,
     * whose one-, two- and three-dimensional overloads this one body
     * calls; overload resolution picks the arm from the type of the size
     * parameter and the number of arguments the index sequences expand
     * to. The launcher, instantiated for the segment, quadrilateral or
     * hexahedron (see the file notes) with APPEND false, runs
     * BwdTransSegKernel, BwdTransQuadKernel or BwdTransHexKernel with the
     * interpolation matrices #m_B as its basis tables -- contracting
     * direction 0 into the first workspace, direction 1 from there into
     * the output or the second workspace, and direction 2 into the output
     * -- so the output is overwritten. The nodal-to-modal matrix it takes
     * is passed as null and #m_isModified is not read by these kernels.
     *
     * Fields stored at an interleave width larger than the SIMD width
     * are handled by reshaping `width_ratio` consecutive groups in
     * place: the input is converted to the implementation width before
     * the kernel and both input and output are converted back to the
     * input's width once the groups have been processed. The output
     * block's recorded interleave width is set to the input's on return.
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
     * @tparam ind1            Workspaces #m_wsp the kernels use: none in
     *                         1D, one in 2D, two in 3D (see
     *                         NumWorkspace()).
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
        // Shape size.
        const auto nmTot = sizeParam.nmTot();
        const auto nqTot = sizeParam.nqTot();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransKernelLauncher<SHAPE_TYPE, false>(
                    sizeParam, m_isModified, m_B[ind0]...,
                    (const simd_t *)nullptr, m_wsp[ind1].data()...,
                    reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr));

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nmTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
