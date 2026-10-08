///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialAVXSumFac.hpp
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
// Description: Serial/AVX sum-factorised implementation of the per-block
// backward transform
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file BwdTransSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block backward transform.
 *
 * Sum factorisation replaces the dense nqTot x nmTot product of the
 * StdMat path by one contraction per direction, so a tensor-product
 * element costs \f$O(n^{d+1})\f$ rather than \f$O(n^{2d})\f$. The
 * per-direction contractions are the kernels in
 * BwdTransSerialAVXSumFacKernels.hpp; this header holds the block
 * operator that caches their inputs and drives them.
 *
 * Which shape a block holds is a runtime value, while the kernels are
 * templated on it, so v_Apply() switches on the shape into the
 * ShapeBlock() specialisations that CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape. NEKTAR_BLOCKOP_SWITCH_CODE, defined below,
 * selects the switch body those definitions expand.
 *
 * @see BwdTransSerialAVXStdMat.hpp for the dense-matrix alternative and
 * BwdTransDeviceSumFac.hpp for the device sum-factorised paths.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOp.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised block implementation of the backward
 * transform: one contraction per direction, vectorised across the
 * elements of a SIMD group.
 *
 * The constructor caches the per-direction 1D basis tables and, for the
 * nodal shapes, the nodal-to-modal matrix from the data warehouse, so
 * they are built once and shared between operators. It also sizes the
 * scratch buffers (#m_wsp) that hold the intermediates between
 * contraction stages: one in two dimensions, two in three, none
 * in one, where the single contraction writes straight to the output.
 *
 * Data is processed at #m_implInterleaveWidth, the SIMD width, with lane
 * j of a vector holding element j's value, so the vectorisation is
 * across the elements of a group. On the Serial execution space simd_t
 * is scalar and a group is a single element.
 *
 * The element geometry never enters: the backward transform is a
 * standard-region operation, so #m_coordDim is recorded only for
 * uniformity with the other element operators, and the two DEFORMED
 * instantiations the generated switch selects between compute the same
 * result.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if (scalar for
 *                         Serial, tinysimd::simd for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::SumFac here).
 * @tparam TData           Floating-point type of the field data.
 *
 * @see BwdTransDeviceSumFac.hpp for the device sum-factorised paths.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class BwdTransBlockOpImpl : public BwdTransBlockOp<TData>
{
    using BlockOpBase = BwdTransBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Cache the block's basis tables and nodal-to-modal matrix
     * and allocate the per-stage workspaces the kernels need.
     *
     * @param   block_idx       Index of the block within the expansion
     *                          list's Collections.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse the basis tables are
     *                          cached in.
     */
    BwdTransBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BwdTransBlockOp<TData>(block_idx, exp, dataWarehouse)
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
                StdRegions::StdMatKey<simd_t>(basisKeys, m_shapeType,
                                              StdRegions::eNodalToModal,
                                              nodalType));
        }
        else
        {
            m_nodToMod = (const simd_t *)nullptr;
        }

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            BwdTrans2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0;
            BwdTrans3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
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
    /// Interleave width the kernels operate at: the SIMD vector width
    /// (1 on Serial), one element per lane.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// 1D basis table per direction.
    std::vector<const TData *> m_B;
    /// Scratch buffers holding the intermediates between contraction
    /// stages; sized by the constructor, one per stage boundary.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// Nodal-to-modal matrix for the nodal shapes; null otherwise.
    const simd_t *m_nodToMod;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    /// Whether the alignment warning has already been issued; it is
    /// emitted once per block operator so that a misaligned field does
    /// not flood the log with one warning per Apply().
    bool m_warnOnce = false;
#endif

    /**
     * @brief Check the block's alignment, then dispatch on the runtime
     * shape to the generated per-shape entry point.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Physical-space output block; overwritten, or
     *                      accumulated into in append mode.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
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
     * @brief Number of scratch buffers the kernels need in @p dim
     * dimensions: one per contraction-stage boundary, so none in 1D.
     *
     * @param   dim     Dimension of the reference element.
     *
     * @return Number of entries of #m_wsp the kernels are handed.
     */
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 2;
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
     * case for, which lets the kernels fix their loop bounds at compile
     * time. Either form is accepted, and the static assertion below is
     * what ties whichever arrives to the dimension of @p SHAPE_TYPE.
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
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>());
    }

    /**
     * @brief Backward-transform every element of the block by sum
     * factorisation.
     *
     * The two index packs are what let one definition serve all three
     * dimensions: @p ind0 expands to one basis table per direction and
     * @p ind1 to one scratch buffer per contraction-stage boundary, so
     * the kernel call names exactly the arguments that dimension needs.
     *
     * The field may be interleaved at a width other than simd_t::width,
     * so work proceeds in chunks of `chunkSize = max(simd width, input
     * width)` elements per component: each chunk is reshaped in place to
     * simd_t::width interleaving, its `width_ratio` lane groups are run
     * through the kernel, and the chunk is reshaped back once its last
     * group is done. In append mode the output chunk is also reshaped
     * forward first, so the kernels accumulate in the layout they write.
     * On return the output block's interleave width is set to the
     * input's.
     *
     * @tparam SHAPE_TYPE      Shape the kernels are instantiated for.
     * @tparam DEFORMED        Deformed-geometry branch.
     * @tparam TSizeParameter  Size-parameter type, runtime or templated.
     * @tparam ind0            One index per direction.
     * @tparam ind1            One index per scratch buffer.
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
        // Shape size.
        const auto nmTot = sizeParam.nmTot();
        const auto nqTot = sizeParam.nqTot();

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
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        nmTot, (TData *)inptr);
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

                    // BwdTrans kernel.
                    BwdTransKernelLauncher<SHAPE_TYPE, true>(
                        sizeParam, m_isModified, m_B[ind0]..., m_nodToMod,
                        m_wsp[ind1].data()...,
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }
                else
                {
                    // BwdTrans kernel.
                    BwdTransKernelLauncher<SHAPE_TYPE, false>(
                        sizeParam, m_isModified, m_B[ind0]..., m_nodToMod,
                        m_wsp[ind1].data()...,
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr));
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
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
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
