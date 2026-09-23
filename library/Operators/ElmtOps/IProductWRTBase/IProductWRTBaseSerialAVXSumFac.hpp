///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialAVXSumFac.hpp
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
// Description: Serial/AVX sum-factorised (SumFac) implementation of
// the per-block inner product with the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block inner product with the basis.
 *
 * @details
 * Sum factorisation exploits the tensor-product structure of the expansion
 * basis: instead of multiplying by the dense nmTot x nqTot standard-region
 * matrix, the basis transpose is applied one coordinate direction at a
 * time through the one-dimensional basis tables, which reduces both the
 * operation count and the matrix data that must be streamed. The
 * quadrature metric is not a separate pass here: the per-direction weights
 * enter the contraction of their own direction and the Jacobian enters the
 * first (direction-0) contraction, both inside the kernels
 * (StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp, reached
 * through the launchers in IProductWRTBaseSerialAVXSumFacKernels.hpp).
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one translation
 * unit per shape: each wraps OperatorND() in the two-level switch over
 * modes and points (BlockOpSwitchCode.h.in, selected by the
 * NEKTAR_BLOCKOP_SWITCH_CODE macro below) so that compile-time element
 * sizes reach the kernels for equi-ordered blocks whose mode count lies in
 * the configured range and whose quadrature order follows the standard
 * relation to the mode order, falling back to a runtime-sized call, with a
 * one-time warning, for other sizes.
 *
 * Vectorisation is across elements: on the AVX execution space the element
 * data is interleaved in groups of simd_t::width elements, and every
 * arithmetic operation in the kernels acts on one SIMD vector holding the
 * same point or mode of width different elements. On Serial simd_t is
 * scalar, giving the one-element-at-a-time reference path from the same
 * source.
 *
 * This header provides the definition of detail::IProductWRTBaseBlockOpImpl
 * for the Serial and AVX execution spaces with the SumFac implementation
 * tag; see IProductWRTBaseSerialAVXStdMat.hpp for the StdMat sibling and
 * the note on the one-definition-per-translation-unit scheme, and
 * IProductWRTBaseDeviceSumFac.hpp for the device counterparts.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTBaseOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated per-shape sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOp.hpp"

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::Operators::detail
{

/**
 * @brief Serial/AVX SumFac inner product with the basis: one element per
 * SIMD lane, the basis transpose applied direction by direction.
 *
 * @details
 * The constructor caches, from the data warehouse, per coordinate
 * direction:
 * - the one-dimensional basis table #m_B (m_B[d][m * nq_d + i] is row m of
 *   direction d's table evaluated at point i; m is a direction-d mode
 *   number for the tensor-product directions, but along a collapsed
 *   direction it is the shape's combined mode index, the table then having
 *   correspondingly more than nm[d] rows);
 * - the one-dimensional quadrature weights #m_W (BasisDataKey with
 *   eWeights). For the collapsed eModified_B/eOrtho_B and
 *   eModified_C/eOrtho_C basis types this table is not the raw Gaussian
 *   weight: it already carries the weight factors of the collapsed
 *   coordinate map, either explicitly or through the Gauss-Radau-Jacobi
 *   rule (see BasisDataWarehouseDef.hpp), so the kernels can treat every
 *   direction alike.
 * For the nodal shapes (NodalTri, NodalTet, NodalPrism) it also fetches the
 * transposed nodal-to-modal matrix #m_nodToModTrans, through which the
 * kernel launchers map the modal result to nodal coefficients after the
 * tensor-product stages -- the transpose of what BwdTrans applies, and in
 * the opposite order. Finally it fetches the block's Jacobians, interleaved
 * at #m_implInterleaveWidth like the field data, and allocates the
 * persistent SIMD workspaces #m_wsp the kernels stage their directional
 * intermediate sums in, sized by IProduct2DWorkspace and
 * IProduct3DWorkspace.
 *
 * #m_isModified records whether the first-direction basis is of eModified_A
 * type; the collapsed-coordinate kernels (Tri, Tet, Prism, Pyr) then add
 * the extra contributions of the modes the modified basis shares between
 * mode columns (see the isModified blocks in the kernels).
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that makes one pass over the element groups
 * per component and homogeneous mode and hands every group to
 * IProductWRTBaseKernelLauncher.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (SumFac).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class IProductWRTBaseBlockOpImpl : public IProductWRTBaseBlockOp<TData>
{
    using BlockOpBase = IProductWRTBaseBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata, fetch the
     * one-dimensional tables, nodal-to-modal matrix and Jacobians, and
     * allocate the kernel workspaces (see the class description).
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    IProductWRTBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTBaseBlockOp<TData>(block_idx, exp, dataWarehouse)
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
            m_nodToModTrans = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(
                    basisKeys, m_shapeType, StdRegions::eNodalToModalTranspose,
                    nodalType));
        }
        else
        {
            m_nodToModTrans = (const simd_t *)nullptr;
        }

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
            IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size, wsp2Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size));
        }
    }

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            IProductWRTBaseBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Element interleave width the kernels expect: the SIMD width (1 on
    /// the Serial execution space), one element per lane. The Jacobians
    /// are fetched at this width too.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// One-dimensional basis tables, one per direction (eBasis):
    /// m_B[d][m * nq_d + i] is table row m at point i, m being a
    /// direction-d mode number along a tensor-product direction and the
    /// shape's combined mode index along a collapsed one (see the class
    /// description).
    std::vector<const TData *> m_B;
    /// One-dimensional quadrature weights, one per direction (eWeights),
    /// collapsed-coordinate factors included (see the class description).
    std::vector<const TData *> m_W;
    /// Kernel workspaces in SIMD vectors (one value per lane): none in 1D,
    /// one in 2D, three in 3D (see NumWorkspace()), sized by
    /// IProduct2DWorkspace and IProduct3DWorkspace for the block's shape
    /// and element size.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// Transposed nodal-to-modal matrix of the nodal shapes, broadcast to
    /// the SIMD type; null otherwise.
    const simd_t *m_nodToModTrans;
    /// Jacobians of the block, interleaved at #m_implInterleaveWidth: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief In debug builds, warn once if the block storage is not
     * aligned for simd_t; then dispatch on the block's shape to the
     * generated ShapeBlock() specialisation. An unsupported shape
     * prints a message and leaves @p outblock untouched.
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Coefficient-space output block.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
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

    // Number of workspaces used by the kernels in dim dimensions.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 3;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
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
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>());
    }

    /**
     * @brief Worker for every dimension: apply the operator to all element
     * groups of the block.
     *
     * @details
     * Makes one pass over the element groups per component and homogeneous
     * mode. Every group is handed to IProductWRTBaseKernelLauncher, whose
     * one-, two- and three-dimensional overloads this one body calls;
     * overload resolution picks the arm from the type of the size parameter
     * and the number of arguments the index sequences expand to. Which of
     * the four instantiations runs is decided per call from the two
     * settings of IProductWRTBaseBlockOp: the quadrature-metric flag picks
     * the overload taking the weights and the Jacobians or the one without
     * them, and a scale factor of exactly 1 picks the SCALE = false
     * variant, which drops the final multiplication. APPEND is set to false:
     * this family never accumulates. The Jacobian pointer advances by one
     * vector per group for a regular geometry and by nqTot vectors for a
     * deformed one, and is rewound for every component.
     *
     * Fields stored at an interleave width larger than the SIMD width are
     * handled by reshaping width_ratio consecutive groups in place: the
     * input is converted to the implementation width before the kernel and
     * both input and output are converted back to the input's width once
     * the groups have been processed. The output block's recorded
     * interleave width is set to the input's on return.
     *
     * @tparam SHAPE_TYPE      Shape of the block, the nodal enumerators
     *                         included.
     * @tparam DEFORMED        Jacobians vary point by point.
     * @tparam TSizeParameter  Size parameter of this dimension, templated
     *                         or not.
     * @tparam ind0            Reference directions, 0 to ndim - 1; selects
     *                         the tables #m_B and #m_W.
     * @tparam ind1            Workspaces #m_wsp the kernels use: none in
     *                         1D, one in 2D, three in 3D (see
     *                         NumWorkspace()).
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Coefficient-space output block.
     * @param   sizeParam   Element sizes, in runtime or compile-time form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        // Shape size.
        const auto nmTot = sizeParam.nmTot();
        const auto nqTot = sizeParam.nqTot();

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

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
            auto jacptr = m_jacptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, interleaveWidth, chunkSize,
                        nqTot, (TData *)inptr);
                }

                // IProduct kernel.
                if (this->m_integration)
                {
                    if (this->m_scale == 1.0)
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false,
                                                      DEFORMED>(
                            sizeParam, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr),
                            m_B[ind0]..., m_W[ind0]..., m_nodToModTrans,
                            reinterpret_cast<const simd_t *>(jacptr),
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, true, false,
                                                      DEFORMED>(
                            sizeParam, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr),
                            m_B[ind0]..., m_W[ind0]..., m_nodToModTrans,
                            reinterpret_cast<const simd_t *>(jacptr),
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }
                else
                {
                    // IProduct kernel with no quadrature B^T op.
                    if (this->m_scale == 1.0)
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
                            sizeParam, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr),
                            m_B[ind0]..., m_nodToModTrans,
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr));
                    }
                    else
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, true, false>(
                            sizeParam, m_isModified,
                            reinterpret_cast<const simd_t *>(inptr),
                            m_B[ind0]..., m_nodToModTrans,
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += nmTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
