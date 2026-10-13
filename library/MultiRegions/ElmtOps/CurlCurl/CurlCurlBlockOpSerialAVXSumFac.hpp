///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlBlockOpSerialAVXSumFac.hpp
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
 * @file CurlCurlBlockOpSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised implementation of the curl-curl block
 * operator.
 *
 * @details
 * Instead of a dense standard-region matrix, the SumFac strategy
 * differentiates through the tensor-product structure of the basis: the
 * PhysDerivTensor{1,2,3}DKernel routines
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp) produce the
 * derivatives along the reference tensor directions from the
 * one-dimensional derivative matrices, and the Curl2DScalarSumFacKernel,
 * Curl2DVectorSumFacKernel and Curl3DSumFacKernel routines
 * (CurlCurlBlockOpSerialAVXSumFacKernels.hpp) apply the collapsed-to-standard
 * coordinate correction for simplex-type shapes and the geometric
 * derivative factors and take the curl in a single sweep. Both curls of
 * the operator, omega = curl u and out = curl omega, are evaluated by one
 * call of CurlCurlKernelLauncher per element group. A multi-plane 3DH1
 * block instead gets the plane part of a single curl on every plane, from
 * Curl3DH1KernelLauncher, which CurlCurlOpImpl completes with the z
 * parts. Each SIMD lane owns one element, so a group of simd_t::width
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
 * This header provides the definition of detail::CurlCurlBlockOpImpl for
 * the Serial and AVX execution spaces with the SumFac implementation tag;
 * see CurlCurlBlockOpSerialAVXStdMat.hpp for the StdMat sibling and the note on
 * the one-definition-per-translation-unit scheme.
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

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp>

#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlBlockOpSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised curl-curl block operator: tensor-product
 * differentiation followed by the fused curl kernels, one element per SIMD
 * lane.
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
 * kernel documentation in CurlCurlBlockOpSerialAVXSumFacKernels.hpp for which
 * factor serves which direction. The constructor also fetches the block's
 * derivative factors and allocates the workspace #m_wsp: NumWorkspace()
 * planes of nqTot SIMD vectors, the tensor-direction derivatives of every
 * component of a group at once plus one plane per component of the
 * vorticity omega (a scalar in 2D, a vector in 3D), so that each curl
 * kernel can consume the derivatives of all components in a single sweep.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that makes one pass over the element
 * groups and hands every group to CurlCurlKernelLauncher
 * (CurlCurlBlockOpSerialAVXSumFacKernels.hpp), which differentiates it with
 * PhysDerivTensor*DKernel into #m_wsp, forms omega there with the first
 * curl kernel, differentiates omega and writes curl(omega) to the output
 * block with the second. A multi-plane 3DH1 block on a 2D shape goes to
 * OperatorCurl3DH1() instead; on any other shape it is a fatal error.
 * Fields stored at a larger interleave width are reshaped in place around
 * the group loop, as in the StdMat implementation.
 *
 * @note The segment path (ShapeBlock<Seg>, the 1D CurlCurlKernelLauncher)
 * evaluates only the first derivative of a single component, the
 * curl-curl being undefined in one dimension; it exists so that the
 * generated per-shape translation units compile. The constructor asserts
 * that the block is two- or three-dimensional, but only with ASSERTL1,
 * which is compiled out of non-debug builds.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (SumFac).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using BlockOpBase = CurlCurlBlockOp<TData>;
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
     * The curl-curl operator is only defined for two- and
     * three-dimensional expansions whose coordinate dimension equals the
     * shape dimension; both are asserted here.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
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

        // Allocate workspace. The curl-curl operator holds the tensorial
        // derivatives of every component at once so that the chain rule and
        // the curl can be applied by a single sweep: that is dimension
        // arrays per component, plus one array per component of omega.
        unsigned int nqTot = m_nq[0];
        for (unsigned int d = 1; d < m_dimension; d++)
        {
            nqTot *= m_nq[d];
        }

        m_wsp = std::vector<simd_t, tinysimd::allocator<simd_t>>(
            NumWorkspace(m_dimension) * nqTot);
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
    /// Element interleave width the kernels expect: the SIMD width (1 on
    /// the Serial execution space).
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// description), broadcast to the SIMD type.
    std::vector<const simd_t *> m_f;
    /// Workspace: NumWorkspace() planes of nqTot SIMD vectors holding the
    /// tensor-direction derivatives of one element group and its
    /// vorticity (see the class description).
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp;
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
     * generated ShapeBlock() specialisation. An unlisted shape prints
     * a message and returns, leaving the output block unwritten.
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

    // Number of workspaces used by the kernels in dim dimensions: the dim
    // tensorial derivatives of each of the dim components, plus one array per
    // component of omega, which is a scalar in 2D and a vector in 3D.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return dim * dim + ((dim == 2) ? 1u : dim);
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
     * element groups of the block.
     *
     * @details
     * Hands a multi-plane 3DH1 block to OperatorCurl3DH1() when ndim is
     * two and raises a fatal error otherwise. Else checks that the input
     * block carries at least ndim components, then makes one pass over the
     * element groups of the first ndim components. Every group is handed
     * to CurlCurlKernelLauncher, whose one-, two- and three-dimensional
     * overloads this one body calls; overload resolution picks the arm
     * from the number of arguments the index sequences expand to. The
     * launcher forms the tensor-direction derivatives of the group's
     * components in #m_wsp, combines them into the vorticity there,
     * differentiates it again and writes curl(curl u) to the ndim output
     * components. The derivative-factor pointer advances by one group's
     * factors: ndim * #m_coordDim of them, times nqTot for deformed
     * geometry.
     *
     * Fields stored at an interleave width larger than the implementation
     * width are handled by reshaping width_ratio consecutive groups in
     * place: the ndim input components are converted to the
     * implementation width before the kernel, and both the input and the
     * output components are converted back to the input's width once the
     * groups have been processed. The output block's recorded interleave
     * width is set to the input's on return.
     *
     * @tparam SHAPE_TYPE          Shape of the block, the nodal
     *                             enumerators included.
     * @tparam DEFORMED            Derivative factors vary point by point.
     * @tparam TPhysSizeParameter  Size parameter of this dimension,
     *                             templated or not.
     * @tparam ind0                Reference directions, 0 to ndim - 1;
     *                             selects the derivative matrices #m_D and
     *                             the components handed to the launcher.
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
        // which is a different sweep over the same kernels.
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
                // all. The guard is L0: without it the sweep below would
                // ignore the extra planes and come back silently wrong in a
                // release build.
                NEKERROR(ErrorUtil::efatal,
                         "CurlCurlBlockOp supports homogeneous expansions "
                         "only on two-dimensional planes.");
            }
        }

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        ASSERTL1(inblock.GetNumComponents() >= ndim,
                 "Input block does not have enough components");

        unsigned int dfsize = ndim * m_coordDim;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Pointer offset between the components of the input and
        // output blocks.
        const auto inCompOffset  = inblock.CompSize();
        const auto outCompOffset = outblock.CompSize();

        // Initialize pointers.
        auto inbase  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outbase = outblock.template GetPtr<MemSpace, WriteOnly>();
        const TData *inptr[ndim];
        TData *outptr[ndim];
        for (unsigned int d = 0; d < ndim; ++d)
        {
            inptr[d]  = inbase + d * inCompOffset;
            outptr[d] = outbase + d * outCompOffset;
        }

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        auto dfptr = m_dfptr;

        // Loop over element groups.
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
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

            // CurlCurl kernel.
            CurlCurlKernelLauncher<SHAPE_TYPE, DEFORMED>(
                sizeParam, m_D[ind0]..., m_f[ind1]...,
                reinterpret_cast<const simd_t *>(dfptr), m_wsp.data(),
                reinterpret_cast<const simd_t *>(inptr[ind0])...,
                reinterpret_cast<simd_t *>(outptr[ind0])...);

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
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)outptr[d] -
                            (width_ratio - 1) * nqTot * simd_t::width);
                }
            }

            // Increment pointers for the next elmt group.
            dfptr += dfsize * simd_t::width;
            for (unsigned int d = 0; d < ndim; ++d)
            {
                inptr[d] += nqTot * simd_t::width;
                outptr[d] += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    /// \brief One curl's plane part on each plane of a 3DH1 block: three
    /// components in, three out.
    ///
    /// The z part is the operator's z-op and the double curl is the pair
    /// taken twice with the output carrying omega in between, so this is one
    /// curl's worth of plane derivatives and nothing more. Each plane holds the
    /// same geometry, so the derivative factors restart with every plane while
    /// the field pointers run on through the block.
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

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        unsigned int dfsize = 2u * 2u;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Offsets between the components of a block: one component spans all
        // of the planes.
        const auto nhomo         = inblock.GetNumHomoModes();
        const auto inCompOffset  = inblock.CompSize() * nhomo;
        const auto outCompOffset = outblock.CompSize() * nhomo;

        // Initialize pointers.
        auto inbase  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outbase = outblock.template GetPtr<MemSpace, WriteOnly>();
        const TData *inptr[nComp];
        TData *outptr[nComp];
        for (unsigned int c = 0; c < nComp; ++c)
        {
            inptr[c]  = inbase + c * inCompOffset;
            outptr[c] = outbase + c * outCompOffset;
        }

        // Loop over the planes. A component's planes are adjacent, so the
        // field pointers simply carry on from one to the next.
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
                    for (unsigned int c = 0; c < nComp; ++c)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, interleaveWidth, chunkSize,
                            nqTot, (TData *)inptr[c]);
                    }
                }

                // Plane part of one curl.
                Curl3DH1KernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_D[ind0]..., m_f[ind1]...,
                    reinterpret_cast<const simd_t *>(dfptr), m_wsp.data(),
                    reinterpret_cast<const simd_t *>(inptr[0]),
                    reinterpret_cast<const simd_t *>(inptr[1]),
                    reinterpret_cast<const simd_t *>(inptr[2]),
                    reinterpret_cast<simd_t *>(outptr[0]),
                    reinterpret_cast<simd_t *>(outptr[1]),
                    reinterpret_cast<simd_t *>(outptr[2]));

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int c = 0; c < nComp; ++c)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)inptr[c] -
                                (width_ratio - 1) * nqTot * simd_t::width);
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)outptr[c] -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                for (unsigned int c = 0; c < nComp; ++c)
                {
                    inptr[c] += nqTot * simd_t::width;
                    outptr[c] += nqTot * simd_t::width;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
