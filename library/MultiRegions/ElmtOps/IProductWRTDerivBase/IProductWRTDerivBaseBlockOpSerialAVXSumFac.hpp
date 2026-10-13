///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseBlockOpSerialAVXSumFac.hpp
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
// the per-block inner product with the gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseBlockOpSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block inner product with the gradient of the basis.
 *
 * @details
 * Instead of one dense matrix per reference direction, the SumFac strategy
 * works through the tensor-product structure of the basis with the
 * one-dimensional basis, derivative and weight tables: a geometric kernel
 * (IProductWRTDerivBaseBlockOpSerialAVXSumFacKernels.hpp) forms one weighted
 * field per reference direction, the SumDerivTensor kernels
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp) apply the
 * transposed one-dimensional derivative matrices and sum the directions,
 * and, for a Coeff output, the metric-free IProductWRTBase kernels
 * (IProductWRTBaseBlockOpSerialAVXSumFacKernels.hpp) contract the result
 * against the basis. Each SIMD lane owns one element, so a group of
 * simd_t::width elements advances in lock-step (width one on the Serial
 * execution space).
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlockTwoOutStates.cpp.in, one
 * translation unit per shape holding both output states: each wraps
 * OperatorND() in a switch over the element sizes so that compile-time
 * constants reach the kernels, falling back to a runtime-sized call for
 * other sizes. The Coeff output uses the two-level switch over modes and
 * points (BlockOpSwitchCode.h.in), the Phys output the switch over the
 * coordinate dimension and points (BlockOpSwitch1DCoordsCode.h.in), as
 * the macros below select.
 *
 * This header provides the definition of
 * detail::IProductWRTDerivBaseBlockOpImpl for the Serial and AVX execution
 * spaces with the SumFac implementation tag; see
 * IProductWRTDerivBaseBlockOpSerialAVXStdMat.hpp for the StdMat sibling and the
 * note on the one-definition-per-translation-unit scheme.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; it should not normally be included by any other
 * translation unit. Other code goes through IProductWRTDerivBaseOp.hpp and the
 * operator factory: including this header directly instantiates the
 * whole template set in that translation unit, which is what the
 * generated sources exist to avoid.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOp.hpp>

#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseBlockOpSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions, one per output field state (see
// LibUtilities/BasicUtils/Switch/BlockOpShapeBlockTwoOutStates.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE_CoeffOut
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS_PhysOut

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised inner product with the gradient of the
 * basis: geometric stage, transposed tensor derivative and, for a Coeff
 * output, basis contraction, one element per SIMD lane.
 *
 * @details
 * The constructor collects, per reference direction, the element sizes and
 * the one-dimensional derivative and weight tables (and, for a Coeff
 * output, the basis tables), plus the
 * pointwise chain-rule factors (1 + eta)/2 and 2/(1 - eta) evaluated at
 * the quadrature points (data-warehouse keys eHalfMultOnePlusZero and
 * eTwoOverOneMinusZero), broadcast to the SIMD type. The factor tables are
 * fetched for every 2D block (two of them) and every 3D block (four of
 * them) whatever the shape, since the condition in the constructor is on
 * the element dimension alone; only the shapes with collapsed coordinates
 * use them, the kernels discarding the tables of quadrilaterals and
 * hexahedra through a compile-time shape test. For the nodal shapes with a
 * Coeff output it also fetches the transposed nodal-to-modal matrix, and
 * for every block the Jacobians and derivative factors. It then allocates
 * the workspaces: #m_df for the factors of the current point, for a Coeff
 * output #m_wsp for the IProductWRTBase kernels (sized by
 * IProduct2DWorkspace and IProduct3DWorkspace), and one plane of nqTot SIMD
 * vectors per reference direction (#m_tmp) plus one for their sum (#m_tmpsum).
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that makes one pass over the element
 * groups per output component and homogeneous mode and runs three stages
 * on every group:
 * -# IProductWRTDerivBaseKernelLauncher
 *    (IProductWRTDerivBaseBlockOpSerialAVXSumFacKernels.hpp) contracts the
 *    input's coordDim components against the derivative factors, applies
 *    the collapsed-coordinate corrections and the quadrature metric, and
 *    leaves one weighted field \f$g_d\f$ per reference direction in
 *    #m_tmp;
 * -# the same launcher applies the transposed one-dimensional derivative
 *    matrix of each direction with SumDerivTensorNDKernel and sums the
 *    directions into #m_tmpsum;
 * -# for a Phys output the sum is stored, scaled by #m_scale; for a Coeff
 *    output IProductWRTBaseKernelLauncher, in the variant that does not
 *    apply the quadrature metric (stage 1 already did), contracts it
 *    against the one-dimensional basis tables direction by direction and,
 *    for the nodal shapes, multiplies by the transposed nodal-to-modal
 *    matrix.
 *
 * For the collapsed shapes the correction factors act on the direction
 * fields before the transposed derivative, which is the transpose of
 * applying them after the derivative in the forward operator.
 *
 * @note On the nodal shapes (NodalTri, NodalTet, NodalPrism) the append
 * flag does not reach the output block on this path: stage 3 hands it to
 * the IProduct kernel, which accumulates into the workspace slice holding
 * the pre-nodal coefficients, and the MatVecKernel
 * (LibUtilities/BasicUtils/Utils/UtilsSerialAVXKernels.hpp) that then
 * applies the transposed nodal-to-modal matrix stores its result
 * unconditionally. See IProductWRTDerivBaseOp::SetAppend.
 *
 * @tparam ExecSpace      NektarSpaces::Serial or NektarSpaces::AVX; selects
 *                        scalar or SIMD arithmetic through simd_t.
 * @tparam Implementation Implementation tag (SumFac).
 * @tparam TFieldOut      Output field state (Coeff or Phys).
 * @tparam TData          Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, FieldState TFieldOut,
          typename TData>
class IProductWRTDerivBaseBlockOpImpl
    : public IProductWRTDerivBaseBlockOp<TFieldOut, TData>
{
    using BlockOpBase = IProductWRTDerivBaseBlockOp<TFieldOut, TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata, fetch the
     * one-dimensional tables, collapsed-coordinate factors,
     * nodal-to-modal matrix, Jacobians and derivative factors, and
     * allocate the workspaces (see the class description).
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    IProductWRTDerivBaseBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : IProductWRTDerivBaseBlockOp<TFieldOut, TData>(block_idx, exp,
                                                        dataWarehouse)
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

            // Fetch basis data. Only the coefficient-space output takes
            // the inner product with the basis.
            if constexpr (TFieldOut == FieldState::Coeff)
            {
                m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        exp->GetBasis(d)->GetBasisKey(),
                        LibUtilities::eBasis)));
            }
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(), LibUtilities::eWeights)));
        }

        if (m_dimension == 2)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            // Fetch geometric factors.
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        if (TFieldOut == FieldState::Coeff &&
            ((m_shapeType == LibUtilities::NodalTri) ||
             (m_shapeType == LibUtilities::NodalPrism) ||
             (m_shapeType == LibUtilities::NodalTet)))
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

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Workspace for kernels - also checks preconditions.
        m_df = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_dimension *
                                                                m_coordDim);
        // The inner product with the basis, taken only for the
        // coefficient-space output, needs its own workspaces.
        if constexpr (TFieldOut == FieldState::Coeff)
        {
            if (m_dimension == 2)
            {
                unsigned int wsp0Size = 0;
                IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0],
                                    m_nq[1], wsp0Size);
                m_wsp.push_back(
                    std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            }
            else if (m_dimension == 3)
            {
                unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
                IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2],
                                    m_nq[0], m_nq[1], m_nq[2], wsp0Size,
                                    wsp1Size, wsp2Size);
                m_wsp.push_back(
                    std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
                m_wsp.push_back(
                    std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
                m_wsp.push_back(
                    std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size));
            }
        }

        // One tensorial derivative per direction, plus the array holding
        // their sum.
        unsigned int nqTot = 1;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            nqTot *= m_nq[d];
        }
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_tmp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot));
        }
        m_tmpsum = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
    }

    /// Registration name for the block-operator factory, defined by the
    /// generated registration unit.
    static std::string className;

    /// @brief Creator function registered with the block-operator
    /// factory; builds one block operator for the given block of
    /// elements.
    static std::unique_ptr<ElmtBlockOp<FieldState::Phys, TFieldOut, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<IProductWRTDerivBaseBlockOpImpl<
            ExecSpace, Implementation, TFieldOut, TData>>(block_idx, exp,
                                                          dataWarehouse);
    }

protected:
    /// Element interleave width the kernels expect: the SIMD width (1 on
    /// the Serial execution space), one element per lane.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// Shape of the block's elements; drives the dispatch in v_Apply().
    LibUtilities::ShapeType m_shapeType;
    /// Whether the geometry is deformed (per-point Jacobian and derivative
    /// factors); selects the DEFORMED branch of the generated dispatch and
    /// hence of the kernels.
    bool m_isDeformed;
    /// Whether the direction-0 basis is of type eModified_A; enables the
    /// collapsed vertex-mode corrections in the IProductWRTBase kernels.
    bool m_isModified;
    /// Reference (shape) dimension of the elements (1, 2 or 3): the number
    /// of reference directions summed over.
    unsigned int m_dimension;
    /// Coordinate dimension of the (possibly embedded) elements: the number
    /// of input components consumed per output component.
    unsigned int m_coordDim;
    /// Modes per reference direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per reference direction.
    std::vector<unsigned int> m_nq;
    /// One-dimensional basis tables, one per direction (eBasis); fetched
    /// for a Coeff output only.
    std::vector<const TData *> m_B;
    /// One-dimensional derivative matrices, one per direction
    /// (eDerivative).
    std::vector<const TData *> m_D;
    /// One-dimensional quadrature weights, one per direction (eWeights).
    std::vector<const TData *> m_W;
    /// Collapsed-coordinate chain-rule factors (see the class
    /// description), broadcast to the SIMD type: in 2D (1 + eta_0)/2 and
    /// 2/(1 - eta_1); in 3D (1 + eta_0)/2, (1 + eta_1)/2, 2/(1 - eta_1)
    /// and 2/(1 - eta_2).
    std::vector<const simd_t *> m_f;
    /// Scratch holding the derivative factors of the point currently being
    /// processed (coordDim * dimension vectors).
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_df;
    /// Workspaces of the IProductWRTBase kernels, allocated for a Coeff
    /// output only: none in 1D, one in 2D, three in 3D (see
    /// NumWorkspace()), sized by IProduct2DWorkspace and
    /// IProduct3DWorkspace for the block's shape and element size.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// One plane of nqTot SIMD vectors per reference direction, receiving
    /// the weighted direction fields \f$g_d\f$ of one element group.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_tmp;
    /// Plane of nqTot SIMD vectors receiving the summed transposed
    /// derivative of one element group.
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_tmpsum;
    /// Transposed nodal-to-modal matrix of the nodal shapes for a Coeff
    /// output, broadcast to the SIMD type; null otherwise.
    const simd_t *m_nodToModTrans = nullptr;
    /// Jacobians of the block, interleaved at #m_implInterleaveWidth: one
    /// value per element, or one per quadrature point on a deformed block.
    const TData *m_jacptr;
    /// Derivative factors of the whole block, interleaved at the SIMD width
    /// and fetched untransposed: a deformed block stores the
    /// coordDim * dimension factors of a point together, point by point,
    /// while a regular block stores a single set per element group and has
    /// no point index at all (see DerivFactorKey).
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief In debug builds, warn once if the block storage is not
     * aligned for simd_t; then dispatch on the block's shape to the
     * generated ShapeBlock() specialisation. An unsupported shape is
     * reported on std::cout and the output block is left untouched.
     *
     * @param   inblock     Physical-space input block holding
     *                      \f${\bf f}\f$, coordDim components per output
     *                      component.
     * @param   outblock    Output block in state @p TFieldOut.
     */
    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock) override
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

    /// Shape specific block operator, specialised for each shape in
    /// LibUtilities/BasicUtils/Switch/BlockOpShapeBlockTwoOutStates.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /// Number of workspaces used by the kernels in dim dimensions.
    static constexpr unsigned int NumWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 3;
    }

    /// Number of collapsed coordinate factors used by the kernels in dim
    /// dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
    }

    /// Generic operator. Builds the index sequences from the shape dimension
    /// and forwards to OperatorNDImpl.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock,
        TSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switches in LibUtilities/BasicUtils/Switch:
        // BlockOpSwitchCode.h.in for a Coeff output,
        // BlockOpSwitch1DCoordsCode.h.in for a Phys output.
        if constexpr (TFieldOut == FieldState::Coeff)
        {
            static_assert(
                (DIM == 1 && IsSizeParameter1D_v<TSizeParameter>) ||
                    (DIM == 2 && IsSizeParameter2D_v<TSizeParameter>) ||
                    (DIM == 3 && IsSizeParameter3D_v<TSizeParameter>),
                "OperatorND expects a size parameter matching the dimension "
                "of the shape.");
        }
        else
        {
            static_assert(
                (DIM == 1 && IsPhysSizeParameter1D_v<TSizeParameter>) ||
                    (DIM == 2 && IsPhysSizeParameter2D_v<TSizeParameter>) ||
                    (DIM == 3 && IsPhysSizeParameter3D_v<TSizeParameter>),
                "OperatorND expects a size parameter matching the dimension "
                "of the shape.");
        }

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    /**
     * @brief Worker for every dimension: apply the operator to all element
     * groups of the block.
     *
     * @details
     * Makes one pass over the element groups per output component and
     * homogeneous mode. Every group is handed to
     * IProductWRTDerivBaseKernelLauncher, whose one-, two- and
     * three-dimensional overloads this one body calls; overload resolution
     * picks the arm from the number of arguments the index sequences
     * expand to. The launcher forms the weighted direction fields in
     * #m_tmp and their summed transposed derivative in #m_tmpsum. For a
     * Phys output that sum, scaled by #m_scale, is stored to (or, in
     * append mode, added onto) the output group; for a Coeff output
     * IProductWRTBaseKernelLauncher contracts it against the basis without
     * re-applying the quadrature metric, the append flag selecting its
     * accumulating variant. The Jacobian and derivative-factor pointers
     * advance by one group's values -- one Jacobian and ndim * coordDim
     * factors per element, times nqTot for deformed geometry -- and after
     * the last homogeneous mode of an output component the input pointer
     * skips the remaining input components of that output component (three
     * in all on a multi-plane 3DH1 expansion, coordDim otherwise).
     *
     * Fields stored at an interleave width larger than the implementation
     * width are handled by reshaping width_ratio consecutive groups in
     * place: the coordDim input components are converted to the
     * implementation width before the kernel, the output only in append
     * mode (when the kernels read it back), and both are converted back to
     * the input's width once the groups have been processed. The output
     * block's recorded interleave width is set to the input's on return.
     *
     * @tparam SHAPE_TYPE      Shape of the block, the nodal enumerators
     *                         included.
     * @tparam DEFORMED        Derivative factors vary point by point.
     * @tparam TSizeParameter  Size parameter of this dimension, templated
     *                         or not; modal for a Coeff output,
     *                         physical-space for a Phys output.
     * @tparam ind0            Reference directions, 0 to ndim - 1; selects
     *                         the tables #m_W, #m_D and #m_B and the planes
     *                         of #m_tmp handed to the launchers.
     * @tparam ind1            Workspaces #m_wsp the IProductWRTBase kernels
     *                         use: none in 1D, one in 2D, three in 3D (see
     *                         NumWorkspace()).
     * @tparam ind2            Collapsed-coordinate factor tables #m_f the
     *                         kernels use: none in 1D, two in 2D, four in
     *                         3D (see NumFactor()).
     *
     * @param   inblock     Physical-space input block.
     * @param   outblock    Output block in state @p TFieldOut.
     * @param   sizeParam   Element sizes, in runtime or compile-time form.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, TFieldOut> &outblock,
        TSizeParameter sizeParam, std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>)
    {
        constexpr unsigned int ndim = sizeof...(ind0);

        // Shape size.
        const auto nqTot = sizeParam.nqTot();

        const auto ndf       = ndim * m_coordDim;
        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Initialize pointers.
        auto inptr    = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr   = (this->m_append)
                            ? outblock.template GetPtr<MemSpace, ReadWrite>()
                            : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto outndata = outblock.GetNumData();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Directions held per variable in the input block. In 3DH1
        // (nhomo > 1) there are always three (x, y, z) regardless of the base
        // mesh coordDim, the z slot being the one DerivZOp reads; the xy pass
        // still consumes only m_coordDim of them.
        const auto inDim = (inblock.GetNumHomoModes() > 1) ? 3u : m_coordDim;

        // Loop over components.
        const auto inoffset = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto inoffset_vec = inoffset / simd_t::width;
        for (unsigned int n = 0;
             n < outblock.GetNumComponents() * outblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            nqTot, (TData *)(inptr + k * inoffset));
                    }
                }

                // IProductWRTDerivBase kernel.
                IProductWRTDerivBaseKernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_coordDim,
                    reinterpret_cast<const simd_t *>(dfptr), m_df.data(),
                    inoffset_vec, reinterpret_cast<const simd_t *>(inptr),
                    m_f[ind2]..., reinterpret_cast<const simd_t *>(jacptr),
                    m_W[ind0]..., m_D[ind0]..., m_tmp[ind0].data()...,
                    m_tmpsum.data());

                if (this->m_append)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, outndata, (TData *)outptr);
                    }
                }

                if constexpr (TFieldOut == FieldState::Phys)
                {
                    simd_t *vecout = reinterpret_cast<simd_t *>(outptr);
                    if (this->m_append)
                    {
                        for (unsigned int i = 0; i < nqTot; ++i)
                        {
                            vecout[i] += this->m_scale * m_tmpsum[i];
                        }
                    }
                    else
                    {
                        for (unsigned int i = 0; i < nqTot; ++i)
                        {
                            vecout[i] = this->m_scale * m_tmpsum[i];
                        }
                    }
                }
                else
                {
                    // Inner product without the quadrature metric wJ.
                    if (this->m_append)
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, true, true>(
                            sizeParam, m_isModified, m_tmpsum.data(),
                            m_B[ind0]..., m_nodToModTrans,
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                    else
                    {
                        IProductWRTBaseKernelLauncher<SHAPE_TYPE, true, false>(
                            sizeParam, m_isModified, m_tmpsum.data(),
                            m_B[ind0]..., m_nodToModTrans,
                            m_wsp[ind1].data()...,
                            reinterpret_cast<simd_t *>(outptr), this->m_scale);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    for (unsigned int k = 0; k < m_coordDim; ++k)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            (TData *)inptr + k * inoffset -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        outndata,
                        (TData *)outptr -
                            (width_ratio - 1) * outndata * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += outndata * simd_t::width;
                jacptr += jacSize * simd_t::width;
                dfptr += jacSize * ndf * simd_t::width;
            }

            // Advance input by inDim-1 components since have already
            // advanced one component in the above.
            if ((n + 1) % inblock.GetNumHomoModes() == 0)
            {
                inptr += (inDim - 1) * inoffset;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
