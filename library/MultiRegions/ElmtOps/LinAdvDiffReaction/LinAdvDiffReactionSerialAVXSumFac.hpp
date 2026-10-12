///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionSerialAVXSumFac.hpp
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
// Description: Serial/AVX sum-factorised implementation of the linearised
// advection-diffusion-reaction block operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LinAdvDiffReactionSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised (SumFac) implementation of the
 * per-block linearised advection-diffusion-reaction operator.
 *
 * @details
 * Instead of the dense standard-region matrices of the StdMat path, the
 * operator is assembled from the one-dimensional tensor-product stages
 * of the simpler families -- the backward transform
 * (BwdTransKernelLauncher), the tensor derivative
 * (PhysDerivTensor{1,2,3}DKernel), the diffusion metric of the Helmholtz
 * family (DiffusionCoeffwithWJ{1,2,3}DKernel), the summed transposed
 * derivative (SumDerivTensor{1,2,3}DKernel) and the inner product
 * (IProductWRTBaseKernelLauncher) -- with the advection term supplied by
 * this family's own AddAdvection{1,2,3}DKernel. The
 * LinAdvDiffReactionKernelLauncher overloads in
 * LinAdvDiffReactionSerialAVXSumFacKernels.hpp string the stages
 * together for one element group; each SIMD lane owns one element, so a
 * group of simd_t::width elements advances in lock-step (width one on
 * the Serial execution space).
 *
 * v_Apply() dispatches on the block's shape type to the ShapeBlock()
 * specialisations, whose definitions CMake generates from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, one
 * translation unit per shape: each wraps OperatorND() in a switch over
 * the element sizes so that compile-time constants reach the kernels,
 * falling back to a runtime-sized call for other sizes.
 *
 * This header provides the definition of
 * detail::LinAdvDiffReactionBlockOpImpl for the Serial and AVX execution
 * spaces with the SumFac implementation tag; see
 * LinAdvDiffReactionSerialAVXStdMat.hpp for the StdMat sibling and the
 * note on the one-definition-per-translation-unit scheme.
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

#include <MultiRegions/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionSerialAVXSumFacKernels.hpp>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_CODE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial/AVX sum-factorised block implementation of the
 * linearised advection-diffusion-reaction operator, one element per SIMD
 * lane.
 *
 * @details
 * The constructor caches, per direction, the 1D basis table, the 1D
 * collocation derivative matrix and the 1D quadrature weights, plus the
 * collapsed-coordinate factors #m_f, the nodal-to-modal matrices of the
 * nodal shapes, and the block's Jacobians and derivative factors -- all
 * from the data warehouse, so they are built once and shared between
 * operators. The factor tables are fetched for every 2D block (two of
 * them) and every 3D block (four of them) whatever the shape, since the
 * condition in the constructor is on the element dimension alone; only
 * the shapes with collapsed coordinates use them. The constructor also
 * sizes the workspaces #m_wsp the shape kernels need, taking the running
 * maximum over the backward-transform and inner-product requirements
 * (BwdTrans{2,3}DWorkspace, IProduct{2,3}DWorkspace), and allocates the
 * quadrature-space buffers #m_bwd and #m_deriv that carry the
 * intermediate fields from stage to stage.
 *
 * The generated ShapeBlock() specialisations land in OperatorND(), which
 * forwards to OperatorNDImpl(): that loops over components and
 * homogeneous modes and, within each, over element groups, handing every
 * group to LinAdvDiffReactionKernelLauncher. The derivative stage there
 * leaves the derivatives in reference (collapsed) space; the advection
 * kernel and the Helmholtz diffusion kernel each apply the chain rule to
 * Cartesian space themselves, which differs from the device
 * sum-factorised paths, whose derivative stage already yields Cartesian
 * derivatives.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX;
 *                         sets simd_t via simd_type_if (scalar for
 *                         Serial, tinysimd::simd for AVX).
 * @tparam Implementation  Implementation tag the including translation
 *                         unit registers (MultiRegions::SumFac here).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class LinAdvDiffReactionBlockOpImpl : public LinAdvDiffReactionBlockOp<TData>
{
    using BlockOpBase = LinAdvDiffReactionBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Capture the block's element metadata, fetch the 1D basis,
     * derivative and weight tables, the collapsed-coordinate factors,
     * the nodal-to-modal matrices, the Jacobians and the derivative
     * factors, and size the workspaces (see the class description).
     *
     * @param   block_idx       Index of the block.
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
            m_nodToModTrans = dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(
                    basisKeys, m_shapeType, StdRegions::eNodalToModalTranspose,
                    nodalType));
        }
        else
        {
            m_nodToMod      = (const simd_t *)nullptr;
            m_nodToModTrans = (const simd_t *)nullptr;
        }

        // Fetch Jacobian and deriv factors.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Workspace for kernels - also checks preconditions.
        if (m_dimension == 2)
        {
            unsigned int wsp0Size = 0;
            BwdTrans2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            IProduct2DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nq[0], m_nq[1],
                                wsp0Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
        }
        else if (m_dimension == 3)
        {
            unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
            BwdTrans3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size);
            IProduct3DWorkspace(m_shapeType, m_nm[0], m_nm[1], m_nm[2], m_nq[0],
                                m_nq[1], m_nq[2], wsp0Size, wsp1Size, wsp2Size);
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size));
            m_wsp.push_back(
                std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp2Size));
        }

        // One tensorial derivative per direction, plus the backward
        // transform of the input.
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
        m_bwd = std::vector<simd_t, tinysimd::allocator<simd_t>>(nqTot);
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
    /// Interleave width the kernels operate at: the SIMD vector width
    /// (1 on Serial), one element per lane.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
    /// diffusivity tensor.
    unsigned int m_coordDim;
    /// Modes per direction.
    std::vector<unsigned int> m_nm;
    /// Quadrature points per direction.
    std::vector<unsigned int> m_nq;
    /// 1D basis table per direction.
    std::vector<const TData *> m_B;
    /// 1D collocation derivative matrix per direction.
    std::vector<const TData *> m_D;
    /// 1D quadrature weights per direction.
    std::vector<const TData *> m_W;
    /// Collapsed-coordinate factors of the non-tensor shapes, tabulated
    /// as SIMD vectors in the order the kernels expect them: in 2D
    /// (1 + eta)/2 of direction 0 and 2/(1 - eta) of direction 1; in 3D
    /// (1 + eta)/2 of directions 0 and 1, then 2/(1 - eta) of
    /// directions 1 and 2. Empty for 1D blocks.
    std::vector<const simd_t *> m_f;
    /// Shape-kernel workspaces, one in 2D and three in 3D (see
    /// NumWorkspace()), each sized at construction to the maximum the
    /// backward-transform and inner-product stages need.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_wsp;
    /// Quadrature-space derivative buffers, one per reference direction
    /// and nqTot vectors long: \f$\partial u/\partial\xi_d\f$ first,
    /// then the diffusion flux with the metric and the quadrature
    /// weighting applied.
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_deriv;
    /// Quadrature-space buffer of nqTot vectors: first the
    /// backward-transformed values \f$u\f$, then the
    /// advection-plus-reaction integrand, and finally the whole
    /// integrand once the diffusion flux has been summed onto it.
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_bwd;
    /// Empty vector passed for the varying-diffusivity arguments of the
    /// Helmholtz diffusion kernels, which this family never uses.
    std::vector<TData> NullTDataVector;
    /// Nodal-to-modal matrix for the nodal shapes; null otherwise.
    const simd_t *m_nodToMod;
    /// Its transpose, used to map the inner-product result back to the
    /// nodal coefficients; null for the non-nodal shapes.
    const simd_t *m_nodToModTrans;
    /// Jacobians of the block's elements, interleaved at
    /// #m_implInterleaveWidth: one value per element when regular, one
    /// per quadrature point when deformed.
    const TData *m_jacptr;
    /// Derivative factors \f$\partial\xi_d/\partial x_k\f$ of the
    /// block's elements, interleaved at #m_implInterleaveWidth in
    /// point-major order (DerivFactorKey with transpose = false):
    /// element/point stride coordDim * dimension, within which the
    /// entry for coordinate k and reference direction d is at
    /// k * dimension + d.
    const TData *m_dfptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    /**
     * @brief In debug builds, warn once if the block storage is not
     * aligned for simd_t; then dispatch on the block's shape to the
     * generated ShapeBlock() specialisation. An unsupported shape
     * prints a message and returns with @p outblock untouched.
     *
     * @param   inblock     Coefficient-space input block.
     * @param   outblock    Coefficient-space output block, overwritten.
     */
    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        // Check alignment.
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
            std::make_integer_sequence<unsigned int, NumWorkspace(DIM)>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    /**
     * @brief Worker for every dimension: apply the operator to all
     * element groups of the block, component by component.
     *
     * @details
     * Reshapes the advection velocity to #m_implInterleaveWidth if it is
     * still held at another width, then loops over the components and
     * homogeneous modes of the block and, within each, over the element
     * groups. Every group is handed to LinAdvDiffReactionKernelLauncher,
     * whose one-, two- and three-dimensional overloads this one body
     * calls; overload resolution picks the arm from the number of
     * arguments the index sequences expand to. The launcher runs the
     * six stages -- backward transform, tensor derivatives, advection
     * and reaction, diffusion metric with quadrature weighting, summed
     * transposed derivative, inner product -- on the buffers #m_bwd,
     * #m_deriv and #m_wsp.
     *
     * The Jacobian and derivative-factor pointers restart at the
     * beginning of the block for every component and homogeneous mode,
     * so the same geometry serves all of them; the velocity pointer
     * starts each at its plane, `n % nhomo`, so the variables share the
     * velocity of their plane. The three velocity components are passed
     * separately, offset from each other by one component's worth of the
     * block (nelmt * nqTot * nhomo values) whatever the coordinate
     * dimension; a 1D or 2D kernel reads only the components its
     * coordinate dimension has. The derivative-factor pointer advances by
     * one group's factors -- coordDim * ndim of them, times nqTot for
     * deformed geometry -- and the Jacobian pointer by one group's values,
     * nqTot of them for deformed geometry.
     *
     * Fields stored at an interleave width larger than the implementation
     * width are handled by reshaping width_ratio consecutive groups in
     * place: the input is converted to the implementation width before
     * the kernel, and both input and output are converted back to the
     * field's width once the chunk's last group has been processed. The
     * output block's recorded interleave width is set to the input's on
     * return.
     *
     * @tparam SHAPE_TYPE      Shape of the block, the nodal enumerators
     *                         included.
     * @tparam DEFORMED        Jacobian and derivative factors vary point
     *                         by point.
     * @tparam TSizeParameter  Size parameter of this dimension,
     *                         templated or not.
     * @tparam ind0            Reference directions, 0 to ndim - 1;
     *                         selects the tables #m_B, #m_D, #m_W and the
     *                         buffers #m_deriv.
     * @tparam ind1            Shape-kernel workspaces #m_wsp: none in 1D,
     *                         one in 2D, three in 3D (see
     *                         NumWorkspace()).
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
        const auto nmTot = sizeParam.nmTot();
        const auto nqTot = sizeParam.nqTot();

        const unsigned int ndf = ndim * m_coordDim;
        unsigned int dfsize    = 1;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        const auto nhomo  = inblock.GetNumHomoModes();
        auto advVelOffset = nelmt * nqTot * nhomo;
        for (unsigned int n = 0; n < inblock.GetNumComponents() * nhomo; ++n)
        {
            // The advection velocity is shared by the variables but not by
            // the planes: the plane this component sits on is n % nhomo, and
            // advVelOffset steps from there to the next velocity component.
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadOnly>() +
                (n % nhomo) * this->m_advVel->CompSize();
            auto jacptr = m_jacptr;
            auto dfptr  = m_dfptr;

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

                // LinAdvDiffReaction kernel.
                LinAdvDiffReactionKernelLauncher<SHAPE_TYPE, DEFORMED>(
                    sizeParam, m_isModified, m_coordDim, diffCoeffPtr,
                    NullTDataVector, m_B[ind0]..., m_D[ind0]..., m_W[ind0]...,
                    m_f[ind2]..., m_nodToMod, m_nodToModTrans,
                    reinterpret_cast<const simd_t *>(jacptr),
                    reinterpret_cast<const simd_t *>(dfptr),
                    reinterpret_cast<const simd_t *>(advVelPtr),
                    reinterpret_cast<const simd_t *>(advVelPtr + advVelOffset),
                    reinterpret_cast<const simd_t *>(advVelPtr +
                                                     2 * advVelOffset),
                    m_wsp[ind1].data()..., m_deriv[ind0].data()...,
                    m_bwd.data(), reinterpret_cast<const simd_t *>(inptr),
                    reinterpret_cast<simd_t *>(outptr), this->m_lambda);

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
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * ndf * simd_t::width;
                jacptr += dfsize * simd_t::width;
                advVelPtr += nqTot * simd_t::width;
                inptr += nmTot * simd_t::width;
                outptr += nmTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
