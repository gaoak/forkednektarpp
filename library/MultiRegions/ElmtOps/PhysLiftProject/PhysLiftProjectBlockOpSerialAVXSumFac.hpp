///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpSerialAVXSumFac.hpp
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
// Description: Serial/AVX implementation of the per-block lift-and-project
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpSerialAVXSumFac.hpp
 * @brief Serial and AVX block implementation of the lift-and-project
 * operator, one element per SIMD lane.
 *
 * @details
 * The constructor fetches the tables of the block's form from the data
 * warehouse, as the separate operators fetch them, and sizes the
 * workspaces of one element group. On a block of the Pointwise or
 * StdInvMass form Apply() walks the block's components and, in each, its
 * element groups: it brings the inputs, and an output it appends to, to the
 * SIMD width a chunk at a time, and for each group lifts the trace onto a
 * copy of the inner products (IProductWRTPhysTrace's launcher) and applies the
 * form in the same sweep: the scale by `scale / (w J)`, or IProductWRTBase's
 * basis transpose, the standard inverse mass over the Jacobian and BwdTrans's
 * backward transform. A block of the ElmtInvMass form takes master's four
 * block operators in sequence (PhysLiftProjectBlockOp::ApplyElmtInvMass()).
 *
 * CMake generates one translation unit per shape and data type from
 * LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in, which expands the
 * trace-extraction switch template
 * (LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in): the
 * trace point counts come compile-time where the switch has them, and with
 * every volume and trace count the same the mode count follows them, as
 * many modes as points or one fewer.
 *
 * @note CMake includes this header into the registration translation
 * units it generates for this operator, execution space and
 * implementation; other code goes through PhysLiftProjectOp.hpp and the
 * operator factory.
 */

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/LinearAlgebra/NekBlas/libXSMMDispatchWrapper.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOpSerialAVXGenericKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpSerialAVXKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/TraceTables.hpp>

#include <algorithm>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Serial and AVX block implementation of the lift-and-project
 * operator, one element per SIMD lane.
 *
 * Naming, as in the trace operators the switch is shared with: #m_nm holds
 * the volume quadrature counts per direction, #m_nq the trace quadrature
 * counts per face pair, and #m_modes the mode counts.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation  MultiRegions::SumFac; not read.
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename TData>
class PhysLiftProjectBlockOpImpl : public PhysLiftProjectBlockOp<TData>
{
    using BlockOpBase = PhysLiftProjectBlockOp<TData>;
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Fetch the tables of the block's form and size the workspaces;
     * on a block of the ElmtInvMass form, create master's four block
     * operators instead.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    PhysLiftProjectBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOpBase(block_idx, exp, dataWarehouse)
    {
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_form      = PhysLiftProjectFormOf(exp);
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->CreateChain(ExecSpace::name, SumFac::name);
        }
        else
        {
            for (unsigned int d = 0; d < m_dimension; ++d)
            {
                m_modes.push_back(exp->GetBasisNumModes(d));
            }
            m_uniformModes = std::all_of(
                m_modes.begin(), m_modes.end(),
                [&](const unsigned int m) { return m == m_modes[0]; });

            FetchLiftTables(exp, block_idx);
            m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianKey<TData>(block_idx,
                                                 m_implInterleaveWidth));
            if (m_form == PhysLiftProjectForm::Pointwise)
            {
                m_weightsptr =
                    this->m_dataWarehouse->template GetData<MemSpace>(
                        LocalRegions::WeightsKey<TData>(block_idx,
                                                        m_implInterleaveWidth));
            }
            else
            {
                FetchProjectionTables(exp);
            }
            SizeWorkspaces();
        }
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            PhysLiftProjectBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    /// Elements the kernels process at once: the SIMD vector width, one
    /// on Serial; also the width the block's geometric factors are fetched
    /// at.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    /// The block's form (PhysLiftProjectFormOf()).
    PhysLiftProjectForm m_form;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    /// Volume quadrature points per direction.
    std::vector<unsigned int> m_nm;
    /// Trace quadrature points per face pair; in one dimension the volume
    /// count, which the generated segment switch dispatches on.
    std::vector<unsigned int> m_nq;
    /// Modes per direction, and whether every direction has the same
    /// count, so that the projection may take compile-time mode counts.
    std::vector<unsigned int> m_modes;
    bool m_uniformModes = false;
    /// The lift's tables, as IProductWRTPhysTrace fetches them
    /// (FetchTraceTables()): the end-point interpolant of each direction,
    /// then the in-face interpolants of the face pairs, broadcast across the
    /// lanes; the trace rules' weights; and where either interpolant is the
    /// identity.
    std::vector<const simd_t *> m_B;
    std::vector<const simd_t *> m_W;
    std::vector<bool> m_isCollocated;
    std::vector<bool> m_endPtsCollocated;
    /// The basis tables of the transpose and the backward transform, as
    /// IProductWRTBase and BwdTrans fetch them, whether the first-direction
    /// basis is eModified_A, and the nodal to modal matrix and its transpose
    /// on a nodal shape (null otherwise).
    std::vector<const TData *> m_basis;
    bool m_isModified             = false;
    const simd_t *m_nodToMod      = nullptr;
    const simd_t *m_nodToModTrans = nullptr;
    /// StdInvMass: the standard inverse mass, its diagonal where it is
    /// diagonal (#m_invMassIsDiagonal) and the whole transposed matrix
    /// otherwise, as MultiplyByElmtInvMass takes them.
    const TData *m_invMassptr = nullptr;
    bool m_invMassIsDiagonal  = false;
    /// The quadrature weights at every point (Pointwise), the Jacobians and
    /// the trace Jacobians (none in one dimension), at
    /// #m_implInterleaveWidth.
    const TData *m_weightsptr  = nullptr;
    const TData *m_jacptr      = nullptr;
    const TData *m_traceJacptr = nullptr;
    /// Workspaces of one element group: the inner products with the trace
    /// lifted onto them, the Pointwise factor at every point, the lift's,
    /// the transpose's and the backward transform's scratch, as their
    /// operators size them, and the coefficients before and after the
    /// inverse mass.
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_vol, m_pointScale,
        m_coef, m_coef2;
    std::vector<std::vector<simd_t, tinysimd::allocator<simd_t>>> m_liftWsp,
        m_iprodWsp, m_bwdWsp;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    /// Only one alignment warning per block operator.
    bool m_warnOnce = false;
#endif

    /// The lift's tables, as IProductWRTPhysTrace fetches them, and the
    /// trace Jacobians.
    void FetchLiftTables(const LocalRegions::ExpansionSharedPtr &exp,
                         const unsigned int block_idx)
    {
        auto tables =
            FetchTraceTables<simd_t, MemSpace>(exp, this->m_dataWarehouse);
        m_nm               = std::move(tables.nm);
        m_B                = std::move(tables.B);
        m_W                = std::move(tables.W);
        m_nq               = std::move(tables.nq);
        m_endPtsCollocated = std::move(tables.endPtsCollocated);
        m_isCollocated     = std::move(tables.isCollocated);
        if (m_dimension > 1)
        {
            m_traceJacptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianLocTraceKey<TData>(
                    block_idx, m_implInterleaveWidth));
        }
        else
        {
            m_nq.push_back(m_nm[0]);
            m_isCollocated.push_back(false);
        }
    }

    /// The projection's tables, as IProductWRTBase, MultiplyByElmtInvMass
    /// and BwdTrans fetch theirs.
    void FetchProjectionTables(const LocalRegions::ExpansionSharedPtr &exp)
    {
        m_isModified = exp->GetBasisType(0) == LibUtilities::eModified_A;
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; ++d)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
            m_basis.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(basisKeys[d],
                                                  LibUtilities::eBasis)));
        }
        LibUtilities::PointsType nodalType = LibUtilities::eNoPointsType;
        if (exp->IsNodalNonTensorialExp())
        {
            nodalType = exp->GetNodalPointsKey().GetPointsType();
        }
        if (m_shapeType == LibUtilities::NodalTri ||
            m_shapeType == LibUtilities::NodalPrism ||
            m_shapeType == LibUtilities::NodalPyr ||
            m_shapeType == LibUtilities::NodalTet)
        {
            m_nodToMod = this->m_dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(basisKeys, m_shapeType,
                                              StdRegions::eNodalToModal,
                                              nodalType));
            m_nodToModTrans = this->m_dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<simd_t>(
                    basisKeys, m_shapeType, StdRegions::eNodalToModalTranspose,
                    nodalType));
        }

        // Only the diagonal where it is diagonal, the whole transposed
        // matrix otherwise.
        m_invMassIsDiagonal = exp->InvMassIsDiagonal();
        StdRegions::StdMatType invMassType =
            StdRegions::eInvMassStdMatTranspose;
        if (m_invMassIsDiagonal)
        {
            invMassType = StdRegions::eInvMassStdMatDiagonal;
        }
        m_invMassptr = this->m_dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType, invMassType,
                                         nodalType));
    }

    /// Size the workspaces for one element group, as the separate operators
    /// size theirs.
    void SizeWorkspaces()
    {
        m_vol.resize(m_nqTot);
        if (m_form == PhysLiftProjectForm::Pointwise)
        {
            m_pointScale.resize(m_nqTot);
        }
        else
        {
            m_coef.resize(m_nmTot);
            m_coef2.resize(m_nmTot);
        }
        if (m_dimension == 2)
        {
            m_liftWsp.emplace_back(std::max(m_nm[0] * 2, m_nm[1] * 2));
            if (m_form == PhysLiftProjectForm::StdInvMass)
            {
                unsigned int wsp0Size = 0;
                IProduct2DWorkspace(m_shapeType, m_modes[0], m_modes[1],
                                    m_nm[0], m_nm[1], wsp0Size);
                m_iprodWsp.emplace_back(wsp0Size);
                wsp0Size = 0;
                BwdTrans2DWorkspace(m_shapeType, m_modes[0], m_modes[1],
                                    m_nm[0], m_nm[1], wsp0Size);
                m_bwdWsp.emplace_back(wsp0Size);
            }
        }
        else if (m_dimension == 3)
        {
            m_liftWsp.emplace_back(
                std::max({m_nm[0] * m_nm[1] * 2, m_nm[1] * m_nm[2] * 2,
                          m_nm[0] * m_nm[2] * 2}));
            m_liftWsp.emplace_back(std::max(
                {m_nm[1] * m_nq[1], m_nm[0] * m_nq[3], m_nm[0] * m_nq[5]}));
            if (m_form == PhysLiftProjectForm::StdInvMass)
            {
                unsigned int wsp0Size = 0, wsp1Size = 0, wsp2Size = 0;
                IProduct3DWorkspace(m_shapeType, m_modes[0], m_modes[1],
                                    m_modes[2], m_nm[0], m_nm[1], m_nm[2],
                                    wsp0Size, wsp1Size, wsp2Size);
                m_iprodWsp.emplace_back(wsp0Size);
                m_iprodWsp.emplace_back(wsp1Size);
                m_iprodWsp.emplace_back(wsp2Size);
                wsp0Size = 0;
                wsp1Size = 0;
                BwdTrans3DWorkspace(m_shapeType, m_modes[0], m_modes[1],
                                    m_modes[2], m_nm[0], m_nm[1], m_nm[2],
                                    wsp0Size, wsp1Size);
                m_bwdWsp.emplace_back(wsp0Size);
                m_bwdWsp.emplace_back(wsp1Size);
            }
        }
    }

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        if (m_form == PhysLiftProjectForm::ElmtInvMass)
        {
            this->template ApplyElmtInvMass<MemSpace>(inblock, outblock);
        }
        else
        {
            this->ShapeSwitch(m_shapeType, [&](auto shape) {
                this->template ShapeBlock<decltype(shape)::value>(inblock,
                                                                  outblock);
            });
        }
    }

    /**
     * @brief The block's operator with the lift of @p trace: the trace is
     * held for this application alone, then the block is applied as by
     * Apply().
     */
    void v_ApplyLift(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &trace,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        this->m_trace = &trace;
        v_Apply(inblock, outblock);
        this->m_trace = nullptr;
    }

    // Shape-specific entry points, specialised for each shape in the
    // translation units generated from
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /// Whether the trace size parameter @p TTraceSizeParameter is
    /// compile-time with every volume and trace count the same.
    template <typename TTraceSizeParameter>
    static constexpr bool UniformCounts()
    {
        using T = TTraceSizeParameter;
        if constexpr (IsTemplatedTraceSizeParameter3D<T>::value)
        {
            constexpr unsigned int n = T::nm0();
            return T::nm1() == n && T::nm2() == n && T::nq00() == n &&
                   T::nq01() == n && T::nq10() == n && T::nq11() == n &&
                   T::nq20() == n && T::nq21() == n;
        }
        else if constexpr (IsTemplatedTraceSizeParameter2D<T>::value)
        {
            constexpr unsigned int n = T::nm0();
            return T::nm1() == n && T::nq00() == n && T::nq10() == n;
        }
        else
        {
            return IsTemplatedTraceSizeParameter1D<T>::value;
        }
    }

    /// The compile-time size parameter of @p NUMMODES modes and
    /// @p NUMPOINTS points in every direction.
    template <LibUtilities::ShapeType SHAPE_TYPE, unsigned int NUMMODES,
              unsigned int NUMPOINTS>
    static auto TemplatedSizeParameter()
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        if constexpr (DIM == 1)
        {
            return TemplatedSizeParameter1D<NUMMODES, NUMPOINTS>();
        }
        else if constexpr (DIM == 2)
        {
            return TemplatedSizeParameter2D<
                NUMMODES, NUMMODES,
                LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, NUMMODES,
                                                      NUMMODES),
                NUMPOINTS, NUMPOINTS>();
        }
        else
        {
            return TemplatedSizeParameter3D<
                NUMMODES, NUMMODES, NUMMODES,
                LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, NUMMODES,
                                                      NUMMODES, NUMMODES),
                NUMPOINTS, NUMPOINTS, NUMPOINTS>();
        }
    }

    /// The runtime size parameter of the block.
    template <unsigned int DIM> auto NonTemplatedSizeParameter() const
    {
        if constexpr (DIM == 1)
        {
            return NonTemplatedSizeParameter1D(m_modes[0], m_nm[0]);
        }
        else if constexpr (DIM == 2)
        {
            return NonTemplatedSizeParameter2D(m_modes[0], m_modes[1], m_nmTot,
                                               m_nm[0], m_nm[1]);
        }
        else
        {
            return NonTemplatedSizeParameter3D(m_modes[0], m_modes[1],
                                               m_modes[2], m_nmTot, m_nm[0],
                                               m_nm[1], m_nm[2]);
        }
    }

    /// Lift, IProductWRTBase and BwdTrans workspaces in @p dim dimensions.
    static constexpr unsigned int NumLiftWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 2;
    }
    static constexpr unsigned int NumIProductWorkspace(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 1 : 3;
    }

    /**
     * @brief The block's form with the trace size parameter the generated
     * switch chose: compile-time mode counts for the StdInvMass form where
     * every volume and trace count is the same and the modes are as many as
     * the points or one fewer, the runtime ones otherwise. A deformed block
     * reaching here is Pointwise.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const TTraceSizeParameter traceSizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        static_assert(
            (DIM == 1 && IsTraceSizeParameter1D_v<TTraceSizeParameter>) ||
                (DIM == 2 && IsTraceSizeParameter2D_v<TTraceSizeParameter>) ||
                (DIM == 3 && IsTraceSizeParameter3D_v<TTraceSizeParameter>),
            "OperatorND expects a trace size parameter matching the "
            "dimension of the shape.");

        WARNINGL1(m_warnOnce ||
                      (inblock.GetAlignment() % simd_t::alignment == 0 &&
                       outblock.GetAlignment() % simd_t::alignment == 0),
                  "Input or output Field are not aligned to the required "
                  "alignment for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

        if constexpr (DEFORMED)
        {
            ApplyGroups<SHAPE_TYPE, true, PhysLiftProjectForm::Pointwise>(
                inblock, outblock, NonTemplatedSizeParameter<DIM>(),
                traceSizeParam);
        }
        else if (m_form == PhysLiftProjectForm::Pointwise)
        {
            ApplyGroups<SHAPE_TYPE, false, PhysLiftProjectForm::Pointwise>(
                inblock, outblock, NonTemplatedSizeParameter<DIM>(),
                traceSizeParam);
        }
        else
        {
            WithStdInvMassSizeParameter<SHAPE_TYPE>(
                traceSizeParam, [&](const auto sizeParam) {
                    this->template ApplyGroups<SHAPE_TYPE, false,
                                               PhysLiftProjectForm::StdInvMass>(
                        inblock, outblock, sizeParam, traceSizeParam);
                });
        }
    }

    /**
     * @brief Calls @p func once with the volume size parameter of the
     * StdInvMass form: compile-time counts of N or N - 1 modes and N points
     * in every direction where the trace size parameter is compile-time with
     * every count N and the block has that many modes in every direction,
     * the runtime counts otherwise. N - 1 modes only where that is a valid
     * mode count: one at least, and two for the nodal shapes, whose
     * coefficient counts assert it even in constant evaluation.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, typename TTraceSizeParameter,
              typename TFunc>
    NEK_FORCE_INLINE void WithStdInvMassSizeParameter(const TTraceSizeParameter,
                                                      TFunc &&func) const
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        if constexpr (UniformCounts<TTraceSizeParameter>())
        {
            constexpr unsigned int N = TTraceSizeParameter::nm0();
            constexpr bool nodal     = SHAPE_TYPE == LibUtilities::NodalTri ||
                                   SHAPE_TYPE == LibUtilities::NodalTet ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism ||
                                   SHAPE_TYPE == LibUtilities::NodalPyr;
            constexpr unsigned int minModes = nodal ? 2u : 1u;
            const unsigned int modes        = m_uniformModes ? m_modes[0] : 0u;
            if constexpr (N >= minModes + 1)
            {
                if (modes == N)
                {
                    func(TemplatedSizeParameter<SHAPE_TYPE, N, N>());
                }
                else if (modes == N - 1)
                {
                    func(TemplatedSizeParameter<SHAPE_TYPE, N - 1, N>());
                }
                else
                {
                    func(NonTemplatedSizeParameter<DIM>());
                }
            }
            else if constexpr (N >= minModes)
            {
                if (modes == N)
                {
                    func(TemplatedSizeParameter<SHAPE_TYPE, N, N>());
                }
                else
                {
                    func(NonTemplatedSizeParameter<DIM>());
                }
            }
            else
            {
                func(NonTemplatedSizeParameter<DIM>());
            }
        }
        else
        {
            func(NonTemplatedSizeParameter<DIM>());
        }
    }

    /// ApplyGroupsImpl() with the index sequences of the shape's dimension.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              PhysLiftProjectForm FORM, typename TSizeParameter,
              typename TTraceSizeParameter>
    NEK_FORCE_INLINE void ApplyGroups(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        const TSizeParameter sizeParam,
        const TTraceSizeParameter traceSizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
        ApplyGroupsImpl<SHAPE_TYPE, DEFORMED, FORM>(
            inblock, outblock, sizeParam, traceSizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, DIM *(DIM - 1)>(),
            std::make_integer_sequence<unsigned int, NumLiftWorkspace(DIM)>(),
            std::make_integer_sequence<unsigned int,
                                       NumIProductWorkspace(DIM)>(),
            std::make_integer_sequence<unsigned int, DIM - 1>());
    }

    /**
     * @brief The block's components and, in each, its element groups: each
     * chunk of the inputs, and of the output when it is appended to,
     * brought to #m_implInterleaveWidth, then for each group the factor of
     * the form, the lift and the form, and the chunk brought back. The
     * output leaves at the input's width.
     *
     * @tparam FORM  The block's form, Pointwise or StdInvMass.
     * @tparam ind0  One index per direction.
     * @tparam ind1  One per face pair.
     * @tparam ind2  One per lift workspace.
     * @tparam ind3  One per IProductWRTBase workspace.
     * @tparam ind4  One per BwdTrans workspace.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              PhysLiftProjectForm FORM, typename TSizeParameter,
              typename TTraceSizeParameter, unsigned int... ind0,
              unsigned int... ind1, unsigned int... ind2, unsigned int... ind3,
              unsigned int... ind4>
    void ApplyGroupsImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        [[maybe_unused]] const TSizeParameter sizeParam,
        const TTraceSizeParameter traceSizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>,
        std::integer_sequence<unsigned int, ind2...>,
        std::integer_sequence<unsigned int, ind3...>,
        std::integer_sequence<unsigned int, ind4...>)
    {
        const unsigned int nqTotTrace =
            traceSizeParam.template nqTotTrace<SHAPE_TYPE>();

        // Initialize pointers. The inputs are brought to the kernels' width
        // in place, a chunk at a time, and restored afterwards.
        TData *inPtr =
            const_cast<TData *>(inblock.template GetPtr<MemSpace, ReadOnly>());
        TData *tracePtr = nullptr;
        if (this->m_trace)
        {
            ASSERTL1(this->m_trace->GetNumElementsWithPadding() ==
                         inblock.GetNumElementsWithPadding(),
                     "Trace and volume blocks differ in their padding.");
            tracePtr = const_cast<TData *>(
                this->m_trace->template GetPtr<MemSpace, ReadOnly>());
        }
        auto outPtr = (this->m_append)
                          ? outblock.template GetPtr<MemSpace, ReadWrite>()
                          : outblock.template GetPtr<MemSpace, WriteOnly>();
        simd_t *vol = m_vol.data();

        // One Jacobian per element and one per trace when affine, one per
        // point when deformed.
        const size_t jacSize = DEFORMED ? m_nqTot : 1;
        const size_t traceJacSize =
            DEFORMED ? nqTotTrace
                     : LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];
        const auto jac      = reinterpret_cast<const simd_t *>(m_jacptr);
        const auto traceJac = reinterpret_cast<const simd_t *>(m_traceJacptr);

        // Get interleave parameter: the trace, and the output when it is
        // appended to, may each arrive at its own width; the output leaves
        // at the input's.
        const unsigned int inInterleaveWidth = inblock.GetInterleaveWidth();
        unsigned int traceInterleaveWidth    = m_implInterleaveWidth;
        if (this->m_trace)
        {
            traceInterleaveWidth = this->m_trace->GetInterleaveWidth();
        }
        unsigned int outInterleaveWidth = m_implInterleaveWidth;
        if (this->m_append)
        {
            outInterleaveWidth = outblock.GetInterleaveWidth();
        }
        const size_t chunkSize =
            std::max({m_implInterleaveWidth, inInterleaveWidth,
                      traceInterleaveWidth, outInterleaveWidth});
        const size_t width_ratio = chunkSize / m_implInterleaveWidth;

        // Dispatch kernel: the product of one element group with the
        // standard inverse mass, which only a StdInvMass block whose inverse
        // mass is not diagonal applies.
        [[maybe_unused]] auto invMassKernel =
            LibxsmmDispatchWrapper<TData>::dispatch(simd_t::width, m_nmTot,
                                                    m_nmTot, 1.0, 0.0);

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
                        m_nqTot, inPtr);
                    if (tracePtr)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, traceInterleaveWidth,
                            chunkSize, nqTotTrace, tracePtr);
                    }
                    if (this->m_append)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, m_nqTot, outPtr);
                    }
                }

                // The factor of the form: scale / (w J) at every point
                // (Pointwise) or scale / J (StdInvMass).
                simd_t coefScale(this->m_scale);
                if constexpr (FORM == PhysLiftProjectForm::Pointwise)
                {
                    PhysLiftProjectPointScaleKernel<DEFORMED>(
                        m_nqTot, this->m_scale,
                        reinterpret_cast<const simd_t *>(m_weightsptr),
                        jac + e * jacSize, m_pointScale.data());
                }
                else
                {
                    coefScale = SafeDivide(coefScale, jac[e]);
                }

                // The inner products copied, and the trace lifted onto them.
                const auto inGroup = reinterpret_cast<const simd_t *>(inPtr);
                std::copy(inGroup, inGroup + m_nqTot, vol);
                if (tracePtr)
                {
                    const simd_t *traceJacGroup = nullptr;
                    if (traceJac)
                    {
                        traceJacGroup = traceJac + e * traceJacSize;
                    }
                    IProductWRTPhysTraceKernelLauncher<SHAPE_TYPE, DEFORMED,
                                                       true>(
                        traceSizeParam, m_B[ind0]...,
                        m_B[sizeof...(ind0) + ind1]..., m_W[ind1]...,
                        traceJacGroup, m_liftWsp[ind2].data()...,
                        reinterpret_cast<const simd_t *>(tracePtr), vol,
                        (bool)m_isCollocated[ind1]...,
                        (bool)m_endPtsCollocated[ind0]...);
                }

                const auto outGroup = reinterpret_cast<simd_t *>(outPtr);
                if constexpr (FORM == PhysLiftProjectForm::Pointwise)
                {
                    if (this->m_append)
                    {
                        PhysLiftProjectPointwiseKernel<true>(
                            m_nqTot, m_pointScale.data(), vol, outGroup);
                    }
                    else
                    {
                        PhysLiftProjectPointwiseKernel<false>(
                            m_nqTot, m_pointScale.data(), vol, outGroup);
                    }
                }
                else
                {
                    // The metric-free basis transpose (the inner products
                    // carry the metric already), the standard inverse mass
                    // over the Jacobian with the scale, then the backward
                    // transform into the output.
                    simd_t *coef  = m_coef.data();
                    simd_t *coef2 = m_coef2.data();
                    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
                        sizeParam, m_isModified, vol, m_basis[ind0]...,
                        m_nodToModTrans, m_iprodWsp[ind3].data()..., coef);
                    if (m_invMassIsDiagonal)
                    {
                        StdInvMassDiagonalScaleKernel(m_nmTot, m_invMassptr,
                                                      coefScale, coef, coef2);
                    }
                    else
                    {
                        invMassKernel(reinterpret_cast<const TData *>(coef),
                                      m_invMassptr,
                                      reinterpret_cast<TData *>(coef2));
                        PhysLiftProjectCoefScaleKernel(m_nmTot, coefScale,
                                                       coef2);
                    }
                    if (this->m_append)
                    {
                        BwdTransKernelLauncher<SHAPE_TYPE, true>(
                            sizeParam, m_isModified, m_basis[ind0]...,
                            m_nodToMod, m_bwdWsp[ind4].data()..., coef2,
                            outGroup);
                    }
                    else
                    {
                        BwdTransKernelLauncher<SHAPE_TYPE, false>(
                            sizeParam, m_isModified, m_basis[ind0]...,
                            m_nodToMod, m_bwdWsp[ind4].data()..., coef2,
                            outGroup);
                    }
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        inPtr - (width_ratio - 1) * m_nqTot * simd_t::width);
                    if (tracePtr)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            traceInterleaveWidth, m_implInterleaveWidth,
                            chunkSize, nqTotTrace,
                            tracePtr -
                                (width_ratio - 1) * nqTotTrace * simd_t::width);
                    }
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        outPtr - (width_ratio - 1) * m_nqTot * simd_t::width);
                }

                // Increment pointers.
                inPtr += m_nqTot * simd_t::width;
                if (tracePtr)
                {
                    tracePtr += nqTotTrace * simd_t::width;
                }
                outPtr += m_nqTot * simd_t::width;
            }
        }

        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::MultiRegions::detail
