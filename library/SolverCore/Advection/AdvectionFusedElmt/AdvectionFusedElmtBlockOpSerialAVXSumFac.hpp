///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionFusedElmtBlockOpSerialAVXSumFac.hpp
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
// Description: Serial/AVX sum-factorised implementation of the per-block fused
// element-local advection operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvectionFusedElmtBlockOpSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised implementation of the per-block fused
 * element-local advection operator, and what the host implementations of
 * the fused element-local operators share.
 *
 * @details
 * One template serves both host execution spaces: `simd_t` resolves to
 * tinysimd::scalarT<TData> for Serial and tinysimd::simd<TData> for AVX (see
 * simd_type_if), each vector lane owning one element. The kernels are in
 * AdvectionFusedElmtBlockOpSerialAVXSumFacKernels.hpp.
 *
 * detail::AdvectionFusedElmtHostBlock holds what the advection and the
 * advection-diffusion implementations share: the basis data fetched from the
 * data warehouse with the keys the separate operators use, the advection
 * workspaces, the size parameters, and the loop over a block's element
 * groups that brings each chunk of the inputs to the kernels' width and the
 * outputs back. It is templated on the block-operator base it derives from,
 * AdvectionFusedElmtBlockOp or AdvDiffFusedElmtBlockOp.
 *
 * ApplyFusedElmt() switches on the block's shape into the generated
 * ShapeBlock(), which switches on the block's point count and calls
 * OperatorND() with a compile-time trace size parameter inside the
 * configured switch range and a runtime one outside it. OperatorND() takes
 * the compile-time sizes only when every volume and trace count is the same,
 * the runtime sizes otherwise.
 *
 * @note The operator is a class template on the flux policy, so the code
 * supplying the policy generates the translation units that include this
 * header, through ADD_POLICY_ELMT_BLOCK_OPERATOR (cmake/NektarCommon.cmake):
 * the registration and one unit per shape for the ShapeBlock()
 * specialisations, built from the trace operators' size switch
 * (LibUtilities/BasicUtils/Switch/BlockOpSwitchPhysTraceExtract.h.in).
 * Other code goes through AdvectionFusedElmtOp.hpp and the operator factory.
 */

#pragma once

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>
#include <MultiRegions/ElmtOps/PhysTraceExtract/PhysTraceExtractBlockOpSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/TraceTables.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOp.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOpSerialAVXSumFacKernels.hpp>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_PHYSTRACE

namespace Nektar::SolverCore::detail
{

/**
 * @brief What the Serial and AVX implementations of the fused element-local
 * operators share, one element per SIMD lane.
 *
 * @details
 * The fields may arrive at any interleave width. A pass over the element
 * groups (BeginGroups()) brings each chunk of the state and the auxiliary
 * fields to #m_implInterleaveWidth and back after it, a no-op for a caller that
 * already carries its fields at that width, and the outputs leave at the
 * state's width.
 *
 * @tparam ExecSpace     NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Physics       The flux policy.
 * @tparam TData         Floating-point type of the field data.
 * @tparam TBlockOpBase  The block-operator base template:
 *                       AdvectionFusedElmtBlockOp or one derived from it,
 *                       instantiated on @p TData.
 */
template <typename ExecSpace, typename Physics, typename TData,
          template <typename> typename TBlockOpBase>
class AdvectionFusedElmtHostBlock : public TBlockOpBase<TData>
{
protected:
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;
    using Block    = LibUtilities::BlockAccessor<TData, FieldState::Phys>;

    static constexpr unsigned int NDIM = Physics::NDIM;
    static constexpr unsigned int NVAR = Physics::NVAR;
    static constexpr unsigned int NAUX = Physics::NAUX;
    static_assert(NDIM >= 1 && NDIM <= 3,
                  "The fused element-local operators take one-, two- and "
                  "three-dimensional meshes");
    /// Line buffer of the line-blocked kernels for runtime sizes: a block
    /// with more points per direction takes the runtime-sized kernels field
    /// by field. Compile-time sizes size the buffers to the element instead
    /// (AdvectionFusedElmtLineLength()).
    static constexpr unsigned int MAXQ = 8;

    /// Elements the kernels process at once: the SIMD vector width, one on
    /// Serial; also the width the block's geometric factors are fetched at.
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    /// The flux policy (InitialisePolicy()).
    Physics m_physics;
    /// The basis data the kernels read (FetchBasisData()).
    AdvectionFusedElmtBasisData<TData, simd_t> m_basisData;
    /// Derivative factors and Jacobians of the block, at
    /// #m_implInterleaveWidth.
    const TData *m_dfptr  = nullptr;
    const TData *m_jacptr = nullptr;
    /// The layout of one element group's workspaces (see
    /// AdvectionFusedElmtWorkspace) in the shared per-stream scratch buffer
    /// (MultiRegions::BlockOperator::GetStaticWorkSpace()): each one's
    /// offset, in vectors, and the size of the layout, which a derived block
    /// operator extends with its own (CarveWorkspace()).
    size_t m_fluxOffset = 0, m_alignedOffset = 0, m_dfWspOffset = 0,
           m_interpWspOffset = 0, m_stageOffset = 0, m_wspSize = 0;
    /// Distance between consecutive variables of the vector workspaces
    /// (PaddedVariableStride()).
    unsigned int m_varStride = 0;
    /// The auxiliary fields and the trace ApplyFusedElmt() reads and writes
    /// besides the state and the derivative inner products, held for that
    /// application alone and null otherwise.
    Block *m_aux   = nullptr;
    Block *m_trace = nullptr;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    /// Only one alignment warning per block operator.
    bool m_warnOnce = false;
#endif

    /**
     * @brief Fetch the block's basis data and lay out the advection
     * workspaces (CarveWorkspace()).
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionFusedElmtHostBlock(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : TBlockOpBase<TData>(block_idx, exp, dataWarehouse)
    {
        FetchBasisData(exp, block_idx);
        const auto extractSizes =
            MultiRegions::detail::PhysTraceExtractWorkspaceSizes(this->m_nm,
                                                                 this->m_nq);
        m_varStride     = PaddedVariableStride(NDIM * this->m_nqTot);
        m_fluxOffset    = CarveWorkspace(size_t(NVAR) * m_varStride);
        m_alignedOffset = CarveWorkspace(size_t(NVAR) * m_varStride);
        m_dfWspOffset   = CarveWorkspace(size_t(NDIM) * NDIM);
        m_interpWspOffset =
            CarveWorkspace(std::max<size_t>(extractSizes[1], 1));
        // NDIM fields as well: the advection-diffusion pass stages the
        // directional components of the viscous flux on deformed elements.
        m_stageOffset = CarveWorkspace(size_t(std::max(NVAR, NDIM)) *
                                       this->m_packing.nqTotTraceVol);
    }

    ~AdvectionFusedElmtHostBlock() override = default;

    /// Whether an implementation takes a block of @p exp: an element of the
    /// policy's dimension in a space of the same dimension, of a shape the
    /// kernels are compiled for (TakesShape()): every shape but the nodal
    /// ones.
    static bool TakesBlock(const LocalRegions::ExpansionSharedPtr &exp)
    {
        return exp->GetShapeDimension() == NDIM && exp->GetCoordim() == NDIM &&
               TBlockOpBase<TData>::TakesShape(exp->DetShapeType(), NDIM);
    }

    /// The block's basis data (the differentiation matrices, the quadrature
    /// weights, the collapsed-coordinate factors and the trace interpolation
    /// tables) and its geometric factors, as PhysDeriv,
    /// IProductWRTDerivBase and PhysTraceExtract fetch them.
    void FetchBasisData(const LocalRegions::ExpansionSharedPtr &exp,
                        const unsigned int block_idx)
    {
        auto &warehouse = this->m_dataWarehouse;
        auto &basisData = m_basisData;

        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const auto bkey          = exp->GetBasis(d)->GetBasisKey();
            basisData.derivMatrix[d] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(bkey,
                                                  LibUtilities::eDerivative));
            basisData.weights[d] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(bkey,
                                                  LibUtilities::eWeights));
        }
        if constexpr (NDIM == 2)
        {
            basisData.factors[0] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero));
            basisData.factors[1] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero));
        }
        else if constexpr (NDIM == 3)
        {
            basisData.factors[0] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero));
            basisData.factors[1] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero));
            basisData.factors[2] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero));
            basisData.factors[3] = warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero));
        }

        const auto traceTables =
            MultiRegions::detail::FetchTraceTables<TData, MemSpace>(exp,
                                                                    warehouse);
        for (size_t i = 0; i < traceTables.B.size(); ++i)
        {
            basisData.traceBasis[i] = traceTables.B[i];
        }

        m_dfptr = warehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));
        m_jacptr = warehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    /**
     * @brief The distance between consecutive variables of a vector
     * workspace whose variables take @p unpadded vectors each.
     *
     * The vector workspaces hold `NVAR * NDIM` components of `nqTot`
     * vectors, which the kernels walk in lockstep. Where the components span
     * a whole number of pages (64 points at 64-byte vectors: P3 hexahedra on
     * AVX-512), every one of them would start at the same offset within a
     * page and compete for the same cache sets, and the step time would
     * depend on where the allocator puts each workspace relative to the
     * fields. So each variable starts about a fifth of a page on from the
     * last, whatever the point count; the `NDIM` components of one variable
     * stay `nqTot` apart, as the line-blocked and metric kernels take them.
     */
    static unsigned int PaddedVariableStride(const unsigned int unpadded)
    {
        constexpr size_t pageBytes   = 4096;
        constexpr size_t vectorBytes = sizeof(simd_t);
        constexpr size_t stepBytes =
            (pageBytes / NVAR + vectorBytes - 1) / vectorBytes * vectorBytes;
        const size_t unpaddedBytes = size_t(unpadded) * vectorBytes;
        const size_t padBytes =
            (stepBytes + pageBytes - unpaddedBytes % pageBytes) % pageBytes;
        return unpadded + static_cast<unsigned int>(padBytes / vectorBytes);
    }

    /// Append @p size vectors to the workspace layout and return their
    /// offset.
    size_t CarveWorkspace(const size_t size)
    {
        const size_t offset = m_wspSize;
        m_wspSize += size;
        return offset;
    }

    /// The start of the workspaces: the stream's scratch buffer, one set of
    /// workspaces for every block operator on the stream. Fetched on each
    /// application, since another block operator may have grown and so
    /// moved it since the last.
    simd_t *WorkspaceBase()
    {
        return reinterpret_cast<simd_t *>(
            MultiRegions::BlockOperator<TData>::template GetStaticWorkSpace<
                MemSpace>(m_wspSize * simd_t::width));
    }

    /// The advection workspaces as the kernels take them, in the buffer
    /// starting at @p base (WorkspaceBase()).
    AdvectionFusedElmtWorkspace<simd_t> AdvectionWorkspace(simd_t *base)
    {
        return AdvectionFusedElmtWorkspace<simd_t>{base + m_fluxOffset,
                                                   base + m_alignedOffset,
                                                   m_varStride,
                                                   base + m_dfWspOffset,
                                                   base + m_interpWspOffset,
                                                   base + m_stageOffset};
    }

    void v_InitialisePolicy(
        const LibUtilities::SessionReaderSharedPtr &session) override
    {
        m_physics = Physics(session);
    }

    /// Whether the trace size parameter @p TTraceSizeParameter is
    /// compile-time with every volume and trace count the same: the sizes
    /// the kernels are compiled for.
    template <typename TTraceSizeParameter>
    static constexpr bool UniformCounts()
    {
        using T = TTraceSizeParameter;
        if constexpr (MultiRegions::detail::IsTemplatedTraceSizeParameter1D<
                          T>::value)
        {
            return true;
        }
        else if constexpr (MultiRegions::detail::
                               IsTemplatedTraceSizeParameter2D<T>::value)
        {
            constexpr unsigned int n = T::nm0();
            return T::nm1() == n && T::nq00() == n && T::nq10() == n;
        }
        else if constexpr (MultiRegions::detail::
                               IsTemplatedTraceSizeParameter3D<T>::value)
        {
            constexpr unsigned int n = T::nm0();
            return T::nm1() == n && T::nm2() == n && T::nq00() == n &&
                   T::nq01() == n && T::nq10() == n && T::nq11() == n &&
                   T::nq20() == n && T::nq21() == n;
        }
        else
        {
            return false;
        }
    }

    /// The physical-space size parameter matching a compile-time trace size
    /// parameter.
    template <typename TTraceSizeParameter> static auto TemplatedPhysSize()
    {
        using T = TTraceSizeParameter;
        if constexpr (NDIM == 1)
        {
            return MultiRegions::detail::TemplatedPhysSizeParameter1D<
                1, T::nm0()>();
        }
        else if constexpr (NDIM == 2)
        {
            return MultiRegions::detail::TemplatedPhysSizeParameter2D<
                2, T::nm0(), T::nm1()>();
        }
        else if constexpr (NDIM == 3)
        {
            return MultiRegions::detail::TemplatedPhysSizeParameter3D<
                T::nm0(), T::nm1(), T::nm2()>();
        }
    }

    /// The runtime physical-space size parameter of the block.
    auto NonTemplatedPhysSize() const
    {
        const auto &nm = this->m_nm;
        if constexpr (NDIM == 1)
        {
            return MultiRegions::detail::NonTemplatedPhysSizeParameter1D(1,
                                                                         nm[0]);
        }
        else if constexpr (NDIM == 2)
        {
            return MultiRegions::detail::NonTemplatedPhysSizeParameter2D(
                2, nm[0], nm[1]);
        }
        else if constexpr (NDIM == 3)
        {
            return MultiRegions::detail::NonTemplatedPhysSizeParameter3D(
                nm[0], nm[1], nm[2]);
        }
    }

    /// Warn, once per block operator, when the fields are not aligned for
    /// the SIMD type.
    void CheckAlignment([[maybe_unused]] Block &inblock,
                        [[maybe_unused]] Block &outblock)
    {
        WARNINGL1(m_warnOnce ||
                      (inblock.GetAlignment() % simd_t::alignment == 0 &&
                       outblock.GetAlignment() % simd_t::alignment == 0),
                  "Input or output Field are not aligned to the required "
                  "alignment for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif
    }

    /// The block-wide pointers, sizes and widths of one pass over the
    /// block's element groups (BeginGroups()).
    struct GroupPass
    {
        TData *statePtr = nullptr;
        TData *auxPtr   = nullptr;
        TData *outPtr   = nullptr;
        std::vector<Block *> traceBlocks;
        std::vector<TData *> tracePtrs;
        size_t volCompSize              = 0;
        size_t traceCompSize            = 0;
        size_t nelmt                    = 0;
        size_t ngroups                  = 0;
        unsigned int inInterleaveWidth  = 0;
        unsigned int auxInterleaveWidth = 0;
        size_t chunkSize                = 0;
        size_t widthRatio               = 1;
    };

    /// One element group's pointers per variable, @c aux per component and
    /// null without auxiliary fields (BeginGroup()).
    struct GroupPointers
    {
        const simd_t *state[NVAR];
        const simd_t *aux[NAUX > 0 ? NAUX : 1] = {};
        simd_t *trace[NVAR];
        simd_t *out[NVAR];

        /// The auxiliary fields as the kernels take them: null for a policy
        /// with none.
        const simd_t *const *Aux() const
        {
            return NAUX > 0 ? aux : nullptr;
        }
    };

    /**
     * @brief Start a pass over the block's element groups.
     *
     * A pass is BeginGroups(), then BeginGroup() and EndGroup() around each
     * group's kernel in turn, then EndGroups(). Each chunk of the state and
     * of the auxiliary fields is brought to #m_implInterleaveWidth for its
     * groups, the padded lanes of the outputs are zeroed, and the chunk is
     * brought back, the outputs to the state's width.
     *
     * @param   inblock       The state.
     * @param   outblock      Out: the derivative inner products.
     * @param   traceOutputs  The outputs laid out as #m_trace besides it
     *                        (null entries skipped), which the group kernel
     *                        writes: their padded lanes are zeroed and their
     *                        width set as the trace's are.
     */
    GroupPass BeginGroups(Block &inblock, Block &outblock,
                          const std::vector<Block *> &traceOutputs)
    {
        GroupPass pass;
        pass.nelmt   = inblock.GetNumElements();
        pass.ngroups = inblock.GetNumElmtGroups(m_implInterleaveWidth);

        // The blocks' pointers, the inputs brought to the kernels' width in
        // place a chunk at a time and restored afterwards.
        pass.statePtr =
            const_cast<TData *>(inblock.template GetPtr<MemSpace, ReadOnly>());
        if (m_aux)
        {
            pass.auxPtr = const_cast<TData *>(
                m_aux->template GetPtr<MemSpace, ReadOnly>());
        }
        pass.outPtr = outblock.template GetPtr<MemSpace, WriteOnly>();
        pass.traceBlocks.push_back(m_trace);
        for (Block *block : traceOutputs)
        {
            if (block)
            {
                pass.traceBlocks.push_back(block);
            }
        }
        for (Block *block : pass.traceBlocks)
        {
            pass.tracePtrs.push_back(
                block->template GetPtr<MemSpace, WriteOnly>());
        }
        // The fields all come from the block's elements: the volume fields
        // share one component size and the element-local traces another.
        ASSERTL1(outblock.CompSize() == inblock.CompSize() &&
                     (!m_aux || m_aux->CompSize() == inblock.CompSize()),
                 "The state, the auxiliary fields and the derivative inner "
                 "products share the block's component size.");
        pass.volCompSize   = inblock.CompSize();
        pass.traceCompSize = m_trace->CompSize();

        // The inputs may each arrive at their own width; the outputs leave
        // at the state's.
        pass.inInterleaveWidth = inblock.GetInterleaveWidth();
        pass.auxInterleaveWidth =
            m_aux ? m_aux->GetInterleaveWidth() : m_implInterleaveWidth;
        pass.chunkSize =
            std::max({m_implInterleaveWidth, pass.inInterleaveWidth,
                      pass.auxInterleaveWidth});
        pass.widthRatio = pass.chunkSize / m_implInterleaveWidth;
        return pass;
    }

    /// Group @p e's pointers, its chunk first brought to the kernels' width
    /// when @p e begins one.
    GroupPointers BeginGroup(const GroupPass &pass, const size_t e)
    {
        constexpr unsigned int width  = simd_t::width;
        const unsigned int nqTot      = this->m_nqTot;
        const unsigned int nqTotTrace = this->m_packing.nqTotTrace;
        const size_t stateOffset      = e * nqTot * width;
        const size_t traceOffset      = e * nqTotTrace * width;

        if (e % pass.widthRatio == 0)
        {
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, pass.inInterleaveWidth,
                    pass.chunkSize, nqTot,
                    pass.statePtr + n * pass.volCompSize + stateOffset);
            }
            for (unsigned int a = 0; a < (m_aux ? NAUX : 0); ++a)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    m_implInterleaveWidth, pass.auxInterleaveWidth,
                    pass.chunkSize, nqTot,
                    pass.auxPtr + a * pass.volCompSize + stateOffset);
            }
        }

        GroupPointers group;
        for (unsigned int n = 0; n < NVAR; ++n)
        {
            group.state[n] = reinterpret_cast<const simd_t *>(
                pass.statePtr + n * pass.volCompSize + stateOffset);
            group.trace[n] = reinterpret_cast<simd_t *>(
                pass.tracePtrs[0] + n * pass.traceCompSize + traceOffset);
            group.out[n] = reinterpret_cast<simd_t *>(
                pass.outPtr + n * pass.volCompSize + stateOffset);
        }
        for (unsigned int a = 0; a < (m_aux ? NAUX : 0); ++a)
        {
            group.aux[a] = reinterpret_cast<const simd_t *>(
                pass.auxPtr + a * pass.volCompSize + stateOffset);
        }
        return group;
    }

    /// Finish group @p e: zero the padded lanes of its outputs, and bring
    /// its chunk back when @p e ends one, the inputs to their widths and the
    /// outputs to the state's.
    void EndGroup(const GroupPass &pass, const size_t e,
                  const GroupPointers &group)
    {
        constexpr unsigned int width  = simd_t::width;
        const unsigned int nqTot      = this->m_nqTot;
        const unsigned int nqTotTrace = this->m_packing.nqTotTrace;
        const size_t traceOffset      = e * nqTotTrace * width;

        // The padded lanes of the outputs: the arithmetic on a padded
        // element's zero state need not give zero.
        if ((e + 1) * width > pass.nelmt)
        {
            for (unsigned int lane = 0; lane < width; ++lane)
            {
                if (e * width + lane < pass.nelmt)
                {
                    continue;
                }
                for (unsigned int n = 0; n < NVAR; ++n)
                {
                    auto outLanes = reinterpret_cast<TData *>(group.out[n]);
                    for (unsigned int i = 0; i < nqTot; ++i)
                    {
                        outLanes[i * width + lane] = 0;
                    }
                    for (TData *tracePtr : pass.tracePtrs)
                    {
                        TData *traceLanes =
                            tracePtr + n * pass.traceCompSize + traceOffset;
                        for (unsigned int i = 0; i < nqTotTrace; ++i)
                        {
                            traceLanes[i * width + lane] = 0;
                        }
                    }
                }
            }
        }

        if (e % pass.widthRatio == pass.widthRatio - 1)
        {
            const size_t chunkState = (e + 1 - pass.widthRatio) * nqTot * width;
            const size_t chunkTrace =
                (e + 1 - pass.widthRatio) * nqTotTrace * width;
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    pass.inInterleaveWidth, m_implInterleaveWidth,
                    pass.chunkSize, nqTot,
                    pass.statePtr + n * pass.volCompSize + chunkState);
                LibUtilities::ReshapeStorage<ExecSpace>(
                    pass.inInterleaveWidth, m_implInterleaveWidth,
                    pass.chunkSize, nqTot,
                    pass.outPtr + n * pass.volCompSize + chunkState);
                for (TData *tracePtr : pass.tracePtrs)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        pass.inInterleaveWidth, m_implInterleaveWidth,
                        pass.chunkSize, nqTotTrace,
                        tracePtr + n * pass.traceCompSize + chunkTrace);
                }
            }
            for (unsigned int a = 0; a < (m_aux ? NAUX : 0); ++a)
            {
                LibUtilities::ReshapeStorage<ExecSpace>(
                    pass.auxInterleaveWidth, m_implInterleaveWidth,
                    pass.chunkSize, nqTot,
                    pass.auxPtr + a * pass.volCompSize + chunkState);
            }
        }
    }

    /// End the pass: the outputs leave at the state's width.
    void EndGroups(const GroupPass &pass, Block &outblock)
    {
        outblock.template SetInterleaveWidth<TData>(pass.inInterleaveWidth);
        for (Block *block : pass.traceBlocks)
        {
            block->template SetInterleaveWidth<TData>(pass.inInterleaveWidth);
        }
    }
};

/**
 * @brief The fused element-local advection operator of one block in the
 * Serial and AVX execution spaces, one element per SIMD lane.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation  MultiRegions::SumFac; not read.
 * @tparam Physics         The flux policy (AdvectionFusedElmtKernels.hpp).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename Physics,
          typename TData>
class AdvectionFusedElmtBlockOpImpl
    : public AdvectionFusedElmtHostBlock<ExecSpace, Physics, TData,
                                         AdvectionFusedElmtBlockOp>
{
    using HostBlock =
        AdvectionFusedElmtHostBlock<ExecSpace, Physics, TData,
                                    SolverCore::AdvectionFusedElmtBlockOp>;
    using BlockOpBase = AdvectionFusedElmtBlockOp<TData>;
    using Block       = typename HostBlock::Block;
    using simd_t      = typename HostBlock::simd_t;

    static constexpr unsigned int NDIM = Physics::NDIM;
    static constexpr unsigned int NAUX = Physics::NAUX;

public:
    /**
     * @brief Fetch the block's basis data and lay out the workspaces.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvectionFusedElmtBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : HostBlock(block_idx, exp, dataWarehouse)
    {
    }

    /// Registration name for BlockOperatorFactory, defined by the
    /// generated factory declaration unit.
    static std::string className;

    /// Creator function registered with BlockOperatorFactory: null for a
    /// block the implementation does not take (TakesBlock()).
    static std::unique_ptr<MultiRegions::BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        if (!HostBlock::TakesBlock(exp))
        {
            return nullptr;
        }
        return std::make_unique<AdvectionFusedElmtBlockOpImpl>(block_idx, exp,
                                                               dataWarehouse);
    }

protected:
    void v_ApplyFusedElmt(Block &state, Block *aux, Block &trace,
                          Block &out) override
    {
        ASSERTL1(static_cast<bool>(aux) == (NAUX > 0),
                 "The auxiliary fields are given exactly when the policy has "
                 "them.");

        this->m_aux   = aux;
        this->m_trace = &trace;

        this->ShapeSwitch(this->m_shapeType, [&](auto shape) {
            if constexpr (BlockOpBase::TakesShape(decltype(shape)::value, NDIM))
            {
                this->template ShapeBlock<decltype(shape)::value>(state, out);
            }
            else
            {
                NEKERROR(ErrorUtil::efatal, "shape type not implemented");
            }
        });

        this->m_aux   = nullptr;
        this->m_trace = nullptr;
    }

    // Shape-specific entry points, specialised for each shape of the
    // policy's dimension in the translation units generated from
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    /// The fused pass with the trace size parameter the generated switch
    /// chose: compile-time sizes for uniform counts, the runtime sizes
    /// otherwise.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TTraceSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        Block &inblock, Block &outblock,
        [[maybe_unused]] const TTraceSizeParameter traceSizeParam)
    {
        this->CheckAlignment(inblock, outblock);
        if constexpr (HostBlock::template UniformCounts<TTraceSizeParameter>())
        {
            ApplyAll<SHAPE_TYPE, DEFORMED>(
                inblock, outblock,
                HostBlock::template TemplatedPhysSize<TTraceSizeParameter>(),
                traceSizeParam);
        }
        else
        {
            ApplyAll<SHAPE_TYPE, DEFORMED>(
                inblock, outblock, this->NonTemplatedPhysSize(),
                this->template NonTemplatedTraceSize<NDIM>());
        }
    }

    /// Every element group of the block through the group kernel.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, typename TTraceSizeParameter>
    void ApplyAll(Block &inblock, Block &outblock,
                  const TPhysSizeParameter sizeParam,
                  const TTraceSizeParameter traceSizeParam)
    {
        const AdvectionFusedElmtWorkspace<simd_t> wsp =
            this->AdvectionWorkspace(this->WorkspaceBase());
        // One Jacobian per element when affine, one per point when deformed.
        const size_t jacSize = DEFORMED ? this->m_nqTot : 1;
        const size_t dfSize  = NDIM * NDIM * jacSize;
        const auto df        = reinterpret_cast<const simd_t *>(this->m_dfptr);
        const auto jac       = reinterpret_cast<const simd_t *>(this->m_jacptr);

        const auto pass = this->BeginGroups(inblock, outblock, {});
        for (size_t e = 0; e < pass.ngroups; ++e)
        {
            const auto group = this->BeginGroup(pass, e);
            AdvectionFusedElmtGroupKernel<SHAPE_TYPE, DEFORMED,
                                          HostBlock::MAXQ>(
                sizeParam, traceSizeParam, this->m_basisData, this->m_packing,
                this->m_physics, df + e * dfSize, jac + e * jacSize,
                group.state, group.Aux(), group.trace, group.out, wsp);
            this->EndGroup(pass, e, group);
        }
        this->EndGroups(pass, outblock);
    }
};

} // namespace Nektar::SolverCore::detail
