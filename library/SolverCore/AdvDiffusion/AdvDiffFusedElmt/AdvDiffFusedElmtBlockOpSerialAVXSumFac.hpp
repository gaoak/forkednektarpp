///////////////////////////////////////////////////////////////////////////////
//
// File: AdvDiffFusedElmtBlockOpSerialAVXSumFac.hpp
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
// element-local advection-diffusion operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file AdvDiffFusedElmtBlockOpSerialAVXSumFac.hpp
 * @brief Serial/AVX sum-factorised implementation of the per-block fused
 * element-local advection-diffusion operator.
 *
 * @details
 * The implementation builds on what the host implementations of the fused
 * element-local operators share
 * (detail::AdvectionFusedElmtHostBlock,
 * AdvectionFusedElmtBlockOpSerialAVXSumFac.hpp): the basis data, the advection
 * workspaces, the size parameters and the loop over the block's element groups.
 * It adds the trace normals, the viscous workspaces, the normal viscous flux on
 * the traces and the gradient trace of the listed elements. The kernels are in
 * AdvDiffFusedElmtBlockOpSerialAVXSumFacKernels.hpp.
 *
 * @note The operator is a class template on the flux policy, so the code
 * supplying the policy generates the translation units that include this
 * header, through ADD_POLICY_ELMT_BLOCK_OPERATOR (cmake/NektarCommon.cmake).
 * Other code goes through AdvDiffFusedElmtOp.hpp and the operator factory.
 */

#pragma once

#include <MultiRegions/DataWarehouse/TraceDataWarehouse.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtBlockOp.hpp>
#include <SolverCore/AdvDiffusion/AdvDiffFusedElmt/AdvDiffFusedElmtBlockOpSerialAVXSumFacKernels.hpp>
#include <SolverCore/Advection/AdvectionFusedElmt/AdvectionFusedElmtBlockOpSerialAVXSumFac.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace Nektar::SolverCore::detail
{

/**
 * @brief The fused element-local advection-diffusion operator of one block
 * in the Serial and AVX execution spaces, one element per SIMD lane.
 *
 * @details
 * The fields arrive at any interleave width and the outputs leave at the
 * state's (AdvectionFusedElmtHostBlock::BeginGroups()). The gradient trace
 * is indexed at its own width.
 *
 * @tparam ExecSpace       NektarSpaces::Serial or NektarSpaces::AVX.
 * @tparam Implementation  MultiRegions::SumFac; not read.
 * @tparam Physics         The flux policy (AdvDiffFusedElmtKernels.hpp).
 * @tparam TData           Floating-point type of the field data.
 */
template <typename ExecSpace, typename Implementation, typename Physics,
          typename TData>
class AdvDiffFusedElmtBlockOpImpl
    : public AdvectionFusedElmtHostBlock<ExecSpace, Physics, TData,
                                         AdvDiffFusedElmtBlockOp>
{
    using HostBlock =
        AdvectionFusedElmtHostBlock<ExecSpace, Physics, TData,
                                    SolverCore::AdvDiffFusedElmtBlockOp>;
    using BlockOpBase = AdvDiffFusedElmtBlockOp<TData>;
    using Block       = typename HostBlock::Block;
    using simd_t      = typename HostBlock::simd_t;
    using MemSpace    = typename HostBlock::MemSpace;

    static constexpr unsigned int NDIM = Physics::NDIM;
    static constexpr unsigned int NVAR = Physics::NVAR;
    static constexpr unsigned int NAUX = Physics::NAUX;

public:
    /**
     * @brief Fetch the block's basis data and trace normals, and add the
     * viscous workspaces to the layout.
     *
     * @param   block_idx       Index of the block.
     * @param   exp             Representative expansion of the block.
     * @param   dataWarehouse   Data warehouse shared with the other
     *                          operators on the expansion list.
     */
    AdvDiffFusedElmtBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : HostBlock(block_idx, exp, dataWarehouse)
    {
        m_normalsptr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::LocalTraceNormalKey<TData>(
                block_idx, HostBlock::m_implInterleaveWidth));

        const unsigned int nqTotTrace = this->m_packing.nqTotTrace;
        m_gradOffset = this->CarveWorkspace(size_t(NVAR) * this->m_varStride);
        m_viscFluxOffset =
            this->CarveWorkspace(size_t(NVAR) * this->m_varStride);
        m_traceCompsOffset = this->CarveWorkspace(size_t(NDIM) * nqTotTrace);
        m_gradTraceWspOffset =
            this->CarveWorkspace(size_t(NVAR) * NDIM * nqTotTrace);
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
        return std::make_unique<AdvDiffFusedElmtBlockOpImpl>(block_idx, exp,
                                                             dataWarehouse);
    }

protected:
    /// One listed element of the gradient trace: its lane in its element
    /// group, and its offset in the gradient trace block at the width that
    /// block was last seen at.
    struct GradTraceElement
    {
        unsigned int lane;
        size_t offset;
    };

    /// Each element's outward normals on its traces, at the
    /// implementation's width.
    const TData *m_normalsptr = nullptr;
    /// The offsets of the viscous workspaces of one element group (see
    /// AdvDiffFusedElmtWorkspace) in the workspace layout, after the
    /// advection ones (AdvectionFusedElmtHostBlock::CarveWorkspace()).
    size_t m_gradOffset = 0, m_viscFluxOffset = 0, m_traceCompsOffset = 0,
           m_gradTraceWspOffset = 0;
    /// The elements whose gradient trace is written, in list order
    /// (SetGradientTraceElements()); for each element group, the first of
    /// them in that group or a later one, so that group e's are
    /// `[m_gradTraceGroupStart[e], m_gradTraceGroupStart[e + 1])`; and the
    /// map of each into the gradient trace block, built for
    /// #m_gradTraceInterleaveWidth, the one thing that depends on the call.
    std::vector<unsigned int> m_gradTraceElements;
    std::vector<size_t> m_gradTraceGroupStart;
    std::vector<GradTraceElement> m_gradTraceMap;
    unsigned int m_gradTraceInterleaveWidth = 0;
    /// The viscous outputs ApplyFusedElmt() writes, held for that
    /// application alone and null otherwise.
    Block *m_normVisc  = nullptr;
    Block *m_gradTrace = nullptr;

    /// The workspaces as the kernels take them.
    AdvDiffFusedElmtWorkspace<simd_t> Workspace()
    {
        simd_t *base = this->WorkspaceBase();
        return AdvDiffFusedElmtWorkspace<simd_t>{
            this->AdvectionWorkspace(base), base + m_gradOffset,
            base + m_viscFluxOffset, base + m_traceCompsOffset,
            base + m_gradTraceWspOffset};
    }

    void v_SetGradientTraceElements(const std::vector<unsigned int> &elements,
                                    const size_t nelmtPad) override
    {
        const unsigned int implInterleaveWidth =
            HostBlock::m_implInterleaveWidth;
        m_gradTraceElements = elements;
        m_gradTraceGroupStart.assign(nelmtPad / implInterleaveWidth + 1, 0);
        size_t k = 0;
        for (size_t e = 0; e < m_gradTraceGroupStart.size(); ++e)
        {
            while (k < elements.size() && elements[k] / implInterleaveWidth < e)
            {
                ++k;
            }
            m_gradTraceGroupStart[e] = k;
        }
        m_gradTraceInterleaveWidth = 0;
        m_gradTraceMap.clear();
    }

    /**
     * @brief Map each listed element onto its lane in its element group, and
     * onto its place in the gradient trace block at @p interleaveWidth: the
     * listed elements in list order, interleaved as that block is.
     */
    void BuildGradTraceMap(const unsigned int interleaveWidth)
    {
        const size_t nqTotTrace = this->m_packing.nqTotTrace;
        m_gradTraceMap.clear();
        for (size_t k = 0; k < m_gradTraceElements.size(); ++k)
        {
            m_gradTraceMap.push_back(
                {m_gradTraceElements[k] % HostBlock::m_implInterleaveWidth,
                 (k / interleaveWidth) * nqTotTrace * interleaveWidth +
                     k % interleaveWidth});
        }
        m_gradTraceInterleaveWidth = interleaveWidth;
    }

    void v_ApplyAdvDiffFusedElmt(Block &state, Block *aux, Block &trace,
                                 Block &traceNormVisc, Block *gradTrace,
                                 Block &out) override
    {
        ASSERTL1(static_cast<bool>(aux) == (NAUX > 0),
                 "The auxiliary fields are given exactly when the policy has "
                 "them.");

        this->m_aux   = aux;
        this->m_trace = &trace;
        m_normVisc    = &traceNormVisc;
        // The listed elements and what depends on them alone were set with
        // the gradient trace (SetGradientTraceElements()); its width is the
        // caller's, and may change between calls.
        m_gradTrace = m_gradTraceElements.empty() ? nullptr : gradTrace;
        if (m_gradTrace &&
            m_gradTrace->GetInterleaveWidth() != m_gradTraceInterleaveWidth)
        {
            BuildGradTraceMap(m_gradTrace->GetInterleaveWidth());
        }

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
        m_normVisc    = nullptr;
        m_gradTrace   = nullptr;
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

    /**
     * @brief Every element group of the block through the group kernel, the
     * listed elements' gradient traces scattered after their group.
     */
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, typename TTraceSizeParameter>
    void ApplyAll(Block &inblock, Block &outblock,
                  const TPhysSizeParameter sizeParam,
                  const TTraceSizeParameter traceSizeParam)
    {
        constexpr unsigned int width  = simd_t::width;
        const unsigned int nqTotTrace = this->m_packing.nqTotTrace;
        const AdvDiffFusedElmtWorkspace<simd_t> wsp = Workspace();

        // One Jacobian per element when affine, one per point when deformed.
        const size_t jacSize = DEFORMED ? this->m_nqTot : 1;
        const size_t dfSize  = NDIM * NDIM * jacSize;
        const auto df        = reinterpret_cast<const simd_t *>(this->m_dfptr);
        const auto jac       = reinterpret_cast<const simd_t *>(this->m_jacptr);
        const auto normals   = reinterpret_cast<const simd_t *>(m_normalsptr);
        const size_t traceCompSize = m_normVisc->CompSize();
        TData *normViscPtr = m_normVisc->template GetPtr<MemSpace, WriteOnly>();
        TData *gradTracePtr =
            m_gradTrace ? m_gradTrace->template GetPtr<MemSpace, ReadWrite>()
                        : nullptr;

        // Each group's listed elements, or null when no gradient trace is
        // written.
        const size_t *groupStart =
            m_gradTrace ? m_gradTraceGroupStart.data() : nullptr;
        const auto pass = this->BeginGroups(inblock, outblock, {m_normVisc});
        for (size_t e = 0; e < pass.ngroups; ++e)
        {
            const auto group         = this->BeginGroup(pass, e);
            const size_t traceOffset = e * nqTotTrace * width;
            const simd_t *groupNormals[NDIM];
            simd_t *normVisc[NVAR];
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                groupNormals[d] =
                    normals + (d * traceCompSize) / width + e * nqTotTrace;
            }
            for (unsigned int n = 0; n < NVAR; ++n)
            {
                normVisc[n] = reinterpret_cast<simd_t *>(
                    normViscPtr + n * traceCompSize + traceOffset);
            }
            const bool writeGradTrace =
                groupStart && groupStart[e] != groupStart[e + 1];

            AdvDiffFusedElmtGroupKernel<SHAPE_TYPE, DEFORMED, HostBlock::MAXQ>(
                sizeParam, traceSizeParam, this->m_basisData, this->m_packing,
                this->m_physics, df + e * dfSize, jac + e * jacSize,
                group.state, group.Aux(), groupNormals, group.trace, normVisc,
                group.out, writeGradTrace, wsp);

            // The listed elements' gradient traces, lane by lane.
            if (writeGradTrace)
            {
                ScatterGradTrace(groupStart[e], groupStart[e + 1], gradTracePtr,
                                 wsp.gradTrace);
            }
            this->EndGroup(pass, e, group);
        }
        this->EndGroups(pass, outblock);
    }

    /// The gradient traces of the listed elements `[first, last)` of one
    /// element group, lane by lane, from the group's workspace into the
    /// gradient trace block.
    void ScatterGradTrace(const size_t first, const size_t last,
                          TData *gradTracePtr, const simd_t *gradTraceWsp)
    {
        constexpr unsigned int width   = simd_t::width;
        const unsigned int nqTotTrace  = this->m_packing.nqTotTrace;
        const size_t gradTraceCompSize = m_gradTrace->CompSize();
        const auto source = reinterpret_cast<const TData *>(gradTraceWsp);
        for (size_t k = first; k < last; ++k)
        {
            const GradTraceElement &elmt = m_gradTraceMap[k];
            for (unsigned int c = 0; c < NVAR * NDIM; ++c)
            {
                TData *dest =
                    gradTracePtr + c * gradTraceCompSize + elmt.offset;
                for (unsigned int i = 0; i < nqTotTrace; ++i)
                {
                    dest[i * m_gradTraceInterleaveWidth] =
                        source[(c * nqTotTrace + i) * width + elmt.lane];
                }
            }
        }
    }
};

} // namespace Nektar::SolverCore::detail
