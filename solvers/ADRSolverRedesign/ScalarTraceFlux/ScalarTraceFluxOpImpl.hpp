///////////////////////////////////////////////////////////////////////////////
//
// File: ScalarTraceFluxOpImpl.hpp
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
// Description: Trace Flux routines with scalar Riemann solvers
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ADRSolverRedesign/ScalarTraceFlux/ScalarTraceFluxOp.hpp"
#include "SolverCore/TraceFlux/TraceFluxOpImpl.hpp"

#include "ADRSolverRedesign/TraceFluxSolver/TraceFluxSolverKernels.hpp"

namespace Nektar::detail
{

/**
 * @brief Advective trace flux for scalar transport, evaluated with a
 * pluggable Riemann solver.
 *
 * Supplies the flux evaluation that SolverCore::detail::TraceFluxOpImpl leaves
 * abstract. The base class owns the trace numbering and the gather/scatter
 * between element-local and global-trace layouts; this class walks the
 * resulting blocks and, for each one, assembles the advection velocity and
 * normals the Riemann solver needs before handing the two trace states to @p
 * FluxKernel.
 *
 * Both InteriorOp() and BoundaryOp() follow the same three steps, marked in
 * the code: gather both sides of the trace, evaluate the flux, scatter the
 * result back. Whether the scatter overwrites or accumulates is controlled by
 * `SolverCore::TraceFluxOp::SetAppend`.
 *
 * The advection velocity is supplied through
 * `SolverCore::TraceFluxOp::SetTraceAdvVel` before the operator is applied.
 * The operator keeps a non-owning pointer to the caller's field, as the other
 * operators do for the fields they are handed, and samples it onto the global
 * trace in #m_gloTraceAdvVel the first time it is applied after the pointer is
 * set; the sample is reused until SetTraceAdvVel() is called again.
 *
 * @tparam FluxKernel Riemann solver, as a class template on the execution
 *                    space. It must provide
 *                    `operator()(blksize, velComps, fluxComps, vel, norm, fwd,
 *                    bwd, flux)`; see `UpwindSolverKernel`.
 * @tparam ExecSpace  Execution space the operator runs in.
 * @tparam TData      Floating-point representation used by the field data.
 *
 * @see SolverCore::detail::TraceFluxOpImpl for the trace numbering and the
 * meaning of T0/T1.
 * @see FluxKernelLauncher in TraceFluxSolverKernels.hpp, which applies
 *      @p FluxKernel over a padded block.
 */
template <template <typename> typename FluxKernel, typename ExecSpace,
          typename TData>
class ScalarTraceFluxOpImpl
    : public SolverCore::detail::TraceFluxOpImpl<ExecSpace, TData,
                                                 ScalarTraceFluxOp<TData>>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator and size its advection-velocity storage.
     *
     * Delegates all trace numbering to SolverCore::detail::TraceFluxOpImpl,
     * then allocates #m_gloTraceAdvVel over the global trace and a scratch
     * buffer large enough for the widest padded trace block found in either the
     * interior or the Dirichlet boundary blocks.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the field components.
     */
    ScalarTraceFluxOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                          const std::vector<std::string> &components)
        : SolverCore::detail::TraceFluxOpImpl<ExecSpace, TData,
                                              ScalarTraceFluxOp<TData>>(
              std::move(expansionList), components),
          m_gloTraceAdvVel(LibUtilities::Field<TData, FieldState::Phys>(
              "Trace Advection Velocity",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              expansionList->GetExp(0)->GetShapeDimension(), 1))
    {
        // initialise trace advection
        m_gloTraceAdvVel.template Initialize<MemSpace>(0.0);

        size_t maxNTraceNPts = 0;
        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            maxNTraceNPts =
                std::max(maxNTraceNPts, this->m_intT0[b].m_nTraceXnPtsPad);
        }
        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            maxNTraceNPts =
                std::max(maxNTraceNPts, this->m_bndT0[b].m_nTraceXnPtsPad);
        }
        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            maxNTraceNPts =
                std::max(maxNTraceNPts, this->m_parT0[b].m_nTraceXnPtsPad);
        }

        auto maxTraceSizeDim = maxNTraceNPts * (this->m_traceDim + 1);

        // set up local workspace for advection velocity
        m_wsploc = LibUtilities::MemoryRegion<TData>(maxTraceSizeDim);
        m_wsploc.template Initialize<MemSpace>(0.0);
        m_advVel = m_wsploc.template GetPtr<MemSpace, ReadWrite>();
    }

    /// Key this operator is registered under in the OperatorFactory. Defined
    /// by the generated factory boilerplate, not in this header.
    // className - for OperatorFactory
    static std::string className;

    /**
     * @brief Factory creator function.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the field components.
     *
     * @return A new operator instance, owned by the caller.
     */
    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            ScalarTraceFluxOpImpl<FluxKernel, ExecSpace, TData>>(expansionList,
                                                                 components);
    }

protected:
    /**
     * @brief Synchronise every block of @p field after the first.
     *
     * Memory is synchronised a block at a time, while the kernels below
     * walk a field from its first block's pointer; taking that pointer
     * therefore leaves every later block unsynchronised. Each field is
     * walked to its own length, since they do not share a block count:
     * the trace normals are built as a single block.
     */
    template <typename MemAccess, typename TField>
    NEK_FORCE_INLINE static void SyncTrailingBlocks(TField &field)
    {
        for (unsigned blk = 1; blk < field.GetBlocks().size(); ++blk)
        {
            field.GetBlocks()[blk].template GetPtr<MemSpace, MemAccess>();
        }
    }
    /// The caller's advection velocity on the local trace, not owned here.
    /// Set by v_SetTraceAdvVel(); the caller keeps it alive.
    LibUtilities::Field<TData, FieldState::Phys> *m_traceAdvVel = nullptr;
    /// #m_traceAdvVel has changed since #m_gloTraceAdvVel was last sampled.
    bool m_gloTraceAdvVelStale = false;
    /// Advection velocity sampled on the global trace, one component per
    /// coordinate direction. Filled from #m_traceAdvVel by
    /// CheckTraceAdvVel().
    LibUtilities::Field<TData, FieldState::Phys> m_gloTraceAdvVel;
    /// Backing allocation for #m_advVel.
    LibUtilities::MemoryRegion<TData> m_wsploc;
    /// Scratch holding the advection velocity for the block currently being
    /// processed, in the same packed layout as
    /// `SolverCore::detail::TraceFluxOpImpl::m_norms`.
    TData *m_advVel;

    // The base declares each hook twice, for the two- and three-field forms;
    // overriding one form alone would hide the other.
    using SolverCore::TraceFluxOp<TData>::v_Apply;
    using SolverCore::TraceFluxOp<TData>::v_ApplyParallel;
    /**
     * @brief Evaluate the advective trace flux on the interior and boundary
     * traces.
     *
     * Zeroes @p flux, then dispatches to InteriorOp() and BoundaryOp() for
     * the trace dimension determined at construction. Neither needs anything
     * from another rank, so a caller may run this between
     * BeginParallelExchange() and EndParallelExchange(); the partition traces
     * are v_ApplyParallel()'s, afterwards.
     *
     * @param in   - Physical-space scalar field to take trace values from.
     * @param flux - Trace-space output.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        // primarily here to ensure all blocks are intialised since
        // the GetPtr() below will only intiialise first block.
        flux.template Initialize<MemSpace>(0);

        CheckTraceAdvVel();

        switch (this->m_traceDim)
        {
            case 0:
                // loop over interior and boundary traces and evaluate flux
                // using the Op methods defined below
                this->template InteriorOp<0>(in, flux);
                this->template BoundaryOp<0>(in, flux);
                break;
            case 1:
                this->template InteriorOp<1>(in, flux);
                this->template BoundaryOp<1>(in, flux);
                break;
            case 2:
                this->template InteriorOp<2>(in, flux);
                this->template BoundaryOp<2>(in, flux);
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Uknown value of m_traceDim");
                break;
        }
    }

    /**
     * @brief Evaluate the advective trace flux on the partition-cut traces.
     *
     * Reads what BeginParallelExchange() / EndParallelExchange() delivered,
     * so it must follow the wait; @p flux must be the field v_Apply() wrote,
     * whose partition slots this fills. Does nothing at all in serial.
     */
    void v_ApplyParallel(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        CheckTraceAdvVel();

        switch (this->m_traceDim)
        {
            case 0:
                this->template ParallelOp<0>(in, flux);
                break;
            case 1:
                this->template ParallelOp<1>(in, flux);
                break;
            case 2:
                this->template ParallelOp<2>(in, flux);
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "Uknown value of m_traceDim");
                break;
        }
    }

    /**
     * @brief Evaluate the flux on every interior trace.
     *
     * For each interior block: gather the forward and backward trace states,
     * gather the advection velocity and trace normals for the same points into
     * #m_advVel and `SolverCore::detail::TraceFluxOpImpl::m_norms`, apply
     * @p FluxKernel, and scatter the single-valued result back to both
     * elements.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space scalar field.
     * @param  flux     Trace-space output.
     *
     * @note The normals and velocity are read from the global trace using the
     *       block-component offsets, so #m_gloTraceAdvVel must already be
     *       populated; it is not derived from @p in.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void InteriorOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux = flux.GetNumComponents();
        auto dim     = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // Initialise every block of the output: the GetPtr() below returns
        // the first block's pointer and the kernel walks the field from it.
        flux.template Initialize<MemSpace>(0);

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();
        auto advVelPtr = m_gloTraceAdvVel.GetBlocks()[0]
                             .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);
        SyncTrailingBlocks<ReadOnly>(m_gloTraceAdvVel);

        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1
            this->template SetInteriorParams<TRACEDIM>(b);
            this->template GetInteriorTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1);

            // Step 2: Solve Riemann proble on close packed variables
            // left and right  traces (m_gloT0, m_gloT1),
            auto numBlock = this->m_intT0[b].offset.size();
            auto npTBlock = this->m_intT0[b].m_nTraceXnPtsPad;
            // gather normal advvel and normals, which share a layout on the
            // global trace and so a single set of offsets
            const auto G =
                this->GetGloTraceOffsetView(this->m_intGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, advVelPtr, G,
                this->m_advVel);
            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr, G,
                this->m_norms);

            // input close packed variables from both traces
            // (m_gloT0, m_gloT1), apply flux Kernel and return flux
            FluxKernelLauncher<FluxKernel, ExecSpace>(
                npTBlock, dim, numflux, this->m_advVel, this->m_norms,
                this->m_gloT0, this->m_gloT1, this->m_flux);

            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackInteriorFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr);
            }
            else
            {
                this->template InterpBackInteriorFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr);
            }
        }
    }

    /**
     * @brief Evaluate the flux on every trace cut by the partitioner.
     *
     * InteriorOp() for traces whose backward element is on another rank. The
     * flux is the interior one throughout - the neighbouring state is a real
     * state, not a boundary condition - and the only differences are that the
     * backward side is read from the receive buffer, which
     * `SolverCore::detail::TraceFluxOpImpl::GetParallelTraces()` handles, and
     * that the result is scattered to the local element alone. The neighbour
     * computes the same flux from the same two states and scatters it to the
     * other element itself.
     *
     * Returns immediately when there are no such traces, which is every case
     * in serial.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space scalar field.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void ParallelOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        if (this->m_parT0.empty())
        {
            return;
        }

        auto numflux = flux.GetNumComponents();
        auto dim     = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();
        auto advVelPtr = m_gloTraceAdvVel.GetBlocks()[0]
                             .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadWrite>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);
        SyncTrailingBlocks<ReadOnly>(m_gloTraceAdvVel);

        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1
            this->template SetParallelParams<TRACEDIM>(b);
            this->template GetParallelTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1);

            // Step 2: Solve Riemann problem on close packed variables
            // left and right traces (m_gloT0, m_gloT1),
            auto numBlock = this->m_parT0[b].offset.size();
            auto npTBlock = this->m_parT0[b].m_nTraceXnPtsPad;
            // gather normal advvel and normals
            const auto G =
                this->GetGloTraceOffsetView(this->m_parGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, advVelPtr, G,
                this->m_advVel);
            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr, G,
                this->m_norms);

            FluxKernelLauncher<FluxKernel, ExecSpace>(
                npTBlock, dim, numflux, this->m_advVel, this->m_norms,
                this->m_gloT0, this->m_gloT1, this->m_flux);

            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackParallelFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr);
            }
            else
            {
                this->template InterpBackParallelFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr);
            }
        }
    }

    /**
     * @brief Evaluate the flux on every Dirichlet boundary trace.
     *
     * As InteriorOp(), but the exterior state comes from the boundary
     * condition operator rather than a neighbouring element, and the result is
     * scattered to the interior element only.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space scalar field.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void BoundaryOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux = flux.GetNumComponents();
        auto dim     = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();
        auto advVelPtr = m_gloTraceAdvVel.GetBlocks()[0]
                             .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);
        SyncTrailingBlocks<ReadOnly>(m_gloTraceAdvVel);

        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            // Step 1: Get global trace 't' and put into m_gloT0 and
            // m_gloT1
            this->template SetBoundaryParams<TRACEDIM>(b);
            this->template GetBoundaryTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1);

            // Step 2: Solve Riemann proble on  close packed variables
            // left and right  traces (m_gloT0, m_gloT1),
            auto numBlock = this->m_bndT0[b].offset.size();
            auto npTBlock = this->m_bndT0[b].m_nTraceXnPtsPad;
            // gather normal advvel and normals
            const auto G =
                this->GetGloTraceOffsetView(this->m_bndGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, advVelPtr, G,
                this->m_advVel);
            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr, G,
                this->m_norms);

            // input close packed variables from both traces
            // (m_gloT0, m_gloT1), apply flux Kernel and
            // return flux
            FluxKernelLauncher<FluxKernel, ExecSpace>(
                npTBlock, dim, numflux, this->m_advVel, this->m_norms,
                this->m_gloT0, this->m_gloT1, this->m_flux);

            // Step 3: Replace flux values from (m_gloT0,m_gloT1) into
            // the local flux storage
            if (this->m_append)
            {
                this->template InterpBackDirichletFlux<TRACEDIM, true>(
                    b, numflux, fluxPtr);
            }
            else
            {
                this->template InterpBackDirichletFlux<TRACEDIM, false>(
                    b, numflux, fluxPtr);
            }
        }
    }

    /**
     * @brief Sample a volume advection velocity onto the global trace.
     *
     * Populates #m_gloTraceAdvVel, which InteriorOp() and BoundaryOp() then
     * read. On interior traces the two sides are averaged, so that both
     * elements see the same velocity and the flux stays single-valued. On
     * boundary traces the interior value is taken unchanged, since the boundary
     * condition operator does not generally carry advection velocity data.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  AdvVel   Physical-space advection velocity, one component per
     *                  coordinate direction.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void SetTraceAdvVel(
        LibUtilities::Field<TData, FieldState::Phys> &AdvVel)
    {
        auto numComp = AdvVel.GetNumComponents();
        this->SetWorkSpace(numComp);

        // get field pointers
        auto inPtr =
            AdvVel.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto advVelPtr = m_gloTraceAdvVel.GetBlocks()[0]
                             .template GetPtr<MemSpace, WriteOnly>();

        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            // Step 1: Get global trace 't' and put into m_gloT0 and
            // m_gloT1
            this->template SetInteriorParams<TRACEDIM>(b);
            this->template GetInteriorTraces<TRACEDIM>(
                b, numComp, inPtr, this->m_gloT0, this->m_gloT1);

            unsigned npT = 0;
            if constexpr (TRACEDIM == 0)
            {
                npT = 1;
            }
            else if constexpr (TRACEDIM == 1)
            {
                npT = this->m_npT[0];
            }
            else if constexpr (TRACEDIM == 2)
            {
                npT = this->m_npT[0] * this->m_npT[1];
            }

            auto numBlock = this->m_intT0[b].offset.size();
            auto npTBlock = this->m_intT0[b].m_nTraceXnPtsPad;

            // Step 2: store the average of the two sides on the global trace
            SolverCore::detail::ScatterGloTraceComponentsKernel<ExecSpace,
                                                                true>(
                numBlock, npT, npTBlock, numComp, this->m_gloT0, this->m_gloT1,
                this->GetGloTraceOffsetView(this->m_intGloTraceDev[b]),
                advVelPtr);
        }

        // A trace cut by the partitioner takes the same average as any other
        // interior trace, so the neighbour's half of it has to be fetched. It
        // is fetched once, here, rather than every apply: the sample is held
        // until the velocity is set again.
        this->ExchangeParallelTracesBlocking(0, inPtr, numComp);

        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            this->template SetParallelParams<TRACEDIM>(b);
            this->template GetParallelTraces<TRACEDIM>(
                b, numComp, inPtr, this->m_gloT0, this->m_gloT1);

            auto numBlock = this->m_parT0[b].offset.size();
            auto npTBlock = this->m_parT0[b].m_nTraceXnPtsPad;

            SolverCore::detail::ScatterGloTraceComponentsKernel<ExecSpace,
                                                                true>(
                numBlock, this->m_npTot, npTBlock, numComp, this->m_gloT0,
                this->m_gloT1,
                this->GetGloTraceOffsetView(this->m_parGloTraceDev[b]),
                advVelPtr);
        }

        // just take interior points rather than average since
        // do not typically have boundary conditions for
        // advection velocity
        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            // Step 1: Get global trace 't' and put into m_gloT0 and
            // m_gloT1
            this->template SetBoundaryParams<TRACEDIM>(b);
            this->template GetBoundaryTraces<TRACEDIM, false>(
                b, numComp, inPtr, this->m_gloT0, this->m_gloT1);

            auto numBlock = this->m_bndT0[b].offset.size();
            auto npTBlock = this->m_bndT0[b].m_nTraceXnPtsPad;

            // Step 2: take the interior point; no averaging, since advection
            // velocity typically has no boundary condition of its own
            SolverCore::detail::ScatterGloTraceComponentsKernel<ExecSpace,
                                                                false>(
                numBlock, this->m_npTot, npTBlock, numComp, this->m_gloT0,
                static_cast<const TData *>(nullptr),
                this->GetGloTraceOffsetView(this->m_bndGloTraceDev[b]),
                advVelPtr);
        }
    }

    /**
     * @brief Keep a pointer to the caller's advection velocity.
     *
     * Nothing is read here: the field is sampled onto the global trace by
     * CheckTraceAdvVel() on the next apply, so the caller must keep it
     * alive for as long as this operator may be applied.
     *
     * @param AdvVel - Physical-space advection velocity on the local trace.
     */
    void v_SetTraceAdvVel(
        LibUtilities::Field<TData, FieldState::Phys> &AdvVel) override
    {
        m_traceAdvVel         = &AdvVel;
        m_gloTraceAdvVelStale = true;
    }

    /**
     * @brief Sample the advection velocity before any exchange is in flight.
     *
     * The sampling averages each interior trace, so a trace the partitioner
     * cut needs its neighbour's half and performs an exchange of its own on
     * the same channel. Doing that from Apply() would resize a channel the
     * caller had already started an exchange on - the solution carries a
     * different number of components - so it happens here instead, at the
     * one entry point that runs before anything is in flight.
     */
    void v_BeginParallelExchange(
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        const unsigned chan) override
    {
        CheckTraceAdvVel();
        SolverCore::detail::TraceFluxOpImpl<
            ExecSpace, TData,
            ScalarTraceFluxOp<TData>>::v_BeginParallelExchange(trace, chan);
    }

    /// Sample #m_traceAdvVel onto the global trace if it has changed since the
    /// last sample, dispatching on the trace dimension. Cheap and does
    /// nothing when it has not. Called from v_BeginParallelExchange(), and
    /// from the applies for a caller that never starts an exchange.
    void CheckTraceAdvVel()
    {
        if (!m_gloTraceAdvVelStale)
        {
            return;
        }

        ASSERTL0(m_traceAdvVel,
                 "The advection velocity was not set before the scalar trace "
                 "flux was applied.");

        switch (this->m_traceDim)
        {
            case 0:
                this->template SetTraceAdvVel<0>(*m_traceAdvVel);
                break;
            case 1:
                this->template SetTraceAdvVel<1>(*m_traceAdvVel);
                break;
            case 2:
                this->template SetTraceAdvVel<2>(*m_traceAdvVel);
                break;
        }

        m_gloTraceAdvVelStale = false;
    }
};

} // namespace Nektar::detail
