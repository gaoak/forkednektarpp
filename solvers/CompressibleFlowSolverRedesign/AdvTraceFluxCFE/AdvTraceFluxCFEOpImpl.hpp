///////////////////////////////////////////////////////////////////////////////
//
// File: AdvTraceFluxCFEOpImpl.hpp
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
// Description: AdvTrace Flux routines with scalar Riemann solvers
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "AdvTraceFluxCFEKernels.hpp"
#include "AdvTraceFluxCFEOp.hpp"
#include "SolverCore/RiemannSolver/RiemannSolverKernels.hpp"
#include "SolverCore/TraceFlux/TraceFluxOpImpl.hpp"

namespace Nektar::detail
{
/**
 * @brief Inviscid trace flux for the compressible Euler equations.
 *
 * Supplies the flux evaluation that SolverCore::detail::TraceFluxOpImpl leaves
 * abstract. The base class owns the trace numbering and the gather/scatter
 * between element-local and global-trace layouts; this class walks the
 * resulting blocks, gathers the trace normals for each one, and hands the two
 * conserved states to RiemannKernelLauncher().
 *
 * The conserved state is ordered as density, momentum components, total
 * energy, so `numflux` is the spatial dimension plus two.
 *
 * The Riemann solvers work in Cartesian components through the trace
 * normal, so the normals are all they need: nothing is rotated or cached.
 *
 * @tparam RiemannKernel Flux function, as a class template on execution space,
 *                       equation-of-state parameters and dimension; see the
 *                       solvers under `RiemannSolver/`.
 * @tparam EoSParamType  Equation-of-state parameter pack, built from the
 *                       session by `SetUpEquationOfState`.
 * @tparam ExecSpace     Execution space the operator runs in.
 * @tparam TData         Floating-point representation used by the field data.
 *
 * @see SolverCore::detail::TraceFluxOpImpl for the trace numbering and the
 * meaning of T0/T1.
 * @see AdvDiffTraceFluxCFEOpImpl for the viscous counterpart, which adds a
 *      diffusion contribution on top of this same Riemann solve.
 */
template <template <typename, typename, unsigned int> typename RiemannKernel,
          typename EoSParamType, typename ExecSpace, typename TData>
class AdvTraceFluxCFEOpImpl
    : public SolverCore::detail::TraceFluxOpImpl<ExecSpace, TData,
                                                 AdvTraceFluxCFEOp<TData>>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator and build the equation of state.
     *
     * Delegates trace numbering to SolverCore::detail::TraceFluxOpImpl.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the conserved variables.
     */
    AdvTraceFluxCFEOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                          const std::vector<std::string> &components)
        : SolverCore::detail::TraceFluxOpImpl<ExecSpace, TData,
                                              AdvTraceFluxCFEOp<TData>>(
              expansionList, components)
    {
        m_dim = expansionList->GetExp(0)->GetShapeDimension();

        SetUpEquationOfState(expansionList->GetSession(), m_EoS);
    }

    /// Key this operator is registered under in the OperatorFactory. Defined
    /// by the generated factory boilerplate, not in this header.
    // className - for OperatorFactory
    static std::string className;

    /**
     * @brief Factory creator function.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the conserved variables.
     *
     * @return A new operator instance, owned by the caller.
     */
    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<AdvTraceFluxCFEOpImpl<
            RiemannKernel, EoSParamType, ExecSpace, TData>>(expansionList,
                                                            components);
    }

protected:
    /**
     * @brief Synchronise every block of @p field after the first.
     *
     * Memory is synchronised a block at a time, while the kernels below walk
     * a field from its first block's pointer; taking that pointer therefore
     * leaves every later block unsynchronised. Each field is walked to its
     * own length, since they do not share a block count.
     */
    template <typename MemAccess, typename TField>
    NEK_FORCE_INLINE void SyncTrailingBlocks(TField &field)
    {
        for (unsigned blk = 1; blk < field.GetBlocks().size(); ++blk)
        {
            field.GetBlocks()[blk].template GetPtr<MemSpace, MemAccess>(
                this->m_opStreamID);
        }
    }

    /// Spatial dimension of the mesh.
    unsigned int m_dim;
    /// Equation-of-state parameters.
    EoSParamType m_EoS;

    // The base declares each hook twice, for the two- and three-field forms;
    // overriding one form alone would hide the other.
    using SolverCore::TraceFluxOp<TData>::v_Apply;
    using SolverCore::TraceFluxOp<TData>::v_ApplyParallel;
    /**
     * @brief Evaluate the inviscid trace flux on the interior and boundary
     * traces.
     *
     * Zeroes @p flux, then dispatches to InteriorOp() and BoundaryOp() for the
     * trace dimension determined at construction. Neither needs anything from
     * another rank, so a caller may run this between BeginParallelExchange()
     * and EndParallelExchange(); the partition traces are v_ApplyParallel()'s,
     * afterwards.
     *
     * @param in   - Physical-space conserved variables.
     * @param flux - Trace-space output.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        this->JoinBlockStreams();

        // loop over interior and boundary traces and evaluate flux
        // using InteriorOp and BoundaryOp defined below
        switch (this->m_traceDim)
        {
            case 0:
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

        this->ReleaseBlockStreams();
    }

    /**
     * @brief Evaluate the inviscid trace flux on the partition-cut traces.
     *
     * Reads what BeginParallelExchange() / EndParallelExchange() delivered,
     * so it must follow the wait; @p flux must be the field v_Apply() wrote,
     * whose partition slots this fills. Does nothing at all in serial.
     */
    void v_ApplyParallel(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        if (this->m_parT0.empty())
        {
            return;
        }

        this->JoinBlockStreams();

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

        this->ReleaseBlockStreams();
    }

    /**
     * @brief Evaluate the inviscid flux on every interior trace.
     *
     * For each interior block: gather the conserved state on both sides,
     * gather the trace normals into
     * `SolverCore::detail::TraceFluxOpImpl::m_norms`, solve the Riemann
     * problem, and scatter the single-valued flux back to both elements.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space conserved variables.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void InteriorOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
            this->m_opStreamID);
        auto fluxPtr = flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>(
            this->m_opStreamID);
        auto normPtr =
            this->m_traceNormals.GetBlocks()[0]
                .template GetPtr<MemSpace, ReadOnly>(this->m_opStreamID);

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

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
            // gather normals
            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr,
                this->GetGloTraceOffsetView(this->m_intGloTraceDev[b]),
                this->m_norms, this->m_opStreamID);

            // Solve the Riemann problem through the gathered normals.
            RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace, dim>(
                m_EoS, npTBlock, this->m_norms, this->m_gloT0, this->m_gloT1,
                this->m_flux, this->m_opStreamID);

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
     * @brief Evaluate the inviscid flux on every trace cut by the partitioner.
     *
     * InteriorOp() for traces whose backward element is on another rank. The
     * Riemann solve is the interior one - the exterior state is a real
     * neighbouring element's, delivered by BeginParallelExchange(), not a
     * boundary condition - and the result is scattered to the local element
     * alone; the neighbour computes the same flux from the same two states
     * and scatters it to the other element itself.
     *
     * Returns immediately when there are no such traces, which is every case
     * in serial.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space conserved variables.
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

        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
            this->m_opStreamID);
        auto fluxPtr = flux.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>(
            this->m_opStreamID);
        auto normPtr =
            this->m_traceNormals.GetBlocks()[0]
                .template GetPtr<MemSpace, ReadOnly>(this->m_opStreamID);

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadWrite>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1; the backward side comes from the receive
            // buffer of channel 0.
            this->template SetParallelParams<TRACEDIM>(b);
            this->template GetParallelTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, 0);

            // Step 2: Solve Riemann problem on close packed variables
            // left and right traces (m_gloT0, m_gloT1),
            auto numBlock = this->m_parT0[b].offset.size();
            auto npTBlock = this->m_parT0[b].m_nTraceXnPtsPad;
            // gather normals
            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr,
                this->GetGloTraceOffsetView(this->m_parGloTraceDev[b]),
                this->m_norms, this->m_opStreamID);

            // Solve the Riemann problem through the gathered normals.
            RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace, dim>(
                m_EoS, npTBlock, this->m_norms, this->m_gloT0, this->m_gloT1,
                this->m_flux, this->m_opStreamID);

            // Step 3: Replace flux values into the local flux storage, this
            // side only.
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
     * @brief Evaluate the inviscid flux on every Dirichlet boundary trace.
     *
     * As InteriorOp(), but the exterior state comes from the boundary
     * condition operator, the result is scattered to the interior element
     * only.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space conserved variables.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void BoundaryOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux       = flux.GetNumComponents();
        const unsigned dim = TRACEDIM + 1;

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>(
            this->m_opStreamID);
        auto fluxPtr = flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>(
            this->m_opStreamID);
        auto normPtr =
            this->m_traceNormals.GetBlocks()[0]
                .template GetPtr<MemSpace, ReadOnly>(this->m_opStreamID);

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

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
            // gather normals
            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, dim, normPtr,
                this->GetGloTraceOffsetView(this->m_bndGloTraceDev[b]),
                this->m_norms, this->m_opStreamID);

            // Solve the Riemann problem through the gathered normals.
            RiemannKernelLauncher<RiemannKernel, EoSParamType, ExecSpace, dim>(
                m_EoS, npTBlock, this->m_norms, this->m_gloT0, this->m_gloT1,
                this->m_flux, this->m_opStreamID);

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
     * @brief Apply the Riemann solver directly to supplied forward and
     * backward states, bypassing the trace machinery.
     *
     * A unit-test hook: it walks the blocks of @p fwd rather than the trace
     * blocks, so no gather, interpolation or scatter takes place and the flux
     * returned is the raw Riemann solution at each point.
     *
     * @param fwd  - Forward-side conserved state.
     * @param bwd  - Backward-side conserved state.
     * @param flux - Output flux.
     *
     * @note The normals are the operator's trace normals, block for block.
     */
    // function to help set up unit test for Riemann solver
    void v_ApplyUnitTest(
        LibUtilities::Field<TData, FieldState::Phys> &fwd,
        LibUtilities::Field<TData, FieldState::Phys> &bwd,
        LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        [[maybe_unused]] auto numflux = flux.GetNumComponents();

        ASSERTL1(fwd.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        for (unsigned int blk = 0; blk < fwd.GetBlocks().size(); ++blk)
        {
            // get field pointers
            auto fwdPtr =
                fwd.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
            auto bwdPtr =
                bwd.GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
            auto fluxPtr =
                flux.GetBlocks()[blk].template GetPtr<MemSpace, WriteOnly>();
            auto normPtr = this->m_traceNormals.GetBlocks()[blk]
                               .template GetPtr<MemSpace, ReadOnly>();

            const auto blksize = fwd.GetBlocks()[blk].CompSize();

            switch (m_dim)
            {
                case 1:
                    RiemannKernelLauncher<RiemannKernel, EoSParamType,
                                          ExecSpace, 1>(
                        m_EoS, blksize, normPtr, fwdPtr, bwdPtr, fluxPtr);
                    break;
                case 2:
                    RiemannKernelLauncher<RiemannKernel, EoSParamType,
                                          ExecSpace, 2>(
                        m_EoS, blksize, normPtr, fwdPtr, bwdPtr, fluxPtr);
                    break;
                case 3:
                    RiemannKernelLauncher<RiemannKernel, EoSParamType,
                                          ExecSpace, 3>(
                        m_EoS, blksize, normPtr, fwdPtr, bwdPtr, fluxPtr);
                    break;
            }
        }
    }
};
} // namespace Nektar::detail
