///////////////////////////////////////////////////////////////////////////////
//
// File: DiffusionScalarIPTraceFluxOpImpl.hpp
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
// Description: Scalar IP diffusion trace flux implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ADRSolverRedesign/DiffusionScalarIPTraceFlux/DiffusionScalarIPTraceFluxKernels.hpp"
#include "ADRSolverRedesign/DiffusionScalarIPTraceFlux/DiffusionScalarIPTraceFluxOp.hpp"
#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "MultiRegions/DataWarehouse/TraceDataWarehouse.hpp"
#include "SolverCore/TraceFlux/TraceFluxOpImpl.hpp"

namespace Nektar::detail
{

/**
 * @brief Interior-penalty diffusion trace flux for a scalar field.
 *
 * Evaluates the interior-penalty (IP) numerical flux for a diffusion term.
 * Where the advective operator needs only the solution on each side of a
 * trace, this one also needs its gradient, so it overrides the three-argument
 * `v_Apply(in, inDeriv, flux)` and gathers both fields onto the global trace
 * before calling DiffuseScalarTraceFluxKernel().
 *
 * The flux combines an averaged normal gradient with a penalty on the jump in
 * the solution. The penalty scales with a per-trace geometric factor taken
 * from the data warehouse, with the diffusion tensor resolved in the trace
 * normal direction, and with the user parameter `IPPenaltyCoeff`.
 *
 * Anisotropic diffusion is supported: #m_diffCoeff holds the upper triangle of
 * the symmetric diffusion tensor, defaulting to the identity. Use
 * `SolverCore::TraceFluxOp::SetDiffCoeff` to replace it.
 *
 * Session parameters read at construction:
 *
 * | Parameter        | Default | Use                                    |
 * |------------------|---------|----------------------------------------|
 * | `IPPenaltyCoeff` | 1.0     | Multiplies the jump penalty term.      |
 * | `IPSymmFluxCoeff`| 0.0     | Loaded into #m_IPSymmFluxCoeff.        |
 * | `IP2ndDervCoeff` | 0.0     | Loaded into #m_IP2ndDervCoeff.         |
 *
 * @tparam ExecSpace Execution space the operator runs in.
 * @tparam TData     Floating-point representation used by the field data.
 *
 * @note The symmetric-flux and second-derivative terms of the IP formulation
 *       are not evaluated by this operator. Their coefficients and the
 *       average/jump/coefficient storage they would need are present but
 *       unused.
 *
 * @see SolverCore::detail::TraceFluxOpImpl for the trace numbering and the
 * meaning of T0/T1.
 * @see DiffuseScalarTraceFluxKernel for the flux itself.
 */
template <typename ExecSpace, typename TData>
class DiffusionScalarIPTraceFluxOpImpl
    : public SolverCore::detail::ScalarDiffusionTraceFluxOpImpl<
          ExecSpace, TData, DiffusionScalarIPTraceFluxOp<TData>>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    /**
     * @brief Construct the operator, size its gradient workspace and read the
     * IP parameters.
     *
     * Delegates trace numbering to SolverCore::detail::TraceFluxOpImpl, sets
     * #m_diffCoeff to the identity tensor, then allocates space for the forward
     * and backward gradient traces and the per-point penalty factor, sized from
     * the widest padded trace block.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the field components.
     */
    DiffusionScalarIPTraceFluxOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : SolverCore::detail::ScalarDiffusionTraceFluxOpImpl<
              ExecSpace, TData, DiffusionScalarIPTraceFluxOp<TData>>(
              expansionList, components),
          m_traceAver(LibUtilities::Field<TData, FieldState::Phys>(
              "Scalar diffusion trace average",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_traceJump(LibUtilities::Field<TData, FieldState::Phys>(
              "Scalar diffusion trace jump",
              MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                  expansionList->GetTrace()),
              components.size(), 1)),
          m_symmCoeff(LibUtilities::Field<TData, FieldState::Coeff>(
              "Scalar diffusion symmetric trace coeff",
              MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                  expansionList),
              components.size(), 1))
    {
        m_nDim  = expansionList->GetCoordim(0);
        m_nComp = components.size();

        // Read-only mesh data spanning every block, one component per block,
        // so it is indexed by the scalar offset and not by the normals'. It
        // does not change, so it is fetched once here rather than per apply.
        m_penFactorPtr = this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::IPTraceScalarKey<TData>(
                0, MultiRegions::IPTraceScalarData::IPPenaltyFactor));

        // default diffusion coefficient
        std::vector<TData> diffCoeff(m_nDim * (m_nDim + 1) / 2, TData(0.0));
        for (unsigned int d = 0; d < m_nDim; ++d)
        {
            diffCoeff[d * (d + 3) / 2] = TData(1.0);
        }
        this->SetDiffCoeff(diffCoeff);

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

        auto maxTraceSizeDimComp = maxNTraceNPts * m_nDim * m_nComp;

        // Vector-aligned strides; see TraceFluxOpImpl::PadToVectorWidth().
        const size_t derivStride = this->PadToVectorWidth(maxTraceSizeDimComp);
        const size_t penStride   = this->PadToVectorWidth(maxNTraceNPts);

        m_wspDeriv =
            LibUtilities::MemoryRegion<TData>(derivStride * 2 + penStride);

        m_gloDerivT0 = m_wspDeriv.template GetPtr<MemSpace, WriteOnly>();
        m_gloDerivT1 = m_gloDerivT0 + derivStride;
        m_penFactor  = m_gloDerivT1 + derivStride;

        auto session = expansionList->GetSession();

        session->LoadParameter("IPSymmFluxCoeff", m_IPSymmFluxCoeff, 0.0);
        session->LoadParameter("IP2ndDervCoeff", m_IP2ndDervCoeff, 0.0);
        session->LoadParameter("IPPenaltyCoeff", m_IPPenaltyCoeff, 1.0);
    }

    /// Key this operator is registered under in the OperatorFactory. Defined
    /// by the generated factory boilerplate, not in this header.
    static std::string className;

    /**
     * @brief Factory creator function.
     *
     * @param expansionList - Expansion list the operator acts on.
     * @param components    - Names of the field components.
     *
     * @return A new operator instance, owned by the caller.
     */
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            DiffusionScalarIPTraceFluxOpImpl<ExecSpace, TData>>(expansionList,
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
    NEK_FORCE_INLINE static void SyncTrailingBlocks(TField &field)
    {
        for (unsigned blk = 1; blk < field.GetBlocks().size(); ++blk)
        {
            field.GetBlocks()[blk].template GetPtr<MemSpace, MemAccess>();
        }
    }

    /// Coordinate dimension of the mesh.
    unsigned int m_nDim;
    /// Number of field components.
    unsigned int m_nComp;
    /// Trace average and jump storage for the symmetric-flux term. Allocated
    /// but not currently read; see the class note.
    LibUtilities::Field<TData, FieldState::Phys> m_traceAver, m_traceJump;
    /// Coefficient-space storage for the symmetric-flux term. Allocated but
    /// not currently read; see the class note.
    LibUtilities::Field<TData, FieldState::Coeff> m_symmCoeff;
    /// Upper triangle of the symmetric diffusion tensor, in the packed
    /// ordering expected by `GetDiffCoeffMapPtr`. Length `nDim*(nDim+1)/2`.
    LibUtilities::MemoryRegion<TData> m_diffCoeff;

    /// Session parameter `IPSymmFluxCoeff`. Not currently read.
    TData m_IPSymmFluxCoeff = 0.0;
    /// Interior-penalty factor per trace, from the data warehouse.
    const TData *m_penFactorPtr = nullptr;
    /// Session parameter `IP2ndDervCoeff`. Not currently read.
    TData m_IP2ndDervCoeff = 0.0;
    /// Session parameter `IPPenaltyCoeff`, multiplying the jump penalty.
    TData m_IPPenaltyCoeff = 1.0;

    // workspace
    /// Backing allocation for the three pointers below.
    LibUtilities::MemoryRegion<TData> m_wspDeriv;
    /// Forward-side solution gradient, packed on the global trace with
    /// `nDim * nComp` components per point.
    TData *m_gloDerivT0;
    /// Backward-side solution gradient, same layout as #m_gloDerivT0.
    TData *m_gloDerivT1;
    /// Per-point geometric penalty factor gathered from the data warehouse.
    TData *m_penFactor;

    /**
     * @brief Replace the diffusion tensor.
     *
     * @param diffCoeff - Upper triangle of the symmetric diffusion tensor:
     *                    1, 3 or 6 entries in 1D, 2D or 3D respectively.
     */
    void v_SetDiffCoeff(std::vector<TData> &diffCoeff) override
    {
        [[maybe_unused]] const auto diffCoeffSize = m_nDim * (m_nDim + 1) / 2;
        ASSERTL1(diffCoeff.size() == diffCoeffSize,
                 "The number of diffusion coefficients must match 1, 3 or 6 "
                 "for a 1D, 2D or 3D case, respectively.");

        m_diffCoeff = LibUtilities::MemoryRegion<TData>::template FromVector<
            MemSpace, TData>(diffCoeff);
    }

    // The base declares each hook twice, for the two- and three-field forms;
    // overriding one form alone would hide the other.
    using SolverCore::TraceFluxOp<TData>::v_Apply;
    using SolverCore::TraceFluxOp<TData>::v_ApplyParallel;
    /**
     * @brief Evaluate the IP diffusion trace flux on the interior and
     * boundary traces.
     *
     * Dispatches to InteriorOp() and BoundaryOp() for the trace dimension
     * determined at construction. Neither needs anything from another rank,
     * so a caller may run this between BeginParallelExchange() and
     * EndParallelExchange(); the partition traces are v_ApplyParallel()'s,
     * afterwards.
     *
     * @param in      - Physical-space scalar field.
     * @param inDeriv - Physical-space gradient of @p in, `nDim` components per
     *                  field component.
     * @param flux    - Trace-space output.
     *
     * @note Unlike the advective operator, @p flux is not zeroed here; every
     *       trace is written by one of the passes, or accumulated into
     *       when `SolverCore::TraceFluxOp::SetAppend` is set.
     */
    void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
                 LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                 LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        switch (this->m_traceDim)
        {
            case 0:
                this->template InteriorOp<0>(in, inDeriv, flux);
                this->template BoundaryOp<0>(in, inDeriv, flux);
                break;
            case 1:
                this->template InteriorOp<1>(in, inDeriv, flux);
                this->template BoundaryOp<1>(in, inDeriv, flux);
                break;
            case 2:
                this->template InteriorOp<2>(in, inDeriv, flux);
                this->template BoundaryOp<2>(in, inDeriv, flux);
                break;
        }
    }

    /**
     * @brief Evaluate the IP diffusion trace flux on the partition-cut
     * traces.
     *
     * Reads the state and gradient the two exchange channels delivered, so it
     * must follow EndParallelExchange(); @p flux must be the field v_Apply()
     * wrote, whose partition slots this fills. Does nothing at all in serial.
     */
    void v_ApplyParallel(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        LibUtilities::Field<TData, FieldState::Phys> &flux) override
    {
        switch (this->m_traceDim)
        {
            case 0:
                this->template ParallelOp<0>(in, inDeriv, flux);
                break;
            case 1:
                this->template ParallelOp<1>(in, inDeriv, flux);
                break;
            case 2:
                this->template ParallelOp<2>(in, inDeriv, flux);
                break;
        }
    }

    /**
     * @brief Evaluate the IP flux on every interior trace.
     *
     * For each interior block: gather the solution and its gradient on both
     * sides of the trace, gather the normals and penalty factor for the same
     * points, apply DiffuseScalarTraceFluxKernel() in its interior form, and
     * scatter the result back to both elements.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space scalar field.
     * @param  inDeriv  Physical-space gradient of @p in.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    void InteriorOp(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                    LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux = flux.GetNumComponents();

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();

        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadOnly>(inDeriv);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        auto diffCoeff = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        for (unsigned b = 0; b < this->m_intT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1
            this->template SetInteriorParams<TRACEDIM>(b);
            this->template GetInteriorTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1);
            this->template GetInteriorTraces<TRACEDIM>(
                b, numflux * m_nDim, inDerivPtr, m_gloDerivT0, m_gloDerivT1);

            // Step 2: Solve Riemann proble on  close packed variables
            // left and right  traces (m_gloT0, m_gloT1),
            auto numBlock = this->m_intT0[b].offset.size();
            auto npTBlock = this->m_intT0[b].m_nTraceXnPtsPad;
            for (unsigned t = 0; t < numBlock; ++t)
            {
                // Single component per block, so not the normals' offset,
                // which is in their coordDim-component layout.
                //
                // Worth checking: the failure this catches - indexing this
                // array with the normals' coordDim-component offset - reads
                // adjacent trace data far more often than anything invalid, so
                // it gives a plausible wrong answer rather than a crash, and
                // only on a mesh with more than one trace block.
                ASSERTL1(
                    this->m_intGloTraceScalarOffset[b][t] + this->m_npTot <=
                        this->m_gloTraceScalarSize,
                    "Single-component trace data is read past the end of "
                    "its block: offset " +
                        std::to_string(this->m_intGloTraceScalarOffset[b][t]) +
                        " + " + std::to_string(this->m_npTot) + " points > " +
                        std::to_string(this->m_gloTraceScalarSize) + ".");
            }

            // gather the normals and the penalty factor
            const auto G =
                this->GetGloTraceOffsetView(this->m_intGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, m_nDim, normPtr, G,
                this->m_norms);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, m_penFactorPtr, G, m_penFactor);

            DiffuseScalarTraceFluxKernel<ExecSpace, true>(
                npTBlock, m_nDim, m_nComp, npTBlock, m_IPPenaltyCoeff,
                diffCoeff, this->m_norms, m_penFactor, this->m_gloT0,
                this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux);

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
     * @brief Evaluate the IP flux on every trace cut by the partitioner.
     *
     * InteriorOp() for traces whose backward element is on another rank, and
     * deliberately identical to it in every respect that matters to the
     * physics. A parallel trace is an *interior* trace: its exterior state and
     * gradient are a real neighbouring element's, not a boundary condition's,
     * so it takes `IsInterior = true` in the flux kernel - average weights of
     * one half, jump weights of one, and no boundary energy-flux weight. The
     * boundary path would reflect a perfectly good neighbour about itself and
     * return something finite, smooth and wrong.
     *
     * Both the state and its gradient are read from receive buffers, on
     * channels 0 and 1; see `SolverCore::TraceFluxOp::BeginParallelExchange`.
     * The result is scattered to the local element alone, the neighbour
     * computing the same flux from the same four traces and scattering it to
     * the other element itself.
     *
     * Returns immediately when there are no such traces, which is every case
     * in serial.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space state.
     * @param  inDeriv  Physical-space gradient of the state.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    NEK_FORCE_INLINE void ParallelOp(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        if (this->m_parT0.empty())
        {
            return;
        }

        auto numflux = flux.GetNumComponents();

        this->SetWorkSpace(numflux);

        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, ReadWrite>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadOnly>(inDeriv);
        SyncTrailingBlocks<ReadWrite>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        auto diffCoeff = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        for (unsigned b = 0; b < this->m_parT0.size(); ++b)
        {
            // Step 1: Get global traces associated with block b and put into
            // m_gloT0 and m_gloT1. The backward side of each comes from the
            // channel it was exchanged on rather than from the local field.
            this->template SetParallelParams<TRACEDIM>(b);
            this->template GetParallelTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1, 0);
            this->template GetParallelTraces<TRACEDIM>(
                b, numflux * m_nDim, inDerivPtr, m_gloDerivT0, m_gloDerivT1, 1);

            // Step 2: gather the normals and penalty factor for these traces
            auto numBlock = this->m_parT0[b].offset.size();
            auto npTBlock = this->m_parT0[b].m_nTraceXnPtsPad;
            for (unsigned t = 0; t < numBlock; ++t)
            {
                // Single component per block, so not the normals' offset; see
                // the note on the same read in InteriorOp(). Partitioning
                // multiplies trace blocks, so this is the case that guard was
                // written for.
                ASSERTL1(
                    this->m_parGloTraceScalarOffset[b][t] + this->m_npTot <=
                        this->m_gloTraceScalarSize,
                    "Single-component trace data is read past the end of "
                    "its block: offset " +
                        std::to_string(this->m_parGloTraceScalarOffset[b][t]) +
                        " + " + std::to_string(this->m_npTot) + " points > " +
                        std::to_string(this->m_gloTraceScalarSize) + ".");
            }

            const auto G =
                this->GetGloTraceOffsetView(this->m_parGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, m_nDim, normPtr, G,
                this->m_norms);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, m_penFactorPtr, G, m_penFactor);

            DiffuseScalarTraceFluxKernel<ExecSpace, true>(
                npTBlock, m_nDim, m_nComp, npTBlock, m_IPPenaltyCoeff,
                diffCoeff, this->m_norms, m_penFactor, this->m_gloT0,
                this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux);

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
     * @brief Evaluate the IP flux on every Dirichlet boundary trace.
     *
     * As InteriorOp(), with two differences. The exterior solution comes from
     * the boundary condition operator on the components carrying a Dirichlet
     * condition, and the exterior gradient from the components carrying a
     * Neumann one; everything else is left equal to its interior value, giving
     * no jump. The kernel is then instantiated in its boundary form, which
     * weights the average and jump terms accordingly.
     *
     * @tparam TRACEDIM Trace dimension; must equal `m_traceDim`.
     * @param  in       Physical-space scalar field.
     * @param  inDeriv  Physical-space gradient of @p in.
     * @param  flux     Trace-space output.
     */
    template <unsigned TRACEDIM>
    void BoundaryOp(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                    LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        auto numflux = flux.GetNumComponents();

        this->SetWorkSpace(numflux);

        ASSERTL1(in.GetNumComponents() == numflux,
                 "Assumption that input components are the "
                 "same as the flux components");

        // get field pointers
        auto inPtr = in.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto inDerivPtr =
            inDeriv.GetBlocks()[0].template GetPtr<MemSpace, ReadOnly>();
        auto fluxPtr =
            flux.GetBlocks()[0].template GetPtr<MemSpace, WriteOnly>();
        auto normPtr = this->m_traceNormals.GetBlocks()[0]
                           .template GetPtr<MemSpace, ReadOnly>();

        SyncTrailingBlocks<ReadOnly>(in);
        SyncTrailingBlocks<ReadOnly>(inDeriv);
        SyncTrailingBlocks<WriteOnly>(flux);
        SyncTrailingBlocks<ReadOnly>(this->m_traceNormals);

        auto diffCoeff = m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // These trace geometry/weight arrays are read-only mesh data, so
        // they live in the data warehouse instead of per-op Field storage.
        for (unsigned b = 0; b < this->m_bndT0.size(); ++b)
        {
            // Step 1: Get global trace 't' and put into m_gloT0 and
            // m_gloT1
            this->template SetBoundaryParams<TRACEDIM>(b);
            this->template GetBoundaryTraces<TRACEDIM>(
                b, numflux, inPtr, this->m_gloT0, this->m_gloT1);

            auto numBlock = this->m_bndT0[b].offset.size();
            auto npTBlock = this->m_bndT0[b].m_nTraceXnPtsPad;

            // The normals are gathered before the gradient traces, not after:
            // imposing a Neumann condition needs them to resolve the normal
            // component of the gradient.
            for (unsigned t = 0; t < numBlock; ++t)
            {
                // Single component per block, so not the normals' offset,
                // which is in their coordDim-component layout; see the note on
                // the same read in InteriorOp().
                ASSERTL1(
                    this->m_bndGloTraceScalarOffset[b][t] + this->m_npTot <=
                        this->m_gloTraceScalarSize,
                    "Single-component trace data is read past the end of "
                    "its block: offset " +
                        std::to_string(this->m_bndGloTraceScalarOffset[b][t]) +
                        " + " + std::to_string(this->m_npTot) + " points > " +
                        std::to_string(this->m_gloTraceScalarSize) + ".");
            }

            const auto G =
                this->GetGloTraceOffsetView(this->m_bndGloTraceDev[b]);

            SolverCore::detail::GatherGloTraceComponentsKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, m_nDim, normPtr, G,
                this->m_norms);
            SolverCore::detail::GatherGloTraceScalarKernel<ExecSpace>(
                numBlock, this->m_npTot, m_penFactorPtr, G, m_penFactor);

            // The gradient carries the Neumann conditions; components with any
            // other condition keep the exterior gradient equal to the interior.
            this->template GetNeumannBoundaryTraces<TRACEDIM>(
                b, numflux, m_nDim, inDerivPtr, this->m_norms, m_gloDerivT0,
                m_gloDerivT1);

            // Turn the exterior state into a ghost. The kernel averages
            // half-and-half on boundary traces as it does in the interior, so
            // imposing u = g needs u- = 2g - u+ rather than g itself: its
            // average with the interior is g and its jump is 2(g - u+), which
            // is what legacy produced with its own boundary weights.
            //
            // This is applied to every component, not only the Dirichlet ones.
            // GetBoundaryTraces() loads only components marked eDirichlet and
            // leaves the rest with u- = u+, for which 2u- - u+ is u+ again, so
            // Neumann and Robin components pass through untouched.
            //
            // Unlike the compressible operator, nothing here builds a reflected
            // exterior state of its own - there is no wall operator on this
            // path - so every boundary value is an imposed one and no
            // classification is needed. Should a reflective condition arrive,
            // it must be excluded here, exactly as AdvDiffTraceFluxCFEOpImpl
            // excludes its walls.
            SolverCore::detail::MakeDirichletGhostStateKernel<ExecSpace>(
                numBlock, this->m_npTot, npTBlock, numflux, this->m_gloT0,
                this->m_gloT1);

            // Step 2: Solve Riemann proble on  close packed variables
            // left and right  traces (m_gloT0, m_gloT1),

            DiffuseScalarTraceFluxKernel<ExecSpace, false>(
                npTBlock, m_nDim, m_nComp, npTBlock, m_IPPenaltyCoeff,
                diffCoeff, this->m_norms, m_penFactor, this->m_gloT0,
                this->m_gloT1, m_gloDerivT0, m_gloDerivT1, this->m_flux);

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
};

} // namespace Nektar::detail