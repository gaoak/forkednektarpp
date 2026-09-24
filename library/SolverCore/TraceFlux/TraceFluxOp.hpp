///////////////////////////////////////////////////////////////////////////////
//
// File: TraceFluxOp.hpp
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
// Description: TraceFlux operator base class.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SolverCore/SolverCore.hpp>

#include "Operators/Common/Operator.hpp"

#include "Operators/BndCondOps/BndCondPhys/BndCondPhysOp.hpp"
#include "SolverCore/BndCond/BndCondUpdateOp.hpp"

namespace Nektar::SolverCore
{

// TraceFlux operator base class
template <typename TData> class TraceFluxOp : public Operators::Operator<TData>
{

public:
    static std::shared_ptr<TraceFluxOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &method = "", const std::string &execStr = "")
    {
        // Force a genuine cross-library symbol reference into
        // libSolverCore - see EnsureLinked() in SolverCore.hpp. Without
        // it a linker that keeps only what is referenced can drop the
        // library, and the operator registrations go with it.
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string method0 = method;
        if (method == "" && session->DefinesSolverInfo("UpwindType"))
        {
            method0 += session->GetSolverInfo("UpwindType");
        }

        std::string execStr0 =
            (execStr == "")
                ? Operators::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        std::string requestedKey = method0 + execStr0;

        Operators::OperatorFactory<TData> &factory =
            Operators::GetOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            std::stringstream msg;
            msg << "No such operator: " << requestedKey << std::endl;
            factory.PrintAvailableClasses(msg);
            NEKERROR(ErrorUtil::efatal, msg.str());
        }

        return std::static_pointer_cast<TraceFluxOp<TData>>(
            factory.CreateInstance(requestedKey, expansionList, components));
    }

    static inline const std::string name = "";

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_Apply(in, flux);
    }

    void ApplyDirBoundary(LibUtilities::Field<TData, FieldState::Phys> &in,
                          LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyDirBoundary(in, flux);
    }

    void ApplyInterior(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyInterior(in, flux);
    }

    /**
     * @brief Evaluate the flux on the traces cut by the partitioner.
     *
     * The counterpart of Apply(), which covers the interior and boundary
     * traces and needs nothing from other ranks - so a caller overlapping
     * communication with work calls Apply() between BeginParallelExchange()
     * and EndParallelExchange(), and this afterwards, since it reads what the
     * exchange delivered. @p flux must be the field Apply() has already
     * written: the partition traces are written into their own slots of it.
     *
     * Does nothing in serial, or on a rank whose partition has no cut traces.
     */
    void ApplyParallel(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyParallel(in, flux);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_Apply(in, flux);
    }

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in,
               LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
               LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_Apply(in, inDeriv, flux);
    }

    void ApplyDirBoundary(LibUtilities::Field<TData, FieldState::Phys> &in,
                          LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                          LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyDirBoundary(in, inDeriv, flux);
    }

    void ApplyInterior(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                       LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyInterior(in, inDeriv, flux);
    }

    /// Two-trace form of ApplyParallel(), for operators whose flux takes the
    /// state and its gradient; same contract as the single-trace overload.
    void ApplyParallel(LibUtilities::Field<TData, FieldState::Phys> &in,
                       LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                       LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_ApplyParallel(in, inDeriv, flux);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in,
                    LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
                    LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        v_Apply(in, inDeriv, flux);
    }

    /**
     * @brief Start exchanging trace values with the neighbouring partitions.
     *
     * A trace cut by the partitioner has a real neighbouring element on
     * another rank, and needs that element's state before its flux can be
     * evaluated. This posts the exchange; EndParallelExchange() waits for it.
     *
     * The two are separate calls, and separate from Apply(), so that a caller
     * can put its local work between them - the boundary condition update, the
     * volume flux, the inner product, and Apply() itself, which covers only
     * the interior and boundary traces and so needs nothing from the exchange
     * - and pay for the messages only to the extent they outlast it. Only
     * ApplyParallel() has to wait. @p trace must be the same trace field the
     * Apply pair will be given, and must not be written to in between.
     *
     * Does nothing in serial, or on a rank whose partition has no cut traces,
     * so a caller need not ask which case it is in.
     *
     * @param trace   The trace field to send, and the one ApplyParallel()
     *                will read.
     * @param channel Which exchange this is. A diffusion operator has two
     *                traces to send - the state and its gradient, the latter
     *                carrying `nDim` times as many components - and starts
     *                them at different moments so each overlaps the work that
     *                follows it, so they must be in flight together. By
     *                convention 0 carries the state and 1 the gradient.
     */
    void BeginParallelExchange(
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        const unsigned channel = 0)
    {
        v_BeginParallelExchange(trace, channel);
    }

    /**
     * @brief Wait for every exchange begun by BeginParallelExchange().
     *
     * Waits on all channels in flight, so a caller that started two need only
     * call this once. Must be called before ApplyParallel(), which reads what
     * they delivered.
     */
    void EndParallelExchange()
    {
        v_EndParallelExchange();
    }

    void SetTraceAdvVel(LibUtilities::Field<TData, FieldState::Phys> &AdvVel)
    {
        v_SetTraceAdvVel(AdvVel);
    }

    /**
     * @brief Supply trace normals from outside, overriding the mesh.
     *
     * An operator normally takes its normals from the geometry, through
     * GlobalTraceNormalKey in the data warehouse. This replaces them and
     * marks the override, so that any later reload of the mesh-derived
     * default leaves the supplied values alone. It exists for callers that
     * have no mesh normals to speak of - the Riemann unit test and profiler
     * build a normals field directly - and is the only way such a caller can
     * drive the operator.
     *
     * Anything cached from the normals is invalidated through
     * v_OnTraceNormalsChanged().
     *
     * @param traceNormals - New trace normals; moved from.
     */
    void SetTraceNormals(
        LibUtilities::Field<TData, FieldState::Phys> &traceNormals)
    {
        v_SetTraceNormals(traceNormals);
    }

    /// True once SetTraceNormals() has overridden the mesh-derived normals.
    bool HasTraceNormalsOverride() const
    {
        return m_traceNormalsOverridden;
    }

    void SetDiffCoeff(std::vector<TData> &diffCoeff)
    {
        v_SetDiffCoeff(diffCoeff);
    }

    void SetAppend(const bool &append)
    {
        this->m_append = append;
    }

    /**
     * @brief Bring the boundary values up to date for time @p time.
     *
     * Forwards to the boundary condition operator, which returns immediately
     * unless a condition actually depends on time. An owner that advances in
     * time should call this each step: the values are otherwise those from
     * when the operator was built.
     */
    void UpdateBndPhys(const TData &time = 0.0)
    {
        if (m_BndCondOp)
        {
            m_BndCondOp->UpdateBndPhys(time);
        }
    }

    /// The boundary values this operator gathers from. Handed out so that an
    /// operator computing boundary states can write into the same storage,
    /// leaving the gather here unchanged.
    const std::shared_ptr<Operators::BndCondPhysOp<TData>> &GetBndCondPhysOp()
        const
    {
        return m_BndCondOp;
    }

    /**
     * @brief Tell this operator which boundary conditions are in play.
     *
     * A condition owns more than the boundary state. It may also declare what
     * kind of exterior state it built, and contribute to the boundary flux
     * itself - an adiabatic wall suppresses the energy component, an
     * isothermal one prescribes the exterior energy the diffusion averages
     * against. Those belong to the condition, but only this operator has the
     * trace layout to apply them in, so it needs to know which conditions
     * exist and which regions each one claims.
     *
     * Attached through AddBndCondUpdateOp(), which binds the storage at the
     * same moment, so the two always agree on the set of conditions.
     */
    void AddBndCondUpdateOp(const std::shared_ptr<BndCondUpdateOp<TData>> &ptr)
    {
        if (!ptr)
        {
            return;
        }

        // Attaching the storage also lets the condition take a copy of
        // anything the session evaluated into it, which it must do now: every
        // block it owns is overwritten with the interior state before its
        // Apply() runs.
        ptr->SetBndCondPhysOp(m_BndCondOp);

        // Let it declare conditions the session could not express - an
        // adiabatic wall's Neumann energy component - and re-resolve the
        // types this operator gathers by.
        ptr->ApplyTypeOverrides();

        m_bndCondUpdateOps.push_back(ptr);
        m_bndCondUpdateOpByBlock.clear();

        RefreshBndCondTypes();
    }

    /**
     * @brief Compute the boundary values that depend on the interior state.
     *
     * The counterpart of UpdateBndPhys(), which refreshes the values the
     * session evaluates; both write into the same storage.
     *
     * Seeds the storage once with the interior state, for the union of the
     * blocks the attached conditions own, and then lets each transform its
     * own blocks in place. Seeding per condition would undo the work of the
     * ones that ran before it, and seeding a block no condition rewrites
     * would destroy the values evaluated from the session.
     *
     * @param   trace   The trace the interior state is gathered from.
     */
    void UpdateBndCond(LibUtilities::Field<TData, FieldState::Phys> &trace)
    {
        // Once, on the first evaluation: by now every condition the solver
        // means to attach has been, which is not true at construction.
        if (!m_bndCondUpdatesValidated)
        {
            m_bndCondUpdatesValidated = true;
            ValidateBndCondTags();
        }

        if (m_bndCondUpdateOps.empty())
        {
            return;
        }

        GatherBndInteriorState(trace, OwnedBndCondBlocks());

        for (auto &op : m_bndCondUpdateOps)
        {
            op->Apply();
        }
    }

    /**
     * @brief Seed the boundary storage with the interior state at each
     * boundary trace.
     *
     * For conditions that are a function of the interior rather than of the
     * session - a no-slip wall, say. Doing the gather here rather than in the
     * operator that needs it keeps the layout knowledge in one place: the
     * source is the *local* trace, whose points sit in element order and need
     * reorienting onto the global trace, which this operator already does for
     * its own interior side. What lands in the storage is therefore in the
     * same orientation as an evaluated boundary condition, and a caller can
     * transform it pointwise without knowing any of that.
     */
    /// Re-resolve the boundary condition types into this operator's device
    /// tables. Needed when an operator overrides a type after construction.
    void RefreshBndCondTypes()
    {
        v_RefreshBndCondTypes();
    }

    void GatherBndInteriorState(
        LibUtilities::Field<TData, FieldState::Phys> &trace,
        const std::vector<bool> &ownedBlocks)
    {
        v_GatherBndInteriorState(trace, ownedBlocks);
    }

protected:
    LibUtilities::Field<TData, FieldState::Phys> m_traceNormals;
    /// Set once SetTraceNormals() has supplied normals from outside. The
    /// mesh-derived default must not overwrite them after that; see
    /// SetTraceNormals().
    bool m_traceNormalsOverridden = false;
    std::shared_ptr<Operators::BndCondPhysOp<TData>> m_BndCondOp;
    std::vector<std::shared_ptr<BndCondUpdateOp<TData>>> m_bndCondUpdateOps;
    /// Block-to-condition lookup, built on first use by GetBndCondUpdateOp()
    /// and dropped whenever a condition is attached.
    mutable std::vector<const BndCondUpdateOp<TData> *>
        m_bndCondUpdateOpByBlock;
    bool m_append = false;
    /// ValidateBndCondTags() runs once, on the first UpdateBndCond().
    bool m_bndCondUpdatesValidated = false;

    /**
     * @brief Fail on a boundary tag that nothing is going to act on.
     *
     * A USERDEFINEDTYPE is a request for a particular condition. If no
     * attached condition claims the region, nothing computes it and the
     * boundary keeps whatever the session evaluated - which for the legacy
     * compressible sessions is VALUE="0" on density and momentum, because the
     * condition was meant to supply them. A zero density reaching a Riemann
     * solver is a wrong answer, and without this it is a silent one.
     */
    void ValidateBndCondTags() const
    {
        if (!m_BndCondOp)
        {
            return;
        }

        const auto owned = OwnedBndCondBlocks();

        for (unsigned blk = 0; blk < m_BndCondOp->GetNumBlocks(); ++blk)
        {
            // A region carries one condition for all of its components, so the
            // first settles it - the same assumption the conditions make when
            // deciding which blocks they own.
            const std::string &tag = m_BndCondOp->GetBndUserDefined(blk, 0);

            if (tag.empty() || HandledWithoutABndCondUpdateOp(tag))
            {
                continue;
            }

            ASSERTL0(blk < owned.size() && owned[blk],
                     "Boundary region " + std::to_string(blk) +
                         " is tagged USERDEFINEDTYPE=\"" + tag +
                         "\", but no boundary condition operator attached to "
                         "this solver claims it. The region would silently "
                         "keep the values the session evaluated instead of the "
                         "condition the tag asks for. Either attach an "
                         "operator that claims this tag, or remove the tag "
                         "from the session.");
        }
    }

    /// Tags that are honoured somewhere other than a BndCondUpdateOp, and so
    /// are not evidence of a missing one.
    static bool HandledWithoutABndCondUpdateOp(const std::string &tag)
    {
        // Re-evaluated from the session each step by
        // BndCondPhysOp::UpdateBndPhys(); it needs no operator, only that the
        // owner calls UpdateBndPhys(), which the DG operators do.
        return tag == "TimeDependent";
    }

    /**
     * @brief Blocks owned by any attached condition, one flag per block.
     *
     * The union, because seeding is done once for all of them and each then
     * transforms only its own blocks. A block owned by none keeps the values
     * the session evaluated.
     */
    std::vector<bool> OwnedBndCondBlocks() const
    {
        std::vector<bool> owned;

        for (const auto &op : m_bndCondUpdateOps)
        {
            const auto blocks = op->GetOwnedBlocks();

            if (owned.size() < blocks.size())
            {
                owned.resize(blocks.size(), false);
            }

            for (size_t k = 0; k < blocks.size(); ++k)
            {
                // Two conditions claiming one region would both transform it,
                // the second working on the first's output, and the answer
                // would depend on the order they were attached in. A region
                // carries one USERDEFINEDTYPE, so this means two conditions
                // claim the same tag.
                ASSERTL0(!(owned[k] && blocks[k]),
                         "More than one boundary condition operator claims "
                         "boundary region " +
                             std::to_string(k) + ".");

                owned[k] = owned[k] || blocks[k];
            }
        }

        return owned;
    }

    /**
     * @brief The condition claiming boundary storage block @p blk, or null.
     *
     * Blocks are claimed disjointly - OwnedBndCondBlocks() asserts it - so at
     * most one condition answers for any block, and a block claimed by none
     * keeps whatever the session evaluated.
     *
     * Resolved once and cached, because GetOwnedBlocks() builds its answer
     * afresh each call while callers want this per boundary *trace*, of which
     * there are many more than there are blocks.
     */
    const BndCondUpdateOp<TData> *GetBndCondUpdateOp(const unsigned blk) const
    {
        if (m_bndCondUpdateOpByBlock.empty() && m_BndCondOp)
        {
            m_bndCondUpdateOpByBlock.assign(m_BndCondOp->GetNumBlocks(),
                                            nullptr);

            for (const auto &op : m_bndCondUpdateOps)
            {
                const auto owned = op->GetOwnedBlocks();

                for (size_t k = 0;
                     k < owned.size() && k < m_bndCondUpdateOpByBlock.size();
                     ++k)
                {
                    if (owned[k])
                    {
                        m_bndCondUpdateOpByBlock[k] = op.get();
                    }
                }
            }
        }

        return blk < m_bndCondUpdateOpByBlock.size()
                   ? m_bndCondUpdateOpByBlock[blk]
                   : nullptr;
    }

    TraceFluxOp(const MultiRegions::ExpListSharedPtr &expansionList,
                const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
        const auto &vars =
            components.size() ==
                    expansionList->GetSession()->GetVariables().size()
                ? components
                : expansionList->GetSession()->GetVariables();
        m_BndCondOp =
            Operators::BndCondPhysOp<TData>::Create(expansionList, vars);
    }

    ~TraceFluxOp() override = default;

    virtual void v_Apply(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This method is not defined for this derived class");
    }

    virtual void v_ApplyDirBoundary(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This method is not defined for this derived class");
    }

    virtual void v_ApplyInterior(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This method is not defined for this derived class");
    }

    virtual void v_Apply(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This method is not defined for this derived class");
    }

    virtual void v_ApplyDirBoundary(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This method is not defined for this derived class");
    }

    virtual void v_ApplyInterior(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
        NEKERROR(ErrorUtil::efatal,
                 "This method is not defined for this derived class");
    }

    /// No-ops by default, like the exchange calls below: an operator with no
    /// parallel trace machinery, and every operator at all in serial, has no
    /// cut traces to evaluate.
    virtual void v_ApplyParallel(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
    }

    virtual void v_ApplyParallel(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &in,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &inDeriv,
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &flux)
    {
    }

    /// No-ops by default: an operator with no parallel trace machinery, and
    /// every operator at all in serial, has nothing to exchange.
    virtual void v_BeginParallelExchange(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &trace,
        [[maybe_unused]] const unsigned channel)
    {
    }

    virtual void v_EndParallelExchange()
    {
    }

    virtual void v_SetTraceAdvVel(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &AdvVel)
    {
        NEKERROR(
            ErrorUtil::efatal,
            "SetTrace Advection Velocity is not available in this operator");
    }

    virtual void v_SetTraceNormals(
        LibUtilities::Field<TData, FieldState::Phys> &traceNormals)
    {
        this->m_traceNormals           = std::move(traceNormals);
        this->m_traceNormalsOverridden = true;
        v_OnTraceNormalsChanged();
    }

    /**
     * @brief Called whenever m_traceNormals has been replaced.
     *
     * The single point at which anything derived from the normals goes
     * stale, whether they came from the warehouse or were supplied through
     * SetTraceNormals(). An operator caching rotation matrices should
     * invalidate them here rather than overriding v_SetTraceNormals(), so
     * that both paths are covered.
     */
    virtual void v_OnTraceNormalsChanged()
    {
    }

    virtual void v_SetDiffCoeff([[maybe_unused]] std::vector<TData> &diffCoeff)
    {
        NEKERROR(ErrorUtil::efatal,
                 "SetDiffCoeff is not available in this operator");
    }

    virtual void v_RefreshBndCondTypes()
    {
    }

    virtual void v_GatherBndInteriorState(
        [[maybe_unused]] LibUtilities::Field<TData, FieldState::Phys> &trace,
        [[maybe_unused]] const std::vector<bool> &ownedBlocks)
    {
        NEKERROR(ErrorUtil::efatal,
                 "GatherBndInteriorState is not available for this operator "
                 "derived implementation");
    }
};

} // namespace Nektar::SolverCore
