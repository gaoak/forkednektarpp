///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondUpdateOp.hpp
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

#pragma once

#include "Operators/BndCondOps/BndCondPhys/BndCondPhysOp.hpp"
#include "Operators/Common/Operator.hpp"

namespace Nektar::SolverCore
{

/**
 * @brief Base for operators that compute boundary values from the interior.
 *
 * Some boundary conditions cannot be evaluated from the session alone: a
 * no-slip wall state, for instance, is a function of the interior trace and so
 * changes every step. Such an operator writes its result into the storage held
 * by BndCondPhysOp, which the trace gathers already read through
 * BndCondPhysOp::GetBlockPtr(). Nothing downstream changes as a result - the
 * flux path neither knows nor cares whether a value came from a session
 * expression or from here.
 *
 * The base carries no name and is not registered: the meaning of a boundary
 * tag is solver specific, so the concrete operator lives with the solver that
 * understands it, in the same way SolverCore::TraceFluxOp is specialised by the
 * compressible flow solver.
 */
template <typename TData>
class BndCondUpdateOp : public Operators::Operator<TData>
{

public:
    static inline const std::string name = "";

    /// Storage to write into. Must be the same instance the trace flux
    /// operator gathers from, or the computed values go nowhere.
    void SetBndCondPhysOp(
        const std::shared_ptr<Operators::BndCondPhysOp<TData>> &ptr)
    {
        m_bndCondPhysOp = ptr;
        v_SetBndCondPhysOp();
    }

    /**
     * @brief Transform the boundary values this operator owns, in place.
     *
     * The caller seeds the storage with the interior state first - through
     * SolverCore::TraceFluxOp::GatherBndInteriorState() - so an implementation
     * here works pointwise on values already in the right place and
     * orientation, and needs to know nothing about the trace layout.
     */
    void Apply()
    {
        v_Apply();
    }

    void operator()()
    {
        v_Apply();
    }

    /**
     * @brief Which storage blocks this operator computes, one flag per block.
     *
     * Only these are seeded with the interior state. Seeding a block this
     * operator does not then transform would destroy the values evaluated from
     * the session, and nothing would put them back: UpdateBndPhys() returns
     * immediately when no condition depends on time.
     */
    std::vector<bool> GetOwnedBlocks() const
    {
        return v_GetOwnedBlocks();
    }

    /**
     * @brief Declare any condition types this operator imposes that the
     * session could not express.
     *
     * Called once, after the storage is attached and before the trace flux
     * operator resolves the types into its own tables.
     */
    void ApplyTypeOverrides()
    {
        v_ApplyTypeOverrides();
    }

    /**
     * @brief Whether the state this operator leaves in the storage is already
     * a ghost, or is the value to be attained.
     *
     * Every average across a trace, on boundary faces as well as interior
     * ones, is the half-and-half (w+ + w-)/2, so a condition that
     * *mirrors* the interior - a no-slip wall reversing momentum - has already
     * produced the exterior state and it stands as it is, while a condition
     * that *imposes* a target g needs the diffusion's exterior state reflected
     * to 2g - Q+ for the average to land on g.
     *
     * Declared here rather than inferred from the boundary tag, because it is
     * a property of what this operator wrote, not of what the session said.
     * The trace flux operator cannot tell the two apart by looking at the
     * storage.
     *
     * Imposed is the default: it is what an ordinary session Dirichlet region
     * with no operator at all needs, so a condition only overrides this if it
     * builds a mirrored state.
     */
    bool GhostIsReflected() const
    {
        return v_GhostIsReflected();
    }

    /**
     * @brief Weight on the diffusive boundary flux of component @p comp in
     * block @p blk.
     *
     * One leaves the flux as the discretisation formed it, and is the default.
     * Zero suppresses it, which is how a condition on the *flux* is imposed
     * rather than one on the state: at a no-slip wall the viscous work
     * u.tau vanishes, so the energy component of the viscous flux is exactly
     * the heat flux q.n, and an adiabatic wall is precisely a zero weight on
     * it.
     *
     * Per block because one operator may claim several regions carrying
     * different conditions - a wall operator holding both the adiabatic and
     * the isothermal tag - and per component because which flux is
     * constrained is part of the condition rather than a property of the
     * solver.
     *
     * A flux operator is free to support only some components and must say so;
     * it should assert on a weight it cannot apply rather than drop it.
     */
    TData BndFluxWeight(const unsigned blk, const unsigned comp) const
    {
        return v_BndFluxWeight(blk, comp);
    }

    /**
     * @brief Transform the exterior state the *diffusion* sees, in place.
     *
     * The last place a condition can act, and the only one that reaches the
     * diffusion without also reaching the Riemann solver. An isothermal wall
     * needs exactly that: legacy imposes its temperature on the averaged state
     * inside the viscous flux, so the inviscid flux never sees the wall
     * temperature, and writing it into the boundary storage instead would
     * change both.
     *
     * Called once per trace this operator owns, after the caller has formed
     * the exterior state - reflected or imposed, per GhostIsReflected() - so
     * an implementation adjusts a state that is already correct in every other
     * respect.
     *
     * @param blk        - Boundary storage block this trace belongs to, so an
     *                     operator claiming regions with different conditions
     *                     can tell which it is acting on.
     * @param ext        - Exterior state to transform, @p nPts values per
     *                     component, components @p compStride apart.
     * @param interior   - Interior state at the same points, same layout.
     * @param nPts       - Points in this trace.
     * @param compStride - Distance between one component and the next.
     *
     * The layout is given as a stride rather than as any container so that an
     * implementation needs to know nothing about how the caller stores its
     * traces - the same contract v_Apply() has against the boundary storage.
     * Note that @p compStride is a whole trace *block*, not @p nPts: a block
     * holds several traces, and this one is a window into it.
     */
    void ApplyDiffusionExteriorState(const unsigned blk, TData *ext,
                                     const TData *interior, const size_t nPts,
                                     const size_t compStride) const
    {
        v_ApplyDiffusionExteriorState(blk, ext, interior, nPts, compStride);
    }

    /// Whether ApplyDiffusionExteriorState() does anything, so a caller can
    /// skip the per-trace bookkeeping for the conditions that do not.
    bool HasDiffusionExteriorState() const
    {
        return v_HasDiffusionExteriorState();
    }

protected:
    std::shared_ptr<Operators::BndCondPhysOp<TData>> m_bndCondPhysOp;

    BndCondUpdateOp(const MultiRegions::ExpListSharedPtr &expansionList,
                    const std::vector<std::string> &components)
        : Operators::Operator<TData>(expansionList, components)
    {
    }

    ~BndCondUpdateOp() override = default;

    virtual void v_Apply() = 0;

    virtual std::vector<bool> v_GetOwnedBlocks() const = 0;

    virtual void v_ApplyTypeOverrides()
    {
    }

    /// Imposed unless a condition says otherwise. See GhostIsReflected().
    virtual bool v_GhostIsReflected() const
    {
        return false;
    }

    /// Unweighted unless a condition says otherwise. See BndFluxWeight().
    virtual TData v_BndFluxWeight([[maybe_unused]] const unsigned blk,
                                  [[maybe_unused]] const unsigned comp) const
    {
        return TData(1.0);
    }

    /// Nothing, unless a condition acts on the diffusion alone. See
    /// ApplyDiffusionExteriorState().
    virtual void v_ApplyDiffusionExteriorState(
        [[maybe_unused]] const unsigned blk, [[maybe_unused]] TData *ext,
        [[maybe_unused]] const TData *interior,
        [[maybe_unused]] const size_t nPts,
        [[maybe_unused]] const size_t compStride) const
    {
    }

    virtual bool v_HasDiffusionExteriorState() const
    {
        return false;
    }

    /**
     * @brief Hook run once the storage is attached, before any Apply().
     *
     * The place to read values the session evaluated into the storage, because
     * by the time v_Apply() runs they are gone: the caller seeds every owned
     * block with the interior state first, which overwrites them. A condition
     * that prescribes a quantity and extrapolates the rest - an outflow
     * pressure, say - has to take its copy here. A condition with nothing to
     * keep says so with an empty body.
     */
    virtual void v_SetBndCondPhysOp() = 0;
};

} // namespace Nektar::SolverCore
