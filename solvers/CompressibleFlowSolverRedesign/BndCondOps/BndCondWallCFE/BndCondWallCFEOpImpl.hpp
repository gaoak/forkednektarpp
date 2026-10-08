///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondWallCFEOpImpl.hpp
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

#include <MultiRegions/DisContField.h>

#include "BndCondWallCFEKernels.hpp"
#include "BndCondWallCFEOp.hpp"

using namespace Nektar;

namespace Nektar::detail
{

/// Tags this operator claims. Anything else is another operator's business.
inline bool IsViscousWallTag(const std::string &tag)
{
    return tag == "WallViscous" || tag == "WallAdiabatic";
}

/// An adiabatic wall additionally insulates: no heat flux through it.
inline bool IsAdiabaticWallTag(const std::string &tag)
{
    return tag == "WallAdiabatic";
}

/// A plain viscous wall is held at the session temperature Twall. This is the
/// complement of IsAdiabaticWallTag() within IsViscousWallTag(): every viscous
/// wall carries exactly one of the two thermal conditions.
inline bool IsIsothermalWallTag(const std::string &tag)
{
    return tag == "WallViscous";
}

template <typename ExecSpace, typename EoSParamType, typename TData>
class BndCondWallCFEOpImpl : public BndCondWallCFEOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BndCondWallCFEOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                         const std::vector<std::string> &components)
        : BndCondWallCFEOp<TData>(expansionList, components)
    {
        m_numComp  = components.size();
        m_coordDim = MultiRegions::GetCollection(expansionList, 0)
                         .GetExpVector()[0]
                         ->GetCoordim();

        ASSERTL0(m_numComp >= m_coordDim + 2,
                 "A compressible wall needs density, one momentum component "
                 "per direction and energy.");

        SetUpEquationOfState(expansionList->GetSession(), m_EoS);

        m_session = expansionList->GetSession();
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            BndCondWallCFEOpImpl<ExecSpace, EoSParamType, TData>>(expansionList,
                                                                  components);
    }

protected:
    unsigned int m_numComp  = 0;
    unsigned int m_coordDim = 0;
    EoSParamType m_EoS;

    LibUtilities::SessionReaderSharedPtr m_session;

    /// Wall temperature each isothermal region is held at, one per storage
    /// block. See v_SetBndCondPhysOp() for where each value comes from.
    std::vector<TData> m_Twall;

    /// A wall mirrors the interior rather than imposing a target, so what
    /// v_Apply() leaves in the storage is already the exterior state and the
    /// diffusion must not reflect it again.
    bool v_GhostIsReflected() const override
    {
        return true;
    }

    /**
     * @brief An adiabatic wall carries no heat flux.
     *
     * With u = 0 at the wall the viscous work u.tau vanishes, so the energy
     * component of the viscous boundary flux is exactly q.n = k dT/dn, and
     * suppressing it is the whole of the condition. Every other component, and
     * an isothermal wall's energy - which is a condition on the state, imposed
     * on the diffusion's exterior energy instead - keeps its flux.
     */
    TData v_BndFluxWeight(const unsigned blk,
                          const unsigned comp) const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const bool isEnergy = (comp + 1 == m_numComp);

        return (isEnergy &&
                IsAdiabaticWallTag(
                    this->m_bndCondPhysOp->GetBndUserDefined(blk, 0)))
                   ? TData(0.0)
                   : TData(1.0);
    }

    /// An isothermal wall acts on the diffusion's exterior state.
    bool v_HasDiffusionExteriorState() const override
    {
        return true;
    }

    /**
     * @brief Resolve the wall temperature of each region.
     *
     * Three places, least specific first, so that adding the mechanism breaks
     * nothing that already works:
     *
     *   1. the built-in default of 300.15;
     *   2. a session `Twall`, which is how every session written before this
     *      existed says it, and which applies to every wall at once;
     *   3. the region's own tag, `USERDEFINEDTYPE="WallViscous:twall=300.15"`.
     *
     * The third is the reason for the mechanism: a session parameter cannot
     * give two walls different temperatures, and a channel heated on one side
     * needs exactly that.
     *
     * Read here rather than in the constructor because the tags arrive with
     * the storage.
     */
    void v_SetBndCondPhysOp() override
    {
        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();

        double sessionTwall = 300.15;
        m_session->LoadParameter("Twall", sessionTwall, sessionTwall);

        m_Twall.assign(numBlocks, TData(sessionTwall));

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            this->m_bndCondPhysOp->GetBndUserDefinedParam(blk, 0, "twall",
                                                          m_Twall[blk]);
        }
    }

    /**
     * @brief Hold an isothermal wall at Twall, in the diffusion's exterior
     * state only.
     *
     * The wall temperature is a condition on the state the diffusion sees,
     * not on the inviscid flux: writing it into the boundary storage would
     * reach the Riemann solver too. This hook supplies the diffusion's
     * exterior state alone.
     *
     * The trace average is not linear in the total energy: it averages the
     * *internal* energy and adds back the kinetic energy of the averaged
     * momentum, so the target has to be expressed in those terms. Solving
     *
     *     {e_int} = 1/2 (eL + eR) = rho_ave * e(rho_ave, Twall)
     *     eL      = E+ - |m+|^2 / (2 rho+)
     *     eR      = E- - |m-|^2 / (2 rho-)
     *
     * for the exterior energy gives
     *
     *     E- = 2 rho_ave e(rho_ave, Twall) - eL + |m-|^2 / (2 rho-)
     *
     * and the kernel supplies the kinetic term of the average on top.
     * Treating the average as linear instead leaves the wall short by a
     * kinetic energy term - an error of order Mach number squared, small at
     * low speed and not zero.
     *
     * An adiabatic wall does nothing here. Its condition is on the heat flux,
     * which is v_BndFluxWeight().
     */
    void v_ApplyDiffusionExteriorState(const unsigned blk, TData *ext,
                                       const TData *interior, const size_t nPts,
                                       const size_t compStride) const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        if (!IsIsothermalWallTag(
                this->m_bndCondPhysOp->GetBndUserDefined(blk, 0)))
        {
            return;
        }

        // Density, one momentum component per direction and energy. A session
        // carrying extra transported components leaves them alone.
        if (m_numComp != m_coordDim + 2)
        {
            return;
        }

        ImposeIsothermalWallEnergyBlock<ExecSpace>(nPts, compStride, m_coordDim,
                                                   m_numComp - 1, m_EoS,
                                                   m_Twall[blk], interior, ext);
    }

    /// The blocks this operator owns, i.e. the regions tagged as viscous or
    /// adiabatic walls. Only these are seeded with the interior state.
    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            // A region carries one condition for all of its components, so the
            // first settles whether this is a wall.
            owned[blk] = IsViscousWallTag(
                this->m_bndCondPhysOp->GetBndUserDefined(blk, 0));
        }

        return owned;
    }

    /**
     * @brief Turn the seeded interior state into a no-slip wall state.
     *
     * The caller has already put the interior trace into the boundary storage,
     * so this is the whole of the state side of the physics: density stays as
     * it is, momentum reverses, and the average across the trace therefore has
     * zero velocity.
     *
     * Neither thermal condition is imposed here, for two different reasons.
     * An adiabatic wall constrains the heat flux rather than the state, so
     * AdvDiffTraceFluxCFEOpImpl suppresses the energy component of the viscous
     * boundary flux instead. An isothermal wall does constrain the state - but
     * only the state the *diffusion* sees, inside the viscous flux. Writing
     * it into the boundary storage here would reach the inviscid flux as
     * well; it goes into m_gloT1Diff instead.
     *
     * What does belong here is any condition the advection should see too - a
     * moving wall's velocity, for instance.
     *
     * Regions that are not walls are left untouched and are never seeded, so
     * their session-evaluated values stand.
     */
    void v_Apply() override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto owned = v_GetOwnedBlocks();

        for (unsigned blk = 0; blk < owned.size(); ++blk)
        {
            if (!owned[blk])
            {
                continue;
            }

            const auto stride = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            auto bndPtr       = this->m_bndCondPhysOp->UpdateBlockPtr(blk);

            ReverseMomentumBlock<ExecSpace>(stride, m_coordDim, bndPtr);
        }
    }
};

} // namespace Nektar::detail
