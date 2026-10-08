///////////////////////////////////////////////////////////////////////////////
//
// File: EnforceEntropyOpImplBase.hpp
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
// Description: Shared machinery of the entropy Riemann boundary conditions.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <boost/algorithm/string/predicate.hpp>

#include <algorithm>

#include <MultiRegions/DisContField.h>

#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

#include "EnforceEntropyKernels.hpp"

namespace Nektar::detail
{

/**
 * @brief Everything the three entropy Riemann conditions have in common.
 *
 * They differ only in the star state at a subsonic boundary, which is a
 * handful of lines; the rest - reading the prescribed state before it is
 * overwritten, deciding which characteristics cross, turning the star state
 * into a conserved one - is identical. That shared part lives here and the
 * difference arrives as @p Policy, a struct of static functions rather than a
 * virtual, so it inlines into the kernel and stays usable in a device space.
 *
 * A @p Policy provides
 *
 * - `tag`, the USERDEFINEDTYPE it claims;
 * - `InflowStar(...)`, the subsonic inflow star state; and
 * - `hasSubsonicOutflow`, with `OutflowStar(...)` when true.
 *
 * ### What is imposed, and where
 *
 * All three write a complete exterior state into the boundary storage, so the
 * Riemann solver and the diffusion both see it: they are *imposed* regions in
 * the sense of SolverCore::BndCondUpdateOp::GhostIsReflected() and leave
 * v_GhostIsReflected() at its default, so the diffusion reflects the state
 * about the imposed value as it does for any Dirichlet region.
 *
 * ### Ideal gas only
 *
 * The entropy relation these are built on, \f$s = p/\rho^\gamma\f$, is the
 * ideal gas one, so none of them is templated on the equation of state as the
 * wall and the pressure outflow are. They read gamma from the session and
 * refuse anything else, which announces the limitation rather than quietly
 * applying the wrong relation. The check sits in v_SetBndCondPhysOp() rather
 * than the constructor, and only bites once a region is actually claimed: a
 * solver attaches every condition it knows about, so checking at construction
 * would stop a van der Waals run that never asks for one of these.
 */
template <typename ExecSpace, typename TData, typename Policy, typename OpBase>
class EnforceEntropyCFEOpImplBase : public OpBase
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    EnforceEntropyCFEOpImplBase(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : OpBase(expansionList, components)
    {
        m_numComp  = components.size();
        m_coordDim = MultiRegions::GetCollection(expansionList, 0)
                         .GetExpVector()[0]
                         ->GetCoordim();

        ASSERTL0(m_numComp == m_coordDim + 2,
                 "An entropy Riemann condition needs density, one momentum "
                 "component per direction and energy.");

        m_session = expansionList->GetSession();
    }

protected:
    LibUtilities::SessionReaderSharedPtr m_session;
    unsigned int m_numComp  = 0;
    unsigned int m_coordDim = 0;
    TData m_gamma           = 0.0;

    /// Prescribed state per owned block, taken from the session before the
    /// storage is overwritten. Empty for unowned blocks. Held in the
    /// execution space's memory: the boundary storage lives there, so both
    /// the copy at setup and the reads in the apply kernel stay on-space.
    std::vector<LibUtilities::MemoryRegion<TData>> m_rhoBC;
    std::vector<LibUtilities::MemoryRegion<TData>> m_pBC;
    /// Prescribed velocity, component major with the block's stride, in the
    /// same layout as the boundary storage and the normals.
    std::vector<LibUtilities::MemoryRegion<TData>> m_velBC;
    /// Outward normal component of the prescribed velocity, n.u.
    std::vector<LibUtilities::MemoryRegion<TData>> m_VnInf;

    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            owned[blk] = (this->m_bndCondPhysOp->GetBndUserDefined(blk, 0) ==
                          std::string(Policy::tag));
        }

        return owned;
    }

    /**
     * @brief Copy the prescribed state out of the session-evaluated storage
     * while it is still there, and reduce it to what the condition needs.
     *
     * The session gives an ordinary conserved state, so the energy slot really
     * holds an energy. Worth stating because a PressureOutflow region puts a
     * static pressure in the same slot, and a session may carry both
     * conventions at once with only the tag to tell them apart.
     *
     * Taken once, because by the time v_Apply() runs every owned block has
     * been overwritten with the interior state. A time-dependent prescribed
     * state is therefore not supported.
     */
    void v_SetBndCondPhysOp() override
    {
        const auto owned = v_GetOwnedBlocks();

        // The ideal gas check belongs here, not in the constructor. A solver
        // attaches every condition it knows about whether or not the session
        // uses them, so refusing at construction would stop a van der Waals
        // run that has no entropy region anywhere in it. Only a session that
        // actually asks for this condition should have to satisfy it.
        if (std::none_of(owned.begin(), owned.end(), [](bool b) { return b; }))
        {
            return;
        }

        ASSERTL0(m_session->DefinesEquationOfState(),
                 "No EquationOfState section defined in session file");

        auto EoS = m_session->GetEquationOfState();

        ASSERTL0(boost::iequals(EoS.type, "IdealGas"),
                 std::string(Policy::tag) +
                     " assumes the ideal gas entropy s = p/rho^gamma and is "
                     "not defined for equation of state '" +
                     EoS.type + "'.");

        ASSERTL0(EoS.params.count("GAMMA"),
                 "Need to specify Gamma in params of EquationOfState "
                 "definition");

        m_gamma = EoS.params["GAMMA"];

        m_rhoBC.clear();
        m_pBC.clear();
        m_velBC.clear();
        m_VnInf.clear();
        m_rhoBC.resize(owned.size());
        m_pBC.resize(owned.size());
        m_velBC.resize(owned.size());
        m_VnInf.resize(owned.size());

        for (unsigned blk = 0; blk < owned.size(); ++blk)
        {
            if (!owned[blk])
            {
                continue;
            }

            const auto stride  = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            const auto *bndPtr = this->m_bndCondPhysOp->GetBlockPtr(blk);
            const auto *norms  = this->m_bndCondPhysOp->GetBlockNormals(blk);

            // The boundary storage and normals live in the execution space's
            // memory, so the reduction runs there as a kernel - a host loop
            // over these pointers is a host dereference of device memory on a
            // real device.
            m_rhoBC[blk] = LibUtilities::MemoryRegion<TData>(stride);
            m_pBC[blk]   = LibUtilities::MemoryRegion<TData>(stride);
            m_velBC[blk] =
                LibUtilities::MemoryRegion<TData>(m_coordDim * stride);
            m_VnInf[blk] = LibUtilities::MemoryRegion<TData>(stride);
            m_rhoBC[blk].template Initialize<MemSpace>(0);
            m_pBC[blk].template Initialize<MemSpace>(0);
            m_velBC[blk].template Initialize<MemSpace>(0);
            m_VnInf[blk].template Initialize<MemSpace>(0);

            TData *rhoBC = m_rhoBC[blk].template GetPtr<MemSpace, ReadWrite>();
            TData *pBC   = m_pBC[blk].template GetPtr<MemSpace, ReadWrite>();
            TData *velBC = m_velBC[blk].template GetPtr<MemSpace, ReadWrite>();
            TData *VnInf = m_VnInf[blk].template GetPtr<MemSpace, ReadWrite>();

            ReduceEntropyStateBlock<ExecSpace>(stride, m_coordDim,
                                               m_numComp - 1, m_gamma, bndPtr,
                                               norms, rhoBC, pBC, velBC, VnInf);
        }
    }

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

            const auto stride  = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            auto bndPtr        = this->m_bndCondPhysOp->UpdateBlockPtr(blk);
            const TData *norms = this->m_bndCondPhysOp->GetBlockNormals(blk);

            const TData *rhoBC =
                m_rhoBC[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *pBC = m_pBC[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *velBC =
                m_velBC[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *VnInf =
                m_VnInf[blk].template GetPtr<MemSpace, ReadOnly>();

            ApplyEnforceEntropyBlock<ExecSpace, Policy>(
                stride, m_coordDim, m_numComp - 1, m_gamma, rhoBC, pBC, velBC,
                VnInf, norms, bndPtr);
        }
    }
};

} // namespace Nektar::detail
