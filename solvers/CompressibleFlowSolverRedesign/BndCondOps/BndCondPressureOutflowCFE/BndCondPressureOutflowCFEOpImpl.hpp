///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondPressureOutflowCFEOpImpl.hpp
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

#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

#include "BndCondPressureOutflowCFEKernels.hpp"
#include "BndCondPressureOutflowCFEOp.hpp"

using namespace Nektar;

namespace Nektar::detail
{

/// Tag this operator claims.
inline bool IsPressureOutflowTag(const std::string &tag)
{
    return tag == "PressureOutflow";
}

/**
 * @brief Subsonic pressure outflow.
 *
 * Prescribes the static pressure at an outflow and extrapolates everything
 * else, which is the well posed choice while the flow leaves the domain
 * subsonically: one characteristic enters, so exactly one quantity may be
 * imposed. Once the outflow goes supersonic no characteristic enters and
 * nothing may be imposed, so the whole state is extrapolated.
 *
 * The caller seeds the storage with the interior trace, which is already the
 * extrapolated state, so the supersonic branch has nothing to do and the
 * subsonic branch has only the energy left to write:
 *
 *     E = rho+ e(rho+, p_out) + |m+|^2 / (2 rho+)
 *
 * the internal energy that goes with the prescribed pressure at the
 * extrapolated density, plus the extrapolated kinetic energy.
 *
 * ### Where the pressure comes from
 *
 * From the session's boundary value for the *energy* variable, so
 *
 *     <D VAR="E" USERDEFINEDTYPE="PressureOutflow" VALUE="pInf" />
 *
 * holds a pressure and not an energy. Existing sessions are written that
 * way, and there is no other slot for the number to arrive in:
 * BndCondPhysOp evaluates every component from the session alike. It remains a
 * trap - the variable is named for a quantity it does not hold on this one
 * boundary type.
 *
 * The value is taken once, in v_SetBndCondPhysOp(), and not re-read. A
 * time-dependent outflow pressure
 * is therefore *not* supported: a session may declare one and it is silently
 * ignored. The copy is needed either way, because every owned block is
 * overwritten with the interior state before v_Apply() runs.
 *
 * ### Why the diffusion side needs nothing
 *
 * This is an imposed region rather than a reflective one, so the trace flux
 * operator reflects it to 2g - Q+ for the diffusion, so the trace average
 * lands on the imposed value.
 *
 * That the reflection lands on the target exactly is a *precondition of this
 * condition*, not a general property. The compressible kernel averages the
 * internal energy and adds back the kinetic energy of the averaged momentum
 * rather than averaging E linearly. Here rho and momentum are extrapolated, so
 * the ghost has rho- = rho+ and m- = m+, the kinetic term added back is exactly
 * the one subtracted, and {E} collapses to the linear (E+ + E-)/2 = E_b. A
 * condition that perturbs rho or m as well - PressureOutflowNonReflective, say
 * - loses that cancellation and has to solve for the exterior energy on the
 * internal energy instead, as the isothermal wall does. The convention is
 * documented on SolverCore::BndCondUpdateOp::GhostIsReflected().
 */
template <typename ExecSpace, typename EoSParamType, typename TData>
class BndCondPressureOutflowCFEOpImpl
    : public BndCondPressureOutflowCFEOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BndCondPressureOutflowCFEOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : BndCondPressureOutflowCFEOp<TData>(expansionList, components)
    {
        m_numComp  = components.size();
        m_coordDim = MultiRegions::GetCollection(expansionList, 0)
                         .GetExpVector()[0]
                         ->GetCoordim();

        ASSERTL0(m_numComp == m_coordDim + 2,
                 "A compressible pressure outflow needs density, one momentum "
                 "component per direction and energy.");

        SetUpEquationOfState(expansionList->GetSession(), m_EoS);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<
            BndCondPressureOutflowCFEOpImpl<ExecSpace, EoSParamType, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_numComp  = 0;
    unsigned int m_coordDim = 0;
    EoSParamType m_EoS;

    /// Prescribed static pressure per owned block, taken from the session
    /// before the storage is overwritten. Empty for unowned blocks. Held in
    /// the execution space's memory: the boundary storage lives there, so
    /// both the copy at setup and the reads in the apply kernel stay
    /// on-space.
    std::vector<LibUtilities::MemoryRegion<TData>> m_pressure;

    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            owned[blk] = IsPressureOutflowTag(
                this->m_bndCondPhysOp->GetBndUserDefined(blk, 0));
        }

        return owned;
    }

    /// Copy the prescribed pressure out of the session-evaluated storage,
    /// while it is still there.
    void v_SetBndCondPhysOp() override
    {
        const auto owned = v_GetOwnedBlocks();
        m_pressure.clear();
        m_pressure.resize(owned.size());

        for (unsigned blk = 0; blk < owned.size(); ++blk)
        {
            if (!owned[blk])
            {
                continue;
            }

            const auto stride  = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            const auto *bndPtr = this->m_bndCondPhysOp->GetBlockPtr(blk);

            // The boundary storage lives in the execution space's memory, so
            // the copy runs there as a kernel - a host-side assign from this
            // pointer is a host dereference of device memory on a real
            // device.
            m_pressure[blk] = LibUtilities::MemoryRegion<TData>(stride);
            m_pressure[blk].template Initialize<MemSpace>(0);
            TData *pOut =
                m_pressure[blk].template GetPtr<MemSpace, ReadWrite>();

            CopyPrescribedPressureBlock<ExecSpace>(stride, m_numComp - 1,
                                                   bndPtr, pOut);
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

            const auto stride = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            auto bndPtr       = this->m_bndCondPhysOp->UpdateBlockPtr(blk);
            const TData *pOut =
                m_pressure[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *norms = this->m_bndCondPhysOp->GetBlockNormals(blk);

            ASSERTL1(m_pressure[blk].size() == stride,
                     "Prescribed outflow pressure was taken for a different "
                     "block size than the storage now reports.");

            ApplyPressureOutflowBlock<ExecSpace>(
                stride, m_coordDim, m_numComp - 1, m_EoS, pOut, norms, bndPtr);
        }
    }
};

} // namespace Nektar::detail
