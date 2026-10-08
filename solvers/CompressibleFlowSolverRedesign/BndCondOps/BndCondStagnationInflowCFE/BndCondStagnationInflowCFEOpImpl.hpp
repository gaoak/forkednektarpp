///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondStagnationInflowCFEOpImpl.hpp
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

#include <algorithm>
#include <cmath>

#include <boost/algorithm/string/predicate.hpp>

#include <MultiRegions/DisContField.h>

#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

#include "BndCondStagnationInflowCFEKernels.hpp"
#include "BndCondStagnationInflowCFEOp.hpp"

namespace Nektar::detail
{

/// Tag this operator claims.
inline bool IsStagnationInflowTag(const std::string &tag)
{
    return tag == "StagnationInflow";
}

template <typename ExecSpace, typename TData>
class BndCondStagnationInflowCFEOpImpl
    : public BndCondStagnationInflowCFEOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BndCondStagnationInflowCFEOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : BndCondStagnationInflowCFEOp<TData>(expansionList, components)
    {
        m_numComp  = components.size();
        m_coordDim = MultiRegions::GetCollection(expansionList, 0)
                         .GetExpVector()[0]
                         ->GetCoordim();

        ASSERTL0(m_numComp == m_coordDim + 2,
                 "A stagnation inflow needs density, one momentum component "
                 "per direction and energy.");

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
            BndCondStagnationInflowCFEOpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    LibUtilities::SessionReaderSharedPtr m_session;
    unsigned int m_numComp  = 0;
    unsigned int m_coordDim = 0;
    TData m_gamma           = 0.0;

    /// Stagnation state and the normalised flow direction, per owned block,
    /// taken from the session before the storage is overwritten. Held in the
    /// execution space's memory: the boundary storage lives there, so both
    /// the copy at setup and the reads in the apply kernel stay on-space.
    std::vector<LibUtilities::MemoryRegion<TData>> m_rhoStag;
    std::vector<LibUtilities::MemoryRegion<TData>> m_EStag;
    std::vector<LibUtilities::MemoryRegion<TData>> m_dir;

    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            owned[blk] = IsStagnationInflowTag(
                this->m_bndCondPhysOp->GetBndUserDefined(blk, 0));
        }

        return owned;
    }

    void v_SetBndCondPhysOp() override
    {
        const auto owned = v_GetOwnedBlocks();

        if (std::none_of(owned.begin(), owned.end(), [](bool b) { return b; }))
        {
            return;
        }

        ASSERTL0(m_session->DefinesEquationOfState(),
                 "No EquationOfState section defined in session file");

        auto EoS = m_session->GetEquationOfState();

        ASSERTL0(boost::iequals(EoS.type, "IdealGas"),
                 "StagnationInflow assumes the ideal gas enthalpy relations "
                 "and is not defined for equation of state '" +
                     EoS.type + "'.");
        ASSERTL0(EoS.params.count("GAMMA"),
                 "Need to specify Gamma in params of EquationOfState "
                 "definition");

        m_gamma = EoS.params["GAMMA"];

        m_rhoStag.clear();
        m_EStag.clear();
        m_dir.clear();
        m_rhoStag.resize(owned.size());
        m_EStag.resize(owned.size());
        m_dir.resize(owned.size());

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
            // over these pointers is a host dereference of device memory on
            // a real device.
            m_rhoStag[blk] = LibUtilities::MemoryRegion<TData>(stride);
            m_EStag[blk]   = LibUtilities::MemoryRegion<TData>(stride);
            m_dir[blk] = LibUtilities::MemoryRegion<TData>(m_coordDim * stride);
            m_rhoStag[blk].template Initialize<MemSpace>(0);
            m_EStag[blk].template Initialize<MemSpace>(0);
            m_dir[blk].template Initialize<MemSpace>(0);

            TData *rhoStag =
                m_rhoStag[blk].template GetPtr<MemSpace, ReadWrite>();
            TData *EStag = m_EStag[blk].template GetPtr<MemSpace, ReadWrite>();
            TData *dir   = m_dir[blk].template GetPtr<MemSpace, ReadWrite>();

            ReduceStagnationStateBlock<ExecSpace>(stride, m_coordDim,
                                                  m_numComp - 1, bndPtr, norms,
                                                  rhoStag, EStag, dir);
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

            const TData *rhoStag =
                m_rhoStag[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *EStag =
                m_EStag[blk].template GetPtr<MemSpace, ReadOnly>();
            const TData *dir = m_dir[blk].template GetPtr<MemSpace, ReadOnly>();

            ApplyStagnationInflowBlock<ExecSpace>(stride, m_coordDim,
                                                  m_numComp - 1, m_gamma,
                                                  rhoStag, EStag, dir, bndPtr);
        }
    }
};

} // namespace Nektar::detail
