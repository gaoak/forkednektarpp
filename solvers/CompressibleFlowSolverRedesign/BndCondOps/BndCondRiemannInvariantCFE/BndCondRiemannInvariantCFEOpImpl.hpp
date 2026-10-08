///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondRiemannInvariantCFEOpImpl.hpp
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

#include <boost/algorithm/string/predicate.hpp>

#include <MultiRegions/DisContField.h>

#include <LibUtilities/BasicUtils/Field/MemoryRegion.hpp>

#include "BndCondRiemannInvariantCFEKernels.hpp"
#include "BndCondRiemannInvariantCFEOp.hpp"

namespace Nektar::detail
{

/// Tag this operator claims.
inline bool IsRiemannInvariantTag(const std::string &tag)
{
    return tag == "RiemannInvariant";
}

template <typename ExecSpace, typename TData>
class BndCondRiemannInvariantCFEOpImpl
    : public BndCondRiemannInvariantCFEOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BndCondRiemannInvariantCFEOpImpl(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
        : BndCondRiemannInvariantCFEOp<TData>(expansionList, components)
    {
        m_numComp  = components.size();
        m_coordDim = MultiRegions::GetCollection(expansionList, 0)
                         .GetExpVector()[0]
                         ->GetCoordim();

        ASSERTL0(m_numComp == m_coordDim + 2,
                 "A Riemann invariant farfield needs density, one momentum "
                 "component per direction and energy.");

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
            BndCondRiemannInvariantCFEOpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    LibUtilities::SessionReaderSharedPtr m_session;
    unsigned int m_numComp  = 0;
    unsigned int m_coordDim = 0;
    TData m_gamma           = 0.0;
    TData m_rhoInf          = 0.0;
    TData m_pInf            = 0.0;
    /// Freestream velocity, one entry per direction.
    std::vector<TData> m_velInf;
    /// Outward normal component of the freestream velocity, per owned block.
    /// Held in the execution space's memory: the normals it is built from
    /// and the apply kernel that reads it both live there.
    std::vector<LibUtilities::MemoryRegion<TData>> m_VnInf;

    std::vector<bool> v_GetOwnedBlocks() const override
    {
        ASSERTL1(this->m_bndCondPhysOp,
                 "No boundary storage attached: call SetBndCondPhysOp().");

        const auto numBlocks = this->m_bndCondPhysOp->GetNumBlocks();
        std::vector<bool> owned(numBlocks, false);

        for (unsigned blk = 0; blk < numBlocks; ++blk)
        {
            owned[blk] = IsRiemannInvariantTag(
                this->m_bndCondPhysOp->GetBndUserDefined(blk, 0));
        }

        return owned;
    }

    /**
     * @brief Read the freestream and project it onto the outward normals.
     *
     * Unlike the entropy conditions this reads nothing out of the boundary
     * storage - the freestream is a session parameter - but it still belongs
     * here rather than in the constructor, because n.u needs the normals and
     * those arrive with the storage. Deferring it also keeps a session that
     * never asks for this condition from having to satisfy its requirements,
     * the operator being attached whether or not it is used.
     */
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
                 "RiemannInvariant assumes the ideal gas entropy "
                 "s = p/rho^gamma and is not defined for equation of state '" +
                     EoS.type + "'.");
        ASSERTL0(EoS.params.count("GAMMA"),
                 "Need to specify Gamma in params of EquationOfState "
                 "definition");

        m_gamma = EoS.params["GAMMA"];

        // Session parameters, with the defaults existing sessions rely on.
        // The region's own VALUE is not consulted.
        double rhoInf, pInf;
        m_session->LoadParameter("rhoInf", rhoInf, 1.225);
        m_session->LoadParameter("pInf", pInf, 101325.0);
        m_rhoInf = rhoInf;
        m_pInf   = pInf;

        const char *names[3] = {"uInf", "vInf", "wInf"};
        const double defs[3] = {0.1, 0.0, 0.0};
        m_velInf.assign(m_coordDim, TData(0.0));
        for (unsigned d = 0; d < m_coordDim; ++d)
        {
            double v;
            m_session->LoadParameter(names[d], v, defs[d]);
            m_velInf[d] = v;
        }

        m_VnInf.clear();
        m_VnInf.resize(owned.size());

        for (unsigned blk = 0; blk < owned.size(); ++blk)
        {
            if (!owned[blk])
            {
                continue;
            }

            const auto stride = this->m_bndCondPhysOp->GetBlockCompStride(blk);
            const auto *norms = this->m_bndCondPhysOp->GetBlockNormals(blk);

            // The normals live in the execution space's memory, so the
            // projection runs there as a kernel - a host loop over this
            // pointer is a host dereference of device memory on a real
            // device.
            m_VnInf[blk] = LibUtilities::MemoryRegion<TData>(stride);
            m_VnInf[blk].template Initialize<MemSpace>(0);
            TData *VnInf = m_VnInf[blk].template GetPtr<MemSpace, ReadWrite>();

            ProjectFreestreamVelocityBlock<ExecSpace>(
                stride, m_coordDim, m_velInf.data(), norms, VnInf);
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
            const TData *VnInf =
                m_VnInf[blk].template GetPtr<MemSpace, ReadOnly>();

            ApplyRiemannInvariantBlock<ExecSpace>(
                stride, m_coordDim, m_numComp - 1, m_gamma, m_rhoInf, m_pInf,
                m_velInf.data(), VnInf, norms, bndPtr);
        }
    }
};

} // namespace Nektar::detail
