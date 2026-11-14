///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2Op.hpp
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

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/NormOps/NormL2/NormL2BlockOp.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief L2 norm of every component of a physical-space field.
 *
 * For each component \f$ f \f$ the operator evaluates
 * \f$ ( \int_\Omega f^2 \, d\Omega )^{1/2} \f$, or, normalised, that divided
 * by \f$ ( \int_\Omega 1 \, d\Omega )^{1/2} \f$. The integrals are
 * accumulated per block in the selected operator execution space by the
 * NormL2 block operators, packed into one MemoryRegion and reduced once
 * across the ranks. The domain volume is computed by the same block
 * operators on the first Apply() and kept.
 *
 * With a homogeneous direction (3DH1) the block operators sum the local
 * planes of each component with their z quadrature weights. The volume is
 * the plane area, reduced over the ranks sharing the planes only, times the
 * homogeneous length.
 *
 * Apply() is collective: every rank must call it, and the first Apply()
 * also computes the volume. GetVolume() is valid on every rank. GetNorms() is
 * valid on every rank, or on rank zero only after SetRootOnly(true), which
 * reduces the norms to rank zero alone.
 */
template <typename TData> class NormL2Op : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<NormL2Op<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        auto op = MultiRegions::Operator<TData>::template Create<NormL2Op>(
            expansionList, components, execStr0);

        // z quadrature weight of each local plane, times Lhom / 2, with which
        // the block ops sum the planes; empty without a homogeneous direction.
        std::vector<TData> planeWeights;
        if (op->m_homogeneous)
        {
            const auto w    = expansionList->GetHomogeneousBasis()->GetW();
            const auto zIDs = expansionList->GetZIDs();
            for (unsigned int n = 0; n < zIDs.size(); ++n)
            {
                planeWeights.push_back(w[zIDs[n]] * op->m_homoLen * 0.5);
            }
        }

        const auto blockAttr =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);

        for (unsigned int block_idx = 0; block_idx < blockAttr.size();
             block_idx++)
        {
            const auto exp =
                MultiRegions::GetCollection(expansionList, block_idx)
                    .GetExpVector()[0];

            op->m_blockOp.push_back(NormL2BlockOp<TData>::Create(
                block_idx, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));
            op->m_blockOp.back()->SetNumElements(
                blockAttr[block_idx].GetNumElements());
            if (op->m_homogeneous)
            {
                op->m_blockOp.back()->SetPlaneWeights(planeWeights);
            }
        }

        return op;
    }

    static inline const std::string name = "NormL2";

    /// Collective: every rank must call it.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        v_Apply(in);
    }

    /// Collective: as Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        v_Apply(in);
    }

    /// Reduce the norms of the next Apply() to rank zero alone if
    /// @p rootOnly, else to every rank.
    void SetRootOnly(const bool rootOnly)
    {
        m_rootOnly = rootOnly;
    }

    /// L2 norm of each component from the last Apply(), divided by the
    /// square root of the volume if @p normalised; valid on rank zero only
    /// after SetRootOnly(true).
    std::vector<TData> GetNorms(const bool &normalised = false)
    {
        ASSERTL0(!m_rootOnly || this->m_expansionList->GetComm()
                                    ->GetSpaceComm()
                                    ->TreatAsRankZero(),
                 "NormL2Op reduced its norms to rank zero only.");
        auto volume = normalised ? GetVolume() : 1.0;
        auto data   = this->m_norm.ToVector();
        for (auto &val : data)
        {
            val = std::sqrt(val / volume);
        }
        return data;
    }

    /// Domain volume, computed by the first Apply() or on the first call,
    /// whichever comes first, and kept; collective on the first call before
    /// any Apply().
    TData GetVolume()
    {
        v_ComputeVolume();
        return m_volume;
    }

protected:
    /// Whether Apply() reduces the norms to rank zero alone.
    bool m_rootOnly       = false;
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    /// Whether the field has a homogeneous direction (3DH1).
    bool m_homogeneous = false;
    /// Homogeneous length, or 1 without a homogeneous direction.
    TData m_homoLen = 1.0;
    /// Reduced integral of the square of each component from Apply(),
    /// allocated by the implementation's constructor.
    LibUtilities::MemoryRegion<TData> m_norm;
    std::vector<std::shared_ptr<NormL2BlockOp<TData>>> m_blockOp;

    NormL2Op(const MultiRegions::ExpListSharedPtr &expansionList,
             const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_homogeneous = true;
            m_homoLen     = expansionList->GetHomoLen();
        }
    }

    ~NormL2Op() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) = 0;
    virtual void v_ComputeVolume()                                         = 0;
};

} // namespace Nektar::SolverCore
