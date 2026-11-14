///////////////////////////////////////////////////////////////////////////////
//
// File: MeanOp.hpp
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
// Description: Device-portable domain average of a physical-space field.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/Mean/MeanBlockOp.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Domain average of every component of a physical-space field.
 *
 * For each component \f$ f \f$ the operator evaluates
 * \f$ \int_\Omega f \, d\Omega / \int_\Omega 1 \, d\Omega \f$. The integrals
 * are accumulated per block in the selected operator execution space by the
 * Mean block operators, packed into one MemoryRegion and reduced once
 * across the ranks. The domain volume is computed by the same block operators
 * on the first Apply() and kept.
 *
 * After Apply(), GetMeans() returns the domain average of each component, in
 * the order of the components the operator was created with, and GetVolume()
 * the domain volume \f$ \int_\Omega 1 \, d\Omega \f$ (the length or area in
 * 1D or 2D).
 *
 * With a homogeneous direction (3DH1) the block operators sum the local
 * planes of each component with their z quadrature weights, and the
 * integrals are reduced over all the spatial ranks. The volume is the plane
 * area, reduced over the ranks sharing the planes only, times the
 * homogeneous length.
 *
 * Apply() is collective: every rank must call it, and the first Apply()
 * also computes the volume. GetVolume() is valid on every rank. GetMeans() is
 * valid on every rank, or on rank zero only after SetRootOnly(true), which
 * reduces the integrals to rank zero alone.
 */
template <typename TData> class MeanOp : public MultiRegions::Operator<TData>
{
public:
    static std::shared_ptr<MeanOp<TData>> Create(
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

        auto op = MultiRegions::Operator<TData>::template Create<MeanOp>(
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

            op->m_blockOp.push_back(MeanBlockOp<TData>::Create(
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

    static inline const std::string name = "Mean";

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

    /// Reduce the integrals of the next Apply() to rank zero alone if
    /// @p rootOnly, else to every rank.
    void SetRootOnly(const bool rootOnly)
    {
        m_rootOnly = rootOnly;
    }

    /// Domain average of each component from the last Apply(), valid on rank
    /// zero only after SetRootOnly(true).
    std::vector<TData> GetMeans()
    {
        ASSERTL0(!m_rootOnly || this->m_expansionList->GetComm()
                                    ->GetSpaceComm()
                                    ->TreatAsRankZero(),
                 "MeanOp reduced its integrals to rank zero only.");
        const auto volume = GetVolume();
        auto means        = m_result.ToVector();
        for (auto &mean : means)
        {
            mean /= volume;
        }
        return means;
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
    /// Whether Apply() reduces the integrals to rank zero alone.
    bool m_rootOnly       = false;
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    /// Whether the field has a homogeneous direction (3DH1).
    bool m_homogeneous = false;
    /// Homogeneous length, or 1 without a homogeneous direction.
    TData m_homoLen = 1.0;
    /// Reduced integral of each component from Apply(), allocated by the
    /// implementation's constructor.
    LibUtilities::MemoryRegion<TData> m_result;
    std::vector<std::shared_ptr<MeanBlockOp<TData>>> m_blockOp;

    MeanOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_homogeneous = true;
            m_homoLen     = expansionList->GetHomoLen();
        }
    }

    ~MeanOp() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) = 0;
    virtual void v_ComputeVolume()                                         = 0;
};

} // namespace Nektar::SolverCore
