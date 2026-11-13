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
        }

        return op;
    }

    static inline const std::string name = "NormL2";

    void Apply(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        v_Apply(in);
    }

    void operator()(LibUtilities::Field<TData, FieldState::Phys> &in)
    {
        v_Apply(in);
    }

    std::vector<TData> GetNorms(const bool &normalised = false)
    {
        auto volume = normalised ? GetVolume() : 1.0;
        auto data   = this->m_norm.ToVector();
        for (auto &val : data)
        {
            val = std::sqrt(val / volume);
        }
        return data;
    }

    /// Domain volume, computed on the first call and kept; collective on the
    /// first call.
    TData GetVolume()
    {
        v_ComputeVolume();
        return m_volume;
    }

protected:
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    /// Homogeneous length, or 1 without a homogeneous direction.
    TData m_homoLen = 1.0;
    /// z quadrature weight of each local plane, times Lhom / 2; empty
    /// without a homogeneous direction.
    std::vector<TData> m_planeWeights;
    LibUtilities::MemoryRegion<TData> m_plane;
    LibUtilities::MemoryRegion<TData> m_norm;
    std::vector<std::shared_ptr<NormL2BlockOp<TData>>> m_blockOp;

    NormL2Op(const MultiRegions::ExpListSharedPtr &expansionList,
             const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
        if (expansionList->GetExpType() == MultiRegions::e3DH1D)
        {
            m_homoLen = expansionList->GetHomoLen();

            const auto w    = expansionList->GetHomogeneousBasis()->GetW();
            const auto zIDs = expansionList->GetZIDs();
            for (unsigned int n = 0; n < zIDs.size(); ++n)
            {
                m_planeWeights.push_back(w[zIDs[n]] * m_homoLen * 0.5);
            }
        }
    }

    ~NormL2Op() override = default;

    virtual void v_Apply(LibUtilities::Field<TData, FieldState::Phys> &in) = 0;
    virtual void v_ComputeVolume()                                         = 0;
};

} // namespace Nektar::SolverCore
