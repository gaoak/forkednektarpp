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
        // Force a genuine cross-library symbol reference into libSolverCore so
        // the NormL2*OpImpl static factory registrations are not dropped by
        // the linker/loader -- see EnsureLinked() in SolverCore.hpp.
        EnsureLinked();

        auto session = expansionList->GetSession();

        std::string execStr0 =
            (execStr == "")
                ? MultiRegions::Operator<TData>::GetOpExecSpace(session)
                : execStr;

        auto op = MultiRegions::Operator<TData>::template Create<NormL2Op>(
            expansionList, components, execStr0);

        auto blockAttr =
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
        if (normalised)
        {
            this->v_ComputeVolume();
        }

        auto volume = normalised ? this->m_volume : 1.0;
        auto data   = this->m_norm.template ToVector<TData>();
        for (unsigned int nc = 0; nc < this->m_norm.size(); ++nc)
        {
            data[nc] = std::sqrt(data[nc] / volume);
        }
        return data;
    }

protected:
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    /// Homogeneous length, or 1 without a homogeneous direction.
    TData m_homoLen = 1.0;
    /// z quadrature weight of each local plane, times Lhom / 2; empty
    /// without a homogeneous direction.
    std::vector<TData> m_planeWeights;
    LibUtilities::MemoryRegion<TData> m_norm;
    LibUtilities::MemoryRegion<TData> m_plane;
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
            for (size_t n = 0; n < zIDs.size(); ++n)
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
