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

#include <vector>

#include <MultiRegions/Common/Operator.hpp>
#include <SolverCore/MeanRemoval/MeanRemovalBlockOp.hpp>
#include <SolverCore/SolverCore.hpp>

namespace Nektar::SolverCore
{

/**
 * @brief Domain average of every component of a physical-space field.
 *
 * For each component \f$ f \f$ the operator evaluates
 * \f$ \int_\Omega f \, d\Omega / \int_\Omega 1 \, d\Omega \f$. The integrals
 * are accumulated per block in the selected operator execution space by the
 * MeanRemoval block operators, packed into one MemoryRegion and reduced once
 * to rank zero. The domain volume is computed by the same block operators on
 * the first Apply() and kept.
 *
 * After Apply(), GetMeans() returns the domain average of each component, in
 * the order of the components the operator was created with, and GetVolume()
 * the domain volume \f$ \int_\Omega 1 \, d\Omega \f$ (the length or area in
 * 1D or 2D).
 *
 * Apply() is collective: every rank must call it. GetMeans() reads the
 * reduced integrals on the host and is valid on rank zero only, the other
 * ranks holding their own local integrals. GetVolume() is valid on every
 * rank.
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

        const std::string execStr0 =
            (execStr == "") ? MultiRegions::Operator<TData>::GetOpExecSpace(
                                  expansionList->GetSession())
                            : execStr;

        auto op = MultiRegions::Operator<TData>::template Create<MeanOp>(
            expansionList, components, execStr0);

        const auto blockAttributes =
            MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                expansionList);
        for (unsigned int block = 0; block < blockAttributes.size(); ++block)
        {
            const auto exp = MultiRegions::GetCollection(expansionList, block)
                                 .GetExpVector()[0];
            op->m_blockOp.push_back(MeanRemovalBlockOp<TData>::Create(
                block, exp, expansionList->GetDataWarehouseSharedPtr(),
                execStr0));
            op->m_blockOp.back()->SetNumElements(
                blockAttributes[block].GetNumElements());
        }

        return op;
    }

    static inline const std::string name = "Mean";

    /// Collective: every rank must call it.
    void Apply(LibUtilities::Field<TData, FieldState::Phys> &field)
    {
        v_Apply(field);
    }

    /// Collective: as Apply().
    void operator()(LibUtilities::Field<TData, FieldState::Phys> &field)
    {
        v_Apply(field);
    }

    /// Domain average of each component from the last Apply(), valid on rank
    /// zero only.
    std::vector<TData> GetMeans()
    {
        const auto result =
            m_result.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        std::vector<TData> means(result, result + this->m_components.size());
        for (auto &mean : means)
        {
            mean /= m_volume;
        }
        return means;
    }

    /// Domain volume, valid on every rank after the first Apply().
    TData GetVolume() const
    {
        return m_volume;
    }

protected:
    bool m_volumeComputed = false;
    TData m_volume        = 1.0;
    std::vector<std::shared_ptr<MeanRemovalBlockOp<TData>>> m_blockOp;
    /// Reduced integral of each component from Apply(), allocated by the
    /// implementation's constructor.
    LibUtilities::MemoryRegion<TData> m_result;

    MeanOp(const MultiRegions::ExpListSharedPtr &expansionList,
           const std::vector<std::string> &components)
        : MultiRegions::Operator<TData>(expansionList, components)
    {
    }

    ~MeanOp() override = default;

    virtual void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &field) = 0;
};

} // namespace Nektar::SolverCore
