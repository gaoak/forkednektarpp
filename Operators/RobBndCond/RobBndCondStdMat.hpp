#pragma once

#include "Operators/OperatorRobBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorRobBndCondImpl<TData, ImplStdMat>
    : public OperatorRobBndCond<TData>
{
public:
    OperatorRobBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorRobBndCond<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &negflag) override
    {
        auto ncoeffs = out.GetStorage().size();
        auto *inptr  = in.GetStorage().GetCPUPtr();
        auto *outptr = out.GetStorage().GetCPUPtr();
        Array<OneD, NekDouble> inarray(ncoeffs, inptr);
        Array<OneD, NekDouble> robin(ncoeffs, 0.0);
        auto *robptr     = robin.get();
        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();
        for (auto &r : robinBCInfo) // add robin mass matrix
        {
            auto n      = r.first;
            auto offset = this->m_expansionList->GetCoeff_Offset(n);
            auto expPtr = this->m_expansionList->GetExp(n);
            Array<OneD, NekDouble> tmp;
            for (auto rBC = r.second; rBC; rBC = rBC->next)
            {
                expPtr->AddRobinTraceContribution(
                    rBC->m_robinID, rBC->m_robinPrimitiveCoeffs,
                    inarray + offset, tmp = robin + offset);
            }
        }
        if (negflag)
        {
            std::transform(robin.get(), robin.get() + ncoeffs, outptr, outptr,
                           [](const TData &rob, const TData &out)
                           { return out - rob; });
        }
        else
        {
            std::transform(robin.get(), robin.get() + ncoeffs, outptr, outptr,
                           [](const TData &rob, const TData &out)
                           { return out + rob; });
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorRobBndCondImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

} // namespace Nektar::Operators::detail
