#pragma once

#include "Operators/OperatorNeuBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorNeuBndCondImpl<TData, ImplStdMat>
    : public OperatorNeuBndCond<TData>
{
public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &locToGloMap       = contfield->GetLocalToGlobalMap();
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &sign = locToGloMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map  = locToGloMap->GetBndCondCoeffsToLocalCoeffsMap();

        size_t bndcnt = 0;
        auto outptr   = inout.GetStorage().GetCPUPtr();
        // Add weak boundary conditions to forcing
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eNeumann ||
                bndConditions[i]->GetBoundaryConditionType() ==
                    SpatialDomains::eRobin)
            {
                auto &bndcoeff = bndCondExpansions[i]->GetCoeffs();
                if (locToGloMap->GetSignChange())
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        *(outptr + map[bndcnt + j]) +=
                            sign[bndcnt + j] * bndcoeff[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        *(outptr + map[bndcnt + j]) += bndcoeff[j];
                    }
                }
            }
            bndcnt += bndCondExpansions[i]->GetNcoeffs();
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNeuBndCondImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

} // namespace Nektar::Operators::detail
