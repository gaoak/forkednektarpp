#pragma once

#include "Operators/OperatorDirBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorDirBndCondImpl<TData, ImplStdMat>
    : public OperatorDirBndCond<TData>
{
public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCond<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &out) override
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &locToGloMap       = contfield->GetLocalToGlobalMap();
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto &sign = locToGloMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map  = locToGloMap->GetBndCondCoeffsToLocalCoeffsMap();
        auto nloc  = locToGloMap->GetNumLocalCoeffs();

        size_t bndcnt = 0;
        auto outptr   = out.GetStorage().GetCPUPtr();
        std::fill(outptr, outptr + nloc, 0.0);
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                auto &bndcoeff = bndCondExpansions[i]->GetCoeffs();
                if (locToGloMap->GetSignChange())
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        *(outptr + map[bndcnt + j]) =
                            sign[bndcnt + j] * bndcoeff[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < bndCondExpansions[i]->GetNcoeffs();
                         j++)
                    {
                        *(outptr + map[bndcnt + j]) = bndcoeff[j];
                    }
                }
            }
            bndcnt += bndCondExpansions[i]->GetNcoeffs();
        }

        // communicate local Dirichlet coeffs that are just
        // touching a dirichlet boundary on another partition
        auto &ParallelDirBndSign = locToGloMap->GetParallelDirBndSign();

        for (auto &it : ParallelDirBndSign)
        {
            *(outptr + it) *= -1;
        }

        Array<OneD, NekDouble> arr(nloc, outptr);
        locToGloMap->UniversalAbsMaxBnd(arr);
        std::copy(arr.get(), arr.get() + nloc, outptr);

        for (auto &it : ParallelDirBndSign)
        {
            *(outptr + it) *= -1;
        }

        auto &copyLocalDirDofs = locToGloMap->GetCopyLocalDirDofs();
        for (auto &it : copyLocalDirDofs)
        {
            *(outptr + std::get<0>(it)) =
                *(outptr + std::get<1>(it)) * std::get<2>(it);
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDirBndCondImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

} // namespace Nektar::Operators::detail
