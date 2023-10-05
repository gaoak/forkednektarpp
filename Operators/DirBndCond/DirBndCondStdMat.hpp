#pragma once

#include "Operators/OperatorDirBndCond.hpp"

#include <set>
#include <tuple>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

void ImposeDirichletConditions(Array<OneD, NekDouble> &arr,
                               const ContFieldSharedPtr &contfield);

template <typename TData>
class OperatorDirBndCondImpl<TData, ImplStdMat>
    : public OperatorDirBndCond<TData>
{
public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCond<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // get number of local coeffs
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto nloc = contfield->GetLocalToGlobalMap()->GetNumLocalCoeffs();

        // Field -> Array (** Needs changing!)
        Array<OneD, NekDouble> arr(nloc, inout.GetStorage().GetCPUPtr());

        // Core of function
        ImposeDirichletConditions(arr, contfield);

        // Array -> Field (** Needs changing!)
        std::copy(arr.data(), arr.data() + nloc,
                  inout.GetStorage().GetCPUPtr());
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

// Lifted code from Nektar (ContField.cpp)
void ImposeDirichletConditions(Array<OneD, NekDouble> &arr,
                               const ContFieldSharedPtr &contfield)
{
    // get attributes from expansion list
    auto locToGloMap = contfield->GetLocalToGlobalMap();

    Array<OneD, ExpListSharedPtr> bndCondExpansions =
        contfield->GetBndCondExpansions();
    Array<OneD, BoundaryConditionShPtr> bndConditions =
        contfield->GetBndConditions();

    // lifted code:
    int i, j;
    int bndcnt = 0;

    Array<OneD, NekDouble> sign =
        locToGloMap->GetBndCondCoeffsToLocalCoeffsSign();
    const Array<OneD, const int> map =
        locToGloMap->GetBndCondCoeffsToLocalCoeffsMap();

    for (i = 0; i < bndCondExpansions.size(); ++i)
    {
        if (bndConditions[i]->GetBoundaryConditionType() ==
            SpatialDomains::eDirichlet)
        {

            const Array<OneD, const NekDouble> bndcoeff =
                (bndCondExpansions[i])->GetCoeffs();

            if (locToGloMap->GetSignChange())
            {
                for (j = 0; j < (bndCondExpansions[i])->GetNcoeffs(); j++)
                {
                    arr[map[bndcnt + j]] = sign[bndcnt + j] * bndcoeff[j];
                }
            }
            else
            {
                for (j = 0; j < (bndCondExpansions[i])->GetNcoeffs(); j++)
                {
                    arr[map[bndcnt + j]] = bndcoeff[j];
                }
            }
        }
        bndcnt += bndCondExpansions[i]->GetNcoeffs();
    }

    // communicate local Dirichlet coeffs that are just
    // touching a dirichlet boundary on another partition
    std::set<int> &ParallelDirBndSign = locToGloMap->GetParallelDirBndSign();

    for (auto &it : ParallelDirBndSign)
    {
        arr[it] *= -1;
    }

    locToGloMap->UniversalAbsMaxBnd(arr);

    for (auto &it : ParallelDirBndSign)
    {
        arr[it] *= -1;
    }

    std::set<ExtraDirDof> &copyLocalDirDofs =
        locToGloMap->GetCopyLocalDirDofs();
    for (auto &it : copyLocalDirDofs)
    {
        arr[std::get<0>(it)] = arr[std::get<1>(it)] * std::get<2>(it);
    }
}

} // namespace Nektar::Operators::detail
