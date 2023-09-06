#pragma once

#include "Operators/OperatorNeuBndCond.hpp"

#include <MultiRegions/ContField.h>
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

void ImposeNeumannConditions(Array<OneD, NekDouble> &arr, const ContFieldSharedPtr &contfield);

template <typename TData>
class OperatorNeuBndCondImpl : public OperatorNeuBndCond<TData>
{
public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &inout)
    {
        // get number of local coeffs
        auto contField = std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto nloc = contField->GetLocalToGlobalMap()->GetNumLocalCoeffs();

        // Field -> Array (** Needs changing!)
        Array<OneD, NekDouble> rhs(nloc, inout.GetStorage().GetCPUPtr());

        // Core of function
        ImposeNeumannConditions(rhs, contField);
        
        // Array -> Field (** Needs changing!)
        std::copy(rhs.data(), rhs.data() + nloc, inout.GetStorage().GetCPUPtr());
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNeuBndCondImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

// Lifted from ContField.cpp (HelmSolve function)
void ImposeNeumannConditions(Array<OneD, NekDouble> &arr, const ContFieldSharedPtr &contfield)
{
    auto locToGloMap = contfield->GetLocalToGlobalMap();
    auto bndCondExpansions = contfield->GetBndCondExpansions();
    auto bndConditions = contfield->GetBndConditions();

    int bndcnt = 0;
    Array<OneD, NekDouble> sign = locToGloMap->GetBndCondCoeffsToLocalCoeffsSign();
    const Array<OneD, const int> map = locToGloMap->GetBndCondCoeffsToLocalCoeffsMap();
    // Add weak boundary conditions to forcing
    for (size_t i = 0; i < bndCondExpansions.size(); ++i)
    {
        if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eNeumann ||
            bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eRobin)
        {

            const Array<OneD, const NekDouble> bndcoeff =
                (bndCondExpansions[i])->GetCoeffs();

            if (locToGloMap->GetSignChange())
            {
                for (size_t j = 0; j < (bndCondExpansions[i])->GetNcoeffs(); j++)
                {
                    arr[map[bndcnt + j]] += sign[bndcnt + j] * bndcoeff[j];
                }
            }
            else
            {
                for (size_t j = 0; j < (bndCondExpansions[i])->GetNcoeffs(); j++)
                {
                    arr[map[bndcnt + j]] += bndcoeff[j];
                }
            }
        }
        bndcnt += bndCondExpansions[i]->GetNcoeffs();
    }
}

}