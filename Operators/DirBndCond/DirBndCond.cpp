#include "DirBndCond.hpp"

#include <tuple>
#include <set>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

// Declare function containing lifted code from Nektar in advance
void ImposeDirichletConditions(Array<OneD, NekDouble> &outarray, const ContFieldSharedPtr &explist);

template <typename TData>
void OperatorDirBndCondImpl<TData>::apply(Field<TData, FieldState::Coeff> &inout)
{
    /* IMPLEMENTATION OF APPLY DIRICHLET BOUNDARY CONDITIONS */

    // attributes required:
    // m_locToGloMap
    // m_bndCondExpansions
    // m_bndConditions

    // get number of local coeffs
    auto contField = std::dynamic_pointer_cast<ContField>(this->m_expansionList);
    auto nloc = contField->GetLocalToGlobalMap()->GetNumLocalCoeffs();

    // Field -> Array
    Array<OneD, NekDouble> outarray(nloc, inout.GetStorage().GetCPUPtr());

    // Core of function
    ImposeDirichletConditions(outarray, contField);
    
    // Array -> Field
    std::copy(outarray.data(), outarray.data() + nloc, inout.GetStorage().GetCPUPtr());
}

// Lifted code from Nektar (ContField.cpp)
void ImposeDirichletConditions(Array<OneD, NekDouble> &outarray, const ContFieldSharedPtr &contField)
{
    // get attributes from expansion list
    auto locToGloMap = contField->GetLocalToGlobalMap();

    Array<OneD, ExpListSharedPtr> bndCondExpansions = contField->GetBndCondExpansions(); // = contField->GetBndCondExpansions() ? -- can't find a getter fcn to access this
    Array<OneD, BoundaryConditionShPtr> bndConditions = contField->GetBndConditions(); // = contField->GetBndConditions() ?

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
                    outarray[map[bndcnt + j]] = sign[bndcnt + j] * bndcoeff[j];
                }
            }
            else
            {
                for (j = 0; j < (bndCondExpansions[i])->GetNcoeffs(); j++)
                {
                    outarray[map[bndcnt + j]] = bndcoeff[j];
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
        outarray[it] *= -1;
    }

    locToGloMap->UniversalAbsMaxBnd(outarray);

    for (auto &it : ParallelDirBndSign)
    {
        outarray[it] *= -1;
    }

    std::set<ExtraDirDof> &copyLocalDirDofs = locToGloMap->GetCopyLocalDirDofs();
    for (auto &it : copyLocalDirDofs)
    {
        outarray[std::get<0>(it)] = outarray[std::get<1>(it)] * std::get<2>(it);
    }
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorDirBndCondImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DirBndCond",
        OperatorDirBndCondImpl<double>::instantiate, 
        ""
    );

}