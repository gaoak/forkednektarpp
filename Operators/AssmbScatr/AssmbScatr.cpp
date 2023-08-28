#include "AssmbScatr.hpp"

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
void OperatorAssmbScatrImpl<TData, TFieldState>::apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out)
{
    /* 
        ! ADD ASSEMBLY + SCATTER CODE HERE !

    // Assemble + Scatter

    // cast expansion list to continuous field
    // then retrieve ptr to the assembly map
    auto contField = std::dynamic_pointer_cast<ContField>(this->m_expansionList);
    auto assmbMap  = contField->GetLocalToGlobalMap();

    // Get the solution type
    GlobalSysSolnType solnType = assmbMap->GetGlobalSysSolnType();

    if (solnType == eIterativeFull)
    {
        assmbMap->Assemble(inArr, outArr);
        
        if (ZeroDir)
        {
            int nDir = assmbMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, outArr, 1);
        }
        assmbMap->GlobalToLocal(outArr, outArr);
    }
    else
    {
        assmbMap->AssembleBnd(inArr, outArr);
        if (ZeroDir)
        {
            int nDir = assmbMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, outArr, 1);
        }
        assmbMap->GlobalToLocalBnd(outArr, outArr);
    }

    */
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorAssmbScatrImpl<double, FieldState::Coeff>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "AssmbScatr",
        OperatorAssmbScatrImpl<double, FieldState::Coeff>::instantiate, 
        ""
    );

}