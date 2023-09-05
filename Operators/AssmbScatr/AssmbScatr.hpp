#pragma once

#include "Operators/OperatorAssmbScatr.hpp"

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/ContField.h>
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorAssmbScatrImpl : public OperatorAssmbScatr<TData>
{
public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
    {
        // cast expansion list to continuous field
        // then retrieve ptr to the assembly map
        auto contField = std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto assmbMap  = contField->GetLocalToGlobalMap();

        // get number of local coeffs
        auto nloc = assmbMap->GetNumLocalCoeffs();

        // copy in_field data to in_array, initialise out_array
        Array<OneD, TData> inArr(nloc, in.GetStorage().GetCPUPtr());
        Array<OneD, TData> outArr(nloc);
        
        bool ZeroDir = true; // set to true in NekLinSysIterCGLoc.cpp

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

        // copy data from out_array to out_field
        std::copy(outArr.data(), outArr.data() + nloc, in.GetStorage().GetCPUPtr());
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorAssmbScatrImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}