#pragma once

#include "Operators/OperatorAssmbScatr.hpp"

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorAssmbScatrImpl<TData, ImplStdMat>
    : public OperatorAssmbScatr<TData>
{
public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out)
    {
        // cast expansion list to continuous field
        // then retrieve ptr to the assembly map
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto assmbMap = contfield->GetLocalToGlobalMap();

        // get number of local coeffs
        auto nloc = assmbMap->GetNumLocalCoeffs();

        // Field -> Array (** Needs changing!)
        Array<OneD, TData> inArr(nloc, in.GetStorage().GetCPUPtr());
        Array<OneD, TData> outArr(nloc);

        // Lifted code:
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

        // Array -> Field (** Needs changing!)
        std::copy(outArr.data(), outArr.data() + nloc,
                  out.GetStorage().GetCPUPtr());
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorAssmbScatrImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

} // namespace Nektar::Operators::detail
