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
        : OperatorAssmbScatr<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false)
    {
        // Get the solution type
        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();

        auto nloc = m_assmbMap->GetNumLocalCoeffs();
        auto nglo = (solnType == eIterativeFull)
                        ? m_assmbMap->GetNumGlobalCoeffs()
                        : m_assmbMap->GetNumGlobalBndCoeffs();
        auto nDir = m_assmbMap->GetNumGlobalDirBndCoeffs();

        // Field -> Array (** Needs changing!)
        Array<OneD, TData> inArr(nloc, in.GetStorage().GetCPUPtr());
        Array<OneD, TData> tmpArr(nglo);
        Array<OneD, TData> outArr(nloc);

        // Lifted code:
        if (solnType == eIterativeFull)
        {
            m_assmbMap->Assemble(inArr, tmpArr);

            if (zeroDir)
            {
                Vmath::Zero(nDir, tmpArr, 1);
            }

            m_assmbMap->GlobalToLocal(tmpArr, outArr);
        }
        else
        {
            m_assmbMap->AssembleBnd(inArr, tmpArr);

            if (zeroDir)
            {
                Vmath::Zero(nDir, tmpArr, 1);
            }

            m_assmbMap->GlobalToLocalBnd(tmpArr, outArr);
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

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
