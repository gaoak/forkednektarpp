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

        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();
        auto nglo                  = (solnType == eIterativeFull)
                                         ? m_assmbMap->GetNumGlobalCoeffs()
                                         : m_assmbMap->GetNumGlobalBndCoeffs();

        m_tmp = Array<OneD, TData>(nglo);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false)
    {
        Assemble(in, m_tmp);

        if (zeroDir)
        {
            auto nDir = m_assmbMap->GetNumGlobalDirBndCoeffs();
            Vmath::Zero(nDir, m_tmp.get(), 1);
        }

        GlobalToLocal(m_tmp, out);
    }

    void Assemble(Field<TData, FieldState::Coeff> &in,
                  Array<OneD, TData> outarray)
    {
        // Get the solution type
        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();

        auto nloc = m_assmbMap->GetNumLocalCoeffs();
        Array<OneD, TData> inarray(nloc);

        // Copy data from input field
        auto *inarrptr = inarray.data();
        auto *inptr    = in.GetStorage().GetCPUPtr();
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = in.GetBlocks()[block_idx].block_size;
            auto nElmts = in.GetBlocks()[block_idx].num_elements;
            auto nmTot  = in.GetBlocks()[block_idx].num_pts;

            std::copy(inptr, inptr + nElmts * nmTot, inarrptr);

            inarrptr += nElmts * nmTot;
            inptr += nSize;
        }

        if (solnType == eIterativeFull)
        {
            m_assmbMap->Assemble(inarray, outarray);
        }
        else
        {
            m_assmbMap->AssembleBnd(inarray, outarray);
        }
    }

    void GlobalToLocal(Array<OneD, TData> inarray,
                       Field<TData, FieldState::Coeff> &out)
    {
        // Get the solution type
        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();

        auto nloc = m_assmbMap->GetNumLocalCoeffs();
        Array<OneD, TData> outarray(nloc);

        if (solnType == eIterativeFull)
        {
            m_assmbMap->GlobalToLocal(inarray, outarray);
        }
        else
        {
            m_assmbMap->GlobalToLocalBnd(inarray, outarray);
        }

        // Copy data to output field
        auto *outarrptr = outarray.data();
        auto *outptr    = out.GetStorage().GetCPUPtr();
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            auto nSize  = out.GetBlocks()[block_idx].block_size;
            auto nElmts = out.GetBlocks()[block_idx].num_elements;
            auto nmTot  = out.GetBlocks()[block_idx].num_pts;

            std::copy(outarrptr, outarrptr + nElmts * nmTot, outptr);

            outarrptr += nElmts * nmTot;
            outptr += nSize;
        }
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
    Array<OneD, TData> m_tmp;
};

} // namespace Nektar::Operators::detail
