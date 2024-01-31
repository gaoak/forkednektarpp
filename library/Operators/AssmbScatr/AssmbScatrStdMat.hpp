#pragma once

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

#include "Operators/OperatorAssmbScatr.hpp"

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

        m_solnType = m_assmbMap->GetGlobalSysSolnType();
        m_nloc     = m_assmbMap->GetNumLocalCoeffs();
        m_nglo     = (m_solnType == eIterativeFull)
                         ? m_assmbMap->GetNumGlobalCoeffs()
                         : m_assmbMap->GetNumGlobalBndCoeffs();
        m_ndir     = m_assmbMap->GetNumGlobalDirBndCoeffs();

        m_glo = Array<OneD, TData>(m_nglo);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false)
    {
        Assemble(in, m_glo);

        // Zeroing Dirichlet BC
        if (zeroDir)
        {
            Vmath::Zero(m_ndir, m_glo.get(), 1);
        }

        GlobalToLocal(m_glo, out);
    }

    void Assemble(Field<TData, FieldState::Coeff> &in,
                  Array<OneD, TData> outarray)
    {
        Array<OneD, TData> inarray(m_nloc);

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

        if (m_solnType == eIterativeFull)
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
        Array<OneD, TData> outarray(m_nloc);

        if (m_solnType == eIterativeFull)
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
    GlobalSysSolnType m_solnType;
    Array<OneD, TData> m_glo;
    size_t m_nloc;
    size_t m_nglo;
    size_t m_ndir;
};

} // namespace Nektar::Operators::detail
