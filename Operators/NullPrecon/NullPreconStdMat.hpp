#pragma once

#include "Field.hpp"
#include "Operators/OperatorNullPrecon.hpp"
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorNullPreconImpl<TData, ImplStdMat>
    : public OperatorNullPrecon<TData>
{
public:
    OperatorNullPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNullPrecon<TData>(std::move(expansionList))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        GlobalSysSolnType solvertype = m_assmbMap->GetGlobalSysSolnType();

        bool isFull = solvertype == eIterativeFull ? true : false;

        size_t nGlobal = (isFull) ? m_assmbMap->GetNumGlobalCoeffs()
                                  : m_assmbMap->GetNumGlobalBndCoeffs();
        size_t nDir    = m_assmbMap->GetNumGlobalDirBndCoeffs();
        size_t nNonDir = nGlobal - nDir;
        size_t ncoeffs = in.GetStorage().size();

        auto *in_ptr  = in.GetStorage().GetCPUPtr();
        auto *out_ptr = out.GetStorage().GetCPUPtr();

        Array<OneD, TData> wk(nGlobal, 0.0);
        Array<OneD, TData> pIn(ncoeffs, in_ptr);
        Array<OneD, TData> pOut(ncoeffs, 0.0);
        (isFull) ? m_assmbMap->Assemble(pIn, wk)
                 : m_assmbMap->AssembleBnd(pIn, wk);
        std::fill(wk.get(), wk.get() + nDir, 0.0);
        (isFull) ? m_assmbMap->GlobalToLocal(wk, pOut)
                 : m_assmbMap->GlobalToLocalBnd(wk, pOut);
        std::copy(pOut.get(), pOut.get() + ncoeffs, out_ptr);
    }

    void configure(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
        boost::ignore_unused(op);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNullPreconImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
