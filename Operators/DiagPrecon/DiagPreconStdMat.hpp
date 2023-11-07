#pragma once

#include "Field.hpp"
#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDiagPreconImpl<TData, ImplStdMat>
    : public OperatorDiagPrecon<TData>
{
public:
    OperatorDiagPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDiagPrecon<TData>(std::move(expansionList))
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
        m_robBCOp  = RobBndCond<TData>::create(this->m_expansionList);
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
        size_t nLocal  = in.GetStorage().size();

        auto *diag_ptr = m_diag.get();
        auto *in_ptr   = in.GetStorage().GetCPUPtr();
        auto *out_ptr  = out.GetStorage().GetCPUPtr();

        Array<OneD, TData> wk(nGlobal, 0.0);
        Array<OneD, TData> pIn(nLocal, in_ptr);
        Array<OneD, TData> pOut(nLocal, 0.0);
        (isFull) ? m_assmbMap->Assemble(pIn, wk)
                 : m_assmbMap->AssembleBnd(pIn, wk);
        std::transform(wk.get() + nDir, wk.get() + nGlobal, diag_ptr + nDir,
                       wk.get() + nDir,
                       [](TData in, TData diag) { return in / diag; });
        std::fill(wk.get(), wk.get() + nDir, 0.0);
        (isFull) ? m_assmbMap->GlobalToLocal(wk, pOut)
                 : m_assmbMap->GlobalToLocalBnd(wk, pOut);
        std::copy(pOut.get(), pOut.get() + nLocal, out_ptr);
    }

    void configure(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
        GlobalSysSolnType solvertype = m_assmbMap->GetGlobalSysSolnType();

        bool isFull = solvertype == eIterativeFull ? true : false;

        size_t nLocal  = m_assmbMap->GetNumLocalCoeffs();
        size_t nGlobal = (isFull) ? m_assmbMap->GetNumGlobalCoeffs()
                                  : m_assmbMap->GetNumGlobalBndCoeffs();
        size_t nDir    = m_assmbMap->GetNumGlobalDirBndCoeffs();
        size_t nNonDir = nGlobal - nDir;

        Array<OneD, TData> diag(nLocal, 0.0);

        m_diag = Array<OneD, TData>(nGlobal, 0.0);

        // create unit vector field to extract diagonal
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::create(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        // create action field to receive column action from unit vector
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::create(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        auto *uvec_ptr = unit_vec.GetStorage().GetCPUPtr();
        auto *actn_ptr = action.GetStorage().GetCPUPtr();
        auto *diag_ptr = diag.get();

        for (size_t i = 0; i < nLocal; ++i)
        {
            // set ith term in unit vector to be 1
            *uvec_ptr = 1.0;

            // apply operator to unit vector and store in action field
            op->apply(unit_vec, action);
            m_robBCOp->apply(unit_vec, action);

            // copy ith row term from the action field to get ith diagonal
            *(diag_ptr++) = *(actn_ptr++);

            // reset ith term in unit vector to be 0
            *(uvec_ptr++) = 0.0;
        }

        // Assembly
        for (size_t i = 0; i < nLocal; ++i)
        {
            size_t gid1 = m_assmbMap->GetLocalToGlobalMap(i);
            m_diag[gid1] += diag[i];
        }
        m_assmbMap->UniversalAssemble(m_diag);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDiagPreconImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBCOp;
    AssemblyMapCGSharedPtr m_assmbMap;
    Array<OneD, TData> m_diag;
};

} // namespace Nektar::Operators::detail
