#pragma once

#include "Operators/Field.hpp"
#include "Operators/AssmbScatr/AssmbScatrStdMat.hpp"
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
        : OperatorDiagPrecon<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        GlobalSysSolnType solvertype = m_assmbMap->GetGlobalSysSolnType();
        bool isFull = solvertype == eIterativeFull ? true : false;
        m_nGlobal   = (isFull) ? m_assmbMap->GetNumGlobalCoeffs()
                               : m_assmbMap->GetNumGlobalBndCoeffs();
        m_nLocal    = m_assmbMap->GetNumLocalCoeffs();
        m_nDir      = m_assmbMap->GetNumGlobalDirBndCoeffs();

        m_wk   = Array<OneD, TData>(m_nGlobal, 0.0);
        m_diag = Array<OneD, TData>(m_nGlobal, 0.0);

        m_assmbScatr =
            std::static_pointer_cast<OperatorAssmbScatrImpl<TData, ImplStdMat>>(
                AssmbScatr<TData>::create(this->m_expansionList));
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_assmbScatr->Assemble(in, m_wk);

        std::transform(m_wk.get() + m_nDir, m_wk.get() + m_nGlobal,
                       m_diag.get() + m_nDir, m_wk.get() + m_nDir,
                       [](TData in, TData diag) { return in / diag; });

        std::fill(m_wk.get(), m_wk.get() + m_nDir, 0.0);

        m_assmbScatr->GlobalToLocal(m_wk, out);
    }

    void configure(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
        auto robBCOp = RobBndCond<TData>::create(this->m_expansionList);

        Array<OneD, TData> diag(m_nLocal);

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

        size_t expIdx = 0;
        for (size_t block_idx = 0; block_idx < unit_vec.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = unit_vec.GetBlocks()[block_idx].num_elements;
            auto nmTot        = expPtr->GetNcoeffs();

            for (size_t i = 0; i < nmTot; ++i)
            {
                for (size_t j = 0; j < nElmts; ++j)
                {
                    // set ith term in unit vector to be 1
                    uvec_ptr[j * nmTot + i] = 1.0;
                }

                // apply operator to unit vector and store in action field
                op->apply(unit_vec, action);
                robBCOp->apply(unit_vec, action);

                for (size_t j = 0; j < nElmts; ++j)
                {
                    // copy ith row term from the action field to get ith
                    // diagonal
                    diag_ptr[j * nmTot + i] = actn_ptr[j * nmTot + i];

                    // reset ith term in unit vector to be 0
                    uvec_ptr[j * nmTot + i] = 0.0;
                }
            }
            uvec_ptr += unit_vec.GetBlocks()[block_idx].block_size;
            diag_ptr += unit_vec.GetBlocks()[block_idx].block_size;
            actn_ptr += unit_vec.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }

        // Assembly
        for (size_t i = 0; i < m_nLocal; ++i)
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
    std::shared_ptr<OperatorAssmbScatrImpl<TData, ImplStdMat>> m_assmbScatr;
    AssemblyMapCGSharedPtr m_assmbMap;
    Array<OneD, TData> m_diag;
    Array<OneD, TData> m_wk;
    size_t m_nGlobal;
    size_t m_nLocal;
    size_t m_nDir;
};

} // namespace Nektar::Operators::detail
