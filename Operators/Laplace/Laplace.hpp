#pragma once

#include "Operators/OperatorLaplace.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

#include <Field.hpp>
#include <Operators/OperatorPhysDeriv.hpp>
#include <Operators/OperatorBwdTrans.hpp>
#include <Operators/OperatorIProductWRTBase.hpp>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorLaplaceImpl : public OperatorLaplace<TData>
{
public:
    OperatorLaplaceImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLaplace<TData>(std::move(expansionList)),
        m_tmp0(Field<TData, FieldState::Phys>::create(GetBlockAttributes(FieldState::Phys, expansionList))),
        m_tmp1(Field<TData, FieldState::Phys>::create(GetBlockAttributes(FieldState::Phys, expansionList))),
        m_tmp2(Field<TData, FieldState::Phys>::create(GetBlockAttributes(FieldState::Phys, expansionList))),
        m_tmp3(Field<TData, FieldState::Phys>::create(GetBlockAttributes(FieldState::Phys, expansionList)))
    {
        m_BwdTransOp = BwdTrans<TData>::create(this->m_expansionList);
        m_IProdOp = IProductWRTBase<TData>::create(this->m_expansionList);
        m_PhysDerivOp = PhysDeriv<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
    {
        // BwdTrans
        //m_BwdTransOp->apply(in, m_tmp0);

        // PhysDeriv
        //m_PhysDerivOp->apply(m_tmp0, m_tmp1, m_tmp2, m_tmp3);

        // IProductWRTDerBase
        //m_IProdOp->apply(m_tmp1, m_tmp2, m_tmp3, out);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorLaplaceImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorPhysDeriv<TData>> m_PhysDerivOp;
    Field<TData, FieldState::Phys> m_tmp0;
    Field<TData, FieldState::Phys> m_tmp1;
    Field<TData, FieldState::Phys> m_tmp2;
    Field<TData, FieldState::Phys> m_tmp3;
};

}