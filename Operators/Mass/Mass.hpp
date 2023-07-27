#pragma once

#include "Operators/OperatorMass.hpp"
#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorMassImpl : public OperatorMass<TData>
{
public:
    OperatorMassImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMass<TData>(std::move(expansionList))
    {
        this->m_BwdTransOp = BwdTrans<TData>::create(this->m_expansionList);
        this->m_IProductWRTBaseOp = IProductWRTBase<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out);

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMassImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
};

}