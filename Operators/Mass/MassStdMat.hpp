#pragma once

#include "Operators/OperatorBwdTrans.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorMass.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorMassImpl<TData, ImplStdMat> : public OperatorMass<TData>
{
public:
    OperatorMassImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMass<TData>(std::move(expansionList)),
          m_field(Field<TData, FieldState::Phys>::create(
              GetBlockAttributes(FieldState::Phys, expansionList)))
    {
        m_BwdTransOp = BwdTrans<TData>::create(this->m_expansionList);
        m_IProductWRTBaseOp =
            IProductWRTBase<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: BwdTrans
        m_BwdTransOp->apply(in, m_field);

        // Step 2: Inner product for mass matrix operation
        m_IProductWRTBaseOp->apply(m_field, out);
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMassImpl<TData, ImplStdMat>>(
            expansionList);
    }

    static std::string className;

protected:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    Field<TData, FieldState::Phys> m_field;
};

} // namespace Nektar::Operators::detail
