#pragma once

#include "Field.hpp"
#include "Operators/OperatorDiagPrecon.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDiagPreconImpl : public OperatorDiagPrecon<TData>
{
public:
    OperatorDiagPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDiagPrecon<TData>(std::move(expansionList)),
        m_diag(Field<TData, FieldState::Coeff>::create(GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out);
    void configure(const std::unique_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op);

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDiagPreconImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
    
protected:
    // diagonal of conditioner
    Field<TData, FieldState::Coeff> m_diag;

};

}