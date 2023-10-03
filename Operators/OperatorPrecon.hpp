#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Precon base class
template <typename TData>
class OperatorPrecon : public OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    virtual ~OperatorPrecon() = default;

    OperatorPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>(expansionList)
    {    
    }

    virtual void apply(Field<TData, FieldState::Coeff> &in,
                       Field<TData, FieldState::Coeff> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Coeff> &in,
                            Field<TData, FieldState::Coeff> &out)
    {
        apply(in, out);
    }

    virtual void configure(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op) = 0;
};

}
