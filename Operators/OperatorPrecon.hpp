#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

template <typename TData>
class OperatorPrecon : public OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>
{

public:
    virtual ~OperatorPrecon() = default;

    OperatorPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>(expansionList)
    {    
    }

    virtual void configure(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op) = 0;
};

}