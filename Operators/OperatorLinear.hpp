#pragma once

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"

namespace Nektar::Operators
{

template <typename TData, FieldState TFieldIn, FieldState TFieldOut>
class OperatorLinear : public Operator<TData>
{

public:
    virtual ~OperatorLinear() = default;

    OperatorLinear(const MultiRegions::ExpListSharedPtr &expansionList)
        : Operator<TData>(expansionList)
    {    
    }

    virtual void apply(Field<TData, TFieldIn> &in, Field<TData, TFieldOut> &out) = 0;
};

}