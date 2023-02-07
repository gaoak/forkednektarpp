#pragma once

#include <variant>

#include "Field.hpp"

enum class OpMethod
{
    LocMat,
    SumFac,
    MatFree
};

static constexpr OpMethod DefaultMethod = OpMethod::SumFac;

template <typename TData, FieldState TStateIn, FieldState TStateOut>
class OperatorBase
{

public:
    using OpBaseT = OperatorBase<TData, TStateIn, TStateOut>;

    virtual void apply(Field<TData, TStateIn> &in,
                       Field<TData, TStateOut> &out) = 0;

    virtual ~OperatorBase(){};
};
