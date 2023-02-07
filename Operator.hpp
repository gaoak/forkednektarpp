#pragma once

#include <variant>

#include "Field.hpp"

struct MethodLocMat;
struct MethodSumFac;
struct MethodMatFree;
using DefaultMethod = MethodSumFac;

template <typename TData, typename TStateIn, typename TStateOut>
class OperatorBase
{

public:
    using OpBaseT = OperatorBase<TData, TStateIn, TStateOut>;

    virtual void apply(Field<TData, TStateIn> &in,
                       Field<TData, TStateOut> &out) = 0;

    virtual ~OperatorBase(){};
};
