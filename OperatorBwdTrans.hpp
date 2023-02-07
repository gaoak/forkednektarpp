#pragma once

#include "NekFactory.hpp"
#include "Operator.hpp"

#include <iostream>

template <typename TData>
using BwdTransBase = OperatorBase<TData, StateCoeff, StatePhys>;

template <typename TData, typename TMethod> class OperatorBwdTrans;

template <typename TData>
class OperatorBwdTrans<TData, MethodMatFree> : public BwdTransBase<TData>
{
public:
    void apply(Field<TData, StateCoeff> &in,
               Field<TData, StatePhys> &out) override
    {
        std::cout << "Op bwd trans mat free\n";
    }

    static std::unique_ptr<BwdTransBase<TData>> create()
    {
        return std::make_unique<OperatorBwdTrans<TData, MethodMatFree>>();
    }

    static std::string className;
};

using OpBwdTransFactory =
    Nektar::LibUtilities::NekFactory<std::string, BwdTransBase<double>>;

OpBwdTransFactory &GetOpBwdTransFactory()
{
    static OpBwdTransFactory instance;
    return instance;
}

template <>
std::string OperatorBwdTrans<double, MethodMatFree>::className =
    GetOpBwdTransFactory().RegisterCreatorFunction(
        "OperatorBwdTransDoubleMatFree",
        OperatorBwdTrans<double, MethodMatFree>::create, "...");

template <typename TData>
class OperatorBwdTrans<TData, MethodSumFac> : public BwdTransBase<TData>
{

    void apply(Field<TData, StateCoeff> &in,
               Field<TData, StatePhys> &out) override
    {
        std::cout << "Op bwd trans sum fac\n";
    }
};
