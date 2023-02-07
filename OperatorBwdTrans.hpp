#pragma once

#include "Field.hpp"
#include "NekFactory.hpp"
#include "Operator.hpp"

#include <iostream>

template <typename TData>
using BwdTransBase = OperatorBase<TData, FieldState::Coeff, FieldState::Phys>;

template <typename TData, OpMethod TMethod> class OperatorBwdTrans;

template <typename TData>
class OperatorBwdTrans<TData, OpMethod::MatFree> : public BwdTransBase<TData>
{
public:
    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        std::cout << "Op bwd trans mat free\n";
    }

    static std::unique_ptr<BwdTransBase<TData>> create()
    {
        return std::make_unique<OperatorBwdTrans<TData, OpMethod::MatFree>>();
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
std::string OperatorBwdTrans<double, OpMethod::MatFree>::className =
    GetOpBwdTransFactory().RegisterCreatorFunction(
        "OperatorBwdTransDoubleMatFree",
        OperatorBwdTrans<double, OpMethod::MatFree>::create, "...");

template <typename TData>
class OperatorBwdTrans<TData, OpMethod::SumFac> : public BwdTransBase<TData>
{
public:
    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        std::cout << "Op bwd trans sum fac\n";
    }

    static std::unique_ptr<BwdTransBase<TData>> create()
    {
        return std::make_unique<OperatorBwdTrans<TData, OpMethod::SumFac>>();
    }

    static std::string className;
};

template <>
std::string OperatorBwdTrans<double, OpMethod::SumFac>::className =
    GetOpBwdTransFactory().RegisterCreatorFunction(
        "OperatorBwdTransDoubleSumFac",
        OperatorBwdTrans<double, OpMethod::SumFac>::create, "...");
