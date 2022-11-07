#pragma once

// Specialisations for CPU-based IProduct implementations

#include "Operator.hpp"

// IProduct local matrix operator
template<typename TData>
class Operator<TData, OpIProduct, MethodLocMat, BackendCPU>
    : public OperatorBase<Operator<TData, OpIProduct, MethodLocMat, BackendCPU>, TData, StatePhys, StateCoeff>
{
public:
    using ClassType = Operator<TData, OpIProduct, MethodLocMat, BackendCPU>;

/*
    static std::unique_ptr<ClassType> create()
    {
        return std::unique_ptr<ClassType>(new ClassType());
    }
*/

    void apply_impl(Field<TData, StatePhys> &in, Field<TData, StateCoeff> &out)
    {
        std::cout << "Perform IProduct op with LocMat" << std::endl;
    }
};

// IProduct mat-free operator
template<typename TData>
class Operator<TData, OpIProduct, MethodMatFree, BackendCPU>
    : public OperatorBase<Operator<TData, OpIProduct, MethodMatFree, BackendCPU>, TData, StatePhys, StateCoeff>
{
public:
    using ClassType = Operator<TData, OpIProduct, MethodMatFree, BackendCPU>;

/*
    static std::unique_ptr<ClassType> create()
    {
        return std::unique_ptr<ClassType>(new ClassType());
    }
*/

    void apply_impl(Field<TData, StatePhys> &in, Field<TData, StateCoeff> &out)
    {
        std::cout << "Perform IProduct op with MatFree" << std::endl;

        // Quad
        // size_t nElmt = 1000;
        // size_t nVW   = 4;
        // size_t nGrpsQ = nElmt / nVW;
        // auto o = OpKernel<TData, OpIProduct, MethodLocMat, BackendCPU, NQ0, NQ1, NP0, NP1>
        // for (int i = 0; i < nGrpsQ; ++i) {
            
        // }
    }
};
