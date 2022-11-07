#pragma once

// Specialisations for CUDA-based BwdTrans implementations

#include "Operator.hpp"

// BwdTrans mat-free operator
template<typename TData>
class Operator<TData, OpBwdTrans, MethodLocMat, BackendCUDA>
    : public OperatorBase<Operator<TData, OpBwdTrans, MethodLocMat, BackendCUDA>, TData, StateCoeff, StatePhys>
{
public:
    using ClassType = Operator<TData, OpBwdTrans, MethodMatFree, BackendCUDA>;
/*
    static std::unique_ptr<ClassType> create()
    {
        return std::unique_ptr<ClassType>(new ClassType());
    }
*/

    void apply_impl(Field<TData, StateCoeff, BackendCUDA> &in, Field<TData, StatePhys, BackendCUDA> &out);
//    {
//        std::cout << "Perform BwdTrans op with CUDA" << std::endl;
//    }
};
