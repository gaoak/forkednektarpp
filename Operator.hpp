#include <memory>
#include <iostream>

#include "Field.hpp"

typedef double NekDouble;

struct OpBwdTrans;
struct OpIProductWRTBase;
struct OpPhysDeriv;

struct MethodLocMat;
struct MethodSumFac;
struct MethodMatFree;
using DefaultMethod = MethodSumFac;

// Base class
template<typename TData>
class OperatorBase
{
    public:
        virtual void apply(TData in, TData out) = 0;
};

// Templated Operator
template<typename TData,
         typename TOp,
         typename TMethod = DefaultMethod,
         typename TDevice = DefaultDevice>
class Operator;


template<typename TData>
class Operator<TData, OpBwdTrans, MethodLocMat, DeviceCPU> 
    : public OperatorBase<TData>
{
public:
    using ClassType = Operator<TData, OpBwdTrans, MethodLocMat, DeviceCPU>;

    static std::unique_ptr<ClassType> create()
    {
        return std::unique_ptr<ClassType>(new ClassType());
    }

    void apply(TData in, TData out) override
    {
        std::cout << "Perform BwdTrans op with LocMat" << std::endl;
    }
};

template<typename TData>
class Operator<TData, OpBwdTrans, MethodMatFree, DeviceCPU> 
    : public OperatorBase<TData>
{
public:
    using ClassType = Operator<TData, OpBwdTrans, MethodMatFree, DeviceCPU>;

    static std::unique_ptr<ClassType> create()
    {
        return std::unique_ptr<ClassType>(new ClassType());
    }

    void apply(TData in, TData out)
    {
        std::cout << "Perform BwdTrans op with MatFree" << std::endl;
    }
};

