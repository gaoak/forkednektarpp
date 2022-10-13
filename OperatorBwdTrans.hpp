// Specialisations for CPU-based BwdTrans implementations

#include "Operator.hpp"

// BwdTrans local matrix operator
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

// BwdTrans mat-free operator
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

        // Quad
        // size_t nElmt = 1000;
        // size_t nVW   = 4;
        // size_t nGrpsQ = nElmt / nVW;
        // auto o = OpKernel<TData, OpBwdTrans, MethodLocMat, DeviceCPU, NQ0, NQ1, NP0, NP1>
        // for (int i = 0; i < nGrpsQ; ++i) {
            
        // }
    }
};
