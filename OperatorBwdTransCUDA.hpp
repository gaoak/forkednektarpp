// Specialisations for CUDA-based BwdTrans implementations

#include "Operator.hpp"

// BwdTrans mat-free operator
template<typename TData>
class Operator<TData, OpBwdTrans, MethodMatFree, DeviceCUDA> 
    : public OperatorBase<TData>
{
public:
    using ClassType = Operator<TData, OpBwdTrans, MethodMatFree, DeviceCUDA>;

    static std::unique_ptr<ClassType> create()
    {
        return std::unique_ptr<ClassType>(new ClassType());
    }

    void apply(TData in, TData out)
    {
        std::cout << "Perform BwdTrans op with CUDA" << std::endl;
    }
};