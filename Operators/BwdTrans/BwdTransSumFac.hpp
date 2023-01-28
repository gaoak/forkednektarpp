#include "Operators/OperatorBwdTrans.hpp"

namespace Nektar::Operators
{
    
// sum-factorisation implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplSumFac> : public OperatorBwdTrans<TData>
{
public:
    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        std::cout << "Op bwd trans sum fac\n";
    }

    static std::unique_ptr<Operator<TData>> instantiate()
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplSumFac>>();
    }

    static std::string className;
};

}