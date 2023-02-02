#include "Operators/OperatorBwdTrans.hpp"

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplCUDA> : public OperatorBwdTrans<TData>
{
public:
    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        std::cout << "Op bwd trans CUDA\n";
    }

    static std::unique_ptr<Operator<TData>> instantiate()
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplMatFree>>();
    }

    static std::string className;
};

}