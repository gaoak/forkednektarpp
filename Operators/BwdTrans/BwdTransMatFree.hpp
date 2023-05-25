#include "Operators/OperatorBwdTrans.hpp"

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplMatFree> : public OperatorBwdTrans<TData>
{
public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        TData *x       = in.GetStorage().GetCPUPtr();
        TData *y       = out.GetStorage().GetCPUPtr();
        const size_t n = in.GetStorage().size();
        for (size_t i = 0; i < n; ++i)
        {
            y[i] = 2.0 * x[i];
        }
        std::cout << "Op bwd trans mat free\n";
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplMatFree>>(
            expansionList);
    }

    static std::string className;
};

} // namespace Nektar::Operators::detail
