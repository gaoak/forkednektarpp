#include "MemoryRegionCUDA.hpp"
#include "Operators/OperatorBwdTrans.hpp"

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplCUDA> : public OperatorBwdTrans<TData>
{
public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        TData *x = in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        TData *y = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        std::cout << "Op bwd trans CUDA\n";
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        MultiRegions::ExpListSharedPtr expansionList)
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplCUDA>>(
            expansionList);
    }

    static std::string className;
};

} // namespace Nektar::Operators::detail
