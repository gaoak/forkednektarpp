#pragma once

#include "Operators/OperatorBwdTrans.hpp"

namespace Nektar::Operators::detail
{

// sum-factorisation implementation
template <typename TData>
class OperatorBwdTransImpl<TData, ImplSumFac> : public OperatorBwdTrans<TData>
{
public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
    }

    void apply([[maybe_unused]] Field<TData, FieldState::Coeff> &in,
               [[maybe_unused]] Field<TData, FieldState::Phys> &out) override
    {
        std::cout << "Op bwd trans sum fac\n";
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorBwdTransImpl<TData, ImplSumFac>>(
            expansionList);
    }

    static std::string className;
};

} // namespace Nektar::Operators::detail
