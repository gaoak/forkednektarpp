#pragma once

#include "Operators/OperatorIdentity.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
class OperatorIdentityImpl<TData, TFieldState, ImplStdMat>
    : public OperatorIdentity<TData, TFieldState>
{
public:
    OperatorIdentityImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIdentity<TData, TFieldState>(expansionList)
    {
    }

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        size_t N  = in.GetStorage().size();
        auto pIn  = in.GetStorage().GetCPUPtr();
        auto pOut = out.GetStorage().GetCPUPtr();

        for (size_t i = 0; i < N; ++i)
        {
            *(pOut++) = *(pIn++);
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIdentityImpl<TData, TFieldState, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

} // namespace Nektar::Operators::detail
