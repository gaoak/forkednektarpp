#pragma once

#include "Operators/OperatorIdentity.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
class OperatorIdentityImpl : public OperatorIdentity<TData, TFieldState>
{
public:
    OperatorIdentityImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIdentity<TData, TFieldState>(std::move(expansionList))
    {
    }

    void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out)
    {
        size_t N = in.GetStorage().size();
        auto pIn = in.GetStorage().GetCPUPtr();
        auto pOut = out.GetStorage().GetCPUPtr();
        
        for (size_t i = 0; i < N; ++i)
            *(pOut++) = *(pIn++);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorIdentityImpl<TData, TFieldState>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}