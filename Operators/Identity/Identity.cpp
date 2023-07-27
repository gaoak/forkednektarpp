#include "Identity.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
void OperatorIdentityImpl<TData, TFieldState>::apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out)
{
    size_t N = in.GetStorage().size();
    auto pIn = in.GetStorage().GetCPUPtr();
    auto pOut = out.GetStorage().GetCPUPtr();

    for (size_t i = 0; i < N; ++i)
        *(pOut++) = *(pIn++);
}

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorIdentityImpl<double, FieldState::Coeff>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityCoeff",
        OperatorIdentityImpl<double, FieldState::Coeff>::instantiate, 
        ""
    );

template <>
std::string OperatorIdentityImpl<double, FieldState::Phys>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "IdentityPhys",
        OperatorIdentityImpl<double, FieldState::Phys>::instantiate, 
        ""
    );
}