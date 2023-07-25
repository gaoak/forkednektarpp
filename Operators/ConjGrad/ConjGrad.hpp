#include "Operators/OperatorConjGrad.hpp"

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorConjGradImpl : public OperatorConjGrad<TData>
{
public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData>(std::move(expansionList))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out);

    // takes array in local dofs, assembles to global dofs, then scatters back to local dofs
    void assembleScatter(
        const size_t &N,
        Field<TData, FieldState::Coeff> &in,
        Field<TData, FieldState::Coeff> &out, 
        const bool &ZeroDir
    );

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorConjGradImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}
