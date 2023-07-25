#include "Operators/OperatorConjGrad.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
class OperatorConjGradImpl : public OperatorConjGrad<TData, TFieldState>
{
public:
    OperatorConjGradImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorConjGrad<TData, TFieldState>(std::move(expansionList))
    {
    }

    void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out);

    // takes array in local dofs, assembles to global dofs, then scatters back to local dofs
    void assembleScatter(
        const size_t &N,
        Field<TData, TFieldState> &in,
        Field<TData, TFieldState> &out, 
        const bool &ZeroDir
    );

    // instantiation function for CreatorFunction in Operator Factory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorConjGradImpl<TData, TFieldState>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;
};

}
