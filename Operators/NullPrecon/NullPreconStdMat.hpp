#pragma once

#include "Field.hpp"

#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorNullPrecon.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorNullPreconImpl<TData, ImplStdMat>
    : public OperatorNullPrecon<TData>
{
public:
    OperatorNullPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNullPrecon<TData>(expansionList)
    {
        m_assmbScatr = AssmbScatr<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_assmbScatr->apply(in, out, true);
    }

    void configure(
        [[maybe_unused]] const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNullPreconImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatr;
};

} // namespace Nektar::Operators::detail
