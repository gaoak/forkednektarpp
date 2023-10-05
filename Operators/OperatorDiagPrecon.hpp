#pragma once

#include "Field.hpp"
#include "OperatorPrecon.hpp"

namespace Nektar::Operators
{

template <typename TData>
class OperatorDiagPrecon : public OperatorPrecon<TData>
{
public:
    OperatorDiagPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPrecon<TData>(std::move(expansionList))
    {
    }
};

// Descriptor / traits class for DiagPrecon to be used by Operator create
// function
template <typename TData> struct DiagPrecon
{
    using class_name = OperatorDiagPrecon<TData>;
    static const std::string key;
    static const std::string default_impl;

    DiagPrecon() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<DiagPrecon<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// declare class for implementation of DiagPrecon operator
template <typename TData, typename Op> class OperatorDiagPreconImpl;
} // namespace detail

} // namespace Nektar::Operators
