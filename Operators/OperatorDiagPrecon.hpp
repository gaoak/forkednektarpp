#pragma once

#include "Field.hpp"
#include "OperatorPrecon.hpp"

namespace Nektar::Operators
{

// DiagPrecon base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorDiagPrecon : public OperatorPrecon<TData>
{
public:
    virtual ~OperatorDiagPrecon() = default;

    OperatorDiagPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPrecon<TData>(std::move(expansionList))
    {
    }
};

// Descriptor / traits class for DiagPrecon to be used by Operator create
// function
template <typename TData = default_fp_type> struct DiagPrecon
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
// Template for implementation of DiagPrecon operator
template <typename TData, typename Op> class OperatorDiagPreconImpl;
} // namespace detail

} // namespace Nektar::Operators
