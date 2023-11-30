#pragma once

#include "Field.hpp"
#include "OperatorPrecon.hpp"

namespace Nektar::Operators
{

// NullPrecon base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorNullPrecon : public OperatorPrecon<TData>
{
public:
    virtual ~OperatorNullPrecon() = default;

    OperatorNullPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPrecon<TData>(expansionList)
    {
    }
};

// Descriptor / traits class for NullPrecon to be used by Operator create
// function
template <typename TData = default_fp_type> struct NullPrecon
{
    using class_name = OperatorNullPrecon<TData>;
    static const std::string key;
    static const std::string default_impl;

    NullPrecon() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<NullPrecon<TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of NullPrecon operator
template <typename TData, typename Op> class OperatorNullPreconImpl;
} // namespace detail

} // namespace Nektar::Operators
