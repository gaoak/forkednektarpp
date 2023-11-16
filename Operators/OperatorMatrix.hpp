#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData, FieldState TFieldState>
class OperatorMatrix : public OperatorLinear<TData, TFieldState, TFieldState>
{

public:
    virtual ~OperatorMatrix() = default;

    OperatorMatrix(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, TFieldState, TFieldState>(expansionList)
    {
    }

    virtual size_t size()                     = 0;
    virtual void fill(std::vector<TData> src) = 0;
    virtual std::string toString()            = 0;
};

// Descriptor / traits class for Matrix to be used by Operator create function
template <FieldState TFieldState, typename TData> struct Matrix
{
    using class_name = OperatorMatrix<TData, TFieldState>;
    static const std::string key;
    static const std::string default_impl;

    Matrix() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Matrix<TFieldState, TData>>(
            expansionList, pKey);
    }
};

namespace detail
{
// Template for implementation of Matrix operator
template <typename TData, FieldState TFieldState, typename Op>
class OperatorMatrixImpl;
} // namespace detail

} // namespace Nektar::Operators
