#pragma once

#include <vector>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// Matrix operator base class
template <typename TData, FieldState TFieldState>
class OperatorMatrix : public OperatorLinear<TData, TFieldState, TFieldState>
{

public:
    OperatorMatrix(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, TFieldState, TFieldState>(expansionList)
    {    
    }

    virtual size_t size() = 0;
    virtual void fill(const TData *src) = 0;
    virtual std::string toString() = 0;
};

// Descriptor / traits class for Matrix to be used by Operator create function
template <typename TData, FieldState TFieldState>
struct Matrix
{
    using class_name = OperatorMatrix<TData, TFieldState>;
    static const std::string key;
    static const std::string default_impl;

    Matrix() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<Matrix<TData, TFieldState>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of Matrix operator
    template <typename TData, FieldState TFieldState> 
    class OperatorMatrixImpl;
}

}