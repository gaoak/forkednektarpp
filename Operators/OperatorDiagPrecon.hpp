#pragma once

#include <vector>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

#include "Field.hpp"
#include "Operator.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

template <typename TData>
class OperatorDiagPrecon : public OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    OperatorDiagPrecon(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>(std::move(expansionList))
    {
    }

    virtual void configure(const std::shared_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op) = 0;
};

// Descriptor / traits class for DiagPrecon to be used by Operator create function
template <typename TData>
struct DiagPrecon
{
    using class_name = OperatorDiagPrecon<TData>;
    static const std::string key;
    static const std::string default_impl;

    DiagPrecon() = delete;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<DiagPrecon<TData>>(expansionList, pKey);
    }
};

namespace detail
{
    // declare class for implementation of DiagPrecon operator
    template <typename TData> 
    class OperatorDiagPreconImpl;
}

}