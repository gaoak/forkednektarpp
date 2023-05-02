#pragma once

#include <string>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>
//#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators
{

// Use typenames to define available implementations to
// allow extension by users without modifying library
using default_fp_type = double;

// Core implementation types
struct ImplStdMat;
struct ImplSumFac;
struct ImplMatFree;
struct ImplCUDA;

// Forward-declare the Operator base class so we can define the factory
template <typename TData> class Operator;

// Typename alias for the factory
template <typename TData>
using OperatorFactory =
    Nektar::LibUtilities::NekFactory<std::string, Operator<TData>,
                                     const MultiRegions::ExpListSharedPtr&>;

// Operator factory singleton
template <typename TData> OperatorFactory<TData> &GetOperatorFactory();

template <typename TData> class Operator
{
public:
    virtual ~Operator() = default;

    Operator(const MultiRegions::ExpListSharedPtr& expansionList)
        : m_expansionList(std::move(expansionList))
    {
    }

    template <typename TDescriptor>
    static std::shared_ptr<typename TDescriptor::class_name> create(
        const MultiRegions::ExpListSharedPtr& expansionList,
        std::string pKey = "")
    {
        std::string key = TDescriptor::key;
        if (pKey.empty())
        {
            key += TDescriptor::default_impl;
        }
        else
        {
            key += pKey;
        }

        return std::static_pointer_cast<typename TDescriptor::class_name>(
            GetOperatorFactory<TData>().CreateInstance(
                key, std::move(expansionList)));
    }

protected:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
