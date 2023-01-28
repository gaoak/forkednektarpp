#pragma once

#include <string>

#include "NekFactory.hpp"

namespace Nektar::Operators
{

// Use typenames to define available implementations to
// allow extension by users without modifying library
using default_fp_type = double;

// Implementation types
struct ImplLocMat;
struct ImplSumFac;
struct ImplMatFree;

// Operators

template< typename TData> class Operator;

template< typename TData>
using OperatorFactory =
    Nektar::LibUtilities::NekFactory<std::string, Operator<TData>>;

template< typename TData>
OperatorFactory<TData> &GetOperatorFactory();

template <typename TData>
class Operator
{
public:
    template< typename TDescriptor>
    static std::unique_ptr<typename TDescriptor::class_name> create(std::string pKey = "")
    {
        std::string key = TDescriptor::key;
        if (pKey.empty()) {
            key += TDescriptor::default_impl;
        }
        else {
            key += pKey;
        }
        auto x = GetOperatorFactory<TData>().CreateInstance(key);
        return std::unique_ptr<typename TDescriptor::class_name>(static_cast<typename TDescriptor::class_name*>(x.release()));
    }
};

}