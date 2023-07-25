#pragma once

#include <string>

#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <MultiRegions/ExpList.h>

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
                                     const MultiRegions::ExpListSharedPtr &>;

// Operator factory singleton
template <typename TData> OperatorFactory<TData> &GetOperatorFactory();

template <typename TData> class Operator
{
public:
    virtual ~Operator() = default;

    Operator(const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    template <typename TDescriptor>
    static std::shared_ptr<typename TDescriptor::class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
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
            GetOperatorFactory<TData>().CreateInstance(key, expansionList));
    }

protected:
    size_t GetGeometricFactorSize(void)
    {
        size_t gfSize    = 0;
        size_t nTotElmts = this->m_expansionList->GetNumElmts();

        // Calculate the jacobian array size
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            // Determine shape and type of the element
            auto const expPtr = this->m_expansionList->GetExp(e);
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                gfSize += expPtr->GetTotPoints();
            }
            else
            {
                gfSize++;
            }
        }

        return gfSize;
    }

    Array<OneD, TData> SetJacobian(size_t jacSize)
    {
        // Allocate memory for the jacobian
        Array<OneD, TData> jac(jacSize, 0.0);

        // Initialise jacobian.
        size_t index     = 0;
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto expPtr = this->m_expansionList->GetExp(e);
            auto &auxJac =
                expPtr->GetMetricInfo()->GetJac(expPtr->GetPointsKeys());
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                size_t nqe = expPtr->GetTotPoints();
                for (size_t i = 0; i < nqe; ++i)
                {
                    jac[index++] = auxJac[i];
                }
            }
            else
            {
                jac[index++] = auxJac[0];
            }
        }
        return jac;
    }

    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
