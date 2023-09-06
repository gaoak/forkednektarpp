#pragma once

#include "Operators/OperatorHelm.hpp"

#include <tuple>
#include <set>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

#include <Operators/OperatorLaplace.hpp>
#include <Operators/OperatorMass.hpp>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

using namespace Nektar::Operators;


namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorHelmImpl : public OperatorHelm<TData>
{
public:
    OperatorHelmImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelm<TData>(std::move(expansionList)),
        m_tmp(Field<TData, FieldState::Coeff>::create(GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        m_LaplaceOp = Laplace<TData>::create(this->m_expansionList);
        m_MassOp = Mass<TData>::create(this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
    {
        auto ncoeffs = in.GetStorage().size();
        
        m_MassOp->apply(in, m_tmp);
        
        auto *ptr = m_tmp.GetStorage().GetCPUPtr();
        for (size_t i = 0; i < ncoeffs; ++i)
        {
            *(ptr++) *= m_lambda;
        }
        
        m_LaplaceOp->apply(in, out);

        ptr = m_tmp.GetStorage().GetCPUPtr();
        auto *ptr2 = out.GetStorage().GetCPUPtr();

        for (size_t i = 0; i < ncoeffs; ++i)
        {
            *(ptr2) = -(*(ptr2) + *(ptr));
            ptr++;
            ptr2++;
        }
    }

    void setLambda(const TData &lambda)
    {
        m_lambda = lambda;
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorHelmImpl<TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    TData m_lambda;
    std::shared_ptr<OperatorLaplace<TData>> m_LaplaceOp;
    std::shared_ptr<OperatorMass<TData>> m_MassOp;
    Field<TData, FieldState::Coeff> m_tmp;

};

}