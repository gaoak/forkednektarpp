#pragma once

#include "Field.hpp"
#include "Operators/OperatorDiagPrecon.hpp"

#include <set>
#include <tuple>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDiagPreconImpl<TData, ImplStdMat>
    : public OperatorDiagPrecon<TData>
{
public:
    OperatorDiagPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDiagPrecon<TData>(std::move(expansionList)),
          m_diag(Field<TData, FieldState::Coeff>::create(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        auto *diag_ptr = this->m_diag.GetStorage().GetCPUPtr();
        auto *in_ptr   = in.GetStorage().GetCPUPtr();
        auto *out_ptr  = out.GetStorage().GetCPUPtr();
        size_t ncoeffs = in.GetStorage().size();

        for (size_t i = 0; i < ncoeffs; ++i)
        {
            *(out_ptr++) = *(in_ptr++) / *(diag_ptr++);
        }
    }

    void configure(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
        // create unit vector field to extract diagonal
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::create(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        // create action field to receive column action from unit vector
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::create(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        size_t ncoeffs = unit_vec.GetStorage().size();

        auto *uvec_ptr = unit_vec.GetStorage().GetCPUPtr();
        auto *actn_ptr = action.GetStorage().GetCPUPtr();
        auto *diag_ptr = this->m_diag.GetStorage().GetCPUPtr();

        for (size_t i = 0; i < ncoeffs; ++i)
        {
            // set ith term in unit vector to be 1 and (i-1)th term to be 0 if i
            // > 0
            if (i > 0)
            {
                *uvec_ptr = 0.;
                uvec_ptr++;
            }
            *uvec_ptr = 1.;

            // apply operator to unit vector and store in action field
            op->apply(unit_vec, action);

            // copy ith row term from the action field to get ith diagonal
            *(diag_ptr) = *(actn_ptr);

            // advance the diagonal and action ptrs
            diag_ptr++;
            actn_ptr++;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDiagPreconImpl<TData, ImplStdMat>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    // diagonal of conditioner
    Field<TData, FieldState::Coeff> m_diag;
};

} // namespace Nektar::Operators::detail
