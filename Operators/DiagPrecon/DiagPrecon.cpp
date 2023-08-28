#include "Field.hpp"
#include "DiagPrecon.hpp"

#include <tuple>
#include <set>

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
void OperatorDiagPreconImpl<TData>::apply(Field<TData, FieldState::Coeff> &in, Field<TData, FieldState::Coeff> &out)
{
    // assert that the in/out sizes conform with the expansion list blocks
    
    auto *diag_ptr = this->m_diag.GetStorage().GetCPUPtr();
    auto *in_ptr = in.GetStorage().GetCPUPtr();
    auto *out_ptr = out.GetStorage().GetCPUPtr();
    size_t ncoeffs = in.GetStorage().size();

    for (size_t i = 0; i < ncoeffs; ++i)
    {
        *(out_ptr++) = *(in_ptr++) / *(diag_ptr++);
    }
}

template <typename TData>
void OperatorDiagPreconImpl<TData>::configure(const std::unique_ptr<OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
{
    // create unit vector field to extract diagonal
    Field<TData, FieldState::Coeff> unit_vec = Field<TData, FieldState::Coeff>::create(
        GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

    // create action field to receive column action from unit vector
    Field<TData, FieldState::Coeff> action = Field<TData, FieldState::Coeff>::create(
        GetBlockAttributes(FieldState::Coeff, this->m_expansionList));
    
    size_t ncoeffs = unit_vec.GetStorage().size();

    auto *uvec_ptr = unit_vec.GetStorage().GetCPUPtr();
    auto *actn_ptr = action.GetStorage().GetCPUPtr();
    auto *diag_ptr = this->m_diag.GetStorage().GetCPUPtr();

    for (size_t i = 0; i < ncoeffs; ++i)
    {
        // set ith term in unit vector to be 1 and (i-1)th term to be 0 if i > 0
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

// ****************************************************************************************************************

// Register implementation with Operator Factory
template <>
std::string OperatorDiagPreconImpl<double>::className =
    GetOperatorFactory<double>().RegisterCreatorFunction(
        "DiagPrecon",
        OperatorDiagPreconImpl<double>::instantiate, 
        ""
    );

}