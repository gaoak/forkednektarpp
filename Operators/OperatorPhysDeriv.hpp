#pragma once

#include "Field.hpp"
#include "OperatorLinear.hpp"

namespace Nektar::Operators
{

// PhysDeriv base class
// Defines the apply operator to enforce apply parameter types
template <typename TData>
class OperatorPhysDeriv
    : public OperatorLinear<TData, FieldState::Phys, FieldState::Phys>
{
public:
    virtual ~OperatorPhysDeriv() = default;

    OperatorPhysDeriv(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorLinear<TData, FieldState::Phys, FieldState::Phys>(
              expansionList)
    {
    }

    virtual void apply(Field<TData, FieldState::Phys> &in,
                       Field<TData, FieldState::Phys> &out) = 0;

    virtual void operator()(Field<TData, FieldState::Phys> &in,
                            Field<TData, FieldState::Phys> &out)
    {
        apply(in, out);
    }
};

// Descriptor / traits class for PhysDeriv
template <typename TData = default_fp_type> struct PhysDeriv
{
    using class_name = OperatorPhysDeriv<TData>;
    using FieldIn    = Field<TData, FieldState::Phys>;
    using FieldOut   = Field<TData, FieldState::Phys>;
    static const std::string key;
    static const std::string default_impl;

    static std::shared_ptr<class_name> create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        std::string pKey = "")
    {
        return Operator<TData>::template create<PhysDeriv>(expansionList, pKey);
    }
};

namespace detail
{
// Template for PhysDeriv implementations
template <typename TData, typename Op> class OperatorPhysDerivImpl;
} // namespace detail

} // namespace Nektar::Operators
