#pragma once

#include "Operators/BwdTrans/BwdTransCUDA.hpp"
#include "Operators/IProductWRTBase/IProductWRTBaseCUDA.hpp"
#include "Operators/OperatorMass.hpp"

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorMassImpl<TData, ImplCUDA> : public OperatorMass<TData>
{
public:
    OperatorMassImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMass<TData>(expansionList),
          m_field(
              Field<TData, FieldState::Phys>::template create<MemoryRegionCUDA>(
                  GetBlockAttributes(FieldState::Phys, expansionList)))
    {
        auto nCoord = this->m_expansionList->GetCoordim(0);

        m_BwdTransOp = BwdTrans<>::create(this->m_expansionList, "CUDA");
        m_IProductWRTBaseOp =
            IProductWRTBase<>::create(this->m_expansionList, "CUDA");
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Step 1: BwdTrans
        m_BwdTransOp->apply(in, m_field);

        // Step 2: Inner product for mass matrix operation
        m_IProductWRTBaseOp->apply(m_field, out);
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMassImpl<TData, ImplCUDA>>(
            expansionList);
    }

    static std::string className;

private:
    std::shared_ptr<OperatorBwdTrans<TData>> m_BwdTransOp;
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProductWRTBaseOp;
    Field<TData, FieldState::Phys> m_field;
    size_t m_blockSize = 32;
    size_t m_gridSize;
};

} // namespace Nektar::Operators::detail
