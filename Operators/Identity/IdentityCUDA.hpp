#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/OperatorIdentity.hpp"

namespace Nektar::Operators::detail
{

// Identity matrix implementation
template <typename TData, FieldState TFieldState>
class OperatorIdentityImpl<TData, TFieldState, ImplCUDA>
    : public OperatorIdentity<TData, TFieldState>
{
public:
    OperatorIdentityImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIdentity<TData, TFieldState>(expansionList)
    {
    }

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        size_t N = in.template GetStorage<MemoryRegionCUDA>().size();
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        cudaMemcpy(outptr, inptr, sizeof(TData) * N, cudaMemcpyDeviceToDevice);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIdentityImpl<TData, TFieldState, ImplCUDA>>(expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    size_t m_gridSize;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
