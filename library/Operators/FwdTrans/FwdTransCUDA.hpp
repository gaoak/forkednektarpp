#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/CUDAMathKernels.cuh"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorFwdTrans.hpp"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorMass.hpp"
#include "Operators/OperatorPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"

using vec_t = tinysimd::simd<double>;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorFwdTransImpl<TData, ImplCUDA> : public OperatorFwdTrans<TData>
{
public:
    OperatorFwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorFwdTrans<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_tmp(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        m_MassOp  = Mass<TData>::create(this->m_expansionList, "CUDA");
        m_DirBCOp = DirBndCond<TData>::create(this->m_expansionList, "CUDA");
        // m_RobBCOp = RobBndCond<TData>::create(this->m_expansionList, "CUDA");
        m_IProdOp =
            IProductWRTBase<TData>::create(this->m_expansionList, "CUDA");
        m_CGOp = ConjGrad<TData>::create(this->m_expansionList, "CUDA");
        m_CGOp->setLHS(m_MassOp);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        size_t nloc  = out.template GetStorage<MemoryRegionCUDA>().size();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *rhsptr =
            m_rhs.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *tmpptr =
            m_tmp.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Deterime CUDA grid size.
        m_gridSize = GetCUDAGridSize(nloc, m_blockSize);

        // IProductWRT of RHS
        m_IProdOp->apply(in, m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(out);
        m_MassOp->apply(out, m_tmp);
        subKernel<<<m_gridSize, m_blockSize>>>(nloc, rhsptr, tmpptr, rhsptr);

        // Handle Robin BCs
        // m_RobBCOp->apply(out, m_rhs, true);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, m_tmp);

        // Add Dirichlet BCs
        addKernel<<<m_gridSize, m_blockSize>>>(nloc, outptr, tmpptr, outptr);
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        precon->configure(m_MassOp);

        m_CGOp->setPrecon(precon);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorFwdTransImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_RobBCOp;
    std::shared_ptr<OperatorMass<TData>> m_MassOp;
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_tmp;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
