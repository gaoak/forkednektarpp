#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/CUDAMathKernels.cuh"
#include "Operators/OperatorConjGrad.hpp"
#include "Operators/OperatorDirBndCond.hpp"
#include "Operators/OperatorHelmSolve.hpp"
#include "Operators/OperatorHelmholtz.hpp"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorIProductWRTBase.hpp"
#include "Operators/OperatorLinear.hpp"
#include "Operators/OperatorNeuBndCond.hpp"
#include "Operators/OperatorPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"

using vec_t = tinysimd::simd<double>;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorHelmSolveImpl<TData, ImplCUDA> : public OperatorHelmSolve<TData>
{
public:
    OperatorHelmSolveImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorHelmSolve<TData>(expansionList),
          m_rhs(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList))),
          m_tmp(Field<TData, FieldState::Coeff>::template create<
                MemoryRegionCUDA>(
              GetBlockAttributes(FieldState::Coeff, expansionList)))
    {
        m_IProdOp =
            IProductWRTBase<TData>::create(this->m_expansionList, "CUDA");
        m_DirBCOp = DirBndCond<TData>::create(this->m_expansionList, "CUDA");
        m_NeuBCOp = NeuBndCond<TData>::create(this->m_expansionList, "CUDA");
        // m_RobBCOp = RobBndCond<TData>::create(this->m_expansionList, "CUDA");
        m_HelmOp = Helmholtz<TData>::create(this->m_expansionList, "CUDA");
        m_CGOp   = ConjGrad<TData>::create(this->m_expansionList, "CUDA");
        m_CGOp->setLHS(m_HelmOp);
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
        negKernel<<<m_gridSize, m_blockSize>>>(nloc, rhsptr, rhsptr);

        // Handle Neumann BCs on RHS
        m_NeuBCOp->apply(m_rhs);

        // Handle Dirichlet BCs
        m_DirBCOp->apply(out);
        m_HelmOp->apply(out, m_tmp);
        subKernel<<<m_gridSize, m_blockSize>>>(nloc, rhsptr, tmpptr, rhsptr);

        // Handle Robin BCs
        // m_RobBCOp->apply(out, m_rhs, true);

        // Solve for u_hat using Conjugate Gradient
        m_CGOp->apply(m_rhs, m_tmp);

        // Add Dirichlet BCs
        addKernel<<<m_gridSize, m_blockSize>>>(nloc, outptr, tmpptr, outptr);
    }

    void setLambda(const TData &lambda)
    {
        m_HelmOp->setLambda(lambda);
    }

    void setPrecon(
        const std::shared_ptr<OperatorPrecon<TData>> &precon) override
    {
        precon->configure(m_HelmOp);

        m_CGOp->setPrecon(precon);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorHelmSolveImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorIProductWRTBase<TData>> m_IProdOp;
    std::shared_ptr<OperatorDirBndCond<TData>> m_DirBCOp;
    std::shared_ptr<OperatorNeuBndCond<TData>> m_NeuBCOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_RobBCOp;
    std::shared_ptr<OperatorHelmholtz<TData>> m_HelmOp;
    std::shared_ptr<OperatorConjGrad<TData>> m_CGOp;
    Field<TData, FieldState::Coeff> m_rhs;
    Field<TData, FieldState::Coeff> m_tmp;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
