#pragma once

#include "Field.hpp"
#include "Operators/AssmbScatr/AssmbScatrCUDA.hpp"
#include "Operators/CUDAMathKernels.cuh"
#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorRobBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDiagPreconImpl<TData, ImplCUDA> : public OperatorDiagPrecon<TData>
{
public:
    OperatorDiagPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDiagPrecon<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        GlobalSysSolnType solvertype = m_assmbMap->GetGlobalSysSolnType();
        bool isFull = solvertype == eIterativeFull ? true : false;
        m_nGlobal   = (isFull) ? m_assmbMap->GetNumGlobalCoeffs()
                               : m_assmbMap->GetNumGlobalBndCoeffs();
        m_nLocal    = m_assmbMap->GetNumLocalCoeffs();
        m_nDir      = m_assmbMap->GetNumGlobalDirBndCoeffs();

        cudaMalloc((void **)&m_wk, sizeof(TData) * m_nGlobal);
        cudaMalloc((void **)&m_diag, sizeof(TData) * m_nGlobal);

        m_assmbScatr =
            std::static_pointer_cast<OperatorAssmbScatrImpl<TData, ImplCUDA>>(
                AssmbScatr<TData>::create(this->m_expansionList, "CUDA"));

        // Deterime CUDA grid size.
        m_gridSize = GetCUDAGridSize(m_nGlobal - m_nDir, m_blockSize);
    }

    ~OperatorDiagPreconImpl(void)
    {
        cudaFree(m_wk);
        cudaFree(m_diag);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_assmbScatr->Assemble(in, m_wk);

        vdivKernel<<<m_gridSize, m_blockSize>>>(
            m_nGlobal - m_nDir, m_wk + m_nDir, m_diag + m_nDir, m_wk + m_nDir);

        cudaMemset(m_wk, 0, sizeof(TData) * m_nDir);

        m_assmbScatr->GlobalToLocal(m_wk, out);
    }

    void configure(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
        // auto robBCOp = RobBndCond<TData>::create(this->m_expansionList,
        // "CUDA");

        Array<OneD, TData> diag(m_nLocal);

        // create unit vector field to extract diagonal
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::template create<MemoryRegionCUDA>(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        // create action field to receive column action from unit vector
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::template create<MemoryRegionCUDA>(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        auto *uvec_ptr =
            unit_vec.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *actn_ptr =
            action.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        TData init = 1.0;
        for (size_t i = 0; i < m_nLocal; ++i)
        {
            // set ith term in unit vector to be 1
            cudaMemcpy(uvec_ptr + i, &init, sizeof(TData),
                       cudaMemcpyHostToDevice);

            // apply operator to unit vector and store in action field
            op->apply(unit_vec, action);
            // robBCOp->apply(unit_vec, action);

            // copy ith row term from the action field to get ith diagonal
            cudaMemcpy(diag.get() + i, actn_ptr + i, sizeof(TData),
                       cudaMemcpyDeviceToHost);

            // reset ith term in unit vector to be 0
            cudaMemset(uvec_ptr + i, 0, sizeof(TData));
        }

        // Assembly
        Array<OneD, TData> tmp(m_nGlobal, 0.0);
        for (size_t i = 0; i < m_nLocal; ++i)
        {
            size_t gid1 = m_assmbMap->GetLocalToGlobalMap(i);
            tmp[gid1] += diag[i];
        }
        m_assmbMap->UniversalAssemble(tmp);

        cudaMemcpy(m_diag, tmp.get(), sizeof(TData) * m_nGlobal,
                   cudaMemcpyHostToDevice);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDiagPreconImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatrImpl<TData, ImplCUDA>> m_assmbScatr;
    AssemblyMapCGSharedPtr m_assmbMap;
    TData *m_diag;
    TData *m_wk;
    size_t m_nGlobal;
    size_t m_nLocal;
    size_t m_nDir;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
