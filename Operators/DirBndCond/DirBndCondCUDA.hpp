#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/DirBndCond/DirBndCondCUDAKernels.cuh"
#include "Operators/OperatorDirBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDirBndCondImpl<TData, ImplCUDA> : public OperatorDirBndCond<TData>
{
public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto assmbMap = contfield->GetLocalToGlobalMap();
        m_bndExpSize  = contfield->GetBndCondExpansions().size();
        m_signChange  = assmbMap->GetSignChange();

        // Memory allocation
        if (m_signChange)
        {
            auto &sign = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
            cudaMalloc((void **)&m_sign, sizeof(TData) * sign.size());
            cudaMemcpy(m_sign, sign.get(), sizeof(TData) * sign.size(),
                       cudaMemcpyHostToDevice);
        }

        auto &map = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        cudaMalloc((void **)&m_map, sizeof(int) * map.size());
        cudaMemcpy(m_map, map.get(), sizeof(int) * map.size(),
                   cudaMemcpyHostToDevice);

        Array<OneD, int> offset(m_bndExpSize, 0);
        for (size_t i = 1; i < offset.size(); ++i)
        {
            offset[i] = offset[i - 1] +
                        contfield->GetBndCondExpansions()[i - 1]->GetNcoeffs();
        }
        cudaMalloc((void **)&m_offset, sizeof(int) * offset.size());
        cudaMemcpy(m_offset, offset.get(), sizeof(int) * offset.size(),
                   cudaMemcpyHostToDevice);

        Array<OneD, int> ncoeff(m_bndExpSize);
        size_t ntotcoeff = 0;
        for (size_t i = 0; i < ncoeff.size(); ++i)
        {
            ncoeff[i] = contfield->GetBndCondExpansions()[i]->GetNcoeffs();
            ntotcoeff += contfield->GetBndCondExpansions()[i]->GetNcoeffs();
        }
        cudaMalloc((void **)&m_ncoeff, sizeof(int) * ncoeff.size());
        cudaMemcpy(m_ncoeff, ncoeff.get(), sizeof(int) * ncoeff.size(),
                   cudaMemcpyHostToDevice);

        Array<OneD, BoundaryConditionType> bctype(m_bndExpSize);
        for (size_t i = 0; i < bctype.size(); ++i)
        {
            bctype[i] =
                contfield->GetBndConditions()[i]->GetBoundaryConditionType();
        }
        cudaMalloc((void **)&m_bctype,
                   sizeof(BoundaryConditionType) * bctype.size());
        cudaMemcpy(m_bctype, bctype.get(),
                   sizeof(BoundaryConditionType) * bctype.size(),
                   cudaMemcpyHostToDevice);

        Array<OneD, TData> coeff(ntotcoeff);
        for (size_t i = 0; i < offset.size(); ++i)
        {
            for (size_t j = 0; j < ncoeff[i]; ++j)
            {
                coeff[offset[i] + j] =
                    contfield->GetBndCondExpansions()[i]->GetCoeffs()[j];
            }
        }
        cudaMalloc((void **)&m_coeff, sizeof(TData) * coeff.size());
        cudaMemcpy(m_coeff, coeff.get(), sizeof(TData) * coeff.size(),
                   cudaMemcpyHostToDevice);

        m_localDirSize = assmbMap->GetCopyLocalDirDofs().size();
        Array<OneD, int> locid0(m_localDirSize);
        Array<OneD, int> locid1(m_localDirSize);
        Array<OneD, TData> locsign(m_localDirSize);
        size_t cnt = 0;
        for (auto &it : assmbMap->GetCopyLocalDirDofs())
        {
            locid0[cnt]  = std::get<0>(it);
            locid1[cnt]  = std::get<1>(it);
            locsign[cnt] = std::get<2>(it);
            cnt++;
        }
        cudaMalloc((void **)&m_locid0, sizeof(int) * locid0.size());
        cudaMemcpy(m_locid0, locid0.get(), sizeof(int) * locid0.size(),
                   cudaMemcpyHostToDevice);
        cudaMalloc((void **)&m_locid1, sizeof(int) * locid1.size());
        cudaMemcpy(m_locid1, locid1.get(), sizeof(int) * locid1.size(),
                   cudaMemcpyHostToDevice);
        cudaMalloc((void **)&m_locsign, sizeof(TData) * locsign.size());
        cudaMemcpy(m_locsign, locsign.get(), sizeof(TData) * locsign.size(),
                   cudaMemcpyHostToDevice);
    }

    ~OperatorDirBndCondImpl(void)
    {
        if (m_signChange)
        {
            cudaFree(m_sign);
        }

        cudaFree(m_map);
        cudaFree(m_offset);
        cudaFree(m_ncoeff);
        cudaFree(m_bctype);
        cudaFree(m_coeff);
        cudaFree(m_locid0);
        cudaFree(m_locid1);
        cudaFree(m_locsign);
    }

    void apply(Field<TData, FieldState::Coeff> &out) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Deterime CUDA grid parameters.
        m_gridSize = m_bndExpSize / m_blockSize;
        m_gridSize += (m_bndExpSize % m_blockSize == 0) ? 0 : 1;

        if (m_signChange)
        {
            DirBndCondKernel<TData><<<m_gridSize, m_blockSize>>>(
                m_bndExpSize, m_offset, m_bctype, m_ncoeff, m_sign, m_map,
                m_coeff, outptr);
        }
        else
        {
            DirBndCondKernel<TData>
                <<<m_gridSize, m_blockSize>>>(m_bndExpSize, m_offset, m_bctype,
                                              m_ncoeff, m_map, m_coeff, outptr);
        }

        // communicate local Dirichlet coeffs that are just
        // touching a dirichlet boundary on another partition
        // auto &ParallelDirBndSign = locToGloMap->GetParallelDirBndSign();

        // for (auto &it : ParallelDirBndSign)
        //{
        //     outarr[it] *= -1;
        // }

        // Array<OneD, NekDouble> arr(nloc, outarr.data());
        // locToGloMap->UniversalAbsMaxBnd(arr);
        // std::copy(arr.get(), arr.get() + nloc, outarr.data());

        // for (auto &it : ParallelDirBndSign)
        //{
        //     outarr[it] *= -1;
        // }

        // Deterime CUDA grid parameters.
        m_gridSize = m_localDirSize / m_blockSize;
        m_gridSize += (m_localDirSize % m_blockSize == 0) ? 0 : 1;
        LocalDirBndCondKernel<TData><<<m_gridSize, m_blockSize>>>(
            m_localDirSize, m_locid0, m_locid1, m_locsign, outptr);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDirBndCondImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    bool m_signChange;
    size_t m_bndExpSize;
    size_t m_localDirSize;
    BoundaryConditionType *m_bctype;
    TData *m_coeff;
    TData *m_sign;
    int *m_map;
    int *m_ncoeff;
    int *m_offset;
    int *m_locid0;
    int *m_locid1;
    TData *m_locsign;
    size_t m_gridSize;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
