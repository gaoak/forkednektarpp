#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/NeuBndCond/NeuBndCondCUDAKernels.cuh"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorNeuBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/Conditions.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
template <typename TData>
class OperatorNeuBndCondImpl<TData, ImplCUDA> : public OperatorNeuBndCond<TData>
{
public:
    OperatorNeuBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorNeuBndCond<TData>(expansionList)
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

        // Deterime CUDA grid parameters.
        m_gridSize = GetCUDAGridSize(m_bndExpSize, m_blockSize);
    }

    ~OperatorNeuBndCondImpl(void)
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
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto *inoutptr =
            inout.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        if (m_signChange)
        {
            NeuBndCondKernel<TData><<<m_gridSize, m_blockSize>>>(
                m_bndExpSize, m_offset, m_bctype, m_ncoeff, m_sign, m_map,
                m_coeff, inoutptr);
        }
        else
        {
            NeuBndCondKernel<TData><<<m_gridSize, m_blockSize>>>(
                m_bndExpSize, m_offset, m_bctype, m_ncoeff, m_map, m_coeff,
                inoutptr);
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorNeuBndCondImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    bool m_signChange;
    size_t m_bndExpSize;
    BoundaryConditionType *m_bctype;
    TData *m_coeff;
    TData *m_sign;
    int *m_map;
    int *m_ncoeff;
    int *m_offset;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
