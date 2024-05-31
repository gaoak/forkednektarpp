///////////////////////////////////////////////////////////////////////////////
//
// File: NeuBndCondImplShared.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/OperatorNeuBndCond.hpp"

#include "Operators/NeuBndCond/NeuBndCondCUDASumFacKernels.cuh"
#include "Operators/NeuBndCond/NeuBndCondKokkosStdMatKernels.hpp"

#include "Operators/OperatorHelper.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
#if defined(NEKTAR_ENABLE_CUDA)
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
#endif
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorNeuBndCondImpl : public OperatorNeuBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

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

        // Sign
        if (m_signChange)
        {
            auto &sign = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                sign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        }

        // Map
        auto &map = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            map, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Offset
        m_offset = MemoryRegion<int>::template create<MemSpace>(m_bndExpSize);

        m_offset.initialize(0);

        int *offset = m_offset.template GetPtr<NektarSpaces::HostSpace>();

        for (size_t i = 1; i < m_bndExpSize; ++i)
        {
            offset[i] = offset[i - 1] +
                        contfield->GetBndCondExpansions()[i - 1]->GetNcoeffs();
        }

        // NCoeff
        m_ncoeff = MemoryRegion<int>::template create<MemSpace>(m_bndExpSize);

        int *ncoeff = m_ncoeff.template GetPtr<NektarSpaces::HostSpace>();

        size_t ntotcoeff = 0;
        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            ncoeff[i] = contfield->GetBndCondExpansions()[i]->GetNcoeffs();
            ntotcoeff += contfield->GetBndCondExpansions()[i]->GetNcoeffs();
        }

        // BC type
        m_bctype =
            MemoryRegion<BoundaryConditionType>::template create<MemSpace>(
                m_bndExpSize);

        BoundaryConditionType *bctype =
            m_bctype.template GetPtr<NektarSpaces::HostSpace>();

        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            bctype[i] =
                contfield->GetBndConditions()[i]->GetBoundaryConditionType();
        }

        // Coeff
        m_coeff = MemoryRegion<TData>::template create<MemSpace>(ntotcoeff);

        TData *coeff = m_coeff.template GetPtr<NektarSpaces::HostSpace>();

        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            for (size_t j = 0; j < ncoeff[i]; ++j)
            {
                coeff[offset[i] + j] =
                    contfield->GetBndCondExpansions()[i]->GetCoeffs()[j];
            }
        }

        // Deterime CUDA grid parameters.
#if defined(NEKTAR_ENABLE_CUDA)
        if constexpr (std::is_same<ExecSpace, NektarSpaces::CUDA>::value)
        {
            m_gridSize = GetCUDAGridSize(m_bndExpSize, m_blockSize);
        }
#endif
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        TData *inOutPtr = inout.template GetPtr<MemSpace>();

        BoundaryConditionType *bctypePtr = m_bctype.template GetPtr<MemSpace>();
        TData *coeffPtr                  = m_coeff.template GetPtr<MemSpace>();
        int *mapPtr                      = m_map.template GetPtr<MemSpace>();
        int *ncoeffPtr                   = m_ncoeff.template GetPtr<MemSpace>();
        int *offsetPtr                   = m_offset.template GetPtr<MemSpace>();

        // Copy memory to the device, if necessary and get raw pointers.
        if (m_signChange)
        {
            TData *signPtr = m_sign.template GetPtr<MemSpace>();

            NeuBndCondKernel<ExecSpace, TData>(
                m_gridSize, m_blockSize, m_bndExpSize, offsetPtr, bctypePtr,
                ncoeffPtr, signPtr, mapPtr, coeffPtr, inOutPtr);
        }
        else
        {
            NeuBndCondKernel<ExecSpace, TData>(
                m_gridSize, m_blockSize, m_bndExpSize, offsetPtr, bctypePtr,
                ncoeffPtr, mapPtr, coeffPtr, inOutPtr);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorNeuBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    bool m_signChange;
    size_t m_bndExpSize;

    MemoryRegion<BoundaryConditionType> m_bctype;
    MemoryRegion<TData> m_coeff;
    MemoryRegion<TData> m_sign;
    MemoryRegion<int> m_map;
    MemoryRegion<int> m_ncoeff;
    MemoryRegion<int> m_offset;

    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
