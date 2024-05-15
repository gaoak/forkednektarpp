///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondImplShared.hpp
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

#include "Operators/DirBndCond/DirBndCondImplBase.hpp"

#include "Operators/DirBndCond/DirBndCondCUDASumFacKernels.cuh"
#include "Operators/DirBndCond/DirBndCondKokkosStdMatKernels.hpp"

#include "Operators/OperatorHelper.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)
#if defined(NEKTAR_ENABLE_CUDA)
              || (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
                  std::is_same<Implementation, Operators::SumFac>::value)
#endif
              >::type>
class OperatorDirBndCondImpl
    : public OperatorDirBndCondImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCondImplBase<ExecSpace, Implementation, TData>(
              expansionList)
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

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                sign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        }

        // Map
        auto &map = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            map, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Offset
        m_offset = MemoryRegion<int>::template create<MemSpace>(m_bndExpSize);

        int *offset = m_offset.template GetPtr<NektarSpaces::HostSpace>();

        offset[0] = 0;

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

        // BC Type
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

        // Local
        m_localDirSize = assmbMap->GetCopyLocalDirDofs().size();

        m_locid0 = MemoryRegion<int>::template create<MemSpace>(m_localDirSize);
        m_locid1 = MemoryRegion<int>::template create<MemSpace>(m_localDirSize);
        m_locsign =
            MemoryRegion<TData>::template create<MemSpace>(m_localDirSize);

        int *locid0    = m_locid0.template GetPtr<NektarSpaces::HostSpace>();
        int *locid1    = m_locid1.template GetPtr<NektarSpaces::HostSpace>();
        TData *locsign = m_locsign.template GetPtr<NektarSpaces::HostSpace>();

        size_t cnt = 0;
        for (auto &it : assmbMap->GetCopyLocalDirDofs())
        {
            locid0[cnt]  = std::get<0>(it);
            locid1[cnt]  = std::get<1>(it);
            locsign[cnt] = std::get<2>(it);
            cnt++;
        }
    }

    void apply(Field<TData, FieldState::Coeff> &out) override
    {
        TData *outptr = out.template GetPtr<MemSpace>();

        BoundaryConditionType *p_bctype = m_bctype.template GetPtr<MemSpace>();
        TData *p_coeff                  = m_coeff.template GetPtr<MemSpace>();
        int *p_map                      = m_map.template GetPtr<MemSpace>();
        int *p_ncoeff                   = m_ncoeff.template GetPtr<MemSpace>();
        int *p_offset                   = m_offset.template GetPtr<MemSpace>();

        int *p_locid0    = m_locid0.template GetPtr<MemSpace>();
        int *p_locid1    = m_locid1.template GetPtr<MemSpace>();
        TData *p_locsign = m_locsign.template GetPtr<MemSpace>();
        // Copy memory to the device, if necessary and get raw pointers.

#if defined(NEKTAR_ENABLE_CUDA)
        // Deterime CUDA grid size.
        m_gridSize = GetCUDAGridSize(m_bndExpSize, m_blockSize);
#endif
        if (m_signChange)
        {
            TData *p_sign = m_sign.template GetPtr<MemSpace>();

            DirBndCondKernel<ExecSpace, TData>(
                m_gridSize, m_blockSize, m_bndExpSize, p_offset, p_bctype,
                p_ncoeff, p_sign, p_map, p_coeff, outptr);
        }
        else
        {
            DirBndCondKernel<ExecSpace, TData>(
                m_gridSize, m_blockSize, m_bndExpSize, p_offset, p_bctype,
                p_ncoeff, p_map, p_coeff, outptr);
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

        // Deterime CUDA grid size.
        m_gridSize = GetCUDAGridSize(m_localDirSize, m_blockSize);

        LocalDirBndCondKernel<ExecSpace, TData>(m_gridSize, m_blockSize,
                                                m_localDirSize, p_locid0,
                                                p_locid1, p_locsign, outptr);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorDirBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    bool m_signChange;
    size_t m_bndExpSize;
    size_t m_localDirSize;

    MemoryRegion<BoundaryConditionType> m_bctype;
    MemoryRegion<TData> m_coeff;
    MemoryRegion<TData> m_sign;
    MemoryRegion<int> m_map;
    MemoryRegion<int> m_ncoeff;
    MemoryRegion<int> m_offset;
    MemoryRegion<int> m_locid0;
    MemoryRegion<int> m_locid1;
    MemoryRegion<TData> m_locsign;

    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
