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

#include "Operators/OperatorDirBndCond.hpp"

#include "Operators/DirBndCond/DirBndCondCUDASumFacKernels.cuh"
#include "Operators/DirBndCond/DirBndCondKokkosStdMatKernels.hpp"

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
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorDirBndCondImpl : public OperatorDirBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

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

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                sign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        }

        // map
        auto &map = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            map, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // offset
        Array<OneD, int> offset(m_bndExpSize);

        offset[0] = 0;
        for (size_t i = 1; i < m_bndExpSize; ++i)
        {
            offset[i] = offset[i - 1] +
                        contfield->GetBndCondExpansions()[i - 1]->GetNcoeffs();
        }

        m_offset = MemoryRegion<int>::template fromArray<MemSpace, int>(
            offset, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // ncoeff
        Array<OneD, int> ncoeff(m_bndExpSize);

        size_t ntotcoeff = 0;
        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            ncoeff[i] = contfield->GetBndCondExpansions()[i]->GetNcoeffs();
            ntotcoeff += contfield->GetBndCondExpansions()[i]->GetNcoeffs();
        }

        m_ncoeff = MemoryRegion<int>::template fromArray<MemSpace, int>(
            ncoeff, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // BC Type
        Array<OneD, BoundaryConditionType> bctype(m_bndExpSize);

        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            bctype[i] =
                contfield->GetBndConditions()[i]->GetBoundaryConditionType();
        }

        m_bctype = MemoryRegion<BoundaryConditionType>::template fromArray<
            MemSpace, BoundaryConditionType>(
            bctype, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // coeff
        Array<OneD, TData> coeff(ntotcoeff);

        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            for (size_t j = 0; j < ncoeff[i]; ++j)
            {
                coeff[offset[i] + j] =
                    contfield->GetBndCondExpansions()[i]->GetCoeffs()[j];
            }
        }

        m_coeff = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            coeff, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // local
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

        m_locid0 = MemoryRegion<int>::template fromArray<MemSpace, int>(
            locid0, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        m_locid1 = MemoryRegion<int>::template fromArray<MemSpace, int>(
            locid1, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        m_locsign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            locsign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
    }

    void apply(Field<TData, FieldState::Coeff> &out) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        TData *outPtr = out.template GetPtr<MemSpace, ReadWrite>();

        const BoundaryConditionType *bctypePtr =
            m_bctype.template GetPtr<MemSpace, ReadOnly>();

        const TData *coeffPtr = m_coeff.template GetPtr<MemSpace, ReadOnly>();
        const int *mapPtr     = m_map.template GetPtr<MemSpace, ReadOnly>();
        const int *ncoeffPtr  = m_ncoeff.template GetPtr<MemSpace, ReadOnly>();
        const int *offsetPtr  = m_offset.template GetPtr<MemSpace, ReadOnly>();
        const int *locid0Ptr  = m_locid0.template GetPtr<MemSpace, ReadOnly>();
        const int *locid1Ptr  = m_locid1.template GetPtr<MemSpace, ReadOnly>();
        const TData *locsignPtr =
            m_locsign.template GetPtr<MemSpace, ReadOnly>();

        if (m_signChange)
        {
            const TData *signPtr = m_sign.template GetPtr<MemSpace, ReadOnly>();

            DirBndCondKernel<ExecSpace, TData>(m_bndExpSize, offsetPtr,
                                               bctypePtr, ncoeffPtr, signPtr,
                                               mapPtr, coeffPtr, outPtr);
        }
        else
        {
            DirBndCondKernel<ExecSpace, TData>(m_bndExpSize, offsetPtr,
                                               bctypePtr, ncoeffPtr, mapPtr,
                                               coeffPtr, outPtr);
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

        LocalDirBndCondKernel<ExecSpace, TData>(m_localDirSize, locid0Ptr,
                                                locid1Ptr, locsignPtr, outPtr);
    }

    // className - for OperatorFactory
    static std::string className;

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
};

} // namespace Nektar::Operators::detail
