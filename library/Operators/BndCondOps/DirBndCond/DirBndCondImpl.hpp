///////////////////////////////////////////////////////////////////////////////
//
// File: DirBndCondImpl.hpp
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

#include "Operators/BndCondOps/OperatorDirBndCond.hpp"

#include "Operators/BndCondOps/DirBndCond/DirBndCondCUDAKernels.cuh"
#include "Operators/BndCondOps/DirBndCond/DirBndCondKernels.hpp"
#include "Operators/BndCondOps/DirBndCond/DirBndCondKokkosKernels.hpp"

#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorDirBndCondImpl : public OperatorDirBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorDirBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDirBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto &bndCondExpansions = contfield->GetBndCondExpansions();
        auto &bndConditions     = contfield->GetBndConditions();
        auto assmbMap           = contfield->GetLocalToGlobalMap();
        auto &sign              = assmbMap->GetBndCondCoeffsToLocalCoeffsSign();
        auto &map               = assmbMap->GetBndCondCoeffsToLocalCoeffsMap();
        m_signChange            = assmbMap->GetSignChange();

        // Compute number boundary coefficients
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                m_nbndcoeff += bndCondExpansions[i]->GetNcoeffs();
            }
        }

        // Collecting boundary coefficients
        Array<OneD, TData> bndcoeff(m_nbndcoeff);
        Array<OneD, int> index(m_nbndcoeff);
        size_t bndcnt = 0, cnt = 0;
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            auto nBndExpCoeff = bndCondExpansions[i]->GetNcoeffs();

            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                auto &bndExpCoeff = bndCondExpansions[i]->GetCoeffs();
                std::copy(bndExpCoeff.data(), bndExpCoeff.data() + nBndExpCoeff,
                          bndcoeff.data() + bndcnt);
                for (size_t j = 0; j < nBndExpCoeff; ++j)
                {
                    index[bndcnt + j] = cnt + j;
                }
                bndcnt += nBndExpCoeff;
            }
            cnt += nBndExpCoeff;
        }

        m_bndcoeff = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
            bndcoeff, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Set mapping to skip over padding elements
        int i = 0, j = 0;

        Array<OneD, int> alignmentMap(expansionList->GetNcoeffs());
        auto blocks =
            GetBlockAttributes(FieldState::Coeff, expansionList, vec_t::width);
        for (auto &block : blocks)
        {
            auto const ncoeff    = block.num_pts;
            auto const nElmts    = block.num_elements;
            auto const nPadElmts = block.num_padding_elements;
            for (unsigned int e = 0; e < nElmts; e++)
            {
                for (unsigned int n = 0; n < ncoeff; n++)
                {
                    alignmentMap[i++] = j++;
                }
            }
            j += nPadElmts * ncoeff;
        }

        // Compute aligned map to skip over padding elements
        Array<OneD, int> alignedMap(m_nbndcoeff);
        for (int i = 0; i < m_nbndcoeff; i++)
        {
            alignedMap[i] = alignmentMap[map[index[i]]];
        }

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            alignedMap, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        if (m_signChange)
        {
            Array<OneD, TData> alignedSign(m_nbndcoeff);
            for (int i = 0; i < m_nbndcoeff; i++)
            {
                alignedSign[i] = sign[index[i]];
            }

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                alignedSign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        }

        // local
        m_localDirSize = assmbMap->GetCopyLocalDirDofs().size();
        Array<OneD, int> locid0(m_localDirSize);
        Array<OneD, int> locid1(m_localDirSize);
        Array<OneD, TData> locsign(m_localDirSize);

        cnt = 0;
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

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        auto *mapPtr      = m_map.template GetPtr<MemSpace, ReadOnly>();
        auto *bndcoeffPtr = m_bndcoeff.template GetPtr<MemSpace, ReadOnly>();
        auto *inoutPtr    = inout.template GetPtr<MemSpace, ReadWrite>();

        if (m_signChange)
        {
            const TData *signPtr = m_sign.template GetPtr<MemSpace, ReadOnly>();

            DirBndCondKernel<ExecSpace>(m_nbndcoeff, signPtr, mapPtr,
                                        bndcoeffPtr, inoutPtr);
        }
        else
        {
            DirBndCondKernel<ExecSpace>(m_nbndcoeff, mapPtr, bndcoeffPtr,
                                        inoutPtr);
        }

        if constexpr (std::is_same<ExecSpace, NektarSpaces::Serial>::value)
        {
            // communicate local Dirichlet coeffs that are just
            // touching a dirichlet boundary on another partition
            auto contfield =
                std::dynamic_pointer_cast<ContField>(this->m_expansionList);
            auto assmbMap            = contfield->GetLocalToGlobalMap();
            auto &ParallelDirBndSign = assmbMap->GetParallelDirBndSign();
            auto nloc                = assmbMap->GetNumLocalCoeffs();

            // Copy the data from the input field.
            auto inoutarr = inout.toArray();
            for (auto &it : ParallelDirBndSign)
            {
                inoutarr[it] *= -1;
            }

            Array<OneD, NekDouble> arr(nloc, inoutarr.data());
            assmbMap->UniversalAbsMaxBnd(arr);
            std::copy(arr.get(), arr.get() + nloc, inoutarr.data());

            for (auto &it : ParallelDirBndSign)
            {
                inoutarr[it] *= -1;
            }

            auto &copyLocalDirDofs = assmbMap->GetCopyLocalDirDofs();
            for (auto &it : copyLocalDirDofs)
            {
                inoutarr[std::get<0>(it)] =
                    inoutarr[std::get<1>(it)] * std::get<2>(it);
            }

            // Copy the data to the output field.
            inout.template copyArray<MemSpace>(inoutarr);
        }
        else
        {
            // TODO: Global assembly
            auto *locid0Ptr  = m_locid0.template GetPtr<MemSpace, ReadOnly>();
            auto *locid1Ptr  = m_locid1.template GetPtr<MemSpace, ReadOnly>();
            auto *locsignPtr = m_locsign.template GetPtr<MemSpace, ReadOnly>();

            LocalDirBndCondKernel<ExecSpace>(m_localDirSize, locid0Ptr,
                                             locid1Ptr, locsignPtr, inoutPtr);
        }
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
    MemoryRegion<int> m_map;
    MemoryRegion<TData> m_sign;
    MemoryRegion<TData> m_bndcoeff;
    MemoryRegion<int> m_locid0;
    MemoryRegion<int> m_locid1;
    MemoryRegion<TData> m_locsign;
    size_t m_nbndcoeff    = 0;
    size_t m_localDirSize = 0;
    bool m_signChange;
};

} // namespace Nektar::Operators::detail
