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
#include "Operators/BndCondOps/DirBndCond/DirBndCondSYCLKernels.hpp"

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
        auto &parallelDirBndSign = assmbMap->GetParallelDirBndSign();
        m_signChange             = assmbMap->GetSignChange();

        // Compute number boundary coefficients
        for (size_t i = 0; i < bndCondExpansions.size(); ++i)
        {
            if (bndConditions[i]->GetBoundaryConditionType() ==
                SpatialDomains::eDirichlet)
            {
                m_nbndcoeff += bndCondExpansions[i]->GetNcoeffs();
            }
        }

        // Return if no Dirichlet boundary condition.
        if (m_nbndcoeff == 0)
        {
            return;
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
            bndcoeff, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());

        // Set mapping to skip over padding elements
        int i = 0, j = 0;

        Array<OneD, int> alignmentMap(expansionList->GetNcoeffs());
        auto blocks = GetBlockAttributes(FieldState::Coeff, expansionList,
                                         ExecSpace::width);
        for (auto &block : blocks)
        {
            const auto ncoeff    = block.num_pts;
            const auto nElmts    = block.num_elements;
            const auto nPadElmts = block.num_padding_elements;
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
            alignedMap, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());

        if (m_signChange)
        {
            Array<OneD, TData> alignedSign(m_nbndcoeff);
            for (int i = 0; i < m_nbndcoeff; i++)
            {
                alignedSign[i] = sign[index[i]];
            }

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                alignedSign,
                EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
        }

        m_parallelDirBndSignSize = parallelDirBndSign.size();
        if (m_parallelDirBndSignSize > 0)
        {
            Array<OneD, int> alignedParallelDirBndSign(
                m_parallelDirBndSignSize);
            for (auto &it : parallelDirBndSign)
            {
                alignedParallelDirBndSign[i] = alignmentMap[it];
            }

            m_parallelDirBndSign =
                MemoryRegion<int>::template fromArray<MemSpace, int>(
                    alignedParallelDirBndSign,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
        }

        // local
        m_localDirSize = assmbMap->GetCopyLocalDirDofs().size();
        if (m_localDirSize > 0)
        {
            Array<OneD, int> locid0(m_localDirSize);
            Array<OneD, int> locid1(m_localDirSize);
            Array<OneD, TData> locsign(m_localDirSize);

            cnt = 0;
            for (auto &it : assmbMap->GetCopyLocalDirDofs())
            {
                locid0[cnt]  = alignmentMap[std::get<0>(it)];
                locid1[cnt]  = alignmentMap[std::get<1>(it)];
                locsign[cnt] = std::get<2>(it);
                cnt++;
            }

            m_locid0 = MemoryRegion<int>::template fromArray<MemSpace, int>(
                locid0, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_locid1 = MemoryRegion<int>::template fromArray<MemSpace, int>(
                locid1, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
            m_locsign =
                MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                    locsign,
                    EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());
        }
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        if (m_nbndcoeff > 0)
        {
            auto *inoutPtr = inout.template GetPtr<MemSpace, ReadWrite>();
            auto *mapPtr   = m_map.template GetPtr<MemSpace, ReadOnly>();
            auto *bndcoeffPtr =
                m_bndcoeff.template GetPtr<MemSpace, ReadOnly>();
            const TData *signPtr =
                m_signChange ? m_sign.template GetPtr<MemSpace, ReadOnly>()
                             : nullptr;

            if (m_signChange)
            {
                DirBndCondKernel<ExecSpace>(m_nbndcoeff, signPtr, mapPtr,
                                            bndcoeffPtr, inoutPtr);
            }
            else
            {
                DirBndCondKernel<ExecSpace>(m_nbndcoeff, mapPtr, bndcoeffPtr,
                                            inoutPtr);
            }
        }

        if (m_parallelDirBndSignSize > 0)
        {
            auto *inoutPtr = inout.template GetPtr<MemSpace, ReadWrite>();
            auto *parallelDirBndSignPtr =
                m_parallelDirBndSign.template GetPtr<MemSpace, ReadOnly>();

            ParallelDirBndSignKernel<ExecSpace>(
                m_parallelDirBndSignSize, parallelDirBndSignPtr, inoutPtr);
        }

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            // Copy the data from the input field.
            auto inoutarr = inout.toArray();

            contfield->GetLocalToGlobalMap()->UniversalAbsMaxBnd(inoutarr);

            // Copy the data to the output field.
            inout.template copyArray<MemSpace>(inoutarr);
        }

        if (m_parallelDirBndSignSize > 0)
        {
            auto *inoutPtr = inout.template GetPtr<MemSpace, ReadWrite>();
            auto *parallelDirBndSignPtr =
                m_parallelDirBndSign.template GetPtr<MemSpace, ReadOnly>();

            ParallelDirBndSignKernel<ExecSpace>(
                m_parallelDirBndSignSize, parallelDirBndSignPtr, inoutPtr);
        }

        if (m_localDirSize > 0)
        {
            auto *inoutPtr   = inout.template GetPtr<MemSpace, ReadWrite>();
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
    MemoryRegion<int> m_parallelDirBndSign;
    MemoryRegion<int> m_locid0;
    MemoryRegion<int> m_locid1;
    MemoryRegion<TData> m_locsign;

    size_t m_nbndcoeff              = 0;
    size_t m_parallelDirBndSignSize = 0;
    size_t m_localDirSize           = 0;
    bool m_signChange;
};

} // namespace Nektar::Operators::detail
