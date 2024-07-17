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

#include "Operators/BndCondOps/OperatorNeuBndCond.hpp"

#include "Operators/BndCondOps/NeuBndCond/NeuBndCondCUDASumFacKernels.cuh"
#include "Operators/BndCondOps/NeuBndCond/NeuBndCondKokkosStdMatKernels.hpp"
#include "Operators/Common/OperatorHelper.hpp"

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

        // sign
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

        // BC type
        Array<OneD, BoundaryConditionType> bctype(m_bndExpSize);

        for (size_t i = 0; i < m_bndExpSize; ++i)
        {
            bctype[i] =
                contfield->GetBndConditions()[i]->GetBoundaryConditionType();
        }

        m_bctype = MemoryRegion<BoundaryConditionType>::template fromArray<
            MemSpace, BoundaryConditionType>(
            bctype, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Coeff
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
    }

    void apply(Field<TData, FieldState::Coeff> &inout) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        TData *inoutPtr = inout.template GetPtr<MemSpace, ReadWrite>();

        const BoundaryConditionType *bctypePtr =
            m_bctype.template GetPtr<MemSpace, ReadOnly>();

        const TData *coeffPtr = m_coeff.template GetPtr<MemSpace, ReadOnly>();
        const int *mapPtr     = m_map.template GetPtr<MemSpace, ReadOnly>();
        const int *ncoeffPtr  = m_ncoeff.template GetPtr<MemSpace, ReadOnly>();
        const int *offsetPtr  = m_offset.template GetPtr<MemSpace, ReadOnly>();

        if (m_signChange)
        {
            const TData *signPtr = m_sign.template GetPtr<MemSpace, ReadOnly>();

            NeuBndCondKernel<ExecSpace, TData>(m_bndExpSize, offsetPtr,
                                               bctypePtr, ncoeffPtr, signPtr,
                                               mapPtr, coeffPtr, inoutPtr);
        }
        else
        {
            NeuBndCondKernel<ExecSpace, TData>(m_bndExpSize, offsetPtr,
                                               bctypePtr, ncoeffPtr, mapPtr,
                                               coeffPtr, inoutPtr);
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
};

} // namespace Nektar::Operators::detail
