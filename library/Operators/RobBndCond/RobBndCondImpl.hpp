///////////////////////////////////////////////////////////////////////////////
//
// File: RobBndCondImpl.hpp
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

#include "Operators/OperatorRobBndCond.hpp"

#include "Operators/MathKernels.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{
// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorRobBndCondImpl : public OperatorRobBndCond<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorRobBndCondImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorRobBndCond<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &negflag) override
    {
        auto ncoeffs = m_assmbMap->GetNumLocalCoeffs();

        // Copy the data from the input field.
        Array<OneD, TData> inArray = in.toArray();
        Array<OneD, NekDouble> robinArray(ncoeffs, 0.0);

        auto robinBCInfo = this->m_expansionList->GetRobinBCInfo();

        for (auto &r : robinBCInfo) // add robin mass matrix
        {
            auto n      = r.first;
            auto offset = this->m_expansionList->GetCoeff_Offset(n);
            auto expPtr = this->m_expansionList->GetExp(n);
            Array<OneD, NekDouble> tmpArray;

            for (auto rBC = r.second; rBC; rBC = rBC->next)
            {
                expPtr->AddRobinTraceContribution(
                    rBC->m_robinID, rBC->m_robinPrimitiveCoeffs,
                    inArray + offset, tmpArray = robinArray + offset);
            }
        }

        const TData *robinPtr;
        MemoryRegion<TData> robinMR;

        // Data is on the host via the array so use that pointer.
        if constexpr (std::is_same<MemSpace, Kokkos::HostSpace>::value ||
                      std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            robinPtr = robinArray.get();
        }
        // Data is on the host via the array so move to the device and
        // use the device pointer.
        else if constexpr (
#if defined(NEKTAR_ENABLE_KOKKOS)
            std::is_same<MemSpace,
                         Kokkos::DefaultExecutionSpace::memory_space>::value ||
#endif
            std::is_same<MemSpace, NektarSpaces::DeviceSpace>::value)
        {
            robinMR = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                "robin", robinArray, __EXECSPACE_MEMORY_REGION_ONLY__);

            robinPtr = out.template GetPtr<MemSpace>();
        }

        // Copy the data to the output field.
        auto *outPtr = out.template GetPtr<MemSpace>();

        for (auto const &block : out.GetBlocks())
        {
            auto nSize  = block.block_size;
            auto nElmts = block.num_elements;
            auto nmTot  = block.num_pts;

            if (negflag)
            {
                subKernel<ExecSpace, TData>(nElmts * nmTot, outPtr, robinPtr,
                                            outPtr);
            }
            else
            {
                addKernel<ExecSpace, TData>(nElmts * nmTot, outPtr, robinPtr,
                                            outPtr);
            }

            robinPtr += nElmts * nmTot;
            outPtr += nSize;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorRobBndCondImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
};

} // namespace Nektar::Operators::detail
