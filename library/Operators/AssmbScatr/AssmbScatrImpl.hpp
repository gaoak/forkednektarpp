///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrImpl.hpp
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

#include "Operators/OperatorAssmbScatr.hpp"

#include "Operators/AssmbScatr/AssmbScatrCUDAKernels.cuh"
#include "Operators/AssmbScatr/AssmbScatrKernels.hpp"
#include "Operators/AssmbScatr/AssmbScatrKokkosKernels.hpp"
#include "Operators/AssmbScatr/AssmbScatrSYCLKernels.hpp"

#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

// Base implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorAssmbScatrImpl : public OperatorAssmbScatr<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        auto assmbMap = contfield->GetLocalToGlobalMap();

        m_nLocal     = assmbMap->GetNumLocalCoeffs();
        m_nGlobal    = assmbMap->GetNumGlobalCoeffs();
        m_nDir       = assmbMap->GetNumGlobalDirBndCoeffs();
        m_signChange = assmbMap->AssemblyMap::GetSignChange();

        m_global = MemoryRegion<TData>::template create<MemSpace>(
            "AssmbScatr global", m_nGlobal, ExecSpace::alignment);

        // Compute aligned map to skip over padding elements
        auto map = assmbMap->GetLocalToGlobalMap();

        const bool device_only = true;

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            map, ExecSpace::alignment, device_only);

        // Memory allocation for sign pointer
        if (m_signChange)
        {
            auto sign = assmbMap->GetLocalToGlobalSign();

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                sign, ExecSpace::alignment, device_only);
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false) override
    {
        this->Assemble(in, m_global);

        // Zeroing Dirichlet BC
        if (zeroDir && m_nDir > 0)
        {
            m_global.template initialize<DeviceOnly>(0, m_nDir);
        }

        this->GlobalToLocal(m_global, out);
    }

    void Assemble(Field<TData, FieldState::Coeff> &in,
                  MemoryRegion<TData> &out) override
    {
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();
        const int *mapPtr  = m_map.template GetPtr<MemSpace, ReadOnly>();
        const TData *signPtr =
            m_signChange ? m_sign.template GetPtr<MemSpace, ReadOnly>()
                         : nullptr;

        // Zero the output
        out.template initialize<DeviceOnly>(0, this->m_nGlobal);

        for (const auto &block : in.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts = block.num_elements;
            auto ncoeff = block.num_pts;

            if (m_signChange)
            {
                AssembleKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr,
                                                 signPtr, inPtr, outPtr);
            }
            else
            {
                AssembleKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr, inPtr,
                                                 outPtr);
            }

            inPtr += block.block_size;
            mapPtr += ncoeff * nElmts;
            signPtr += ncoeff * nElmts;
        }

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            auto outArr = out.toArray();

            contfield->GetLocalToGlobalMap()->UniversalAssemble(outArr);

            out.template copyArray<MemSpace, TData>(outArr);
        }
    }

    void GlobalToLocal(MemoryRegion<TData> &in,
                       Field<TData, FieldState::Coeff> &out) override
    {
        const TData *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();
        const int *mapPtr  = m_map.template GetPtr<MemSpace, ReadOnly>();
        const TData *signPtr =
            m_signChange ? m_sign.template GetPtr<MemSpace, ReadOnly>()
                         : nullptr;

        // Zero the output
        out.template initialize<DeviceOnly>(0, this->m_nLocal);

        for (const auto &block : out.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts = block.num_elements;
            auto ncoeff = block.num_pts;

            if (m_signChange)
            {
                GlobalToLocalKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr,
                                                      signPtr, inPtr, outPtr);
            }
            else
            {
                GlobalToLocalKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr,
                                                      inPtr, outPtr);
            }

            mapPtr += ncoeff * nElmts;
            signPtr += ncoeff * nElmts;
            outPtr += block.block_size;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorAssmbScatrImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    MemoryRegion<TData> m_global;
    MemoryRegion<TData> m_sign;
    MemoryRegion<int> m_map;

    bool m_signChange = false;
    size_t m_nLocal;
    size_t m_nGlobal;
    size_t m_nDir;
};

} // namespace Nektar::Operators::detail
