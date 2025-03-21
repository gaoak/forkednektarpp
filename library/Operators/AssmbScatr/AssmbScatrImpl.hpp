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

#include <MultiRegions/ContField.h>

#include "Operators/AssmbScatr/OperatorAssmbScatr.hpp"

#include "Operators/AssmbScatr/AssmbScatrDeviceKernels.hpp"
#include "Operators/AssmbScatr/AssmbScatrSerialAVXKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
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

        m_nDir       = assmbMap->GetNumGlobalDirBndCoeffs();
        m_signChange = assmbMap->AssemblyMap::GetSignChange();

        auto nGlobal = assmbMap->GetNumGlobalCoeffs();

        m_global = MemoryRegion<TData>::Create("AssmbScatr global", nGlobal,
                                               ExecSpace::alignment);

        auto map = assmbMap->GetLocalToGlobalMap();

        m_map = MemoryRegion<int>::template FromArray<MemSpace, int>(
            map, ExecSpace::alignment);

        if (m_signChange)
        {
            auto sign = assmbMap->GetLocalToGlobalSign();

            m_sign =
                MemoryRegion<TData>::template FromArray<MemSpace, NekDouble>(
                    sign, ExecSpace::alignment);
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false) override
    {
        // Local to global.
        this->Assemble(in, m_global);

        // Zeroing Dirichlet BC.
        if (zeroDir && m_nDir > 0)
        {
            m_global.template Initialize<MemSpace>(0, m_nDir);
        }

        // Global to local.
        this->GlobalToLocal(m_global, out);
    }

    void Assemble(Field<TData, FieldState::Coeff> &local,
                  MemoryRegion<TData> &global,
                  const bool &signChange = true) override
    {
        // Initialize MemoryRegion pointers.
        auto globalPtr = global.template GetPtr<MemSpace, WriteOnly>();
        auto mapPtr    = m_map.template GetPtr<MemSpace, ReadOnly>();
        auto signPtr   = m_signChange
                             ? m_sign.template GetPtr<MemSpace, ReadOnly>()
                             : nullptr;

        // Zero the output.
        global.template Initialize<MemSpace>(0);

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < local.GetBlocks().size(); ++blk)
        {
            // Determine shape and type of the element.
            auto &localblock = local.GetBlocks()[blk];
            auto nElmts      = localblock.GetNumElements();
            auto ncoeff      = localblock.GetNumData();

            // Initialize pointer.
            auto localPtr = localblock.template GetPtr<MemSpace, ReadOnly>();

            if (m_signChange && signChange)
            {
                AssembleKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr,
                                                 signPtr, localPtr, globalPtr);
            }
            else
            {
                AssembleKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr,
                                                 localPtr, globalPtr);
            }

            // Increment pointers for the next element type.
            localPtr += localblock.size();
            mapPtr += ncoeff * nElmts;
            signPtr += ncoeff * nElmts;
        }

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            auto globalArr = global.template ToArray<NekDouble>();

            contfield->GetLocalToGlobalMap()->UniversalAssemble(globalArr);

            global.template CopyArray<MemSpace, NekDouble>(globalArr);
        }
    }

    void GlobalToLocal(MemoryRegion<TData> &global,
                       Field<TData, FieldState::Coeff> &local) override
    {
        // Initialize MemoryRegion pointers.
        auto globalPtr = global.template GetPtr<MemSpace, ReadOnly>();
        auto mapPtr    = m_map.template GetPtr<MemSpace, ReadOnly>();
        auto signPtr   = m_signChange
                             ? m_sign.template GetPtr<MemSpace, ReadOnly>()
                             : nullptr;

        // Zero the output.
        local.template Initialize<MemSpace>(0);

        // Loop over the blocks.
        for (unsigned int blk = 0; blk < local.GetBlocks().size(); ++blk)
        {
            // Determine shape and type of the element.
            auto &block = local.GetBlocks()[blk];
            auto nElmts = block.GetNumElements();
            auto ncoeff = block.GetNumData();

            // Initialize pointer.
            auto localPtr = block.template GetPtr<MemSpace, WriteOnly>();

            if (m_signChange)
            {
                GlobalToLocalKernel<ExecSpace, TData>(
                    ncoeff * nElmts, mapPtr, signPtr, globalPtr, localPtr);
            }
            else
            {
                GlobalToLocalKernel<ExecSpace, TData>(ncoeff * nElmts, mapPtr,
                                                      globalPtr, localPtr);
            }

            // Increment pointers for the next element type.
            mapPtr += ncoeff * nElmts;
            signPtr += ncoeff * nElmts;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorAssmbScatrImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    MemoryRegion<TData> m_global;
    MemoryRegion<TData> m_sign;
    MemoryRegion<int> m_map;

    bool m_signChange = false;
    unsigned int m_nDir;
};

} // namespace Nektar::Operators::detail
