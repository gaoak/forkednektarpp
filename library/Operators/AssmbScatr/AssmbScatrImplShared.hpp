///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrImplShared.hpp
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

#include "AssmbScatrImplBase.hpp"

#include "Operators/AssmbScatr/AssmbScatrCUDAKernels.cuh"
#include "Operators/AssmbScatr/AssmbScatrKokkosKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::CUDA>::value ||
              std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value>::type>
class OperatorAssmbScatrImpl
    : public OperatorAssmbScatrImplBase<ExecSpace, Implementation, TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatrImplBase<ExecSpace, Implementation, TData>(
              expansionList)
    {
        // Memory allocation for assemble pointer
        auto assmb = this->m_assmbMap->GetLocalToGlobalMap();

        m_assmb = MemoryRegion<int>::template fromArray<MemSpace, int>(
            assmb, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());

        // Memory allocation for sign pointer
        m_signChange = this->m_assmbMap->AssemblyMap::GetSignChange();

        if (m_signChange)
        {
            auto sign = this->m_assmbMap->GetLocalToGlobalSign();

            m_sign = MemoryRegion<TData>::template fromArray<MemSpace, TData>(
                sign, EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        }
    }

    void Assemble(Field<TData, FieldState::Coeff> &in,
                  MemoryRegion<TData> &out) override
    {
        const TData *inPtr  = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr       = out.template GetPtr<MemSpace, WriteOnly>();
        const int *assmbPtr = m_assmb.template GetPtr<MemSpace, ReadOnly>();

        // Zero the output
        out.initialize(0, this->m_nGlobal);

        if (this->m_solnType == eIterativeFull)
        {
            size_t offset = 0;

            for (const auto &block : in.GetBlocks())
            {
                // Determine shape and type of the element.
                auto nElmts = block.num_elements;
                auto ncoeff = block.num_pts;

                if (m_signChange)
                {
                    const TData *signPtr =
                        m_sign.template GetPtr<MemSpace, ReadOnly>();

                    AssembleKernel<ExecSpace, TData>(ncoeff * nElmts, offset,
                                                     assmbPtr, signPtr, inPtr,
                                                     outPtr);
                }
                else
                {
                    AssembleKernel<ExecSpace, TData>(ncoeff * nElmts, offset,
                                                     assmbPtr, inPtr, outPtr);
                }

                offset += block.block_size;
            }
        }
    }

    void GlobalToLocal(MemoryRegion<TData> &in,
                       Field<TData, FieldState::Coeff> &out) override
    {
        const TData *inPtr  = in.template GetPtr<MemSpace, ReadOnly>();
        TData *outPtr       = out.template GetPtr<MemSpace, WriteOnly>();
        const int *assmbPtr = m_assmb.template GetPtr<MemSpace, ReadOnly>();

        // Zero the output
        out.initialize(0, this->m_nLocal);

        if (this->m_solnType == eIterativeFull)
        {
            size_t offset = 0;

            for (const auto &block : out.GetBlocks())
            {
                // Determine shape and type of the element.
                auto nElmts = block.num_elements;
                auto ncoeff = block.num_pts;

                if (m_signChange)
                {
                    const TData *signptr =
                        m_sign.template GetPtr<MemSpace, ReadOnly>();

                    GlobalToLocalKernel<ExecSpace, TData>(
                        ncoeff * nElmts, offset, assmbPtr, signptr, inPtr,
                        outPtr);
                }
                else
                {
                    GlobalToLocalKernel<ExecSpace, TData>(
                        ncoeff * nElmts, offset, assmbPtr, inPtr, outPtr);
                }

                offset += block.block_size;
            }
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorAssmbScatrImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    bool m_signChange = false;

    MemoryRegion<TData> m_sign;
    MemoryRegion<int> m_assmb;
};

} // namespace Nektar::Operators::detail
