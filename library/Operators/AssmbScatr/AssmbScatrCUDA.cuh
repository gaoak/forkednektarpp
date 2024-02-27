///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrCUDA.hpp
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

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

#include "Operators/AssmbScatr/AssmbScatrCUDAKernels.cuh"
#include "Operators/MemoryRegionCUDA.hpp"
#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/OperatorHelper.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorAssmbScatrImpl<TData, ImplCUDA> : public OperatorAssmbScatr<TData>
{
public:
    OperatorAssmbScatrImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorAssmbScatr<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        m_solnType = m_assmbMap->GetGlobalSysSolnType();
        m_nloc     = m_assmbMap->GetNumLocalCoeffs();
        m_nglo     = (m_solnType == eIterativeFull)
                         ? m_assmbMap->GetNumGlobalCoeffs()
                         : m_assmbMap->GetNumGlobalBndCoeffs();
        m_ndir     = m_assmbMap->GetNumGlobalDirBndCoeffs();

        // Memory allocation for global space pointer
        cudaMalloc((void **)&m_gloptr, sizeof(TData) * m_nglo);

        // Memory allocation for assemble pointer
        cudaMalloc((void **)&m_assmbptr, sizeof(int) * m_nloc);
        auto assmbptr = m_assmbMap->GetLocalToGlobalMap().get();
        cudaMemcpy(m_assmbptr, assmbptr, sizeof(int) * m_nloc,
                   cudaMemcpyHostToDevice);

        m_signChange = m_assmbMap->AssemblyMap::GetSignChange();

        if (m_signChange)
        {
            // Memory allocation for sign pointer
            cudaMalloc((void **)&m_signptr, sizeof(TData) * m_nloc);
            auto signptr = m_assmbMap->GetLocalToGlobalSign().get();
            cudaMemcpy(m_signptr, signptr, sizeof(TData) * m_nloc,
                       cudaMemcpyHostToDevice);
        }
    }

    ~OperatorAssmbScatrImpl()
    {
        cudaFree(m_gloptr);
        cudaFree(m_assmbptr);
        if (m_signChange)
        {
            cudaFree(m_signptr);
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out,
               const bool &zeroDir = false)
    {
        Assemble(in, m_gloptr);

        // Zeroing Dirichlet BC
        if (zeroDir)
        {
            cudaMemset(m_gloptr, 0, sizeof(TData) * m_ndir);
        }

        GlobalToLocal(m_gloptr, out);
    }

    void Assemble(Field<TData, FieldState::Coeff> &in, TData *outptr)
    {
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Zero output
        cudaMemset(outptr, 0, sizeof(TData) * m_nglo);

        if (m_solnType == eIterativeFull)
        {
            // Initialise index
            size_t expIdx = 0;
            size_t offset = 0;

            for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
                 ++block_idx)
            {
                // Determine shape and type of the element.
                auto const expPtr = this->m_expansionList->GetExp(expIdx);
                auto nElmts       = in.GetBlocks()[block_idx].num_elements;
                auto ncoeff       = expPtr->GetNcoeffs();

                // Deterime CUDA grid size.
                m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

                if (m_signChange)
                {
                    AssembleKernel<<<m_gridSize, m_blockSize>>>(
                        ncoeff, nElmts, offset, m_assmbptr, m_signptr, inptr,
                        outptr);
                }
                else
                {
                    AssembleKernel<<<m_gridSize, m_blockSize>>>(
                        ncoeff, nElmts, offset, m_assmbptr, inptr, outptr);
                }

                // Increment pointer and index for next element type.
                offset += ncoeff * nElmts;
                expIdx += nElmts;
            }
        }
    }

    void GlobalToLocal(TData *inptr, Field<TData, FieldState::Coeff> &out)
    {
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Zero output
        cudaMemset(outptr, 0, sizeof(TData) * m_nloc);

        if (m_solnType == eIterativeFull)
        {
            // Initialise index
            size_t expIdx = 0;
            size_t offset = 0;

            for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
                 ++block_idx)
            {
                // Determine shape and type of the element.
                auto const expPtr = this->m_expansionList->GetExp(expIdx);
                auto nElmts       = out.GetBlocks()[block_idx].num_elements;
                auto ncoeff       = expPtr->GetNcoeffs();

                // Deterime CUDA grid size.
                m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

                if (m_signChange)
                {
                    GlobalToLocalKernel<<<m_gridSize, m_blockSize>>>(
                        ncoeff, nElmts, offset, m_assmbptr, m_signptr, inptr,
                        outptr);
                }
                else
                {
                    GlobalToLocalKernel<<<m_gridSize, m_blockSize>>>(
                        ncoeff, nElmts, offset, m_assmbptr, inptr, outptr);
                }

                // Increment pointer and index for next element type.
                offset += ncoeff * nElmts;
                expIdx += nElmts;
            }
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorAssmbScatrImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    AssemblyMapCGSharedPtr m_assmbMap;
    GlobalSysSolnType m_solnType;
    TData *m_gloptr;
    bool m_signChange;
    TData *m_signptr;
    int *m_assmbptr;
    size_t m_nloc;
    size_t m_nglo;
    size_t m_ndir;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
