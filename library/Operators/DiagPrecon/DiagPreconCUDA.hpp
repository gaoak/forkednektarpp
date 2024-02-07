///////////////////////////////////////////////////////////////////////////////
//
// File: DiagPreconCUDA.hpp
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

#include "Field.hpp"
#include "Operators/AssmbScatr/AssmbScatrCUDA.hpp"
#include "Operators/CUDAMathKernels.cuh"
#include "Operators/DiagPrecon/DiagPreconCUDAKernel.cuh"
#include "Operators/OperatorDiagPrecon.hpp"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorRobBndCond.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename TData>
class OperatorDiagPreconImpl<TData, ImplCUDA> : public OperatorDiagPrecon<TData>
{
public:
    OperatorDiagPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDiagPrecon<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        GlobalSysSolnType solvertype = m_assmbMap->GetGlobalSysSolnType();
        bool isFull = solvertype == eIterativeFull ? true : false;
        m_nGlobal   = (isFull) ? m_assmbMap->GetNumGlobalCoeffs()
                               : m_assmbMap->GetNumGlobalBndCoeffs();
        m_nLocal    = m_assmbMap->GetNumLocalCoeffs();
        m_nDir      = m_assmbMap->GetNumGlobalDirBndCoeffs();

        cudaMalloc((void **)&m_wk, sizeof(TData) * m_nGlobal);
        cudaMalloc((void **)&m_diag, sizeof(TData) * m_nGlobal);

        m_assmbScatr =
            std::static_pointer_cast<OperatorAssmbScatrImpl<TData, ImplCUDA>>(
                AssmbScatr<TData>::create(this->m_expansionList, "CUDA"));
    }

    ~OperatorDiagPreconImpl(void)
    {
        cudaFree(m_wk);
        cudaFree(m_diag);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Deterime CUDA grid size.
        m_gridSize = GetCUDAGridSize(m_nGlobal - m_nDir, m_blockSize);

        m_assmbScatr->Assemble(in, m_wk);

        vdivKernel<<<m_gridSize, m_blockSize>>>(
            m_nGlobal - m_nDir, m_wk + m_nDir, m_diag + m_nDir, m_wk + m_nDir);

        cudaMemset(m_wk, 0, sizeof(TData) * m_nDir);

        m_assmbScatr->GlobalToLocal(m_wk, out);
    }

    void configure(
        const std::shared_ptr<
            OperatorLinear<TData, FieldState::Coeff, FieldState::Coeff>> &op)
    {
        // auto robBCOp = RobBndCond<TData>::create(this->m_expansionList,
        // "CUDA");

        TData *diag;
        cudaMalloc((void **)&diag, sizeof(TData) * m_nLocal);

        // create unit vector field to extract diagonal
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::template create<MemoryRegionCUDA>(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        // create action field to receive column action from unit vector
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::template create<MemoryRegionCUDA>(
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList));

        auto *uvec_ptr =
            unit_vec.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *actn_ptr =
            action.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto diag_ptr = diag;

        size_t expIdx = 0;
        for (size_t block_idx = 0; block_idx < unit_vec.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = unit_vec.GetBlocks()[block_idx].num_elements;
            auto nmTot        = expPtr->GetNcoeffs();

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            for (size_t i = 0; i < nmTot; ++i)
            {
                // set ith term in unit vector to be 1
                SetDiagonalKernel<<<m_gridSize, m_blockSize>>>(nmTot, nElmts, i,
                                                               1.0, uvec_ptr);

                // apply operator to unit vector and store in action field
                op->apply(unit_vec, action);
                // robBCOp->apply(unit_vec, action);

                // copy ith row term from the action field to get ith diagonal
                CopyDiagonalKernel<<<m_gridSize, m_blockSize>>>(
                    nmTot, nElmts, i, actn_ptr, diag_ptr);

                // reset ith term in unit vector to be 0
                SetDiagonalKernel<<<m_gridSize, m_blockSize>>>(nmTot, nElmts, i,
                                                               0.0, uvec_ptr);
            }

            uvec_ptr += unit_vec.GetBlocks()[block_idx].block_size;
            diag_ptr += unit_vec.GetBlocks()[block_idx].block_size;
            actn_ptr += unit_vec.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }

        // Assembly
        Array<OneD, TData> locdiag(m_nLocal);
        Array<OneD, TData> glodiag(m_nGlobal, 0.0);
        cudaMemcpy(locdiag.get(), diag, sizeof(TData) * m_nLocal,
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < m_nLocal; ++i)
        {
            size_t gid1 = m_assmbMap->GetLocalToGlobalMap(i);
            glodiag[gid1] += locdiag[i];
        }
        m_assmbMap->UniversalAssemble(glodiag);

        cudaMemcpy(m_diag, glodiag.get(), sizeof(TData) * m_nGlobal,
                   cudaMemcpyHostToDevice);
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDiagPreconImpl<TData, ImplCUDA>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    std::shared_ptr<OperatorAssmbScatrImpl<TData, ImplCUDA>> m_assmbScatr;
    AssemblyMapCGSharedPtr m_assmbMap;
    TData *m_diag;
    TData *m_wk;
    size_t m_nGlobal;
    size_t m_nLocal;
    size_t m_nDir;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
