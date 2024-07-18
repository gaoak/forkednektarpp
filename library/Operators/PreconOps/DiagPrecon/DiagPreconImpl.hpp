///////////////////////////////////////////////////////////////////////////////
//
// File: DiagPreconImpl.hpp
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

#include "Operators/PreconOps/DiagPrecon/DiagPreconKernels.cuh"
#include "Operators/PreconOps/DiagPrecon/DiagPreconKernels.hpp"
#include "Operators/PreconOps/OperatorDiagPrecon.hpp"

#include "Operators/BndCondOps/OperatorRobBndCond.hpp"
#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/Field/Field.hpp"
#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/OperatorAssmbScatr.hpp"

#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorDiagPreconImpl : public OperatorDiagPrecon<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorDiagPreconImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorDiagPrecon<TData>(expansionList)
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        m_assmbMap = contfield->GetLocalToGlobalMap();

        GlobalSysSolnType solnType = m_assmbMap->GetGlobalSysSolnType();
        bool isFull                = solnType == eIterativeFull ? true : false;
        m_nGlobal                  = (isFull) ? m_assmbMap->GetNumGlobalCoeffs()
                                              : m_assmbMap->GetNumGlobalBndCoeffs();
        m_nLocal                   = m_assmbMap->GetNumLocalCoeffs();
        m_nDir                     = m_assmbMap->GetNumGlobalDirBndCoeffs();

        m_diag = MemoryRegion<TData>::template create<MemSpace>(
            "DiagPrecon diag", m_nGlobal,
            EXECSPACE_MEMORY_REGION_ONLY<MemSpace>());
        m_wk = MemoryRegion<TData>::template create<MemSpace>("DiagPrecon wk",
                                                              m_nGlobal);

        m_assmbScatrOp =
            AssmbScatr<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_assmbScatrOp->Assemble(in, m_wk);

        const TData *diagPtr = m_diag.template GetPtr<MemSpace, ReadOnly>();
        TData *wkPtr         = m_wk.template GetPtr<MemSpace, ReadWrite>();

        divKernel<ExecSpace, TData>(m_nGlobal - m_nDir, wkPtr + m_nDir,
                                    diagPtr + m_nDir, wkPtr + m_nDir);

        m_wk.initialize(0, m_nDir);

        m_assmbScatrOp->GlobalToLocal(m_wk, out);
    }

    void configure(const std::shared_ptr<
                   OperatorLinear<FieldState::Coeff, FieldState::Coeff, TData>>
                       &op) override
    {
        m_robBCOp =
            RobBndCond<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);

        MemoryRegion<TData> diagMR =
            MemoryRegion<TData>::template create<MemSpace>("DiagPrecon local",
                                                           m_nLocal);

        // create unit vector field to extract diagonal
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::template create<MemSpace>(
                "DiagPrecon unit vec",
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList,
                                   vec_t::width),
                1, vec_t::alignment);

        // create action field to receive column action from unit vector
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::template create<MemSpace>(
                "DiagPrecon action",
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList,
                                   vec_t::width),
                1, vec_t::alignment);

        diagMR.template initialize<HostDevice>(0);
        unit_vec.template initialize<HostDevice>(0);

        size_t offset1 = 0;
        size_t offset2 = 0;
        size_t exp_idx = 0;

        for (auto const &block : unit_vec.GetBlocks())
        {
            // Block dependent
            auto const nElmts    = block.num_elements;
            auto const nPadElmts = block.num_padding_elements;
            auto const nmTot     = block.num_pts;

            for (size_t i = 0; i < nmTot; ++i)
            {
                // Set ith term in unit vector to be 1.
                SetDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, i, offset1,
                                                    1.0, unit_vec);

                // Apply the operator to unit vector and store in the
                // action field.
                op->apply(unit_vec, action);

                if constexpr (std::is_same<ExecSpace,
                                           NektarSpaces::Serial>::value)
                {
                    m_robBCOp->apply(unit_vec, action);
                }

                // Copy the ith row term from the action field to get
                // the ith diagonal.
                CopyDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, i, offset1,
                                                     offset2, action, diagMR);

                // Reset the ith term in the unit vector to be 0
                SetDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, i, offset1,
                                                    0.0, unit_vec);
            }

            offset1 += (nElmts + nPadElmts) * nmTot;
            offset2 += nElmts * nmTot;
            exp_idx += nElmts;
        }

        // Assembly
        const TData *diagHost =
            diagMR.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        Array<OneD, TData> glodiag(m_nGlobal, 0.0);

        for (size_t i = 0; i < m_nLocal; ++i)
        {
            size_t gid1 = m_assmbMap->GetLocalToGlobalMap(i);
            glodiag[gid1] += diagHost[i];
        }

        m_assmbMap->UniversalAssemble(glodiag);

        m_diag.template copyArray<MemSpace, TData>(glodiag);
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorDiagPreconImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatrOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBCOp;

    AssemblyMapCGSharedPtr m_assmbMap;

    MemoryRegion<TData> m_diag;
    MemoryRegion<TData> m_wk;

    size_t m_nGlobal;
    size_t m_nLocal;
    size_t m_nDir;
};

} // namespace Nektar::Operators::detail
