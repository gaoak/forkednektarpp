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

#include "Operators/AssmbScatr/AssmbScatrImpl.hpp"
#include "Operators/BndCondOps/OperatorRobBndCond.hpp"
#include "Operators/Common/OperatorHelper.hpp"
#include "Operators/Field/Field.hpp"
#include "Operators/MathKernels/MathKernels.hpp"
#include "Operators/OperatorAssmbScatr.hpp"
#include "Operators/PreconOps/OperatorDiagPrecon.hpp"

#include "Operators/PreconOps/DiagPrecon/DiagPreconCUDAKernels.cuh"
#include "Operators/PreconOps/DiagPrecon/DiagPreconKernels.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconSYCLKernels.hpp"

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
        auto assmbMap = contfield->GetLocalToGlobalMap();

        bool isFull = assmbMap->GetGlobalSysSolnType() == eIterativeFull;
        m_nGlobal   = (isFull) ? assmbMap->GetNumGlobalCoeffs()
                               : assmbMap->GetNumGlobalBndCoeffs();
        m_nLocal    = assmbMap->GetNumLocalCoeffs();
        m_nDir      = assmbMap->GetNumGlobalDirBndCoeffs();

        m_diag = MemoryRegion<TData>::template create<MemSpace>(
            "DiagPrecon diag", m_nGlobal, ExecSpace::alignment);

        m_wk = MemoryRegion<TData>::template create<MemSpace>(
            "DiagPrecon wk", m_nGlobal, ExecSpace::alignment);

        auto map = assmbMap->GetLocalToGlobalMap();

        m_map = MemoryRegion<int>::template fromArray<MemSpace, int>(
            map, EXECSPACE_MEMORY_REGION_ONLY<MemSpace, ExecSpace>());

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

        m_wk.template initialize<DeviceOnly>(0, m_nDir);

        m_assmbScatrOp->GlobalToLocal(m_wk, out);
    }

    void configure(const std::shared_ptr<
                   OperatorLinear<FieldState::Coeff, FieldState::Coeff, TData>>
                       &op) override
    {
        m_robBCOp =
            RobBndCond<TData>::template create<ExecSpace, Implementation>(
                this->m_expansionList);

        MemoryRegion<TData> diag =
            MemoryRegion<TData>::template create<MemSpace>(
                "DiagPrecon local", m_nLocal, ExecSpace::alignment);

        // create unit vector field to extract diagonal
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::template create<MemSpace>(
                "DiagPrecon unit vec",
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList,
                                   ExecSpace::width),
                1, ExecSpace::alignment);

        // create action field to receive column action from unit vector
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::template create<MemSpace>(
                "DiagPrecon action",
                GetBlockAttributes(FieldState::Coeff, this->m_expansionList,
                                   ExecSpace::width),
                1, ExecSpace::alignment);

        m_diag.template initialize<DeviceOnly>(0);
        diag.template initialize<DeviceOnly>(0);
        unit_vec.template initialize<DeviceOnly>(0);

        size_t offset1 = 0;
        size_t offset2 = 0;
        size_t exp_idx = 0;

        for (const auto &block : unit_vec.GetBlocks())
        {
            // Block dependent
            const auto nElmts    = block.num_elements;
            const auto nElmtsPad = block.num_elmt_groups * block.width;
            const auto nmTot     = block.num_pts;

            for (size_t i = 0; i < nmTot; ++i)
            {
                // Set ith term in unit vector to be 1.
                SetDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, i, offset1,
                                                    1.0, unit_vec);

                // Apply the operator to unit vector and store in the
                // action field.
                op->apply(unit_vec, action);
                m_robBCOp->apply(unit_vec, action);

                // Copy the ith row term from the action field to get
                // the ith diagonal.
                CopyDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, i, offset1,
                                                     offset2, action, diag);

                // Reset the ith term in the unit vector to be 0
                SetDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, i, offset1,
                                                    0.0, unit_vec);
            }

            offset1 += nElmtsPad * nmTot;
            offset2 += nElmts * nmTot;
            exp_idx += nElmts;
        }

        // Assembly
        const int *mapPtr    = m_map.template GetPtr<MemSpace, ReadOnly>();
        const TData *diagPtr = diag.template GetPtr<MemSpace, ReadOnly>();
        TData *glodiagPtr    = m_diag.template GetPtr<MemSpace, WriteOnly>();

        AssembleKernel<ExecSpace>(m_nLocal, mapPtr, diagPtr, glodiagPtr);

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            auto glodiagArr = m_diag.toArray();

            contfield->GetLocalToGlobalMap()->UniversalAssemble(glodiagArr);

            m_diag.template copyArray<MemSpace, TData>(glodiagArr);
        }
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

    MemoryRegion<TData> m_diag;
    MemoryRegion<TData> m_wk;
    MemoryRegion<int> m_map;

    size_t m_nGlobal;
    size_t m_nLocal;
    size_t m_nDir;
};

} // namespace Nektar::Operators::detail
