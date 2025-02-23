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

#include <MultiRegions/ContField.h>

#include "Operators/PreconOps/DiagPrecon/OperatorDiagPrecon.hpp"

#include "Operators/AssmbScatr/OperatorAssmbScatr.hpp"
#include "Operators/BndCondOps/RobBndCond/OperatorRobBndCond.hpp"
#include "Operators/MathKernels/MathKernels.hpp"

#include "Operators/PreconOps/DiagPrecon/DiagPreconImplKernels.hpp"

using namespace Nektar;
using namespace Nektar::MultiRegions;
using namespace Nektar::SpatialDomains;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
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
        m_nDir      = assmbMap->GetNumGlobalDirBndCoeffs();

        m_glodiag = MemoryRegion<TData>::template Create<MemSpace>(
            "DiagPrecon diag", m_nGlobal, ExecSpace::alignment);

        m_wk = MemoryRegion<TData>::template Create<MemSpace>(
            "DiagPrecon wk", m_nGlobal, ExecSpace::alignment);

        m_assmbScatrOp = OperatorAssmbScatr<TData>::Create(
            this->m_expansionList, ExecSpace::name);

        m_robBCOp = OperatorRobBndCond<TData>::Create(this->m_expansionList,
                                                      ExecSpace::name);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_assmbScatrOp->Assemble(in, m_wk);

        auto diagPtr = m_glodiag.template GetPtr<MemSpace, ReadOnly>();
        auto wkPtr   = m_wk.template GetPtr<MemSpace, ReadWrite>();

        divKernel<ExecSpace, TData>(m_nGlobal - m_nDir, wkPtr + m_nDir,
                                    diagPtr + m_nDir, wkPtr + m_nDir);

        m_wk.template Initialize<MemSpace>(0, m_nDir);

        m_assmbScatrOp->GlobalToLocal(m_wk, out);
    }

    void configure(const std::shared_ptr<
                   OperatorElmt<FieldState::Coeff, FieldState::Coeff, TData>>
                       &op) override
    {
        // Create block attributes.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, this->m_expansionList);

        // Create local diagonal field.
        Field<TData, FieldState::Coeff> locdiag =
            Field<TData, FieldState::Coeff>::template Create<MemSpace>(
                "Local diagonal", blocks, 1, ExecSpace::alignment);

        // Create unit vector field to extract diagonal.
        Field<TData, FieldState::Coeff> unit_vec =
            Field<TData, FieldState::Coeff>::template Create<MemSpace>(
                "DiagPrecon unit vec", blocks, 1, ExecSpace::alignment);

        // Create action field to receive column action from unit vector.
        Field<TData, FieldState::Coeff> action =
            Field<TData, FieldState::Coeff>::template Create<MemSpace>(
                "DiagPrecon action", blocks, 1, ExecSpace::alignment);

        // Initialize field.
        locdiag.template Initialize<MemSpace>(0);
        unit_vec.template Initialize<MemSpace>(0);
        m_glodiag.template Initialize<MemSpace>(0);

        for (unsigned int blk = 0; blk < unit_vec.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &unitvecblock = unit_vec.GetBlocks()[blk];
            auto &actionblock  = action.GetBlocks()[blk];
            auto &diagblock    = locdiag.GetBlocks()[blk];
            const auto nmTot   = unitvecblock.GetNumData();
            const auto nElmts  = unitvecblock.GetNumElements();

            for (unsigned int mode = 0; mode < nmTot; ++mode)
            {
                // Set ith term in unit vector to be 1.
                auto unitptr =
                    unitvecblock.template GetPtr<MemSpace, WriteOnly>();
                SetDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, mode, 1.0,
                                                    unitptr);

                // Apply the operator to unit vector and store in the
                // action field.
                op->apply(unit_vec, action);
                m_robBCOp->apply(unit_vec, action);

                // Copy the ith row term from the action field to get
                // the ith diagonal.
                auto actptr = actionblock.template GetPtr<MemSpace, ReadOnly>();
                auto diagptr = diagblock.template GetPtr<MemSpace, WriteOnly>();
                CopyDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, mode,
                                                     actptr, diagptr);

                // Reset the ith term in the unit vector to be 0.
                unitptr = unitvecblock.template GetPtr<MemSpace, WriteOnly>();
                SetDiagonalKernel<ExecSpace, TData>(nmTot, nElmts, mode, 0.0,
                                                    unitptr);
            }
        }

        // Assembly.
        m_assmbScatrOp->Assemble(locdiag, m_glodiag, false);

        // TODO: Universal assembly on device.
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        if (contfield->GetSession()->GetComm()->GetRowComm()->GetSize() > 1)
        {
            auto glodiagArr = m_glodiag.template ToArray<NekDouble>();

            contfield->GetLocalToGlobalMap()->UniversalAssemble(glodiagArr);

            m_glodiag.template CopyArray<MemSpace, NekDouble>(glodiagArr);
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorDiagPreconImpl<ExecSpace, TData>>(
            expansionList);
    }

protected:
    std::shared_ptr<OperatorAssmbScatr<TData>> m_assmbScatrOp;
    std::shared_ptr<OperatorRobBndCond<TData>> m_robBCOp;

    MemoryRegion<TData> m_glodiag;
    MemoryRegion<TData> m_wk;

    unsigned int m_nGlobal;
    unsigned int m_nDir;
};

} // namespace Nektar::Operators::detail
