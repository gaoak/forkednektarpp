///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassSerialGeneric.hpp
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
// Description: Implementation of the elemental inverse mass operator for the
// standard matrix approach.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LocalRegions/Expansion.h>

#include "ElmtOps/OperatorMultiplyByElmtInvMass.hpp"
#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorMultiplyByElmtInvMassImpl
    : public OperatorMultiplyByElmtInvMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMultiplyByElmtInvMassImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMultiplyByElmtInvMass<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();

        // Memory allocation.
        size_t scaleSize = 0;
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            const auto expPtr   = this->m_expansionList->GetExp(e);
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            if (!deformed)
            {
                scaleSize++;
            }
        }

        // Loop over the elements of expansionList.
        std::vector<TData> scale(scaleSize);
        TData *scalePtr = scale.data();
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            // Copy scaling factor.
            const auto expPtr   = this->m_expansionList->GetExp(e);
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto &InvMass = expPtr->GetLocMatrix(StdRegions::eInvMass);
            if (!deformed)
            {
                (*scalePtr++) = InvMass->Scale();
            }
        }

        m_scale = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            scale, ExecSpace::alignment);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize pointers.
        auto *inPtr    = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr   = out.template GetPtr<MemSpace, WriteOnly>();
        auto *scalePtr = m_scale.template GetPtr<MemSpace, ReadOnly>();

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent.
            const auto &inblock  = in.GetBlocks()[block_idx];
            const auto nElmts    = inblock.num_elements;
            const auto nElmtsPad = nElmts + inblock.num_padding_elements;

            // Determine shape and type of the element.
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto nmTot    = expPtr->GetNcoeffs();
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;

            const TData alpha = 1.0;
            const TData beta  = 0.0;
            if (deformed)
            {
                // Perform matrix-vector multiply.
                for (size_t e = 0; e < nElmts; e++)
                {
                    const auto expPtr =
                        this->m_expansionList->GetExp(exp_idx + e);
                    const auto &InvMass =
                        expPtr->GetLocMatrix(StdRegions::eInvMass);
                    const auto matPtr = InvMass->GetRawPtr();
                    Blas::Dgemv('N', nmTot, nmTot, alpha, matPtr, nmTot, inPtr,
                                1, beta, outPtr, 1);
                    inPtr += nmTot;
                    outPtr += nmTot;
                }
            }
            else
            {
                // Perform matrix-matrix multiply.
                const auto &InvMass =
                    expPtr->GetLocMatrix(StdRegions::eInvMass);
                const auto matPtr = InvMass->GetRawPtr();
                Blas::Dgemm('N', 'N', nmTot, nElmts, nmTot, alpha, matPtr,
                            nmTot, inPtr, nmTot, beta, outPtr, nmTot);
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts * nmTot, NEKTAR_LAMBDA(const unsigned int i) {
                        outPtr[i] *= scalePtr[i / nmTot];
                    });
                inPtr += nElmts * nmTot;
                outPtr += nElmts * nmTot;
                scalePtr += nElmts;
            }

            // Increment pointer and index for next element type.
            inPtr += (nElmtsPad - nElmts) * nmTot;
            outPtr += (nElmtsPad - nElmts) * nmTot;
            exp_idx += nElmts;
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMultiplyByElmtInvMassImpl<
            ExecSpace, Implementation, TData>>(expansionList);
    }

private:
    MemoryRegion<TData> m_scale;
};

} // namespace Nektar::Operators::detail
