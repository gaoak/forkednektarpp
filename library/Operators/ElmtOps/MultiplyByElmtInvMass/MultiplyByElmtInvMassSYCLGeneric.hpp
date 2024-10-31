///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassSYCLGeneric.hpp
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

#include "Operators/ElmtOps/OperatorMultiplyByElmtInvMass.hpp"
#include "Operators/LoopExecution/LoopExecution.hpp"
#include "Operators/Utils/OneMKL.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
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
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        const bool device_only = true;

        // Memory allocation.
        size_t dmatSize  = 0;
        size_t scaleSize = 0;
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            const auto expPtr   = this->m_expansionList->GetExp(e);
            const auto nmTot    = expPtr->GetNcoeffs();
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            if (deformed)
            {
                dmatSize += nmTot * nmTot;
            }
            else
            {
                scaleSize++;
            }
        }

        // Loop over the elements of expansionList.
        std::vector<TData> scale(scaleSize);
        std::vector<TData> dmat(dmatSize);
        TData *scalePtr = scale.data();
        TData *dmatPtr  = dmat.data();
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            const auto expPtr   = this->m_expansionList->GetExp(e);
            const auto nmTot    = expPtr->GetNcoeffs();
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto &InvMass = expPtr->GetLocMatrix(StdRegions::eInvMass);

            // Copy inv mass matrix.
            if (deformed)
            {
                std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, dmatPtr);
                dmatPtr += nmTot * nmTot;
            }
            else
            {
                // Copy scaling factor.
                (*scalePtr++) = InvMass->Scale();

                // Fetch basiskeys of current element.
                for (size_t d = 0; d < dimension; d++)
                {
                    basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                }

                // Copy data to m_matPtr, if necessary.
                if (m_matPtr.find(basisKeys) == m_matPtr.end())
                {
                    std::vector<TData> matArray(InvMass->GetStorageSize());
                    std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot,
                                matArray.data());
                    m_matPtr[basisKeys] =
                        MemoryRegion<TData>::template fromVector<MemSpace,
                                                                 TData>(
                            matArray, ExecSpace::alignment, device_only);
                }
            }
        }

        m_scale = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            scale, ExecSpace::alignment, device_only);
        m_dmat = MemoryRegion<TData>::template fromVector<MemSpace, TData>(
            dmat, ExecSpace::alignment, device_only);
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Get SYCL queue.
        auto queue = SYCLQueue::GetInstance();

        // Initialize pointers.
        auto *inPtr    = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr   = out.template GetPtr<MemSpace, WriteOnly>();
        auto *scalePtr = m_scale.template GetPtr<MemSpace, ReadOnly>();
        auto *dmatPtr  = m_dmat.template GetPtr<MemSpace, ReadOnly>();

        // Initialize index.
        size_t exp_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

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
                // Perform batched matrix-vector multiply.
                OneMKL::dgemm_batch(queue, "N", "N", nmTot, 1, nmTot, alpha,
                                    dmatPtr, nmTot, nmTot * nmTot, inPtr, nmTot,
                                    nmTot, beta, outPtr, nmTot, nmTot, nElmts);
                dmatPtr += nElmts * nmTot * nmTot;
            }
            else
            {
                // Fetch basis key for the current element type.
                for (size_t d = 0; d < dimension; d++)
                {
                    basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                }

                // Perform matrix-matrix multiply.
                const auto matPtr =
                    m_matPtr[basisKeys].template GetPtr<MemSpace, ReadOnly>();
                OneMKL::dgemm(queue, "N", "N", nmTot, nElmts, nmTot, alpha,
                              matPtr, nmTot, inPtr, nmTot, beta, outPtr, nmTot);
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts * nmTot, NEKTAR_LAMBDA(const unsigned int i) {
                        outPtr[i] *= scalePtr[i / nmTot];
                    });
                scalePtr += nElmts;
            }

            // Increment pointer and index for next element type.
            inPtr += nElmtsPad * nmTot;
            outPtr += nElmtsPad * nmTot;
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
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<TData>> m_matPtr;
    MemoryRegion<TData> m_dmat;
    MemoryRegion<TData> m_scale;
};

} // namespace Nektar::Operators::detail
