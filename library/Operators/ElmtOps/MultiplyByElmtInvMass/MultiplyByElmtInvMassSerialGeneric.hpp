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
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        const bool device_only = true;

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the elements of expansionList.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        std::vector<TData> dmat, scale;
        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto nmTot    = expPtr->GetNcoeffs();
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nElmts = blocks[blk].GetNumElements();

            // Copy inv mass matrix.
            if (deformed)
            {
                dmat.resize(nElmts * nmTot * nmTot);
                auto dmatPtr = dmat.data();
                for (size_t e = 0; e < nElmts; ++e, ++exp_idx)
                {
                    const auto expPtr = this->m_expansionList->GetExp(exp_idx);
                    const auto &InvMass =
                        expPtr->GetLocMatrix(StdRegions::eInvMass);
                    std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, dmatPtr);
                    dmatPtr += nmTot * nmTot;
                }
            }
            else
            {
                scale.resize(nElmts);
                auto scalePtr = scale.data();
                for (size_t e = 0; e < nElmts; ++e, ++exp_idx)
                {
                    const auto expPtr = this->m_expansionList->GetExp(exp_idx);
                    const auto &InvMass =
                        expPtr->GetLocMatrix(StdRegions::eInvMass);

                    // Copy scaling factor.
                    (*scalePtr++) = InvMass->Scale();

                    // Fetch basiskeys of current element.
                    for (size_t d = 0; d < dimension; d++)
                    {
                        basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
                    }

                    // Copy data to m_mat, if necessary.
                    if (m_mat.find(basisKeys) == m_mat.end())
                    {
                        std::vector<TData> matArray(InvMass->GetStorageSize());
                        std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot,
                                    matArray.data());
                        m_mat[basisKeys] =
                            MemoryRegion<TData>::template FromVector<MemSpace,
                                                                     TData>(
                                matArray, ExecSpace::alignment, device_only);
                    }
                }
            }

            m_scale.push_back(
                MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                    scale, ExecSpace::alignment, device_only));
            m_dmat.push_back(
                MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                    dmat, ExecSpace::alignment, device_only));
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize index.
        size_t exp_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &inblock     = in.GetBlocks()[blk];
            auto &outblock    = out.GetBlocks()[blk];
            const auto nElmts = inblock.GetNumElements();

            // Initialize pointers.
            auto inPtr  = inblock.template GetPtr<MemSpace, ReadOnly>();
            auto outPtr = outblock.template GetPtr<MemSpace, WriteOnly>();

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
                auto dmatPtr =
                    m_dmat[blk].template GetPtr<MemSpace, ReadOnly>();
                for (size_t e = 0; e < nElmts; e++)
                {
                    Blas::Gemv('N', nmTot, nmTot, alpha, dmatPtr, nmTot, inPtr,
                               1, beta, outPtr, 1);
                    inPtr += nmTot;
                    outPtr += nmTot;
                    dmatPtr += nmTot * nmTot;
                }
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
                    m_mat[basisKeys].template GetPtr<MemSpace, ReadOnly>();
                const auto scalePtr =
                    m_scale[blk].template GetPtr<MemSpace, ReadOnly>();
                Blas::Gemm('N', 'N', nmTot, nElmts, nmTot, alpha, matPtr, nmTot,
                           inPtr, nmTot, beta, outPtr, nmTot);
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts * nmTot, NEKTAR_LAMBDA(const unsigned int i) {
                        outPtr[i] *= scalePtr[i / nmTot];
                    });
            }

            // Increment index for next element type.
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
    std::map<std::vector<LibUtilities::BasisKey>, MemoryRegion<TData>> m_mat;
    std::vector<MemoryRegion<TData>> m_dmat;
    std::vector<MemoryRegion<TData>> m_scale;
};

} // namespace Nektar::Operators::detail
