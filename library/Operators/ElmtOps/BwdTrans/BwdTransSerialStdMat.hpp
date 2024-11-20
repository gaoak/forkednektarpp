///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransSerialStdMat.hpp
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

#include <StdRegions/StdExpansion.h>

#include "Operators/ElmtOps/OperatorBwdTrans.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        const auto dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Loop over the elements of expansionList.
        const auto nTotElmts = this->m_expansionList->GetNumElmts();
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            const auto expPtr = this->m_expansionList->GetExp(e);

            // Fetch basiskeys of current element.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Copy data to m_mat, if necessary.
            if (m_mat.find(basisKeys) == m_mat.end())
            {
                const auto nqTot = expPtr->GetTotPoints();
                const auto nmTot = expPtr->GetNcoeffs();
                Array<OneD, NekDouble> tmp(nmTot), t;
                // Get BwdTrans matrix.
                auto &matPtr = m_mat[basisKeys];
                matPtr       = Array<OneD, TData>(nmTot * nqTot);
                Array<OneD, NekDouble> temp(nmTot * nqTot);
                for (size_t i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    expPtr->GetStdExp()->BwdTrans(tmp, t = temp + i * nqTot);
                }
                // copy temp to matPtr
                for (size_t i = 0; i < nmTot; ++i)
                {
                    for (size_t j = 0; j < nqTot; ++j)
                    {
                        matPtr[j + i * nqTot] = temp[j + i * nqTot];
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        const auto dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            // Block dependent.
            auto &inblock        = in.GetBlocks()[blk];
            auto &outblock       = out.GetBlocks()[blk];
            const auto nElmts    = inblock.GetNumElements();
            const auto nElmtsPad = inblock.GetNumElementsWithPadding();

            // Initialize pointers.
            auto inPtr = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                             ? inblock.template GetPtr<MemSpace, ReadOnly>()
                             : inblock.template GetPtr<MemSpace, ReadWrite>();
            auto outPtr = outblock.template GetPtr<MemSpace, WriteOnly>();

            // Determine shape and type of the element.
            const auto expPtr = this->m_expansionList->GetExp(exp_idx);
            const auto nmTot  = expPtr->GetNcoeffs();
            const auto nqTot  = expPtr->GetTotPoints();

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

            // Fetch basis key for the current element type.
            for (unsigned int d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            const auto &matPtr = m_mat[basisKeys];

            // Perform matrix-matrix multiply.
            Blas::Gemm('N', 'N', nqTot, nElmts, nmTot, 1.0, matPtr.data(),
                       nqTot, inPtr, nmTot, 0.0, outPtr, nqTot);

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
        return std::make_unique<
            OperatorBwdTransImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::map<std::vector<LibUtilities::BasisKey>, Array<OneD, TData>> m_mat;
    static constexpr size_t m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
