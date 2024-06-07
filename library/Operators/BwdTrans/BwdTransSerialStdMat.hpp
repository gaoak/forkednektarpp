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

#include "Operators/OperatorBwdTrans.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorBwdTransImpl : public OperatorBwdTrans<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorBwdTransImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorBwdTrans<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the elements of expansionList.
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto const expPtr = this->m_expansionList->GetExp(e);

            // Fetch basiskeys of current element.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Copy data to m_matPtr, if necessary.
            if (m_matPtr.find(basisKeys) == m_matPtr.end())
            {
                size_t nqTot = expPtr->GetTotPoints();
                size_t nmTot = expPtr->GetNcoeffs();
                Array<OneD, TData> tmp(nmTot), t;
                // Get BwdTrans matrix.
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, TData>(nmTot * nqTot);
                for (size_t i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    expPtr->GetStdExp()->BwdTrans(tmp, t = matPtr + i * nqTot);
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize pointers.
        auto *inPtr  = in.template GetConstPtr<MemSpace>();
        auto *outPtr = out.template GetPtr<MemSpace>();

        // Initialize index.
        size_t exp_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const nElmts    = inblock.num_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const dimension = expPtr->GetShapeDimension();
            auto const nmTot     = expPtr->GetNcoeffs();
            auto const nqTot     = expPtr->GetTotPoints();

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            auto const &matPtr = m_matPtr[basisKeys];

            // Perform matrix-matrix multiply.
            Blas::Dgemm('N', 'N', nqTot, nElmts, nmTot, 1.0, matPtr.data(),
                        nqTot, inPtr, nmTot, 0.0, outPtr, nqTot);

            // Increment pointer and index for next element type.
            inPtr += inblock.block_size;
            outPtr += outblock.block_size;
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
    std::map<std::vector<LibUtilities::BasisKey>, Array<OneD, TData>> m_matPtr;
};

} // namespace Nektar::Operators::detail
