///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSerialStdMat.hpp
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

#include "Common/OperatorHelper.hpp"
#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialise jacobian.
        auto locblocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        m_jac = SetJacobian<MemSpace, TData>(expansionList, locblocks,
                                             ExecSpace::alignment);

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Loop over the elements of expansionList.
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
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
                size_t nqTot = expPtr->GetTotPoints();
                size_t nmTot = expPtr->GetNcoeffs();
                Array<OneD, NekDouble> tmp(nqTot), t;
                // Get IProductWRTBase matrix.
                auto &matPtr = m_mat[basisKeys];
                matPtr       = Array<OneD, TData>(nqTot * nmTot);
                Array<OneD, NekDouble> temp(nqTot * nmTot);
                for (size_t i = 0; i < nqTot; ++i)
                {
                    Vmath::Zero(nqTot, tmp, 1);
                    tmp[i] = 1.0;
                    expPtr->GetStdExp()->IProductWRTBase(tmp,
                                                         t = temp + i * nmTot);
                }
                // copy temp to matPtr
                for (size_t i = 0; i < nqTot; ++i)
                {
                    for (size_t j = 0; j < nmTot; ++j)
                    {
                        matPtr[j + i * nmTot] = temp[j + i * nmTot];
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

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
            auto jacPtr = m_jac[blk].template GetPtr<MemSpace, ReadOnly>();

            // Determine shape and type of the element.
            const auto expPtr = this->m_expansionList->GetExp(exp_idx);
            const auto nqTot  = expPtr->GetTotPoints();
            const auto nmTot  = expPtr->GetNcoeffs();

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            if (lambda == 1.0)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    outblock.GetInterleaveWidth(), nElmtsPad,
                    outblock.GetNumData(), outPtr);
            }
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

            // Allocate storate.
            if (m_wsp.size() <= blk)
            {
                m_wsp.push_back(std::vector<TData>(nElmts * nqTot));
            }

            // Get workspace pointer.
            auto wspPtr = m_wsp[blk].data();

            // Multiply by jacobian.
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                for (size_t i = 0; i < nElmts * nqTot; ++i)
                {
                    wspPtr[i] = jacPtr[i] * inPtr[i];
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        wspPtr[e * nqTot + i] =
                            jacPtr[e] * inPtr[e * nqTot + i];
                    }
                }
            }

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            const auto &matPtr = m_mat[basisKeys];

            // Perform matrix-matrix multiply.
            Blas::Gemm('N', 'N', nmTot, nElmts, nqTot, lambda, matPtr.data(),
                       nmTot, wspPtr, nqTot, 0.0, outPtr, nmTot);

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
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::map<std::vector<LibUtilities::BasisKey>, Array<OneD, TData>> m_mat;
    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<std::vector<TData>> m_wsp;
    static constexpr size_t m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
