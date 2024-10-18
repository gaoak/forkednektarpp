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

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorIProductWRTBaseImpl : public OperatorIProductWRTBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialise jacobian.
        auto locblocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        size_t jacSize = GetGeometricFactorSize(expansionList, locblocks);
        m_jac          = SetJacobian<TData>(expansionList, jacSize, locblocks);

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Loop over the elements of expansionList.
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            const auto expPtr = this->m_expansionList->GetExp(e);

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
                Array<OneD, TData> tmp(nqTot), t;
                // Get IProductWRTBase matrix.
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, TData>(nqTot * nmTot);
                for (size_t i = 0; i < nqTot; ++i)
                {
                    Vmath::Zero(nqTot, tmp, 1);
                    tmp[i] = 1.0;
                    expPtr->GetStdExp()->IProductWRTBase(tmp, t = matPtr +
                                                                  i * nmTot);
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Get raw pointers.
        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();

        // Initialize index.
        size_t exp_idx = 0;
        size_t jac_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            const auto &inblock  = in.GetBlocks()[block_idx];
            const auto &outblock = out.GetBlocks()[block_idx];
            const auto nElmts    = inblock.num_elements;
            const auto nPadElmts = inblock.num_padding_elements;

            // Determine shape and type of the element.
            const auto expPtr = this->m_expansionList->GetExp(exp_idx);
            const auto nqTot  = expPtr->GetTotPoints();
            const auto nmTot  = expPtr->GetNcoeffs();

            // Multiply by jacobian.
            if (m_wspsize < inblock.block_size)
            {
                m_wspsize = inblock.block_size;

                m_wsp = MemoryRegion<TData>::template create<MemSpace>(
                    inblock.block_size, ExecSpace::alignment);
            }

            auto wspptr = m_wsp.template GetPtr<MemSpace, WriteOnly>();

            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                for (size_t i = 0; i < inblock.block_size; ++i)
                {
                    wspptr[i] = (*m_jac)[jac_idx++] * inPtr[i];
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        wspptr[e * nqTot + i] =
                            (*m_jac)[jac_idx] * inPtr[e * nqTot + i];
                    }

                    jac_idx++;
                }
                jac_idx += nPadElmts;
            }

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            const auto &matPtr = m_matPtr[basisKeys];

            // Perform matrix-matrix multiply.
            Blas::Dgemm('N', 'N', nmTot, nElmts, nqTot, lambda, matPtr.data(),
                        nmTot, wspptr, nqTot, 0.0, outPtr, nmTot);

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
            OperatorIProductWRTBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::shared_ptr<std::vector<TData>> m_jac;
    std::map<std::vector<LibUtilities::BasisKey>, Array<OneD, TData>> m_matPtr;
    MemoryRegion<TData> m_wsp;
    size_t m_wspsize = 0;
};

} // namespace Nektar::Operators::detail
