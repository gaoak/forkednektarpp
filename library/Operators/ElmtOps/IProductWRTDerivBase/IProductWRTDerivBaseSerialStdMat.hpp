///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialStdMat.hpp
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
#include "Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorIProductWRTDerivBaseImpl
    : public OperatorIProductWRTDerivBase<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialise jacobian.
        auto locblocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        m_jac = SetJacobian<MemSpace, TData>(expansionList, locblocks,
                                             ExecSpace::alignment);
        m_df  = SetDerivativeFactor<MemSpace, TData>(expansionList, locblocks,
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
                Array<OneD, TData> tmp(nqTot), t;
                // Get IProductWRTDerivBase matrix.
                auto &matPtr = m_mat[basisKeys];
                for (size_t d = 0; d < dimension; ++d)
                {
                    matPtr.push_back(Array<OneD, TData>(nqTot * nmTot));
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        expPtr->GetStdExp()->IProductWRTDerivBase(
                            d, tmp, t = matPtr[d] + i * nmTot);
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (size_t blk = 0; blk < out.GetBlocks().size(); ++blk)
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
            auto outPtr = APPEND
                              ? outblock.template GetPtr<MemSpace, ReadWrite>()
                              : outblock.template GetPtr<MemSpace, WriteOnly>();
            auto dfPtr  = m_df[blk].template GetPtr<MemSpace, ReadOnly>();
            auto jacPtr = m_jac[blk].template GetPtr<MemSpace, ReadOnly>();

            // Determine shape and type of the element.
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nCoord = expPtr->GetCoordim();
            const auto nqTot  = expPtr->GetTotPoints();
            const auto nmTot  = expPtr->GetNcoeffs();
            const auto ndf    = dimension * nCoord;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            if (nCoord > 1)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(), nElmtsPad,
                    inblock.GetNumData(), (TData *)inPtr + inblock.size());
            }
            if (nCoord > 2)
            {
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(), nElmtsPad,
                    inblock.GetNumData(), (TData *)inPtr + 2 * inblock.size());
            }
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                outblock.GetInterleaveWidth(), nElmtsPad, outblock.GetNumData(),
                outPtr);
            inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
            outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);

            // Allocate storate.
            if (m_wsp.size() <= blk)
            {
                m_wsp.push_back(std::vector<TData>(dimension * nElmts * nqTot));
            }

            // Get workspace pointer.
            auto wspPtr = m_wsp[blk].data();

            // Calculate dx/dxi in[0] + dy/dxi in[1] + dz/dxi in[2].
            if (deformed)
            {
                for (size_t d = 0; d < dimension; ++d)
                {
                    Vmath::Vmul(nElmts * nqTot, dfPtr + d, ndf, inPtr, 1,
                                wspPtr + d * nElmts * nqTot, 1);
                    for (size_t i = 1; i < nCoord; ++i)
                    {
                        Vmath::Vvtvp(nElmts * nqTot, dfPtr + d + i * dimension,
                                     ndf, inPtr + i * inblock.size(), 1,
                                     wspPtr + d * nElmts * nqTot, 1,
                                     wspPtr + d * nElmts * nqTot, 1);
                    }
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < dimension; ++d)
                    {
                        Vmath::Smul(nqTot, dfPtr[ndf * e + d],
                                    inPtr + e * nqTot, 1,
                                    wspPtr + d * nElmts * nqTot + e * nqTot, 1);
                        for (size_t i = 1; i < nCoord; ++i)
                        {
                            Vmath::Svtvp(
                                nqTot, dfPtr[ndf * e + d + i * dimension],
                                inPtr + i * inblock.size() + e * nqTot, 1,
                                wspPtr + d * nElmts * nqTot + e * nqTot, 1,
                                wspPtr + d * nElmts * nqTot + e * nqTot, 1);
                        }
                    }
                }
            }

            // Multiply by jacobian.
            if (deformed)
            {
                for (size_t d = 0; d < dimension; ++d)
                {
                    Vmath::Vmul(nElmts * nqTot, jacPtr, 1,
                                wspPtr + d * nElmts * nqTot, 1,
                                wspPtr + d * nElmts * nqTot, 1);
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < dimension; ++d)
                    {
                        Vmath::Smul(nqTot, jacPtr[e],
                                    wspPtr + d * nElmts * nqTot + e * nqTot, 1,
                                    wspPtr + d * nElmts * nqTot + e * nqTot, 1);
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
            for (size_t d = 0; d < dimension; d++)
            {
                TData alpha = (d != 0 || APPEND);
                Blas::Dgemm('N', 'N', nmTot, nElmts, nqTot, 1.0,
                            matPtr[d].data(), nmTot,
                            wspPtr + d * nElmts * nqTot, nqTot, alpha, outPtr,
                            nmTot);
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
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<MemoryRegion<TData>> m_df;
    std::vector<std::vector<TData>> m_wsp;
    std::map<std::vector<LibUtilities::BasisKey>,
             std::vector<Array<OneD, TData>>>
        m_mat;
    static constexpr size_t m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
