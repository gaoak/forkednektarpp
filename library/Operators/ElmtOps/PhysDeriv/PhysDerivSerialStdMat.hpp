///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSerialStdMat.hpp
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
#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialise derivative factor.
        auto locblocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        m_df = SetDerivativeFactor<MemSpace, TData>(expansionList, locblocks,
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
                Array<OneD, TData> tmp(nqTot), t;
                // Get deriv matrix.
                auto &matPtr = m_mat[basisKeys];
                matPtr       = std::vector<Array<OneD, TData>>(dimension);
                for (size_t d = 0; d < dimension; ++d)
                {
                    matPtr[d] = Array<OneD, TData>(nqTot * nqTot);
                    for (int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        expPtr->GetStdExp()->PhysDeriv(
                            d, tmp, t = matPtr[d] + i * nqTot);
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Initialize index.
        size_t exp_idx = 0;

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
            auto dfPtr  = m_df[blk].template GetPtr<MemSpace, ReadOnly>();

            // Determine shape and type of the element.
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nCoord  = expPtr->GetCoordim();
            const auto nqTot   = expPtr->GetTotPoints();
            const auto ptsKeys = expPtr->GetPointsKeys();
            const auto ndf     = nCoord * dimension;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(), nElmtsPad, inblock.GetNumData(),
                (TData *)inPtr);
            inblock.SetInterleaveWidth(m_implInterleaveWidth);
            outblock.SetInterleaveWidth(m_implInterleaveWidth);

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Get derivative matrix.
            const auto &matPtr = m_mat[basisKeys];

            // Allocate storate.
            if (m_deriv.size() <= blk)
            {
                m_deriv.push_back(
                    std::vector<TData>(dimension * nqTot * nElmts));
            }

            // Get pointer.
            auto derivPtr = m_deriv[blk].data();

            for (size_t d = 0; d < dimension; ++d)
            {
                // Perform matrix-matrix multiply.
                Blas::Dgemm('N', 'N', nqTot, nElmts, nqTot, 1.0,
                            matPtr[d].data(), nqTot, inPtr, nqTot, 0.0,
                            derivPtr + d * nqTot * nElmts, nqTot);
            }

            if (deformed)
            {
                for (size_t i = 0; i < nCoord; i++)
                {
                    Vmath::Vmul(nqTot * nElmts, dfPtr + i * dimension, ndf,
                                derivPtr, 1, outPtr + i * outblock.size(), 1);
                    for (size_t d = 1; d < dimension; d++)
                    {
                        Vmath::Vvtvp(nqTot * nElmts, dfPtr + i * dimension + d,
                                     ndf, derivPtr + d * nqTot * nElmts, 1,
                                     outPtr + i * outblock.size(), 1,
                                     outPtr + i * outblock.size(), 1);
                    }
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nCoord; i++)
                    {
                        Vmath::Smul(
                            nqTot, dfPtr[i * dimension], derivPtr + e * nqTot,
                            1, outPtr + i * outblock.size() + e * nqTot, 1);
                        for (size_t d = 1; d < dimension; d++)
                        {
                            Vmath::Svtvp(
                                nqTot, dfPtr[i * dimension + d],
                                derivPtr + d * nqTot * nElmts + e * nqTot, 1,
                                outPtr + i * outblock.size() + e * nqTot, 1,
                                outPtr + i * outblock.size() + e * nqTot, 1);
                        }
                    }
                    dfPtr += ndf;
                }
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
            OperatorPhysDerivImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::vector<MemoryRegion<TData>> m_df;
    std::vector<std::vector<TData>> m_deriv;
    std::map<std::vector<LibUtilities::BasisKey>,
             std::vector<Array<OneD, TData>>>
        m_mat;
    static constexpr size_t m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
