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
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialise jacobian.
        auto locblocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        size_t gFacSize = GetGeometricFactorSize(expansionList, locblocks);
        m_jac = SetJacobian<TData>(expansionList, gFacSize, locblocks);
        m_derivFac =
            SetDerivativeFactor<TData>(expansionList, gFacSize, locblocks);

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
                // Get IProductWRTDerivBase matrix.
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, Array<OneD, TData>>(dimension);
                for (size_t d = 0; d < dimension; ++d)
                {
                    matPtr[d] = Array<OneD, TData>(nqTot * nmTot);
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

        // Initialize workspace memory.
        auto nStorage = this->m_expansionList->GetTotPoints();
        m_wsp         = std::vector<TData>(nStorage * dimension);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Copy memory to the host, if necessary and get raw pointers.
        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();
        auto *wspPtr      = m_wsp.data();
        auto *dfPtr       = m_derivFac->data();
        auto *jacPtr      = m_jac->data();

        // Initialize index.
        size_t exp_idx = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            dimension, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            const auto &inblock  = in.GetBlocks()[block_idx];
            const auto &outblock = out.GetBlocks()[block_idx];
            const auto nElmts    = outblock.num_elements;
            const auto nElmtsPad =
                outblock.num_elements + outblock.num_padding_elements;

            // Determine shape and type of the element.
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nCoord = expPtr->GetCoordim();
            const auto nqTot  = expPtr->GetTotPoints();
            const auto nmTot  = expPtr->GetNcoeffs();

            const auto ndf = dimension * nCoord;
            // calculate dx/dxi in[0] + dy/dxi in[1] + dz/dxi in[2]
            if (deformed)
            {
                for (size_t d = 0; d < dimension; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, dfPtr + d, ndf, inPtr, 1,
                                wspPtr + d * nElmts * nqTot, 1);
                    for (size_t i = 1; i < nCoord; ++i)
                    {
                        Vmath::Vvtvp(nqTot * nElmts, dfPtr + d + i * dimension,
                                     ndf, inPtr + i * nElmtsPad * nqTot, 1,
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
                                inPtr + i * nElmtsPad * nqTot + e * nqTot, 1,
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
                    Vmath::Vmul(nqTot * nElmts, jacPtr, 1,
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
            const auto &matPtr = m_matPtr[basisKeys];

            // Perform matrix-matrix multiply.
            for (size_t d = 0; d < dimension; d++)
            {
                TData alpha = (d == 0 && !APPEND) ? 0.0 : 1.0;
                Blas::Dgemm('N', 'N', nmTot, nElmts, nqTot, 1.0,
                            matPtr[d].get(), nmTot, wspPtr + d * nElmts * nqTot,
                            nqTot, alpha, outPtr, nmTot);
            }

            // Increment pointer and index for next element type.
            dfPtr += deformed ? ndf * nElmtsPad * nqTot : ndf * nElmtsPad;
            jacPtr += deformed ? nElmtsPad * nqTot : nElmtsPad;
            wspPtr += dimension * nElmts * nqTot;
            inPtr += inblock.block_size * nCoord;
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
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    std::shared_ptr<std::vector<TData>> m_jac;
    std::shared_ptr<std::vector<TData>> m_derivFac;
    std::vector<TData> m_wsp;
    std::map<std::vector<LibUtilities::BasisKey>,
             Array<OneD, Array<OneD, TData>>>
        m_matPtr;
};

} // namespace Nektar::Operators::detail
