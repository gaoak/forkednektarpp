///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseStdMat.hpp
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

#include "Operators/OperatorIProductWRTDerivBase.hpp"

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorIProductWRTDerivBaseImpl<TData, ImplStdMat>
    : public OperatorIProductWRTDerivBase<TData>
{
public:
    OperatorIProductWRTDerivBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTDerivBase<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t nDim      = this->m_expansionList->GetShapeDimension();

        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
        m_derivFac     = Operator<TData>::SetDerivativeFactor(jacSize);

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the elements of expansionList.
        for (size_t e = 0; e < nTotElmts; ++e)
        {
            auto const expPtr = this->m_expansionList->GetExp(e);

            // Fetch basiskeys of current element.
            for (size_t d = 0; d < nDim; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Copy data to m_matPtr, if necessary.
            if (m_matPtr.find(basisKeys) == m_matPtr.end())
            {
                size_t nqTot = expPtr->GetTotPoints();
                size_t nmTot = expPtr->GetNcoeffs();
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, Array<OneD, TData>>(nDim);
                Array<OneD, TData> tmp(nqTot), t;
                for (size_t d = 0; d < nDim; ++d)
                {
                    // Get IProductWRTDerivBase matrix.
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
        m_wsp         = Array<OneD, TData>(nStorage * nDim);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               bool APPEND = false) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto *inptr   = in.GetStorage().GetCPUPtr();
        auto *outptr  = out.GetStorage().GetCPUPtr();
        auto *wspptr  = m_wsp.get();
        auto nSize    = in.GetFieldSize();
        auto nStorage = this->m_expansionList->GetTotPoints();

        // Initialize index.
        size_t expIdx = 0;
        size_t jacIdx = 0;
        size_t dfIdx  = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = out.GetBlocks()[block_idx].num_elements;
            auto nDim         = expPtr->GetShapeDimension();
            auto nCoord       = expPtr->GetCoordim();
            auto nqTot        = expPtr->GetTotPoints();
            auto nmTot        = expPtr->GetNcoeffs();
            auto deformed     = expPtr->GetMetricInfo()->GetGtype() ==
                            SpatialDomains::eDeformed;

            // calculate dx/dxi in[0] + dy/dxi in[1] + dz/dxi in[2]
            if (deformed)
            {
                for (size_t d = 0; d < nDim; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, m_derivFac[d].get() + dfIdx, 1,
                                inptr, 1, wspptr + d * nStorage, 1);
                    for (size_t i = 1; i < nCoord; ++i)
                    {
                        Vmath::Vvtvp(nqTot * nElmts,
                                     m_derivFac[d + i * nDim].get() + dfIdx, 1,
                                     inptr + i * nSize, 1,
                                     wspptr + d * nStorage, 1,
                                     wspptr + d * nStorage, 1);
                    }
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < nDim; ++d)
                    {
                        Vmath::Smul(nqTot, m_derivFac[d][dfIdx + e],
                                    inptr + e * nqTot, 1,
                                    wspptr + d * nStorage + e * nqTot, 1);
                        for (size_t i = 1; i < nCoord; ++i)
                        {
                            Vmath::Svtvp(nqTot,
                                         m_derivFac[d + i * nDim][dfIdx + e],
                                         inptr + i * nSize + e * nqTot, 1,
                                         wspptr + d * nStorage + e * nqTot, 1,
                                         wspptr + d * nStorage + e * nqTot, 1);
                        }
                    }
                }
            }

            // Multiply by jacobian.
            if (deformed)
            {
                for (size_t d = 0; d < nDim; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, m_jac.get() + jacIdx, 1,
                                wspptr + d * nStorage, 1, wspptr + d * nStorage,
                                1);
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < nDim; ++d)
                    {
                        Vmath::Smul(nqTot, m_jac[jacIdx + e],
                                    wspptr + d * nStorage + e * nqTot, 1,
                                    wspptr + d * nStorage + e * nqTot, 1);
                    }
                }
            }

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < nDim; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            auto matPtr = m_matPtr[basisKeys];

            // Matrix products.
            for (size_t d = 0; d < nDim; d++)
            {
                TData alpha = (d == 0 && !APPEND) ? 0.0 : 1.0;
                Blas::Dgemm('N', 'N', nmTot, nElmts, nqTot, 1.0,
                            matPtr[d].get(), nmTot, wspptr + d * nStorage,
                            nqTot, alpha, outptr, nmTot);
            }
            jacIdx += deformed ? nqTot * nElmts : nElmts;
            dfIdx += deformed ? nqTot * nElmts : nElmts;
            wspptr += nqTot * nElmts;
            inptr += in.GetBlocks()[block_idx].block_size;
            outptr += out.GetBlocks()[block_idx].block_size;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorIProductWRTDerivBaseImpl<TData, ImplStdMat>>(expansionList);
    }

    static std::string className;

private:
    Array<OneD, TData> m_jac;
    Array<OneD, Array<OneD, TData>> m_derivFac;
    std::map<std::vector<LibUtilities::BasisKey>,
             Array<OneD, Array<OneD, TData>>>
        m_matPtr;
    Array<OneD, TData> m_wsp;
};

} // namespace Nektar::Operators::detail
