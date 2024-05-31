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

#include "Operators/OperatorIProductWRTDerivBase.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
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
        // Copy memory to the host, if necessary and get raw pointers.
        auto *inPtr   = in.template GetConstPtr<MemSpace>();
        auto *outPtr  = out.template GetPtr<MemSpace>();
        auto *wspPtr  = m_wsp.get();
        auto nSize    = in.GetFieldSize();
        auto nStorage = this->m_expansionList->GetTotPoints();

        // Initialize index.
        size_t exp_idx = 0;
        size_t jac_idx = 0;
        size_t df_idx  = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < out.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const nElmts    = outblock.num_elements;

            // Determine shape and type of the element.
            auto const expPtr    = this->m_expansionList->GetExp(exp_idx);
            auto const dimension = expPtr->GetShapeDimension();
            auto const deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            auto const nCoord = expPtr->GetCoordim();
            auto const nqTot  = expPtr->GetTotPoints();
            auto const nmTot  = expPtr->GetNcoeffs();

            // calculate dx/dxi in[0] + dy/dxi in[1] + dz/dxi in[2]
            if (deformed)
            {
                for (size_t d = 0; d < dimension; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, m_derivFac[d].get() + df_idx, 1,
                                inPtr, 1, wspPtr + d * nStorage, 1);
                    for (size_t i = 1; i < nCoord; ++i)
                    {
                        Vmath::Vvtvp(
                            nqTot * nElmts,
                            m_derivFac[d + i * dimension].get() + df_idx, 1,
                            inPtr + i * nSize, 1, wspPtr + d * nStorage, 1,
                            wspPtr + d * nStorage, 1);
                    }
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < dimension; ++d)
                    {
                        Vmath::Smul(nqTot, m_derivFac[d][df_idx + e],
                                    inPtr + e * nqTot, 1,
                                    wspPtr + d * nStorage + e * nqTot, 1);
                        for (size_t i = 1; i < nCoord; ++i)
                        {
                            Vmath::Svtvp(
                                nqTot,
                                m_derivFac[d + i * dimension][df_idx + e],
                                inPtr + i * nSize + e * nqTot, 1,
                                wspPtr + d * nStorage + e * nqTot, 1,
                                wspPtr + d * nStorage + e * nqTot, 1);
                        }
                    }
                }
            }

            // Multiply by jacobian.
            if (deformed)
            {
                for (size_t d = 0; d < dimension; ++d)
                {
                    Vmath::Vmul(nqTot * nElmts, m_jac.get() + jac_idx, 1,
                                wspPtr + d * nStorage, 1, wspPtr + d * nStorage,
                                1);
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t d = 0; d < dimension; ++d)
                    {
                        Vmath::Smul(nqTot, m_jac[jac_idx + e],
                                    wspPtr + d * nStorage + e * nqTot, 1,
                                    wspPtr + d * nStorage + e * nqTot, 1);
                    }
                }
            }

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Fetch matrix.
            auto matPtr = m_matPtr[basisKeys];

            // Matrix products.
            for (size_t d = 0; d < dimension; d++)
            {
                TData alpha = (d == 0 && !APPEND) ? 0.0 : 1.0;
                Blas::Dgemm('N', 'N', nmTot, nElmts, nqTot, 1.0,
                            matPtr[d].get(), nmTot, wspPtr + d * nStorage,
                            nqTot, alpha, outPtr, nmTot);
            }

            // Increment pointer and index for next element type.
            jac_idx += deformed ? nqTot * nElmts : nElmts;
            df_idx += deformed ? nqTot * nElmts : nElmts;

            wspPtr += nqTot * nElmts;

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
            OperatorIProductWRTDerivBaseImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

private:
    Array<OneD, TData> m_jac;
    Array<OneD, TData> m_wsp;
    Array<OneD, Array<OneD, TData>> m_derivFac;

    std::map<std::vector<LibUtilities::BasisKey>,
             Array<OneD, Array<OneD, TData>>>
        m_matPtr;
};

} // namespace Nektar::Operators::detail
