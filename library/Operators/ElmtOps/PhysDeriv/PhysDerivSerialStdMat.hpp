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

#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"

namespace Nektar::Operators::detail
{

// Standard matrix implementation
template <typename ExecSpace, typename Implementation, typename TData,
          typename = typename std::enable_if<
              std::is_same<ExecSpace, NektarSpaces::Serial>::value &&
              std::is_same<Implementation, Operators::StdMat>::value>::type>
class OperatorPhysDerivImpl : public OperatorPhysDeriv<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysDerivImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysDeriv<TData>(expansionList)
    {
        size_t nTotElmts = this->m_expansionList->GetNumElmts();
        size_t dimension = this->m_expansionList->GetShapeDimension();

        // Initialise derivative factor.
        size_t dfSize = Operator<TData>::GetGeometricFactorSize();
        m_derivFac    = Operator<TData>::SetDerivativeFactor(dfSize);

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

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
                auto &matPtr = m_matPtr[basisKeys];
                matPtr       = Array<OneD, Array<OneD, TData>>(dimension);
                Array<OneD, NekDouble> tmp(nqTot), t;
                for (size_t d = 0; d < dimension; ++d)
                {
                    // Get deriv matrix.
                    matPtr[d] = Array<OneD, TData>(nqTot * nqTot);
                    for (int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        // TODO: Use redesign kernel
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
        // Initialize pointers.
        const auto *inPtr = in.template GetPtr<MemSpace, ReadOnly>();
        auto *outPtr      = out.template GetPtr<MemSpace, WriteOnly>();
        auto nSize        = out.GetFieldSize();

        // Initialize index.
        size_t exp_idx = 0;
        size_t df_idx  = 0;

        // Initialize basiskey.
        std::vector<LibUtilities::BasisKey> basisKeys(
            3, LibUtilities::NullBasisKey);

        for (const auto &block : in.GetBlocks())
        {
            // Block dependent
            const auto nElmts    = block.num_elements;
            const auto nPadElmts = block.num_padding_elements;

            // Determine shape and type of the element.
            const auto expPtr    = this->m_expansionList->GetExp(exp_idx);
            const auto dimension = expPtr->GetShapeDimension();
            const auto deformed  = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nCoord  = expPtr->GetCoordim();
            const auto nqTot   = expPtr->GetTotPoints();
            const auto ptsKeys = expPtr->GetPointsKeys();

            Array<OneD, Array<OneD, TData>> deriv(dimension);

            // Fetch basis key for the current element type.
            for (size_t d = 0; d < expPtr->GetShapeDimension(); d++)
            {
                basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
            }

            // Get derivative matrix.
            const auto &matPtr = m_matPtr[basisKeys];

            for (size_t d = 0; d < dimension; ++d)
            {
                // Perform matrix-matrix multiply.
                deriv[d] = Array<OneD, TData>(nqTot * nElmts);
                Blas::Dgemm('N', 'N', nqTot, nElmts, nqTot, 1.0,
                            matPtr[d].get(), nqTot, inPtr, nqTot, 0.0,
                            deriv[d].get(), nqTot);
            }

            if (deformed)
            {
                for (size_t i = 0; i < nCoord; i++)
                {
                    Vmath::Vmul(nqTot * nElmts,
                                m_derivFac[i * dimension].get() + df_idx, 1,
                                deriv[0].get(), 1, outPtr + i * nSize, 1);
                    for (size_t d = 1; d < dimension; d++)
                    {
                        Vmath::Vvtvp(nqTot * nElmts,
                                     m_derivFac[i * dimension + d].get() +
                                         df_idx,
                                     1, deriv[d].get(), 1, outPtr + i * nSize,
                                     1, outPtr + i * nSize, 1);
                    }
                }

                outPtr += (nPadElmts + nElmts) * nqTot;
                df_idx += nqTot * nElmts;
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nCoord; i++)
                    {
                        Vmath::Smul(nqTot,
                                    m_derivFac[i * dimension][df_idx + e],
                                    deriv[0].get() + e * nqTot, 1,
                                    outPtr + i * nSize, 1);
                        for (size_t d = 1; d < dimension; d++)
                        {
                            Vmath::Svtvp(
                                nqTot,
                                m_derivFac[i * dimension + d][df_idx + e],
                                deriv[d].get() + e * nqTot, 1,
                                outPtr + i * nSize, 1, outPtr + i * nSize, 1);
                        }
                    }

                    outPtr += nqTot;
                }

                outPtr += nPadElmts * nqTot;
                df_idx += nElmts;
            }

            inPtr += (nPadElmts + nElmts) * nqTot;
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
    Array<OneD, Array<OneD, TData>> m_derivFac;
    std::map<std::vector<LibUtilities::BasisKey>,
             Array<OneD, Array<OneD, TData>>>
        m_matPtr;
};

} // namespace Nektar::Operators::detail
