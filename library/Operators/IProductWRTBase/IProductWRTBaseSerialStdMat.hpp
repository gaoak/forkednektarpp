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

#include "Operators/OperatorIProductWRTBase.hpp"

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
        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        auto *inPtr  = in.template GetConstPtr<MemSpace>();
        auto *outPtr = out.template GetPtr<MemSpace>();

        size_t exp_idx = 0;
        size_t jac_idx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Block dependent
            auto const &inblock  = in.GetBlocks()[block_idx];
            auto const &outblock = out.GetBlocks()[block_idx];
            auto const nElmts    = inblock.num_elements;

            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(exp_idx);
            auto const nqTot  = expPtr->GetTotPoints();
            auto const nmTot  = expPtr->GetNcoeffs();

            Nektar::StdRegions::StdMatrixKey key(
                StdRegions::eIProductWRTBase, expPtr->DetShapeType(), *expPtr);

            // This is the B^{T} matrix
            auto const matPtr = expPtr->GetStdMatrix(key);

            Array<OneD, TData> wsp(nqTot * nElmts, 0.0);
            if (expPtr->GetMetricInfo()->GetGtype() ==
                SpatialDomains::eDeformed)
            {
                for (size_t i = 0; i < nElmts * nqTot; ++i)
                {
                    wsp[i] = m_jac[jac_idx++] * inPtr[i];
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        wsp[e * nqTot + i] =
                            m_jac[jac_idx] * inPtr[e * nqTot + i];
                    }

                    jac_idx++;
                }
            }

            Blas::Dgemm('N', 'N', matPtr->GetRows(), nElmts,
                        matPtr->GetColumns(), lambda, matPtr->GetRawPtr(),
                        matPtr->GetRows(), wsp.get(), nqTot, 0.0, outPtr,
                        nmTot);

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
    Array<OneD, TData> m_jac;
};

} // namespace Nektar::Operators::detail
