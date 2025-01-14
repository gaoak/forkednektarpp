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
               Field<TData, FieldState::Coeff> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents(),
                 "Number of input and output components differ");

        // Loop over the blocks.
        for (m_blk = 0; m_blk < in.GetBlocks().size(); ++m_blk)
        {
            m_expPtr = this->m_expansionList->GetExp(exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[m_blk];
            auto &outblock = out.GetBlocks()[m_blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            exp_idx += inblock.GetNumElements();
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

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Determine shape and type of the element.
        const auto dimension = m_expPtr->GetShapeDimension();
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        const auto nmTot = m_expPtr->GetNcoeffs();
        const auto nqTot = m_expPtr->GetTotPoints();

        auto nElmts = inblock.GetNumElements();

        auto jacPtr = m_jac[m_blk].template GetPtr<MemSpace, ReadOnly>();

        // Allocate storate.
        if (m_wsp.size() <= m_blk)
        {
            m_wsp.push_back(std::vector<TData>(nElmts * nqTot));
        }

        // Get workspace pointer.
        auto wspptr = m_wsp[m_blk].data();

        // Loop over components.
        for (unsigned int nc = 0; nc < m_nComps; ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                inblock.GetInterleaveWidth(),
                inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                (TData *)inptr);

            // Multiply by jacobian.

            if (deformed)
            {
                for (size_t i = 0; i < nElmts * nqTot; ++i)
                {
                    wspptr[i] = jacPtr[i] * inptr[i];
                }
            }
            else
            {
                for (size_t e = 0; e < nElmts; ++e)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        wspptr[e * nqTot + i] =
                            jacPtr[e] * inptr[e * nqTot + i];
                    }
                }
            }

            // Fetch matrix.
            std::vector<LibUtilities::BasisKey> basisKeys(
                dimension, LibUtilities::NullBasisKey);
            for (unsigned int d = 0; d < dimension; d++)
            {
                basisKeys[d] = m_expPtr->GetBasis(d)->GetBasisKey();
            }
            const auto &matPtr = m_mat[basisKeys];

            // Perform matrix-matrix multiply.
            Blas::Gemm('N', 'N', nmTot, nElmts, nqTot, this->m_scale,
                       matPtr.data(), nmTot, wspptr, nqTot, 0.0, outptr, nmTot);

            inptr += inblock.size();
            outptr += outblock.size();
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

private:
    unsigned int m_blk;
    size_t m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    std::map<std::vector<LibUtilities::BasisKey>, Array<OneD, TData>> m_mat;
    std::vector<MemoryRegion<TData>> m_jac;
    std::vector<std::vector<TData>> m_wsp;
    static constexpr size_t m_implInterleaveWidth = 1;
};

} // namespace Nektar::Operators::detail
