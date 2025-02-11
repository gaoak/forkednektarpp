///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassSerialGeneric.hpp
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
// Description: Implementation of the elemental inverse mass operator for the
// standard matrix approach.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LocalRegions/Expansion.h>

#include "Operators/ElmtOps/OperatorMultiplyByElmtInvMass.hpp"
#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class OperatorMultiplyByElmtInvMassImpl
    : public OperatorMultiplyByElmtInvMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMultiplyByElmtInvMassImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMultiplyByElmtInvMass<TData>(expansionList)
    {
        const bool device_only = true;

        // Initialize index.
        size_t exp_idx = 0;

        // Loop over the elements of expansionList.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Phys, expansionList);
        std::vector<TData> dmat;
        for (size_t blk = 0; blk < blocks.size(); ++blk)
        {
            const auto expPtr   = this->m_expansionList->GetExp(exp_idx);
            const auto nmTot    = expPtr->GetNcoeffs();
            const auto deformed = expPtr->GetMetricInfo()->GetGtype() ==
                                  SpatialDomains::eDeformed;
            const auto nElmts = blocks[blk].GetNumElements();

            // Copy inv mass matrix.
            if (deformed)
            {
                dmat.resize(nElmts * nmTot * nmTot);
                auto dmatptr = dmat.data();
                for (size_t e = 0; e < nElmts; ++e, ++exp_idx)
                {
                    const auto expPtr = this->m_expansionList->GetExp(exp_idx);
                    const auto &InvMass =
                        expPtr->GetLocMatrix(StdRegions::eInvMass);
                    std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, dmatptr);
                    dmatptr += nmTot * nmTot;
                }
            }
            else
            {
                exp_idx += nElmts;
            }

            m_dmat.push_back(
                MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                    dmat, ExecSpace::alignment, device_only));
        }
    }

    void apply(Field<TData, FieldState::Coeff> &in,
               Field<TData, FieldState::Coeff> &out) override
    {
        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents(),
                 "Number of input and output components differ");

        // Initialize index.
        m_exp_idx = 0;

        // Loop over the blocks.
        for (m_blk = 0; m_blk < in.GetBlocks().size(); ++m_blk)
        {
            m_expPtr = this->m_expansionList->GetExp(m_exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[m_blk];
            auto &outblock = out.GetBlocks()[m_blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            m_exp_idx += inblock.GetNumElements();
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMultiplyByElmtInvMassImpl<
            ExecSpace, Implementation, TData>>(expansionList);
    }

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Determine shape and type of the element.
        const auto shapeType = m_expPtr->DetShapeType();
        const auto dimension = m_expPtr->GetShapeDimension();
        const auto nmTot     = m_expPtr->GetNcoeffs();
        const auto deformed =
            m_expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        const auto nElmts        = inblock.GetNumElements();
        const auto nElmtsWithPad = inblock.GetNumElementsWithPadding();

        const TData alpha = 1.0;
        const TData beta  = 0.0;
        if (deformed)
        {
            // Loop over components.
            for (size_t nc = 0; nc < m_nComps; ++nc)
            {
                // Perform matrix-vector multiply.
                auto dmatptr =
                    m_dmat[m_blk].template GetPtr<MemSpace, ReadOnly>();

                unsigned int e = 0;
                for (; e < nElmts; e++)
                {
                    Blas::Gemv('N', nmTot, nmTot, alpha, dmatptr, nmTot, inptr,
                               1, beta, outptr, 1);
                    inptr += nmTot;
                    outptr += nmTot;
                    dmatptr += nmTot * nmTot;
                }
                inptr += nmTot * (nElmtsWithPad - e);
                outptr += nmTot * (nElmtsWithPad - e);
            }
        }
        else
        {
            // Fetch basis key for the current element type.
            std::vector<LibUtilities::BasisKey> basisKeys(
                dimension, LibUtilities::NullBasisKey);
            for (size_t d = 0; d < dimension; d++)
            {
                basisKeys[d] = m_expPtr->GetBasis(d)->GetBasisKey();
            }
            auto matptr = this->m_dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, shapeType,
                                 eMultiplyByElmtInvMassStdMat));

            // Fetch jacobian.
            auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(m_exp_idx, 1, inblock.GetNumElements()));

            // Loop over components.
            for (size_t nc = 0; nc < m_nComps; ++nc)
            {
                Blas::Gemm('N', 'N', nmTot, nElmts, nmTot, alpha, matptr, nmTot,
                           inptr, nmTot, beta, outptr, nmTot);
                Nektar::parallel_for<ExecSpace>(
                    0, nElmts * nmTot, NEKTAR_LAMBDA(const unsigned int i) {
                        outptr[i] /= jacptr[i / nmTot];
                    });
                inptr += inblock.size();
                outptr += outblock.size();
            }
        }
    }

private:
    unsigned int m_exp_idx;
    unsigned int m_blk;
    unsigned int m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;

    std::vector<MemoryRegion<TData>> m_dmat;
};

} // namespace Nektar::Operators::detail
