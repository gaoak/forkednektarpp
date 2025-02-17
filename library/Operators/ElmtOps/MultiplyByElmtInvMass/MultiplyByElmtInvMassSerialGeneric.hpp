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
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorMultiplyByElmtInvMassImpl
    : public BlockOperatorMultiplyByElmtInvMass<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorMultiplyByElmtInvMassImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorMultiplyByElmtInvMass<TData>(exp, dataWarehouse)
    {
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Determine shape and type of the element.
        const auto shapeType = this->m_exp->DetShapeType();
        const auto dimension = this->m_exp->GetShapeDimension();
        const auto nmTot     = this->m_exp->GetNcoeffs();
        const auto deformed  = this->m_exp->GetMetricInfo()->GetGtype() ==
                              SpatialDomains::eDeformed;
        const auto nElmts        = inblock.GetNumElements();
        const auto nElmtsWithPad = inblock.GetNumElementsWithPadding();

        const TData alpha = 1.0;
        const TData beta  = 0.0;
        if (deformed)
        {
            // Loop over components.
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                // Reshape, if necessary.
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr);

                // Perform matrix-vector multiply.
                auto dmatptr =
                    this->m_invmass.template GetPtr<MemSpace, ReadOnly>();

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
            for (unsigned int d = 0; d < dimension; d++)
            {
                basisKeys[d] = this->m_exp->GetBasis(d)->GetBasisKey();
            }
            auto matptr = this->m_dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, shapeType,
                                 eMultiplyByElmtInvMassStdMat));

            // Fetch jacobian.
            auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
                JacobianKey<TData>(inblock.GetExpIdx(), 1,
                                   inblock.GetNumElements()));

            // Loop over components.
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                // Reshape, if necessary.
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr);

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

    void v_SetInvMassMatrix(std::vector<TData> &invmass) override
    {
        const bool device_only = true;

        this->m_invmass =
            MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                invmass, ExecSpace::alignment, device_only);
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<BlockOperatorMultiplyByElmtInvMassImpl<
            ExecSpace, Implementation, TData>>(exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;
    MemoryRegion<TData> m_invmass;
};

} // namespace Nektar::Operators::detail
