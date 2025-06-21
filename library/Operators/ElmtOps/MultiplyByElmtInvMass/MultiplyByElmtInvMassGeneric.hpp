///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassGeneric.hpp
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

#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassDeviceKernels.hpp"
#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassSerialAVXKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class MultiplyByElmtInvMassBlockOpImpl
    : public MultiplyByElmtInvMassBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MultiplyByElmtInvMassBlockOpImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : MultiplyByElmtInvMassBlockOp<TData>(exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }
        m_matptr = dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eMultiplyByElmtInvMassStdMat));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            MultiplyByElmtInvMassBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    MemoryRegion<TData> m_invmass;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance();

        const auto nelmt = inblock.GetNumElements();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        const TData alpha = 1.0;
        const TData beta  = 0.0;
        if (m_isDeformed)
        {
            auto dmatptr =
                this->m_invmass.template GetPtr<MemSpace, ReadOnly>();
            // Loop over components.
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                // Reshape, if necessary.
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr);

                // Perform batched matrix-vector multiply.
                NekGemmStridedBatched(
                    handle, "N", "N", m_nmTot, 1, m_nmTot, alpha, dmatptr,
                    m_nmTot, m_nmTot * m_nmTot, inptr, m_nmTot, m_nmTot, beta,
                    outptr, m_nmTot, m_nmTot, nelmt);

                // Increment pointer.
                inptr += inblock.size();
                outptr += outblock.size();
            }
        }
        else
        {
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

                // Perform matrix-matrix multiply.
                NekGemm(handle, "N", "N", m_nmTot, nelmt, m_nmTot, alpha,
                        m_matptr, m_nmTot, inptr, m_nmTot, beta, outptr,
                        m_nmTot);

                // Divide by Jacobian.
                DivideByJacobianKernel<ExecSpace>(nelmt, m_nmTot, jacptr,
                                                  outptr);

                // Increment pointer.
                inptr += inblock.size();
                outptr += outblock.size();
            }
        }
    }

    void v_SetInvMassMatrix(std::vector<TData> &invmass) override
    {
        this->m_invmass =
            MemoryRegion<TData>::template FromVector<MemSpace, TData>(
                invmass, ExecSpace::alignment);
    }
};

} // namespace Nektar::Operators::detail
