///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceDeviceStdMat.hpp
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

#include "Operators/ElmtOps/Divergence/DivergenceBlockOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class DivergenceBlockOpImpl : public DivergenceBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    DivergenceBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : DivergenceBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nmTot     = exp->GetNcoeffs();
        m_nqTot     = exp->GetTotPoints();

        ASSERTL1(m_dimension == m_coordDim,
                 "Only setup for coordinate dimension to be the same as shape "
                 "dimenions");

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType, ePhysDerivStdMat));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            DerivFactorKey<TData>(block_idx, m_implInterleaveWidth, true));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            DivergenceBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_matptr;
    const TData *m_dfptr;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance();

        ASSERTL0(inblock.GetNumHomoModes() == 1,
                 "Currently only setup for one homogenous plane");

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get static workspace pointer.
        auto derivptr = BlockOperator<TData>::template GetWorkSpace<MemSpace>(
            m_dimension * nelmt * m_nqTot);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        const auto inoffset    = inblock.CompSize();
        const auto derivoffset = m_nqTot * nelmt;

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr + d * inoffset);

            // Perform matrix-matrix multiply.
            NekGemm(handle, "N", "N", m_nqTot, nelmt, m_nqTot, (TData)1.0,
                    m_matptr + d * m_nqTot * m_nqTot, m_nqTot,
                    inptr + d * inoffset, m_nqTot, (TData)0.0,
                    derivptr + d * m_nqTot * nelmt, m_nqTot);
        }

        // Multiply by derivative factor.
        if (m_isDeformed)
        {
            MultiplyByDerivDirFactorKernel<ExecSpace, true, false>(
                0, m_nqTot, m_coordDim, m_dimension, nelmt, derivoffset,
                m_dfptr, derivptr, outptr);

            for (unsigned int d = 1; d < m_dimension; d++)
            {
                MultiplyByDerivDirFactorKernel<ExecSpace, true, true>(
                    d, m_nqTot, m_coordDim, m_dimension, nelmt, derivoffset,
                    m_dfptr, derivptr, outptr);
            }
        }
        else
        {
            MultiplyByDerivDirFactorKernel<ExecSpace, false, false>(
                0, m_nqTot, m_coordDim, m_dimension, nelmt, derivoffset,
                m_dfptr, derivptr, outptr);
            for (unsigned int d = 1; d < m_dimension; d++)
            {
                MultiplyByDerivDirFactorKernel<ExecSpace, false, true>(
                    d, m_nqTot, m_coordDim, m_dimension, nelmt, derivoffset,
                    m_dfptr, derivptr, outptr);
            }
        }

        for (unsigned int k = 0; k < m_coordDim; k++)
        {
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, m_nqTot,
                                      (TData *)inptr + k * inoffset);
        }
        ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth, nelmt,
                                  m_nqTot, (TData *)outptr);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
