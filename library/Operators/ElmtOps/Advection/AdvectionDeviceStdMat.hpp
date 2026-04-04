///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDeviceStdMat.hpp
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

#include "Operators/ElmtOps/Advection/AdvectionBlockOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Advection/AdvectionDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionBlockOpImpl : public AdvectionBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvectionBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
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
            AdvectionBlockOpImpl<ExecSpace, Implementation, TData>>(
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
    TData *m_advVel;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance();

        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nelmtTot =
            inblock.GetNumElementsWithPadding() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Get static workspace pointer.
        auto derivptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                m_coordDim * nelmtTot * m_nqTot);

        // Loop over components.
        const auto advelsize =
            m_nqTot * inblock.GetNumElmtGroups(m_implInterleaveWidth);
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            auto advptr = this->m_advVel;

            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr);

            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekGemm(handle, "N", "N", m_nqTot, nelmtTot, m_nqTot,
                        (TData)1.0, m_matptr + d * m_nqTot * m_nqTot, m_nqTot,
                        inptr, m_nqTot, (TData)0.0, derivptr + d * outoffset,
                        m_nqTot);
            }

            // Multiply by derivative factor.
            if (m_isDeformed)
            {
                MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, outoffset,
                    m_dfptr, advptr, advelsize, derivptr, outptr);
            }
            else
            {
                MultiplyByDerivFactorAndAdvecVelKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, outoffset,
                    m_dfptr, advptr, advelsize, derivptr, outptr);
            }

            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, m_nqTot, (TData *)outptr);

            // Increment pointer.
            inptr += inoffset;
            outptr += outoffset;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetAdvVel(BlockAccessor<TData, FieldState::Phys> &advVel) override
    {
        const auto interleaveWidth = advVel.GetInterleaveWidth();
        this->m_advVel = advVel.template GetPtr<MemSpace, ReadWrite>();
        for (unsigned int n = 0; n < this->m_exp->GetCoordim(); n++)
        {
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      advVel.GetNumElementsWithPadding(),
                                      advVel.GetNumData(),
                                      this->m_advVel + n * advVel.CompSize());
        }
        advVel.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
