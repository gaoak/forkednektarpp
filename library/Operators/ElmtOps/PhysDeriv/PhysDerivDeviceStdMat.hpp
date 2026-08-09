///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceStdMat.hpp
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

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysDerivBlockOpImpl : public PhysDerivBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    PhysDerivBlockOpImpl(const unsigned int block_idx,
                         const LocalRegions::ExpansionSharedPtr &exp,
                         NekDataWarehouseSharedPtr dataWarehouse)
        : PhysDerivBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

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
            PhysDerivBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_streamID;
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
        auto handle = NekHandle<ExecSpace>::GetInstance(m_streamID);

        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nelmtTot =
            inblock.GetNumElementsWithPadding() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        const auto outDim = (outblock.GetNumHomoModes() > 1) ? 3u : m_coordDim;
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekGemm(handle, "N", "N", m_nqTot, nelmtTot, m_nqTot,
                        (TData)1.0, m_matptr + d * m_nqTot * m_nqTot, m_nqTot,
                        inptr, m_nqTot, (TData)0.0, outptr + d * outoffset,
                        m_nqTot);
            }

            // Multiply by derivative factor.
            if (m_isDeformed)
            {
                MultiplyByDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, outoffset,
                    outoffset, m_dfptr, outptr, outptr, m_streamID);
            }
            else
            {
                MultiplyByDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, outoffset,
                    outoffset, m_dfptr, outptr, outptr, m_streamID);
            }

            for (unsigned int k = 0; k < outDim; k++)
            {
                ReshapeStorage<ExecSpace>(
                    interleaveWidth, m_implInterleaveWidth, nelmtTot, m_nqTot,
                    (TData *)outptr + k * outoffset, m_streamID);
            }

            // In 3DH1 (nhomo > 1) the output has 3 slots per component
            // (x, y, z) regardless of base mesh coordDim, leaving slot 2
            // free for the z-derivative written by PhysDerivZOpHost/Device.
            inptr += inoffset;
            outptr += outDim * outoffset;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
