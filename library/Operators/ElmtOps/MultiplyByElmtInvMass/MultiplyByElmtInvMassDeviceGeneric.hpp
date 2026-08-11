///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassDeviceGeneric.hpp
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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassBlockOp.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class MultiplyByElmtInvMassBlockOpImpl
    : public MultiplyByElmtInvMassBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    MultiplyByElmtInvMassBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : MultiplyByElmtInvMassBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_invmassptr =
            dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
                basisKeys, m_shapeType, eInvMassStdMat, nodalType));

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Coeff, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            MultiplyByElmtInvMassBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_invmassptr;
    const TData *m_jacptr;
    LibUtilities::MemoryRegion<TData> m_dinvmass;

    void v_Apply(MultiRegions::BlockAccessor<TData, FieldState::Coeff> &inblock,
                 MultiRegions::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

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
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);

            if (m_isDeformed)
            {
                // Fetch deformed mass matrix.
                auto dmatptr =
                    this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>(
                        m_streamID);

                // Perform batched matrix-vector multiply.
                NekBlas::GemmStridedBatched(
                    handle, "N", "N", m_nmTot, 1, m_nmTot, (TData)1.0, dmatptr,
                    m_nmTot, m_nmTot * m_nmTot, inptr, m_nmTot, m_nmTot,
                    (TData)0.0, outptr, m_nmTot, m_nmTot, nelmtTot);
            }
            else
            {
                // Perform matrix-matrix multiply.
                NekBlas::Gemm(handle, "N", "N", m_nmTot, nelmtTot, m_nmTot,
                              (TData)1.0, m_invmassptr, m_nmTot, inptr, m_nmTot,
                              (TData)0.0, outptr, m_nmTot);

                // Divide by Jacobian.
                DivideByJacobian<ExecSpace, false>(nelmt, m_nmTot, nhomo,
                                                   m_jacptr, outptr, outptr,
                                                   m_streamID);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr, m_streamID);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, outblock.GetNumData(), outptr,
                                      m_streamID);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetInvMassMatrix(std::vector<TData> &invmass) override
    {
        this->m_dinvmass = LibUtilities::MemoryRegion<
            TData>::template FromVector<MemSpace, TData>(invmass);
    }
};

} // namespace Nektar::Operators::detail
