///////////////////////////////////////////////////////////////////////////////
//
// File: MultiplyByElmtInvMassSerialAVXGeneric.hpp
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

#include "Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassSerialAVXGenericKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class MultiplyByElmtInvMassBlockOpImpl
    : public MultiplyByElmtInvMassBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;

    using MemSpace = typename ExecSpace::memory_space;

public:
    MultiplyByElmtInvMassBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : MultiplyByElmtInvMassBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_invmassptr =
            dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
                basisKeys, m_shapeType, eInvMassStdMatTranspose, nodalType));

        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            MultiplyByElmtInvMassBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_invmassptr;
    const TData *m_jacptr;
    MemoryRegion<TData> m_dinvmass;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            if (m_isDeformed)
            {
                // Fetch deformed mass matrix.
                auto dmatptr =
                    this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>();

                // Loop over element groups.
                for (size_t e = 0;
                     e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                                  interleaveWidth, chunkSize,
                                                  m_nmTot, (TData *)inptr);
                    }

                    // Perform batched matrix-vector multiply.
                    if constexpr (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
                    {
                        NekGemvStridedBatched(
                            blasHandle_t(), "N", m_nmTot, m_nmTot, 1.0, dmatptr,
                            m_nmTot, m_nmTot * m_nmTot, inptr, simd_t::width, 1,
                            0.0, outptr, simd_t::width, 1, simd_t::width);
                    }
                    else
                    {
                        NekGemvStridedBatched(
                            blasHandle_t(), "N", m_nmTot, m_nmTot, 1.0, dmatptr,
                            m_nmTot, m_nmTot * m_nmTot, inptr, 1, m_nmTot, 0.0,
                            outptr, 1, m_nmTot, simd_t::width);
                    }

                    // Reshape back, if necessary.
                    if (e % width_ratio == width_ratio - 1)
                    {
                        ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nmTot,
                            (TData *)inptr -
                                (width_ratio - 1) * m_nmTot * simd_t::width);
                        ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nmTot,
                            (TData *)outptr -
                                (width_ratio - 1) * m_nmTot * simd_t::width);
                    }

                    // Increment pointers.
                    inptr += m_nmTot * simd_t::width;
                    outptr += m_nmTot * simd_t::width;
                    dmatptr += m_nmTot * m_nmTot * simd_t::width;
                }
            }
            else
            {
                auto jacptr = m_jacptr;

                // Dispatch kernel.
                auto invmass_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    simd_t::width, m_nmTot, m_nmTot, 1.0, 0.0);

                // Loop over element groups.
                for (size_t e = 0;
                     e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                                  interleaveWidth, chunkSize,
                                                  m_nmTot, (TData *)inptr);
                    }
                    // Perform matrix-matrix multiply.
                    invmass_kernel(inptr, m_invmassptr, outptr);

                    // Divide by Jacobian.
                    DivideByJacobianKernel<ExecSpace, false>(
                        1, m_nmTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(outptr),
                        reinterpret_cast<simd_t *>(outptr));

                    // Reshape back, if necessary.
                    if (e % width_ratio == width_ratio - 1)
                    {
                        ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nmTot,
                            (TData *)inptr -
                                (width_ratio - 1) * m_nmTot * simd_t::width);
                        ReshapeStorage<ExecSpace>(
                            interleaveWidth, m_implInterleaveWidth, chunkSize,
                            m_nmTot,
                            (TData *)outptr -
                                (width_ratio - 1) * m_nmTot * simd_t::width);
                    }

                    // Increment pointers.
                    inptr += m_nmTot * simd_t::width;
                    outptr += m_nmTot * simd_t::width;
                    jacptr += simd_t::width;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetInvMassMatrix(std::vector<TData> &invmass) override
    {
        this->m_dinvmass =
            MemoryRegion<TData>::template FromVector<MemSpace, TData>(invmass);
    }
};

} // namespace Nektar::Operators::detail
