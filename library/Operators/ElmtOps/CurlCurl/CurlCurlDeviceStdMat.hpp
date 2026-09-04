///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlDeviceStdMat.hpp
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

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp"

#include "Operators/ElmtOps/CurlCurl/CurlCurlDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : CurlCurlBlockOp<TData>(block_idx, exp, dataWarehouse)
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

        ASSERTL1(m_coordDim == 2 || m_coordDim == 3,
                 "CurlCurl operator only defined for 2D and 3D.");

        ASSERTL1(m_dimension == m_coordDim,
                 "Shape dimension and coordinate dimension are not the same.");

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_matptr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::ePhysDerivStdMat));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Phys, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            CurlCurlBlockOpImpl<ExecSpace, Implementation, TData>>(
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

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        ASSERTL0(inblock.GetNumHomoModes() == 1,
                 "Currently only setup for one homogenous plane");

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Offsets between the components of a block.
        const auto outoffset   = outblock.CompSize();
        const auto derivoffset = m_nqTot * nelmt;

        // omega = curl(u) is a scalar in 2D and a vector in 3D.
        const unsigned int nOmega = (m_dimension == 2u) ? 1u : 3u;

        // Get static workspace pointer. The standard derivatives of every
        // component are held at once so that the chain rule and the curl can
        // be applied by a single kernel.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                (m_dimension * m_coordDim + nOmega) * derivoffset, m_streamID);
        auto derivptr = wspptr;
        auto omegaptr = derivptr + m_dimension * m_coordDim * derivoffset;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, interleaveWidth, m_coordDim * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);

        // Standard derivatives of all the components, one matrix-matrix
        // multiply per direction. The components of a block are contiguous,
        // so they are all handled by the same multiply.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            NekBlas::Gemm(handle, "N", "N", m_nqTot, m_coordDim * nelmt,
                          m_nqTot, (TData)1.0, m_matptr + d * m_nqTot * m_nqTot,
                          m_nqTot, inptr, m_nqTot, (TData)0.0,
                          derivptr + d * m_coordDim * derivoffset, m_nqTot);
        }

        // omega = curl(u)
        if (m_dimension == 2u)
        {
            if (m_isDeformed)
            {
                Curl2DScalarStdMatKernel<ExecSpace, true>(
                    m_nqTot, nelmt, derivoffset, m_dfptr, derivptr, omegaptr,
                    m_streamID);
            }
            else
            {
                Curl2DScalarStdMatKernel<ExecSpace, false>(
                    m_nqTot, nelmt, derivoffset, m_dfptr, derivptr, omegaptr,
                    m_streamID);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                Curl3DStdMatKernel<ExecSpace, true>(
                    m_nqTot, nelmt, derivoffset, derivoffset, m_dfptr, derivptr,
                    omegaptr, m_streamID);
            }
            else
            {
                Curl3DStdMatKernel<ExecSpace, false>(
                    m_nqTot, nelmt, derivoffset, derivoffset, m_dfptr, derivptr,
                    omegaptr, m_streamID);
            }
        }

        // Standard derivatives of omega.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            NekBlas::Gemm(handle, "N", "N", m_nqTot, nOmega * nelmt, m_nqTot,
                          (TData)1.0, m_matptr + d * m_nqTot * m_nqTot, m_nqTot,
                          omegaptr, m_nqTot, (TData)0.0,
                          derivptr + d * nOmega * derivoffset, m_nqTot);
        }

        // out = curl(omega)
        if (m_dimension == 2u)
        {
            if (m_isDeformed)
            {
                Curl2DVectorStdMatKernel<ExecSpace, true>(
                    m_nqTot, nelmt, derivoffset, outoffset, m_dfptr, derivptr,
                    outptr, m_streamID);
            }
            else
            {
                Curl2DVectorStdMatKernel<ExecSpace, false>(
                    m_nqTot, nelmt, derivoffset, outoffset, m_dfptr, derivptr,
                    outptr, m_streamID);
            }
        }
        else
        {
            if (m_isDeformed)
            {
                Curl3DStdMatKernel<ExecSpace, true>(
                    m_nqTot, nelmt, derivoffset, outoffset, m_dfptr, derivptr,
                    outptr, m_streamID);
            }
            else
            {
                Curl3DStdMatKernel<ExecSpace, false>(
                    m_nqTot, nelmt, derivoffset, outoffset, m_dfptr, derivptr,
                    outptr, m_streamID);
            }
        }

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, m_coordDim * nelmt,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            interleaveWidth, m_implInterleaveWidth, m_coordDim * nelmt,
            outblock.GetNumData(), (TData *)outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
