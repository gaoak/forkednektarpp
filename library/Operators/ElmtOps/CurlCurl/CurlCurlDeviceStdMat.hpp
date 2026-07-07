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

#include "Operators/ElmtOps/CurlCurl/CurlCurlBlockOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CombineOmega2DStdMat(
    const size_t npts, const size_t outoffset, const TData *NEK_RESTRICT grad0,
    const TData *NEK_RESTRICT grad1, TData *NEK_RESTRICT omega,
    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, npts, NEKTAR_LAMBDA(const size_t idx) {
            omega[idx] = grad1[idx] - grad0[outoffset + idx];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void AssembleCurlCurl2DStdMat(
    const size_t npts, const size_t outoffset,
    const TData *NEK_RESTRICT gradOmega, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, npts, NEKTAR_LAMBDA(const size_t idx) {
            out0[idx] = gradOmega[outoffset + idx];
            out1[idx] = -gradOmega[idx];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void CombineOmega3DStdMat(
    const size_t npts, const size_t outoffset, const TData *NEK_RESTRICT grad0,
    const TData *NEK_RESTRICT grad1, const TData *NEK_RESTRICT grad2,
    TData *NEK_RESTRICT omega, const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, npts, NEKTAR_LAMBDA(const size_t idx) {
            omega[idx] = grad2[outoffset + idx] - grad1[2u * outoffset + idx];
            omega[outoffset + idx] = grad0[2u * outoffset + idx] - grad2[idx];
            omega[2u * outoffset + idx] = grad1[idx] - grad0[outoffset + idx];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void AssembleCurlCurl3DStdMat(
    const size_t npts, const size_t outoffset,
    const TData *NEK_RESTRICT gradOmega0, const TData *NEK_RESTRICT gradOmega1,
    const TData *NEK_RESTRICT gradOmega2, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *NEK_RESTRICT out2,
    const unsigned int streamID)
{
    Nektar::LoopExecutionSetStreamID(streamID);

    Nektar::parallel_for<ExecSpace>(
        0, npts, NEKTAR_LAMBDA(const size_t idx) {
            out0[idx] =
                gradOmega2[outoffset + idx] - gradOmega1[2u * outoffset + idx];
            out1[idx] = gradOmega0[2u * outoffset + idx] - gradOmega2[idx];
            out2[idx] = gradOmega1[idx] - gradOmega0[outoffset + idx];
        });

    Nektar::LoopExecutionSetStreamID(0);
}

template <typename ExecSpace, typename Implementation, typename TData>
class CurlCurlBlockOpImpl : public CurlCurlBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    CurlCurlBlockOpImpl(const unsigned int block_idx,
                        const LocalRegions::ExpansionSharedPtr &exp,
                        NekDataWarehouseSharedPtr dataWarehouse)
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

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance(m_streamID);

        ASSERTL0(inblock.GetNumHomoModes() == 1,
                 "Currently only setup for one homogenous plane");

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get static workspace pointer.
        auto wspptr =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                8 * m_dimension * nelmt * m_nqTot, m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        const auto inoffset    = inblock.CompSize();
        const auto derivoffset = m_nqTot * nelmt;
        const auto outoffset   = outblock.CompSize();

        auto derivRef = wspptr;
        auto grad0    = derivRef + m_dimension * derivoffset;
        auto grad1    = grad0 + m_coordDim * derivoffset;
        auto grad2    = grad1 + m_coordDim * derivoffset;
        auto omega    = grad2 + m_coordDim * derivoffset;
        auto gradW0   = omega + m_coordDim * derivoffset;
        auto gradW1   = gradW0 + m_coordDim * derivoffset;
        auto gradW2   = gradW1 + m_coordDim * derivoffset;

        auto computePhysDeriv = [&](const TData *fieldptr, TData *physOut) {
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekGemm(handle, "N", "N", m_nqTot, nelmt, m_nqTot, (TData)1.0,
                        m_matptr + d * m_nqTot * m_nqTot, m_nqTot, fieldptr,
                        m_nqTot, (TData)0.0, derivRef + d * derivoffset,
                        m_nqTot);
            }

            if (m_isDeformed)
            {
                MultiplyByDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, 1, derivoffset,
                    derivoffset, m_dfptr, derivRef, physOut, m_streamID);
            }
            else
            {
                MultiplyByDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, 1, derivoffset,
                    derivoffset, m_dfptr, derivRef, physOut, m_streamID);
            }
        };

        for (unsigned int d = 0; d < m_coordDim; d++)
        {
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr + d * inoffset,
                                      m_streamID);
        }

        computePhysDeriv(inptr, grad0);
        computePhysDeriv(inptr + inoffset, grad1);

        if (m_dimension == 2)
        {
            CombineOmega2DStdMat<ExecSpace>(derivoffset, derivoffset, grad0,
                                            grad1, omega, m_streamID);
            computePhysDeriv(omega, gradW0);
            AssembleCurlCurl2DStdMat<ExecSpace>(derivoffset, derivoffset,
                                                gradW0, outptr,
                                                outptr + outoffset, m_streamID);
        }
        else
        {
            computePhysDeriv(inptr + 2 * inoffset, grad2);
            CombineOmega3DStdMat<ExecSpace>(derivoffset, derivoffset, grad0,
                                            grad1, grad2, omega, m_streamID);
            computePhysDeriv(omega, gradW0);
            computePhysDeriv(omega + derivoffset, gradW1);
            computePhysDeriv(omega + 2 * derivoffset, gradW2);
            AssembleCurlCurl3DStdMat<ExecSpace>(
                derivoffset, derivoffset, gradW0, gradW1, gradW2, outptr,
                outptr + outoffset, outptr + 2 * outoffset, m_streamID);
        }

        for (unsigned int k = 0; k < m_coordDim; k++)
        {
            ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt, m_nqTot,
                (TData *)inptr + k * inoffset, m_streamID);
            ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt, m_nqTot,
                (TData *)outptr + k * outoffset, m_streamID);
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
