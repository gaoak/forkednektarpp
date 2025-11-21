///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzDeviceStdMat.hpp
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

#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzDeviceStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseDeviceStdMatKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class HelmholtzBlockOpImpl : public HelmholtzBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    HelmholtzBlockOpImpl(const LocalRegions::ExpansionSharedPtr &exp,
                         NekDataWarehouseSharedPtr dataWarehouse)
        : HelmholtzBlockOp<TData>(exp, dataWarehouse),
          m_diffCoeff(MemoryRegion<TData>("Helmholtz diffCoeff",
                                          exp->GetCoordim() * exp->GetCoordim(),
                                          __STDCPP_DEFAULT_NEW_ALIGNMENT__))
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

        m_bwdmat = dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eBwdTransStdMat, nodalType));
        m_ipbmat = dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eIProductWRTBaseStdMat, nodalType));
        m_derivmat = dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType, ePhysDerivStdMat));
        m_ipdmat = dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
            basisKeys, m_shapeType, eIProductWRTDerivBaseStdMat, nodalType));

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        for (unsigned int d = 0; d < m_coordDim; d++)
        {
            diffCoeff[d * m_coordDim + d] = 1.0; // temporary solution
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            HelmholtzBlockOpImpl<ExecSpace, Implementation, TData>>(
            exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    const TData *m_bwdmat;
    const TData *m_ipbmat;
    const TData *m_ipdmat;
    const TData *m_derivmat;
    MemoryRegion<TData> m_bwd;
    MemoryRegion<TData> m_deriv;
    MemoryRegion<TData> m_diffCoeff;

    void v_Apply(BlockAccessor<TData> &inblock,
                 BlockAccessor<TData> &outblock) override
    {
        auto handle = NekHandle<ExecSpace>::GetInstance();

        const auto nhomo = inblock.GetNumHomoModes();
        const auto nelmt = inblock.GetNumElementsWithPadding();
        const auto nelmtTot =
            inblock.GetNumElementsWithPadding() * inblock.GetNumHomoModes();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian and deriv factors.
        auto jacptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), true));
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Allocate storage.
        if (m_bwd.size() == 0)
        {
            m_bwd =
                MemoryRegion<TData>(nelmtTot * m_nqTot, ExecSpace::alignment);
            m_deriv = MemoryRegion<TData>(m_coordDim * nelmtTot * m_nqTot,
                                          ExecSpace::alignment);
        }

        // Get workspace pointer.
        auto bwdptr   = m_bwd.template GetPtr<MemSpace, WriteOnly>();
        auto derivptr = m_deriv.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        const auto derivoffset = m_nqTot * nelmtTot;
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Step 1: BwdTrans
            // Perform matrix-matrix multiply.
            NekGemm(handle, "N", "N", m_nqTot, nelmtTot, m_nmTot, 1.0, m_bwdmat,
                    m_nqTot, inptr, m_nmTot, 0.0, bwdptr, m_nqTot);

            // Step 2: PhysDeriv
            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekGemm(handle, "N", "N", m_nqTot, nelmtTot, m_nqTot, 1.0,
                        m_derivmat + d * m_nqTot * m_nqTot, m_nqTot, bwdptr,
                        m_nqTot, 0.0, derivptr + d * m_nqTot * nelmtTot,
                        m_nqTot);
            }

            // Multiply by derivative factor.
            if (m_isDeformed)
            {
                MultiplyByDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, dfptr, derivptr, derivptr);
            }
            else
            {
                MultiplyByDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, dfptr, derivptr, derivptr);
            }

            // Step 3: IProduct
            // Multiply by jacobian.
            if (m_isDeformed)
            {
                MultiplyByJacobianKernel<ExecSpace, true>(
                    nelmt, m_nqTot, nhomo, jacptr, bwdptr, bwdptr,
                    this->m_lambda);
            }
            else
            {
                MultiplyByJacobianKernel<ExecSpace, false>(
                    nelmt, m_nqTot, nhomo, jacptr, bwdptr, bwdptr,
                    this->m_lambda);
            }

            // Perform matrix-matrix multiply.
            NekGemm(handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, 1.0, m_ipbmat,
                    m_nmTot, bwdptr, m_nqTot, 0.0, outptr, m_nmTot);

            // Step 4: Multiply by diffusion coefficient
            MultiplyByDiffusionCoeff<ExecSpace>(nelmtTot, m_nqTot, m_coordDim,
                                                derivoffset, diffCoeffPtr,
                                                derivptr);

            // Step 5: IProductWRTDerivBase
            // Multiply by derivative factor and Jacobian.
            if (m_isDeformed)
            {
                MultiplyByJacobianAndDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, jacptr, dfptr, derivptr, derivptr);
            }
            else
            {
                MultiplyByJacobianAndDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, jacptr, dfptr, derivptr, derivptr);
            }

            // Perform matrix-matrix multiply.
            for (unsigned int d = 0; d < m_dimension; d++)
            {
                NekGemm(handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, 1.0,
                        m_ipdmat + d * m_nqTot * m_nmTot, m_nmTot,
                        derivptr + d * nelmtTot * m_nqTot, m_nqTot, 1.0, outptr,
                        m_nmTot);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmtTot, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.size() * inblock.GetNumHomoModes();
            outptr += outblock.size() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetLambda(const TData &lambda) override
    {
        this->m_lambda = lambda;
    }
};

} // namespace Nektar::Operators::detail
