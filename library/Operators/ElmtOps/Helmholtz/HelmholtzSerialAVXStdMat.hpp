///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSerialAVXStdMat.hpp
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

#include "Operators/ElmtOps/Helmholtz/HelmholtzSerialAVXStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseSerialAVXStdMatKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class HelmholtzBlockOpImpl : public HelmholtzBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
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
            basisKeys, m_shapeType, eBwdTransStdMatTranspose, nodalType));
        m_ipbmat = dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType,
                             eIProductWRTBaseStdMatTranspose, nodalType));
        m_derivmat =
            dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                basisKeys, m_shapeType, ePhysDerivStdMatTranspose));
        m_ipdmat = dataWarehouse->template GetData<ExecSpace>(
            StdMatKey<TData>(basisKeys, m_shapeType,
                             eIProductWRTDerivBaseStdMatTranspose, nodalType));

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
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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
        // Fetch Jacobian and deriv factors.
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), false));
        auto diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadOnly>();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Allocate storage.
        if (m_bwd.size() == 0)
        {
            m_bwd   = MemoryRegion<TData>(simd_t::width * m_nqTot,
                                        ExecSpace::alignment);
            m_deriv = MemoryRegion<TData>(m_coordDim * simd_t::width * m_nqTot,
                                          ExecSpace::alignment);
        }

        // Get workspace pointer.
        auto bwdptr   = m_bwd.template GetPtr<MemSpace, WriteOnly>();
        auto derivptr = m_deriv.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Dispatch kernel.
        auto bwd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nmTot, 1.0, 0.0);
        auto ipb_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 0.0);
        auto deriv_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);
        auto ipd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nmTot, m_nqTot, 1.0, 1.0);

        // Loop over components.
        const auto derivsize = m_nqTot;
        for (unsigned int n = 0;
             n < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++n)
        {
            auto jacptr = jacptr_init;
            auto dfptr  = dfptr_init;

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

                // Step 1: BwdTrans
                // Perform matrix-matrix multiply.
                bwd_kernel(inptr, m_bwdmat, bwdptr);

                // Step 2: PhysDeriv
                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    deriv_kernel(bwdptr, m_derivmat + d * m_nqTot * m_nqTot,
                                 derivptr + d * m_nqTot * simd_t::width);
                }

                // Multiply by derivative factor.
                if (m_isDeformed)
                {
                    MultiplyByDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr));
                }
                else
                {
                    MultiplyByDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr));
                }

                // Step 3: IProduct
                // Multiply by jacobian.
                if (m_isDeformed)
                {
                    MultiplyByJacobianKernel<ExecSpace, true>(
                        1, m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(bwdptr),
                        reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                }
                else
                {
                    MultiplyByJacobianKernel<ExecSpace, false>(
                        1, m_nqTot, reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(bwdptr),
                        reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                }

                // Perform matrix-matrix multiply.
                ipb_kernel(bwdptr, m_ipbmat, outptr);

                // Step 4: Multiply by diffusion coefficient, derivative
                // factor and Jacobian.
                if (m_isDeformed)
                {
                    ApplyMetricKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, diffCoeffPtr,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr));
                    jacptr += m_nqTot * simd_t::width;
                    dfptr += m_coordDim * m_dimension * m_nqTot * simd_t::width;
                }
                else
                {
                    ApplyMetricKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                        derivsize, diffCoeffPtr,
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(derivptr),
                        reinterpret_cast<simd_t *>(derivptr));
                    jacptr += simd_t::width;
                    dfptr += m_coordDim * m_dimension * simd_t::width;
                }

                // Step 5: IProductWRTDerivBase
                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    ipd_kernel(derivptr + d * simd_t::width * m_nqTot,
                               m_ipdmat + d * m_nqTot * m_nmTot, outptr);
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
            }
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
