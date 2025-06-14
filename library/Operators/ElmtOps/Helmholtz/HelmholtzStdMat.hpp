///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzStdMat.hpp
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

#include "Operators/ElmtOps/Helmholtz/OperatorHelmholtz.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseStdMatKernels.hpp"
#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseStdMatKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorHelmholtzImpl : public BlockOperatorHelmholtz<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorHelmholtzImpl(const LocalRegions::ExpansionSharedPtr &exp,
                               NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorHelmholtz<TData>(exp, dataWarehouse),
          m_diffCoeff(MemoryRegion<TData>::Create(
              "Helmholtz diffCoeff", exp->GetCoordim() * exp->GetCoordim(),
              __STDCPP_DEFAULT_NEW_ALIGNMENT__))
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

        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        // Specialization for AVX/libXSMM
        if constexpr (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
        {
            m_bwdmat = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType,
                                 eBwdTransStdMatTranspose, nodalType));
            m_ipbmat = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType,
                                 eIProductWRTBaseStdMatTranspose, nodalType));
            m_derivmat =
                dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, ePhysDerivStdMatTranspose));
            m_ipdmat =
                dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType,
                    eIProductWRTDerivBaseStdMatTranspose, nodalType));
        }
        else
        {
            m_bwdmat =
                dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, eBwdTransStdMat, nodalType));
            m_ipbmat =
                dataWarehouse->template GetData<ExecSpace>(StdMatKey<TData>(
                    basisKeys, m_shapeType, eIProductWRTBaseStdMat, nodalType));
            m_derivmat = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType, ePhysDerivStdMat));
            m_ipdmat = dataWarehouse->template GetData<ExecSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType,
                                 eIProductWRTDerivBaseStdMat, nodalType));
        }

        m_diffCoeff.template Initialize<MemSpace>(0);

        TData *diffCoeff =
            m_diffCoeff.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();

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
            BlockOperatorHelmholtzImpl<ExecSpace, Implementation, TData>>(
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
        auto handle = NekHandle<ExecSpace>::GetInstance();

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = (inblock.GetInterleaveWidth() == m_implInterleaveWidth)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Fetch Jacobian and deriv factors.
        constexpr bool transpose =
            std::is_same_v<ExecSpace, NektarSpaces::Device>;
        auto jacptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            JacobianKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                               inblock.GetNumElements()));
        auto dfptr_init = this->m_dataWarehouse->template GetData<ExecSpace>(
            DerivFactorKey<TData>(inblock.GetExpIdx(), m_implInterleaveWidth,
                                  inblock.GetNumElements(), transpose));
        TData *diffCoeffPtr =
            this->m_diffCoeff.template GetPtr<MemSpace, ReadWrite>();

        // Specialization for AVX/libXSMM
        if constexpr (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
        {
            // Get interleave parameter.
            const auto interleave_width = inblock.GetInterleaveWidth();
            const auto width_ratio =
                (interleave_width == 1)
                    ? 1
                    : interleave_width / m_implInterleaveWidth;
            const auto chunkSize =
                std::max(m_implInterleaveWidth, interleave_width);

            // Allocate storage.
            if (m_bwd.size() == 0)
            {
                m_bwd   = MemoryRegion<TData>::Create(simd_t::width * m_nqTot,
                                                      ExecSpace::alignment);
                m_deriv = MemoryRegion<TData>::Create(
                    m_coordDim * simd_t::width * m_nqTot, ExecSpace::alignment);
            }

            // Get workspace pointer.
            auto bwdptr   = m_bwd.template GetPtr<MemSpace, WriteOnly>();
            auto derivptr = m_deriv.template GetPtr<MemSpace, WriteOnly>();

            const auto derivsize = m_nqTot;
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                auto jacptr = jacptr_init;
                auto dfptr  = dfptr_init;

                int flags    = 0;
                int prefetch = LIBXSMM_PREFETCH_NONE;

                // Dispatch kernel.
                auto bwd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    static_cast<int>(simd_t::width), static_cast<int>(m_nqTot),
                    static_cast<int>(m_nmTot), 1.0, 0.0, flags, prefetch);
                auto ipb_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    static_cast<int>(simd_t::width), static_cast<int>(m_nmTot),
                    static_cast<int>(m_nqTot), 1.0, 0.0, flags, prefetch);
                auto deriv_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    static_cast<int>(simd_t::width), static_cast<int>(m_nqTot),
                    static_cast<int>(m_nqTot), 1.0, 0.0, flags, prefetch);
                auto ipd_kernel = LibxsmmDispatchWrapper<TData>::dispatch(
                    static_cast<int>(simd_t::width), static_cast<int>(m_nmTot),
                    static_cast<int>(m_nqTot), 1.0, 1.0, flags, prefetch);

                for (size_t e = 0; e < nelmt / simd_t::width; ++e)
                {
                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                            interleave_width, chunkSize, m_nmTot,
                            (TData *)inptr);
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
                            m_nqTot, 1,
                            reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(bwdptr),
                            reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                    }
                    else
                    {
                        MultiplyByJacobianKernel<ExecSpace, false>(
                            m_nqTot, 1,
                            reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(bwdptr),
                            reinterpret_cast<simd_t *>(bwdptr), this->m_lambda);
                    }

                    // Perform matrix-matrix multiply.
                    ipb_kernel(bwdptr, m_ipbmat, outptr);

                    // Step 4: Multiply by diffusion coefficient
                    MultiplyByDiffusionCoeff<ExecSpace>(
                        1, m_nqTot, m_coordDim, derivsize, diffCoeffPtr,
                        reinterpret_cast<simd_t *>(derivptr));

                    // Step 5: IProductWRTDerivBase
                    // Multiply by derivative factor and Jacobian.
                    if (m_isDeformed)
                    {
                        MultiplyByJacobianAndDerivFactorKernel<ExecSpace, true>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            derivsize, reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(derivptr));
                        jacptr += m_nqTot * simd_t::width;
                        dfptr +=
                            m_coordDim * m_dimension * m_nqTot * simd_t::width;
                    }
                    else
                    {
                        MultiplyByJacobianAndDerivFactorKernel<ExecSpace,
                                                               false>(
                            m_nqTot, m_coordDim, m_dimension, 1, derivsize,
                            derivsize, reinterpret_cast<const simd_t *>(jacptr),
                            reinterpret_cast<const simd_t *>(dfptr),
                            reinterpret_cast<const simd_t *>(derivptr),
                            reinterpret_cast<simd_t *>(derivptr));
                        jacptr += simd_t::width;
                        dfptr += m_coordDim * m_dimension * simd_t::width;
                    }

                    // Perform matrix-matrix multiply.
                    for (unsigned int d = 0; d < m_dimension; d++)
                    {
                        ipd_kernel(derivptr + d * simd_t::width * m_nqTot,
                                   m_ipdmat + d * m_nqTot * m_nmTot, outptr);
                    }

                    // Increment pointers.
                    inptr += m_nmTot * simd_t::width;
                    outptr += m_nmTot * simd_t::width;
                }
            }
        }
        else
        {
            // Allocate storage.
            if (m_bwd.size() == 0)
            {
                m_bwd   = MemoryRegion<TData>::Create(nelmt * m_nqTot,
                                                      ExecSpace::alignment);
                m_deriv = MemoryRegion<TData>::Create(
                    m_coordDim * nelmt * m_nqTot, ExecSpace::alignment);
            }

            // Get workspace pointer.
            auto bwdptr   = m_bwd.template GetPtr<MemSpace, WriteOnly>();
            auto derivptr = m_deriv.template GetPtr<MemSpace, WriteOnly>();

            const auto derivsize = m_nqTot * nelmt;
            for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
            {
                auto jacptr = jacptr_init;
                auto dfptr  = dfptr_init;

                // Reshape, if necessary.
                ReshapeStorage<ExecSpace, m_implInterleaveWidth>(
                    inblock.GetInterleaveWidth(),
                    inblock.GetNumElementsWithPadding(), inblock.GetNumData(),
                    (TData *)inptr);

                // Step 1: BwdTrans
                // Perform matrix-matrix multiply.
                NekGemm(handle, "N", "N", m_nqTot, nelmt, m_nmTot, 1.0,
                        m_bwdmat, m_nqTot, inptr, m_nmTot, 0.0, bwdptr,
                        m_nqTot);

                // Step 2: PhysDeriv
                // Perform matrix-matrix multiply.
                NekGemmStridedBatched(
                    handle, "N", "N", m_nqTot, nelmt, m_nqTot, 1.0, m_derivmat,
                    m_nqTot, m_nqTot * m_nqTot, bwdptr, m_nqTot, 0, 0.0,
                    derivptr, m_nqTot, m_nqTot * nelmt, m_dimension);

                // Multiply by derivative factor.
                if (m_isDeformed)
                {
                    MultiplyByDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, derivsize,
                        derivsize, dfptr, derivptr, derivptr);
                }
                else
                {
                    MultiplyByDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, derivsize,
                        derivsize, dfptr, derivptr, derivptr);
                }

                // Step 3: IProduct
                // Multiply by jacobian.
                if (m_isDeformed)
                {
                    MultiplyByJacobianKernel<ExecSpace, true>(
                        m_nqTot, nelmt, jacptr, bwdptr, bwdptr, this->m_lambda);
                }
                else
                {
                    MultiplyByJacobianKernel<ExecSpace, false>(
                        m_nqTot, nelmt, jacptr, bwdptr, bwdptr, this->m_lambda);
                }

                // Perform matrix-matrix multiply.
                NekGemm(handle, "N", "N", m_nmTot, nelmt, m_nqTot, 1.0,
                        m_ipbmat, m_nmTot, bwdptr, m_nqTot, 0.0, outptr,
                        m_nmTot);

                // Step 4: Multiply by diffusion coefficient
                MultiplyByDiffusionCoeff<ExecSpace>(nelmt, m_nqTot, m_coordDim,
                                                    derivsize, diffCoeffPtr,
                                                    derivptr);

                // Step 5: IProductWRTDerivBase
                // Multiply by derivative factor and Jacobian.
                if (m_isDeformed)
                {
                    MultiplyByJacobianAndDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, derivsize,
                        derivsize, jacptr, dfptr, derivptr, derivptr);
                }
                else
                {
                    MultiplyByJacobianAndDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, nelmt, derivsize,
                        derivsize, jacptr, dfptr, derivptr, derivptr);
                }

                // Perform matrix-matrix multiply.
                for (unsigned int d = 0; d < m_dimension; d++)
                {
                    NekGemm(handle, "N", "N", m_nmTot, nelmt, m_nqTot, 1.0,
                            m_ipdmat + d * m_nqTot * m_nmTot, m_nmTot,
                            derivptr + d * nelmt * m_nqTot, m_nqTot, 1.0,
                            outptr, m_nmTot);
                }

                // Increment pointers.
                inptr += inblock.size();
                outptr += outblock.size();
            }
        }

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
        outblock.template SetInterleaveWidth<TData>(m_implInterleaveWidth);
    }

    void v_SetLambda(const TData &lambda) override
    {
        this->m_lambda = lambda;
    }
};

} // namespace Nektar::Operators::detail
