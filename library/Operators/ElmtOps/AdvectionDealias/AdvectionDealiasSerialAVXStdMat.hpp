///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasSerialAVXStdMat.hpp
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
// Description: Fused 3/2-rule dealiased advection, StdMat implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasBlockOp.hpp"

#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionDealiasBlockOpImpl : public AdvectionDealiasBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvectionDealiasBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionDealiasBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();
        m_nqTot     = exp->GetTotPoints();

        // Fetch matrix.
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nq.push_back(exp->GetNumPoints(d));
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        const auto nodalType = (exp->IsNodalNonTensorialExp())
                                   ? exp->GetNodalPointsKey().GetPointsType()
                                   : LibUtilities::eNoPointsType;

        // Fetch reference derivative matrix.
        m_derivMatPtr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType, StdRegions::ePhysDerivStdMatTranspose));

        // Fetch derivative factor.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Fine (3/2 over-integrated) point counts
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            unsigned int nqFine;
            if (d == 0)
            {
                nqFine = static_cast<unsigned int>(m_dealiasScale * m_nq[0]);
            }
            else
            {
                nqFine =
                    (m_nq[0] - m_nq[d] == 1)
                        ? static_cast<unsigned int>(m_dealiasScale * m_nq[0]) -
                              1
                        : static_cast<unsigned int>(m_dealiasScale * m_nq[d]);
            }
            m_nqFine.push_back(nqFine);
        }
        m_nqFineTot = NqTot(m_nqFine);

        // Fetch interpolation matrix, native -> fine quadrature.
        m_interpMatPtr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::ePhysInterpStdMatTranspose,
                                         nodalType, m_nqFine));

        // Fetch Galerkin projection matrix, fine -> native quadrature.
        m_projectMatPtr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(
                basisKeys, m_shapeType,
                StdRegions::eGalerkinProjectStdMatTranspose, nodalType,
                m_nqFine));
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
            AdvectionDealiasBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;
    static constexpr TData m_dealiasScale               = TData(1.5);

    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    unsigned int m_nqFineTot;
    std::vector<unsigned int> m_nq;
    std::vector<unsigned int> m_nqFine;
    const TData *m_derivMatPtr;
    const TData *m_interpMatPtr;
    const TData *m_projectMatPtr;
    const TData *m_dfptr;
    static unsigned int NqTot(const std::vector<unsigned int> &nq)
    {
        unsigned int total = 1;
        for (auto n : nq)
        {
            total *= n;
        }
        return total;
    }

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>();
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_exp->GetCoordim(),
                this->m_advVel->GetNumData(), advVelPtr);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

        // Initialize pointers.
        auto inptr     = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr    = (this->m_append)
                             ? outblock.template GetPtr<MemSpace, ReadWrite>()
                             : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto advVelPtr = this->m_advVel->template GetPtr<MemSpace, ReadOnly>();
        auto dfptr     = m_dfptr;

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Get static workspace pointers. Only one component's derivative,
        // gradient and combined-result are held at a time - the fine-grid
        // advection velocity is shared across all components.
        const unsigned int derivSize        = m_coordDim * m_nqTot;
        const unsigned int advVelFineSize   = m_coordDim * m_nqFineTot;
        const unsigned int gradFineSize     = m_coordDim * m_nqFineTot;
        const unsigned int combinedFineSize = m_nqFineTot;
        auto wspBase =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                (derivSize + advVelFineSize + gradFineSize + combinedFineSize) *
                simd_t::width);
        auto derivNative  = reinterpret_cast<simd_t *>(wspBase);
        auto advVelFine   = derivNative + derivSize;
        auto gradFine     = advVelFine + advVelFineSize;
        auto combinedFine = gradFine + gradFineSize;

        // Dispatch kernels.
        auto gemm_deriv = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqTot, 1.0, 0.0);
        auto gemm_interp = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqFineTot, m_nqTot, 1.0, 0.0);
        auto gemm_project = LibxsmmDispatchWrapper<TData>::dispatch(
            simd_t::width, m_nqTot, m_nqFineTot, 1.0,
            this->m_append ? 1.0 : 0.0);

        // Loop over element groups.
        const unsigned int ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const auto inCompSize  = inblock.CompSize();
        const auto outCompSize = outblock.CompSize();
        const auto advVelCompSize =
            this->m_advVel->CompSize() * this->m_advVel->GetNumHomoModes();
        const unsigned int dfsize = m_isDeformed
                                        ? m_coordDim * m_dimension * m_nqTot
                                        : m_coordDim * m_dimension;
        for (size_t e = 0; e < inblock.GetNumElmtGroups(m_implInterleaveWidth);
             ++e)
        {
            // Interpolate the advection velocity to the fine grid once - it
            // is shared by every advected component below.
            for (unsigned int d = 0; d < m_coordDim; ++d)
            {
                TData *dst = reinterpret_cast<TData *>(advVelFine) +
                             d * m_nqFineTot * simd_t::width;
                gemm_interp((TData *)advVelPtr + d * advVelCompSize,
                            m_interpMatPtr, dst);
            }

            // Loop over components. Each reshapes its own chunk in and
            // back around its own work - components touch only their own
            // slice of the input and output.
            for (unsigned int c = 0; c < ncomp; ++c)
            {
                TData *inCompPtr  = (TData *)inptr + c * inCompSize;
                TData *outCompPtr = (TData *)outptr + c * outCompSize;

                // Reshape this component's chunk, if necessary.
                if (e % width_ratio == 0)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                        m_nqTot, inCompPtr);
                    if (this->m_append)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, outInterleaveWidth,
                            chunkSize, m_nqTot, outCompPtr);
                    }
                }

                TData *derivOutData = reinterpret_cast<TData *>(derivNative);

                // Native-grid reference derivatives of this component.
                for (unsigned int d = 0; d < m_dimension; ++d)
                {
                    gemm_deriv(inCompPtr, m_derivMatPtr + d * m_nqTot * m_nqTot,
                               derivOutData + d * m_nqTot * simd_t::width);
                }

                // Apply derivative factors to get the physical gradient.
                if (m_isDeformed)
                {
                    MultiplyByDerivFactorKernel<ExecSpace, true>(
                        m_nqTot, m_coordDim, m_dimension, 1, m_nqTot, m_nqTot,
                        reinterpret_cast<const simd_t *>(dfptr), derivNative,
                        derivNative);
                }
                else
                {
                    MultiplyByDerivFactorKernel<ExecSpace, false>(
                        m_nqTot, m_coordDim, m_dimension, 1, m_nqTot, m_nqTot,
                        reinterpret_cast<const simd_t *>(dfptr), derivNative,
                        derivNative);
                }

                // Interpolate the physical gradient to the fine grid.
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    const TData *src =
                        derivOutData + d * m_nqTot * simd_t::width;
                    TData *dst = reinterpret_cast<TData *>(gradFine) +
                                 d * m_nqFineTot * simd_t::width;
                    gemm_interp(src, m_interpMatPtr, dst);
                }

                // Form scale * advVel . grad(u) on the fine grid.
                AdvectionDealiasCombineKernel(
                    m_nqFineTot, m_coordDim, advVelFine, m_nqFineTot, gradFine,
                    m_nqFineTot, combinedFine, this->m_scale);

                // Project directly into this component's output.
                const TData *combinedPtr =
                    reinterpret_cast<TData *>(combinedFine);
                gemm_project(combinedPtr, m_projectMatPtr, outCompPtr);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        inCompPtr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    LibUtilities::ReshapeStorage<ExecSpace>(
                        inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        outCompPtr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                }
            }

            // Increment pointers.
            inptr += m_nqTot * simd_t::width;
            outptr += m_nqTot * simd_t::width;
            advVelPtr += m_nqTot * simd_t::width;
            dfptr += dfsize * simd_t::width;
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
