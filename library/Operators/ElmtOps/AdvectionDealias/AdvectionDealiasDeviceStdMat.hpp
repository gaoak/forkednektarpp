///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasDeviceStdMat.hpp
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
// Description: Fused 3/2-rule dealiased advection, Device StdMat
// implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasBlockOp.hpp"

#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasDeviceStdMatKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceStdMatKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionDealiasBlockOpImpl : public AdvectionDealiasBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    AdvectionDealiasBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : AdvectionDealiasBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;

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
        std::vector<unsigned int> nq;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            nq.push_back(exp->GetNumPoints(d));
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        const auto nodalType = (exp->IsNodalNonTensorialExp())
                                   ? exp->GetNodalPointsKey().GetPointsType()
                                   : LibUtilities::eNoPointsType;

        // Stage A matrix: reference-space derivative, native grid -> native
        // grid, same matrix AdvectionOp's/PhysDerivOp's own Device StdMat
        // backends already use.
        m_derivMatPtr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::ePhysDerivStdMat));

        // Fetch deriv factors (Stage A chain rule) - same key/layout the
        // Device SumFac backend would use with transpose=true, matching
        // AdvectionOp's/PhysDerivOp's own Device StdMat convention.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, true));

        // Fine (over-integrated) point counts.
        const auto nqFine = this->GetScaledNumPoints(nq, m_dealiasScale);
        m_nqFineTot       = NqTot(nqFine);

        // Stage B matrix: native -> fine, same matrix PhysInterp1DScaledOp's
        // own Device StdMat backend uses.
        m_interpMatPtr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::ePhysInterpStdMat,
                                         nodalType, nqFine));

        // Stage D matrix: fine -> native (the new StdMatType added for this
        // operator - see StdMatDataWarehouse.hpp/StdMatDataWarehouseDef.hpp).
        m_projectMatPtr = dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                         StdRegions::eGalerkinProjectStdMat,
                                         nodalType, nqFine));
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
    static constexpr unsigned int m_implInterleaveWidth = 1u;
    static constexpr TData m_dealiasScale               = TData(1.5);

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nqTot;
    unsigned int m_nqFineTot;
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
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>(
                    m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr, m_streamID);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto nhomo    = inblock.GetNumHomoModes();
        const auto ncomp    = inblock.GetNumComponents() * nhomo;
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * nhomo;

        // Initialize pointers.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);
        auto advVelPtr =
            this->m_advVel->template GetPtr<MemSpace, ReadOnly>(m_streamID);

        // Get static workspace pointer. The derivative, the fine-grid
        // gradient and the fine-grid product of every component are held at
        // once - the fine-grid advection velocity is shared across all of
        // them.
        const size_t derivSize      = m_coordDim * m_nqTot * nelmt * ncomp;
        const size_t advVelFineSize = m_coordDim * m_nqFineTot * nelmtTot;
        const size_t gradFineSize   = m_coordDim * m_nqFineTot * nelmt * ncomp;
        const size_t combinedFineSize = m_nqFineTot * nelmt * ncomp;

        auto wspBase =
            BlockOperator<TData>::template GetStaticWorkSpace<MemSpace>(
                derivSize + advVelFineSize + gradFineSize + combinedFineSize,
                m_streamID);
        auto derivWsp      = wspBase;
        auto advVelFineWsp = derivWsp + derivSize;
        auto gradFineWsp   = advVelFineWsp + advVelFineSize;
        auto combinedWsp   = gradFineWsp + gradFineSize;

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Offsets between the components of a block. The derivative and the
        // fine-grid gradient of every component are held at once, so the
        // offset between two directions spans all of them, while the
        // advection velocity is shared by them.
        const auto advVelOffset   = this->m_advVel->CompSize() * nhomo;
        const auto outoffset      = outblock.CompSize() * nhomo;
        const auto derivoffset    = m_nqTot * nelmt * ncomp;
        const auto gradoffset     = m_nqFineTot * nelmt * ncomp;
        const auto compoffset     = m_nqTot * nelmtTot;
        const auto compFineOffset = m_nqFineTot * nelmtTot;

        // Reshape, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            m_implInterleaveWidth, inInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        if (this->m_append)
        {
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, outInterleaveWidth, nelmt * ncomp,
                outblock.GetNumData(), (TData *)outptr, m_streamID);
        }

        // Interpolate the advection velocity to the fine grid once - it is
        // shared by every advected component below. One multiply per
        // coordinate direction, with the homogeneous modes held in the
        // columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqFineTot, nelmtTot, m_nqTot, (TData)1.0,
            m_interpMatPtr, m_nqFineTot, 0, advVelPtr, m_nqTot, advVelOffset,
            (TData)0.0, advVelFineWsp, m_nqFineTot, compFineOffset, m_coordDim);

        // Reference-space derivative. One multiply per direction, with the
        // components and homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmt * ncomp, m_nqTot, (TData)1.0,
            m_derivMatPtr, m_nqTot, m_nqTot * m_nqTot, inptr, m_nqTot, 0,
            (TData)0.0, derivWsp, m_nqTot, derivoffset, m_dimension);

        // Multiply by derivative factor. The derivative factors are indexed
        // by the elements of a single component, so the components are taken
        // one at a time.
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            if (m_isDeformed)
            {
                MultiplyByDerivFactorKernel<ExecSpace, true>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, m_dfptr, derivWsp + n * compoffset,
                    derivWsp + n * compoffset, m_streamID);
            }
            else
            {
                MultiplyByDerivFactorKernel<ExecSpace, false>(
                    m_nqTot, m_coordDim, m_dimension, nelmt, nhomo, derivoffset,
                    derivoffset, m_dfptr, derivWsp + n * compoffset,
                    derivWsp + n * compoffset, m_streamID);
            }
        }

        // Interpolate the physical gradient to the fine grid. One multiply
        // per coordinate direction, with the components and homogeneous modes
        // held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqFineTot, nelmt * ncomp, m_nqTot, (TData)1.0,
            m_interpMatPtr, m_nqFineTot, 0, derivWsp, m_nqTot, derivoffset,
            (TData)0.0, gradFineWsp, m_nqFineTot, gradoffset, m_coordDim);

        // Form scale * advVel . grad(u) on the fine grid. The advection
        // velocity is indexed by the elements of a single component, so the
        // components are taken one at a time.
        for (unsigned int n = 0; n < inblock.GetNumComponents(); ++n)
        {
            AdvectionDealiasCombineStdMatKernel<ExecSpace, false>(
                compFineOffset, m_coordDim, advVelFineWsp, compFineOffset,
                gradFineWsp + n * compFineOffset, gradoffset,
                combinedWsp + n * compFineOffset, this->m_scale, m_streamID);
        }

        // Project back onto the native grid. One multiply per component, with
        // the homogeneous modes held in the columns.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nqTot, nelmtTot, m_nqFineTot, (TData)1.0,
            m_projectMatPtr, m_nqTot, 0, combinedWsp, m_nqFineTot,
            compFineOffset, (TData)this->m_append, outptr, m_nqTot, outoffset,
            inblock.GetNumComponents());

        // Reshape back, if necessary.
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            inblock.GetNumData(), (TData *)inptr, m_streamID);
        LibUtilities::ReshapeStorage<ExecSpace>(
            inInterleaveWidth, m_implInterleaveWidth, nelmt * ncomp,
            outblock.GetNumData(), outptr, m_streamID);

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }
};

} // namespace Nektar::Operators::detail
