///////////////////////////////////////////////////////////////////////////////
//
// File: PhysGalerkinProject1DScaledDeviceStdMat.hpp
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
#include "Operators/ElmtOps/PhysGalerkinProject1DScaled/PhysGalerkinProject1DScaledBlockOp.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class PhysGalerkinProject1DScaledBlockOpImpl
    : public PhysGalerkinProject1DScaledBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    PhysGalerkinProject1DScaledBlockOpImpl(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : PhysGalerkinProject1DScaledBlockOp<TData>(block_idx, exp,
                                                    dataWarehouse)
    {
        m_streamID = block_idx + 1;

        // Determine shape and type of the element.
        m_shapeType = exp->DetShapeType();
        m_isDeformed =
            exp->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed;
        m_dimension = exp->GetShapeDimension();
        m_coordDim  = exp->GetCoordim();

        // Native (target) quadrature point counts - fixed, independent of
        // the over-integration scale factor.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_nm.push_back(exp->GetNumPoints(d));
        }

        // Fetch basis key.
        m_basisKeys = std::vector<LibUtilities::BasisKey>(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_basisKeys[d] = exp->GetBasis(d)->GetBasisKey();
        }

        m_nodalType = (exp->IsNodalNonTensorialExp())
                          ? exp->GetNodalPointsKey().GetPointsType()
                          : LibUtilities::eNoPointsType;
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
        return std::make_unique<PhysGalerkinProject1DScaledBlockOpImpl<
            ExecSpace, Implementation, TData>>(block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_streamID;
    std::vector<LibUtilities::BasisKey> m_basisKeys;
    LibUtilities::ShapeType m_shapeType;
    LibUtilities::PointsType m_nodalType;
    bool m_isDeformed;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    // m_nm is the native (target/output) grid, m_nq the scaled
    // (source/input) grid v_SetScaleFactor derives from it as
    // m_scale * m_nm - the same naming the data warehouse uses for this
    // matrix, and the same PhysInterp1DScaled uses for the reverse
    // direction.
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nq;
    const TData *m_matptr;

    void v_Apply(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock) override
    {
        ASSERTL1(this->m_scale != -1.0,
                 "Scale factor has not been initialised");

        // Get BLAS handle.
        auto handle = NekBlas::Handle<ExecSpace>::GetInstance(m_streamID);

        // Get block sizes.
        const auto ncomp =
            inblock.GetNumComponents() * inblock.GetNumHomoModes();
        const auto nelmt    = inblock.GetNumElementsWithPadding();
        const auto nelmtTot = nelmt * inblock.GetNumHomoModes();

        // Initialize pointers. Appending accumulates into the pre-existing
        // output values, so they are read as well as written.
        auto inptr = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr =
            (this->m_append)
                ? outblock.template GetPtr<MemSpace, ReadWrite>(m_streamID)
                : outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();

        // Offsets between the components of a block.
        const auto inoffset  = inblock.CompSize() * inblock.GetNumHomoModes();
        const auto outoffset = outblock.CompSize() * outblock.GetNumHomoModes();

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

        // Perform batched matrix-matrix multiply, one multiply per component,
        // with the homogeneous modes held in the columns. Appending is a beta
        // of one on the projection.
        NekBlas::GemmStridedBatched(
            handle, "N", "N", m_nmTot, nelmtTot, m_nqTot, (TData)1.0, m_matptr,
            m_nmTot, 0, inptr, m_nqTot, inoffset, (TData)this->m_append, outptr,
            m_nmTot, outoffset, inblock.GetNumComponents());

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

    void v_SetScaleFactor(const TData &scale) override
    {
        this->m_scale = scale;
        m_nq          = this->GetScaledNumPoints(m_nm, this->m_scale);

        m_nmTot =
            std::accumulate(m_nm.begin(), m_nm.end(), 1, std::multiplies());
        m_nqTot =
            std::accumulate(m_nq.begin(), m_nq.end(), 1, std::multiplies());

        // The Galerkin projection matrix, from the scaled grid (m_nq) down
        // to this basis' native quadrature (m_nm).
        m_matptr = this->m_dataWarehouse->template GetData<MemSpace>(
            StdRegions::StdMatKey<TData>(m_basisKeys, m_shapeType,
                                         StdRegions::eGalerkinProjectStdMat,
                                         m_nodalType, m_nq));
    }
};

} // namespace Nektar::Operators::detail
