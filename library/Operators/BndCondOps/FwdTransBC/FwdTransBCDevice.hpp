///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCDevice.hpp
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

#include "Operators/BndCondOps/FwdTransBC/FwdTransBCBlockOp.hpp"
#include "Operators/BndCondOps/FwdTransBC/FwdTransBCDeviceKernels.hpp"

#include "LibUtilities/BasicUtils/Math/MathKernels.hpp"
#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class FwdTransBCBlockOpImpl : public FwdTransBCBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    FwdTransBCBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : FwdTransBCBlockOp<TData>(block_idx, exp, dataWarehouse)
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
        m_nmTotBnd  = exp->NumBndryCoeffs();
        m_nmTotInt  = m_nmTot - m_nmTotBnd;
        m_numEdges  = exp->GetGeom()->GetNumEdges();

        // Points are always collocation and do not need the below routines.
        if (m_dimension == 0 && m_shapeType == LibUtilities::Point)
        {
            m_isCollocation = true;
            return;
        }

        // Flag for collapsed coordinate correction.
        m_isModified = (exp->GetBasisType(0) == LibUtilities::eModified_A);

        // Check whether nodal-type points are used
        LibUtilities::PointsType nodalType =
            (exp->IsNodalNonTensorialExp())
                ? exp->GetNodalPointsKey().GetPointsType()
                : LibUtilities::eNoPointsType;

        m_isNodal = nodalType != LibUtilities::eNoPointsType;

        ASSERTL0(m_shapeType != LibUtilities::NodalTri,
                 "The shape type NodalTri is not implemented for the "
                 "FwdTransBC operator.")

        // Offset for interior points on segments
        if (m_isModified)
        {
            m_offset_seg = 2;
        }
        else if (exp->GetBasisType(0) == LibUtilities::eGLL_Lagrange)
        {
            m_offset_seg = 1;
        }
        else if (exp->GetBasisType(0) == LibUtilities::eOrtho_A ||
                 exp->GetBasisType(0) == LibUtilities::eGauss_Lagrange)
        {
            m_offset_seg = 0;
            ASSERTL0(
                false,
                "The Gauss_Lagrange or Ortho_A basis requires a FwdTrans "
                "operator for segments. This is not currently implemented.")
        }
        else
        {
            ASSERTL0(false, "FwdTransBC operator only implemented for modal or "
                            "nodal basis type.")
        }

        // Get basis keys for fetching matrices
        std::vector<LibUtilities::BasisKey> basisKeys(
            m_dimension, LibUtilities::NullBasisKey);
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            basisKeys[d] = exp->GetBasis(d)->GetBasisKey();

            // Fetch element size.
            m_nm.push_back(exp->GetBasisNumModes(d));
            m_nq.push_back(exp->GetNumPoints(d));

            // Compute interior size
            m_nmInt.push_back(m_nm[d] - 2);

            // Fetch basis data.
            m_B.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(basisKeys[d],
                                                  LibUtilities::eBasis)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(basisKeys[d],
                                                  LibUtilities::eWeights)));
        }

        // Check whether points are collocated
        // Note do this after basic data m_nm, m_nq, etc has been fetched.
        m_isCollocation = true;
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            m_isCollocation =
                m_isCollocation && exp->GetBasis(d)->Collocation();
        }

        // No need to fetch anything for collocation.
        if (m_isCollocation)
        {
            return;
        }

        // Fetch Jacobian.
        m_jacptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::JacobianKey<TData>(block_idx, m_implInterleaveWidth));

        // Fetch data for Operator2D only
        if ((m_shapeType == LibUtilities::Quad) ||
            (m_shapeType == LibUtilities::Tri) ||
            (m_shapeType == LibUtilities::NodalTri))
        {
            // Fetch interpolation matrix.
            // Note it is from basis[1] to basis[0]
            m_interp1to0 = this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    basisKeys[1], LibUtilities::eInterpTranspose, m_nq[0],
                    basisKeys[0].GetPointsType()));

            // Fetch Jacobian for each segment.
            m_jacTraceptr = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::JacobianTraceKey<TData>(block_idx,
                                                      m_implInterleaveWidth));

            // Fetch TraceToElementMap
            m_traceElmtMapptr =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LocalRegions::TraceToElmtMapKey<TData>(
                        block_idx, m_implInterleaveWidth));
            m_traceElmtSignptr =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LocalRegions::TraceToElmtSignKey<TData>(
                        block_idx, m_implInterleaveWidth));

            // Fetch InteriorMap (volume to interior DoF)
            m_interiorMapptr =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LocalRegions::InteriorMapKey<TData>(block_idx,
                                                        m_implInterleaveWidth));

            // Fetch inverse interior mass matrix for 2D elements
            m_massint = this->m_dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eInvMassInteriorStdMat,
                                             nodalType));
        }

        // Fetch inverse interior mass matrix for each basis
        // Note required for 2D elements to apply to the 1D edges (segments)
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // extract 1D basis key
            auto bkeys = std::vector<LibUtilities::BasisKey>{basisKeys[d]};
            m_massint_seg.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    StdRegions::StdMatKey<TData>(
                        bkeys, LibUtilities::eSegment,
                        StdRegions::eInvMassInteriorStdMat, nodalType)));
        }

        // Fetch NodalToModal Matrix if required.
        if (m_shapeType == LibUtilities::NodalTri)
        {
            m_nodToMod = this->m_dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eNodalToModal,
                                             nodalType));
            m_nodToModTrans = this->m_dataWarehouse->template GetData<MemSpace>(
                StdRegions::StdMatKey<TData>(basisKeys, m_shapeType,
                                             StdRegions::eNodalToModalTranspose,
                                             nodalType));
        }
        else
        {
            m_nodToMod      = (const TData *)nullptr;
            m_nodToModTrans = (const TData *)nullptr;
        }

        // Precompute index, if necessary.
        if (m_dimension == 2)
        {
            const bool indexing = (m_shapeType == LibUtilities::Tri ||
                                   m_shapeType == LibUtilities::NodalTri);
            m_index.push_back(
                indexing ? this->m_dataWarehouse->template GetData<MemSpace>(
                               LibUtilities::ModeIndexKey(m_shapeType, m_nm[0],
                                                          m_nm[1], 0))
                         : nullptr);
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<FwdTransBCBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

    unsigned int m_streamID;
    LibUtilities::ShapeType m_shapeType;
    bool m_isDeformed;
    bool m_isModified;
    bool m_isCollocation;
    bool m_isNodal;
    unsigned int m_dimension;
    unsigned int m_coordDim;
    unsigned int m_nmTot;
    unsigned int m_nqTot;
    unsigned int m_nmTotBnd;
    unsigned int m_nmTotInt;
    unsigned int m_numEdges;
    unsigned int m_offset_seg;
    const unsigned int m_nmBndSeg = 2;

    const TData *m_jacptr;
    const TData *m_jacTraceptr;
    const unsigned int *m_traceElmtMapptr;
    const int *m_traceElmtSignptr;
    const unsigned int *m_interiorMapptr;

    LibUtilities::MemoryRegion<TData> m_dinvmass;

    // Extension to use Generic routines
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nmInt;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_W;
    std::vector<const TData *> m_massint_seg;
    const TData *m_massint;
    const TData *m_interp1to0;
    const TData *m_nodToMod;
    const TData *m_nodToModTrans;
    std::vector<const unsigned int *> m_index;

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::BlockAccessor<TData, FieldState::Coeff>
                     &outblock) override
    {
        switch (m_shapeType)
        {
            // Point
            case LibUtilities::Point:
            {
                Operator0D<LibUtilities::ePoint>(inblock, outblock);
                break;
            }
            // Segment
            case LibUtilities::Seg:
            {
                if (m_isDeformed)
                {
                    Operator1D<LibUtilities::eSegment, true>(inblock, outblock);
                }
                else
                {
                    Operator1D<LibUtilities::eSegment, false>(inblock,
                                                              outblock);
                }
                break;
            }
            // Quads
            case LibUtilities::Quad:
            {
                if (m_isDeformed)
                {
                    Operator2D<LibUtilities::Quad, true>(inblock, outblock);
                }
                else
                {
                    Operator2D<LibUtilities::Quad, false>(inblock, outblock);
                }
                break;
            }
            // Triangles
            case LibUtilities::Tri:
            {
                if (m_isDeformed)
                {
                    Operator2D<LibUtilities::Tri, true>(inblock, outblock);
                }
                else
                {
                    Operator2D<LibUtilities::Tri, false>(inblock, outblock);
                }
                break;
            }
            // // Nodal Triangles (not implemented yet)
            // case LibUtilities::NodalTri:
            // {
            //     if (m_isDeformed)
            //     {
            //         Operator2D<LibUtilities::NodalTri, true>(inblock,
            //         outblock);
            //     }
            //     else
            //     {
            //         Operator2D<LibUtilities::NodalTri, false>(inblock,
            //         outblock);
            //     }
            //     break;
            // }
            default:
                std::cout << "Shape type "
                          << LibUtilities::ShapeTypeMap[m_shapeType]
                          << " not implemented." << std::endl;
        }
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void Operator0D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Reshape, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, interleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);

            // Simple copy for 0D points. No loop required.
            size_t nsize = m_nqTot * nelmt;
            Math::copyKernel<ExecSpace, TData>(nsize, inptr, outptr,
                                               m_streamID);

            // Reshape back, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                outblock.GetNumData(), outptr, m_streamID);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int shmemsize =
            sizeof(TData) * FwdTransBCSharedMemorySize(nm0, nq0);
        const unsigned int blocksize = GetDeviceBlockSize<SumFacTOP>(nq0);
        const unsigned int gridsize =
            GetDeviceGridSize<SumFacTOP>(nelmt, blocksize, shmemsize);

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Reshape, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, interleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);

            // Simple copy for collocated data. Copy per component
            if (m_isCollocation)
            {
                size_t nsize = m_nqTot * nelmt;
                Math::copyKernel<ExecSpace, TData>(nsize, inptr, outptr,
                                                   m_streamID);
            }
            else
            {
                // Launch FwdTransBC kernel
                DEVICE_1DGRID_KERNEL_LAUNCHER(
                    (FwdTransBC1DKernelLauncher<DEFORMED>), gridsize, blocksize,
                    shmemsize, m_streamID, nm0, nq0, nelmt, m_B[0], m_W[0],
                    m_offset_seg, m_massint_seg[0], m_jacptr, inptr, outptr);
            }

            // Reshape back, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                outblock.GetNumData(), outptr, m_streamID);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nqTot = nq0 * nq1;

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>(m_streamID);

        // Fetch deformed mass matrix.
        const TData *massintptr =
            (DEFORMED) ? this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>(
                             m_streamID)
                       : m_massint;

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Set Kernel parameters.
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const unsigned int shmemsize =
            sizeof(TData) *
            FwdTransBCSharedMemorySize(SHAPE_TYPE, nm0, nm1, nq0, nq1);
        const unsigned int blocksize = GetDeviceBlockSize<SumFacTOP>(nqTot);
        const unsigned int gridsize =
            GetDeviceGridSize<SumFacTOP>(nelmt, blocksize, shmemsize);

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Reshape, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, interleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);

            // Simple copy for collocated data. Copy per component
            if (m_isCollocation)
            {
                size_t nsize = nqTot * nelmt;
                Math::copyKernel<ExecSpace, TData>(nsize, inptr, outptr,
                                                   m_streamID);
            }
            else
            {
                DEVICE_1DGRID_KERNEL_LAUNCHER(
                    (FwdTransBC2DKernelLauncher<SHAPE_TYPE, DEFORMED>),
                    gridsize, blocksize, shmemsize, m_streamID, nm0, nm1, nmTot,
                    nq0, nq1, nelmt, m_isModified, m_index[0], m_B[0], m_B[1],
                    m_W[0], m_W[1], m_interp1to0, m_offset_seg,
                    m_massint_seg[0], m_massint_seg[1], m_jacTraceptr,
                    m_traceElmtMapptr, m_traceElmtSignptr, m_nmTotInt,
                    m_interiorMapptr, massintptr, m_jacptr, inptr, outptr);
            }

            // Reshape back, if necessary.
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                inblock.GetNumData(), (TData *)inptr, m_streamID);
            LibUtilities::ReshapeStorage<ExecSpace>(
                interleaveWidth, m_implInterleaveWidth, nelmt,
                outblock.GetNumData(), outptr, m_streamID);

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
            TData>::template FromVector<MemSpace, TData>(invmass, m_streamID);
    }
};

} // namespace Nektar::Operators::detail
