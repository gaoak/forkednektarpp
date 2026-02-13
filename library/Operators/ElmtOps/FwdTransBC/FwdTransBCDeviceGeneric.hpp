///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCDeviceGeneric.hpp
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

#include "Operators/ElmtOps/FwdTransBC/FwdTransBCBlockOp.hpp"
#include "Operators/ElmtOps/FwdTransBC/FwdTransBCDeviceGenericKernels.hpp"

#include "Operators/Math/MathKernels.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class FwdTransBCBlockOpImpl : public FwdTransBCBlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    FwdTransBCBlockOpImpl(const unsigned int block_idx,
                          const LocalRegions::ExpansionSharedPtr &exp,
                          NekDataWarehouseSharedPtr dataWarehouse)
        : FwdTransBCBlockOp<TData>(block_idx, exp, dataWarehouse)
    {
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
                BasisDataKey<TData>(basisKeys[d], eBasis)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(basisKeys[d], eWeights)));
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
            JacobianKey<TData>(block_idx, m_implInterleaveWidth));

        // Fetch data for Operator2D only
        if ((m_shapeType == LibUtilities::Quad) ||
            (m_shapeType == LibUtilities::Tri) ||
            (m_shapeType == LibUtilities::NodalTri))
        {
            // Fetch interpolation matrix.
            // Note it is from basis[1] to basis[0]
            m_interp1to0 = this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<TData>(basisKeys[1], eInterpTranspose, m_nq[0],
                                    basisKeys[0].GetPointsType()));

            // Fetch Jacobian for each segment.
            m_jacTraceptr = this->m_dataWarehouse->template GetData<MemSpace>(
                JacobianTraceKey<TData>(block_idx, m_implInterleaveWidth));

            // Fetch TraceToElementMap
            m_traceElmtMapptr =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    TraceToElmtMapKey<TData>(block_idx, m_implInterleaveWidth));
            m_traceElmtSignptr =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    TraceToElmtSignKey<TData>(block_idx,
                                              m_implInterleaveWidth));

            // Fetch InteriorMap (volume to interior DoF)
            m_interiorMapptr =
                this->m_dataWarehouse->template GetData<MemSpace>(
                    InteriorMapKey<TData>(block_idx, m_implInterleaveWidth));
        }

        // Fetch inverse interior mass matrix for each basis
        // Note required for 2D elements to apply to the 1D edges (segments)
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // extract 1D basis key
            auto bkeys = std::vector<LibUtilities::BasisKey>{basisKeys[d]};
            m_massint_seg.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    StdMatKey<TData>(bkeys, LibUtilities::eSegment,
                                     eInvMassInteriorStdMat, nodalType)));
        }
        m_massint =
            this->m_dataWarehouse->template GetData<MemSpace>(StdMatKey<TData>(
                basisKeys, m_shapeType, eInvMassInteriorStdMat, nodalType));

        // Fetch NodalToModal Matrix if required.
        if (m_shapeType == LibUtilities::NodalTri)
        {
            m_nodToMod = this->m_dataWarehouse->template GetData<MemSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType, eNodalToModal,
                                 nodalType));
            m_nodToModTrans = this->m_dataWarehouse->template GetData<MemSpace>(
                StdMatKey<TData>(basisKeys, m_shapeType, eNodalToModalTranspose,
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
                               ModeIndexKey(m_shapeType, m_nm[0], m_nm[1], 0))
                         : nullptr);
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<
        ElmtBlockOp<FieldState::Phys, FieldState::Coeff, TData>>
    Instantiate(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<
            FwdTransBCBlockOpImpl<ExecSpace, Implementation, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = 1u;

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
    const TData *m_massint;
    std::vector<const TData *> m_massint_seg;

    MemoryRegion<TData> m_dinvmass;
    MemoryRegion<TData> m_wsp1;
    MemoryRegion<TData> m_wsp2;
    MemoryRegion<TData> m_wsp3;
    MemoryRegion<TData> m_wsp4;

    // Extension to use Generic routines
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nmInt;
    std::vector<unsigned int> m_nq;
    std::vector<const TData *> m_B;
    std::vector<const TData *> m_W;
    const TData *m_interp1to0;
    const TData *m_nodToMod;
    const TData *m_nodToModTrans;
    std::vector<const unsigned int *> m_index;

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Coeff> &outblock) override
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

    size_t GetWorkspaceSize(const LibUtilities::ShapeType shapeType,
                            const size_t nelmt,
                            [[maybe_unused]] const unsigned int nq0,
                            [[maybe_unused]] const unsigned int nq1,
                            [[maybe_unused]] const unsigned int nq2,
                            [[maybe_unused]] const unsigned int nm0,
                            [[maybe_unused]] const unsigned int nm1,
                            [[maybe_unused]] const unsigned int nm2)
    {
        size_t wspsize = 0;

        if (shapeType == LibUtilities::Point)
        {
            wspsize = 0;
        }
        else if (shapeType == LibUtilities::Seg)
        {
            wspsize = nq0 * nelmt;
        }
        else if (shapeType == LibUtilities::Quad)
        {
            wspsize = nq0 * nq1 * nelmt;
        }
        else if (shapeType == LibUtilities::Tri)
        {
            wspsize = nq0 * nq1 * nelmt;
        }
        else if (shapeType == LibUtilities::NodalTri)
        {
            wspsize = (nq1 + nm0 * (nm0 + 1) / 2) * nelmt;
        }

        return wspsize;
    }

    MemoryRegion<TData> SetWorkspace(
        const LibUtilities::ShapeType shapeType, const size_t nelmt,
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
    {
        auto wspsize =
            GetWorkspaceSize(shapeType, nelmt, nq0, nq1, nq2, nm0, nm1, nm2);

        return MemoryRegion<TData>(wspsize);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void Operator0D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Simple copy for 0D points. No loop required.
            size_t nsize = m_nqTot * nelmt;
            copyKernel<ExecSpace, TData>(nsize, inptr, outptr);

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator1D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nq0 = m_nq[0];

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if (m_wsp1.size() == 0)
        {
            m_wsp1 = SetWorkspace(SHAPE_TYPE, nelmt, nq0, 0, 0, nm0, 0, 0);
        }
        if (m_wsp2.size() == 0)
        {
            m_wsp2 = SetWorkspace(SHAPE_TYPE, nelmt, nq0, 0, 0, nm0, 0, 0);
        }

        // Get workspace pointer.
        auto wspptr1 = m_wsp1.template GetPtr<MemSpace, WriteOnly>();
        auto wspptr2 = m_wsp2.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Simple copy for collocated data. Copy per component
            if (m_isCollocation)
            {
                size_t nsize = m_nqTot * nelmt;
                copyKernel<ExecSpace, TData>(nsize, inptr, outptr);
            }
            else
            {
                // Launch FwdTransBC kernel
                FwdTransBC1DKernel<ExecSpace, DEFORMED>(
                    nm0, nq0, nelmt, m_B[0], m_W[0], m_offset_seg,
                    m_massint_seg[0], m_jacptr, inptr, outptr, wspptr1,
                    wspptr2);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    // Non-size based operator.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED>
    void Operator2D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Shape size.
        const auto nm0 = m_nm[0];
        const auto nm1 = m_nm[1];

        const auto nq0 = m_nq[0];
        const auto nq1 = m_nq[1];

        const auto nqTot = nq0 * nq1;

        const auto nelmt = inblock.GetNumElementsWithPadding();

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Set workspace.
        if (m_wsp1.size() == 0)
        {
            m_wsp1 = SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, 0, nm0, nm1, 0);
        }
        if (m_wsp2.size() == 0)
        {
            m_wsp2 = SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, 0, nm0, nm1, 0);
        }
        if (m_wsp3.size() == 0)
        {
            m_wsp3 = SetWorkspace(SHAPE_TYPE, nelmt, nq0, nq1, 0, nm0, nm1, 0);
        }
        if (m_wsp4.size() == 0)
        {
            m_wsp4 = MemoryRegion<TData>(
                static_cast<size_t>(std::max(nq0, nq1)) * nelmt);
        }

        // Get workspace pointer.
        auto wspptr1 = m_wsp1.template GetPtr<MemSpace, WriteOnly>();
        auto wspptr2 = m_wsp2.template GetPtr<MemSpace, WriteOnly>();
        auto wspptr3 = m_wsp3.template GetPtr<MemSpace, WriteOnly>();
        auto wspptr4 = m_wsp4.template GetPtr<MemSpace, WriteOnly>();

        // Fetch deformed mass matrix.
        const TData *dmatptr = nullptr;
        if constexpr (DEFORMED)
        {
            dmatptr = this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>();
        }

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Reshape, if necessary.
            ReshapeStorage<ExecSpace>(m_implInterleaveWidth, interleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);

            // Simple copy for collocated data. Copy per component
            if (m_isCollocation)
            {
                size_t nsize = nqTot * nelmt;
                copyKernel<ExecSpace, TData>(nsize, inptr, outptr);
            }
            else
            {
                FwdTransBC2DKernel<SHAPE_TYPE, ExecSpace, DEFORMED>(
                    nm0, nm1, nq0, nq1, nelmt, m_isModified, m_index[0], m_B[0],
                    m_B[1], m_W[0], m_W[1], m_interp1to0, m_offset_seg,
                    m_massint_seg[0], m_massint_seg[1], m_jacTraceptr,
                    m_traceElmtMapptr, m_traceElmtSignptr, m_nmTotInt,
                    m_interiorMapptr, m_massint, dmatptr, m_jacptr, inptr,
                    outptr, wspptr1, wspptr2, wspptr3, wspptr4);
            }

            // Reshape back, if necessary.
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, inblock.GetNumData(),
                                      (TData *)inptr);
            ReshapeStorage<ExecSpace>(interleaveWidth, m_implInterleaveWidth,
                                      nelmt, outblock.GetNumData(), outptr);

            // Increment pointers.
            inptr += inblock.CompSize() * inblock.GetNumHomoModes();
            outptr += outblock.CompSize() * outblock.GetNumHomoModes();
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
