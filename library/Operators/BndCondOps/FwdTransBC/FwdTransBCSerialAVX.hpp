///////////////////////////////////////////////////////////////////////////////
//
// File: FwdTransBCSerialAVX.hpp
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

#include <LibUtilities/SimdLib/tinysimd.hpp>
#include <boost/intrusive/pointer_traits.hpp>

#include "Operators/BndCondOps/FwdTransBC/FwdTransBCBlockOp.hpp"
#include "Operators/BndCondOps/FwdTransBC/FwdTransBCSerialAVXKernels.hpp"

#include "Operators/Math/MathKernels.hpp"
#include "Operators/NekBlas/NekBlas.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class FwdTransBCBlockOpImpl : public FwdTransBCBlockOp<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
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
                BasisDataKey<simd_t>(basisKeys[d], eBasis)));
            m_W.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                BasisDataKey<simd_t>(basisKeys[d], eWeights)));
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
                BasisDataKey<simd_t>(basisKeys[1], eInterpTranspose, m_nq[0],
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

            // Fetch inverse interior mass matrix for 2D elements
            m_massint = this->m_dataWarehouse->template GetData<MemSpace>(
                StdMatKey<simd_t>(basisKeys, m_shapeType,
                                  eInvMassInteriorStdMat, nodalType));
        }

        // Fetch inverse interior mass matrix for each basis
        // Note required for 2D elements to apply to the 1D edges (segments)
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // extract 1D basis key
            auto bkeys = std::vector<LibUtilities::BasisKey>{basisKeys[d]};
            m_massint_seg.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    StdMatKey<simd_t>(bkeys, LibUtilities::eSegment,
                                      eInvMassInteriorStdMat, nodalType)));
        }

        // Fetch NodalToModal Matrix if required.
        if (m_shapeType == LibUtilities::NodalTri)
        {
            m_nodToMod = this->m_dataWarehouse->template GetData<MemSpace>(
                StdMatKey<simd_t>(basisKeys, m_shapeType, eNodalToModal,
                                  nodalType));
            m_nodToModTrans = this->m_dataWarehouse->template GetData<MemSpace>(
                StdMatKey<simd_t>(basisKeys, m_shapeType,
                                  eNodalToModalTranspose, nodalType));
        }
        else
        {
            m_nodToMod      = (const simd_t *)nullptr;
            m_nodToModTrans = (const simd_t *)nullptr;
        }

        // Allocate workspace.
        if (m_dimension == 1)
        {
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_nqTot);
            m_wsp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_nqTot);
        }
        else if (m_dimension == 2)
        {
            m_wsp1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_nqTot);
            m_wsp2 = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_nqTot);
            m_wsp3 = std::vector<simd_t, tinysimd::allocator<simd_t>>(m_nqTot);
            m_wsp4 = std::vector<simd_t, tinysimd::allocator<simd_t>>(
                std::max(m_nq[0], m_nq[1]));
        }
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<FwdTransBCBlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    static constexpr unsigned int m_implInterleaveWidth = simd_t::width;

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

    MemoryRegion<TData> m_dinvmass;

    // Extension to use Generic routines
    std::vector<unsigned int> m_nm;
    std::vector<unsigned int> m_nmInt;
    std::vector<unsigned int> m_nq;
    std::vector<const simd_t *> m_B;
    std::vector<const simd_t *> m_W;
    std::vector<const simd_t *> m_massint_seg;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp1;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp2;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp3;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp4;
    const simd_t *m_interp1to0;
    const simd_t *m_massint;
    const simd_t *m_nodToMod;
    const simd_t *m_nodToModTrans;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI system
    // is saturated with warnings
    bool m_warnOnce = false;
#endif

    void v_Apply(BlockAccessor<TData, FieldState::Phys> &inblock,
                 BlockAccessor<TData, FieldState::Coeff> &outblock) override
    {
        WARNINGL1(
            m_warnOnce || (inblock.GetAlignment() % simd_t::alignment == 0 &&
                           outblock.GetAlignment() % simd_t::alignment == 0),
            "Input or output Field are not aligned to the required alignment "
            "for the SIMD vector type.");
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
        m_warnOnce = true;
#endif

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
    void Operator0D(BlockAccessor<TData, FieldState::Phys> &inblock,
                    BlockAccessor<TData, FieldState::Coeff> &outblock)
    {
        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize,
                                              m_nqTot, (TData *)inptr);
                }

                // Simple copy for 0D points. No loop required.
                size_t nsize = m_nqTot * simd_t::width;
                copyKernel<ExecSpace, TData>(nsize, inptr, outptr);

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * m_nqTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        m_nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * m_nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += m_nqTot * simd_t::width;
                outptr += m_nmTot * simd_t::width;
            }
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

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            auto jacptr = m_jacptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nqTot,
                                              (TData *)inptr);
                }

                // Simple copy for collocated data. Copy per component
                if (m_isCollocation)
                {
                    size_t nsize = nqTot * simd_t::width;
                    copyKernel<ExecSpace>(nsize, inptr, outptr);
                }
                else
                {
                    FwdTransBC1DKernel<ExecSpace, SHAPE_TYPE, DEFORMED>(
                        nm0, nq0, m_B[0], m_W[0], m_offset_seg,
                        reinterpret_cast<const simd_t *>(m_massint_seg[0]),
                        reinterpret_cast<const simd_t *>(jacptr),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr), m_wsp1.data(),
                        m_wsp2.data());
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += nmTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
            }
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

        const auto nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);
        const auto nqTot = nq0 * nq1;

        unsigned int jacSize = 1;
        if constexpr (DEFORMED)
        {
            jacSize *= nqTot;
        }

        // Initialize pointers.
        auto inptr  = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outptr = outblock.template GetPtr<MemSpace, WriteOnly>();

        // Get interleave parameter.
        const auto interleaveWidth = inblock.GetInterleaveWidth();
        const auto width_ratio     = (interleaveWidth == 1)
                                         ? 1
                                         : interleaveWidth / m_implInterleaveWidth;
        const auto chunkSize = std::max(m_implInterleaveWidth, interleaveWidth);

        // Loop over components.
        for (unsigned int nc = 0;
             nc < inblock.GetNumComponents() * inblock.GetNumHomoModes(); ++nc)
        {
            auto jacptr      = m_jacptr;
            auto jacTraceptr = m_jacTraceptr;

            // Fetch deformed mass matrix.
            auto massintptr =
                (DEFORMED)
                    ? this->m_dinvmass.template GetPtr<MemSpace, ReadOnly>()
                    : (const TData *)m_massint;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace>(m_implInterleaveWidth,
                                              interleaveWidth, chunkSize, nqTot,
                                              (TData *)inptr);
                }

                // Simple copy for collocated data. Copy per component
                if (m_isCollocation)
                {
                    size_t nsize = nqTot * simd_t::width;
                    copyKernel<ExecSpace>(nsize, inptr, outptr);
                }
                else
                {
                    FwdTransBC2DKernel<ExecSpace, SHAPE_TYPE, DEFORMED>(
                        nm0, nm1, nq0, nq1, m_isModified, m_B[0], m_B[1],
                        m_W[0], m_W[1],
                        reinterpret_cast<const simd_t *>(m_interp1to0),
                        m_offset_seg,
                        reinterpret_cast<const simd_t *>(m_massint_seg[0]),
                        reinterpret_cast<const simd_t *>(m_massint_seg[1]),
                        reinterpret_cast<const simd_t *>(m_jacTraceptr),
                        m_traceElmtMapptr, m_traceElmtSignptr, m_nmTotInt,
                        m_interiorMapptr,
                        reinterpret_cast<const simd_t *>(massintptr),
                        reinterpret_cast<const simd_t *>(m_jacptr),
                        reinterpret_cast<const simd_t *>(inptr),
                        reinterpret_cast<simd_t *>(outptr), m_wsp1.data(),
                        m_wsp2.data(), m_wsp3.data(), m_wsp4.data());
                }

                // Reshape back, if necessary.
                if (e % width_ratio == width_ratio - 1)
                {
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nqTot,
                        (TData *)inptr -
                            (width_ratio - 1) * nqTot * simd_t::width);
                    ReshapeStorage<ExecSpace>(
                        interleaveWidth, m_implInterleaveWidth, chunkSize,
                        nmTot,
                        (TData *)outptr -
                            (width_ratio - 1) * nmTot * simd_t::width);
                }

                // Increment pointers for the next elmt group.
                inptr += nqTot * simd_t::width;
                outptr += nmTot * simd_t::width;
                jacptr += jacSize * simd_t::width;
                jacTraceptr += m_numEdges * simd_t::width;
                if constexpr (DEFORMED)
                {
                    massintptr += m_nmTotInt * m_nmTotInt * simd_t::width;
                }
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(interleaveWidth);
    }

    void v_SetInvMassMatrix(std::vector<TData> &invmass) override
    {
        this->m_dinvmass =
            MemoryRegion<TData>::template FromVector<MemSpace, TData>(invmass);
        auto dmatptr = this->m_dinvmass.template GetPtr<MemSpace, ReadWrite>();

        if (this->m_dinvmass.size())
        {
            ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, 1,
                this->m_dinvmass.size() / (m_nmTotInt * m_nmTotInt),
                m_nmTotInt * m_nmTotInt, (TData *)dmatptr);
        }
    }
};

} // namespace Nektar::Operators::detail
