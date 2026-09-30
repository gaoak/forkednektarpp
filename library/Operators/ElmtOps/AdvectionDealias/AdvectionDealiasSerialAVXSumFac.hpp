///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionDealiasSerialAVXSumFac.hpp
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
// Description: Fused 3/2-rule dealiased advection, SumFac implementation.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "LibUtilities/BasicUtils/Utils/UtilsKernels.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasBlockOp.hpp"
#include "Operators/ElmtOps/AdvectionDealias/AdvectionDealiasSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

// Selects the switch construction used by the generated ShapeBlock
// definitions (see LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in).
#define NEKTAR_BLOCKOP_SWITCH_1DCOORDS

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class AdvectionDealiasBlockOpImpl : public AdvectionDealiasBlockOp<TData>
{
    using BlockOpBase = AdvectionDealiasBlockOp<TData>;
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

        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Native element size.
            m_nq.push_back(exp->GetNumPoints(d));

            // Basis derivative matrix (Stage A).
            m_D.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TData>(
                    exp->GetBasis(d)->GetBasisKey(),
                    LibUtilities::eDerivative)));
        }

        // Fetch geometric factors.
        if (m_dimension == 2)
        {
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }
        else if (m_dimension == 3)
        {
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(0)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eHalfMultOnePlusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(1)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
            m_f.push_back(this->m_dataWarehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<simd_t>(
                    exp->GetBasis(2)->GetBasisKey(),
                    LibUtilities::eTwoOverOneMinusZero)));
        }

        // Fetch derivative factors.
        m_dfptr = this->m_dataWarehouse->template GetData<MemSpace>(
            LocalRegions::DerivFactorKey<TData>(block_idx,
                                                m_implInterleaveWidth, false));

        // Fine (over-integrated) point counts.
        m_nqFine = this->GetScaledNumPoints(m_nq, m_dealiasScale);

        // Fetch interpolation and projection basis data.
        for (unsigned int d = 0; d < m_dimension; d++)
        {
            // Interpolation matrix: native -> fine.
            m_Binterp.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        exp->GetBasis(d)->GetBasisKey(), LibUtilities::eInterp,
                        m_nqFine[d])));
            // Galerkin projection matrix: fine -> native.
            m_Bproject.push_back(
                this->m_dataWarehouse->template GetData<MemSpace>(
                    LibUtilities::BasisDataKey<TData>(
                        exp->GetBasis(d)->GetBasisKey(),
                        LibUtilities::eGalerkinProject, m_nqFine[d])));
        }

        // Reuse the same tensor-product workspace for interpolation and
        // projection.
        unsigned int wsp0Size = 0, wsp1Size = 0;
        if (m_dimension == 2)
        {
            unsigned int interpSize = 0, projectSize = 0;
            BwdTrans2DWorkspace(LibUtilities::Quad, m_nq[0], m_nq[1],
                                m_nqFine[0], m_nqFine[1], interpSize);
            BwdTrans2DWorkspace(LibUtilities::Quad, m_nqFine[0], m_nqFine[1],
                                m_nq[0], m_nq[1], projectSize);
            wsp0Size = std::max(interpSize, projectSize);
        }
        else if (m_dimension == 3)
        {
            unsigned int interpSize0 = 0, interpSize1 = 0, projectSize0 = 0,
                         projectSize1 = 0;
            BwdTrans3DWorkspace(LibUtilities::Hex, m_nq[0], m_nq[1], m_nq[2],
                                m_nqFine[0], m_nqFine[1], m_nqFine[2],
                                interpSize0, interpSize1);
            BwdTrans3DWorkspace(LibUtilities::Hex, m_nqFine[0], m_nqFine[1],
                                m_nqFine[2], m_nq[0], m_nq[1], m_nq[2],
                                projectSize0, projectSize1);
            wsp0Size = std::max(interpSize0, projectSize0);
            wsp1Size = std::max(interpSize1, projectSize1);
        }
        m_wspBT0 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp0Size);
        m_wspBT1 = std::vector<simd_t, tinysimd::allocator<simd_t>>(wsp1Size);

        // Main workspace. Only one component's derivative, gradient and
        // combined-result are held at a time - the fine-grid advection
        // velocity is shared across all components.
        const unsigned int nqTot            = NqTot(m_nq);
        const unsigned int nqTotFine        = NqTot(m_nqFine);
        const unsigned int derivSize        = m_coordDim * nqTot;
        const unsigned int advVelFineSize   = m_coordDim * nqTotFine;
        const unsigned int gradFineSize     = m_coordDim * nqTotFine;
        const unsigned int combinedFineSize = nqTotFine;
        m_wsp = std::vector<simd_t, tinysimd::allocator<simd_t>>(
            derivSize + advVelFineSize + gradFineSize + combinedFineSize);
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
    std::vector<unsigned int> m_nq;
    std::vector<unsigned int> m_nqFine;
    std::vector<const TData *> m_D;
    std::vector<const simd_t *> m_f;
    std::vector<const TData *> m_Binterp;
    std::vector<const TData *> m_Bproject;
    const TData *m_dfptr;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wspBT0;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wspBT1;
    std::vector<simd_t, tinysimd::allocator<simd_t>> m_wsp;
#if defined(NEKTAR_DEBUG) || defined(NEKTAR_FULLDEBUG)
    // flag to ensure we only get one warning for alignment otherwise CI
    // system is saturated with warnings
    bool m_warnOnce = false;
#endif

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
            // Segment
            case LibUtilities::Seg:
            {
                ShapeBlock<LibUtilities::Seg>(inblock, outblock);
                break;
            }
            // Quadrilateral
            case LibUtilities::Quad:
            {
                ShapeBlock<LibUtilities::Quad>(inblock, outblock);
                break;
            }
            // Triangle
            case LibUtilities::Tri:
            {
                ShapeBlock<LibUtilities::Tri>(inblock, outblock);
                break;
            }
            // Nodal triangle
            case LibUtilities::NodalTri:
            {
                ShapeBlock<LibUtilities::NodalTri>(inblock, outblock);
                break;
            }
            // Hexahedron
            case LibUtilities::Hex:
            {
                ShapeBlock<LibUtilities::Hex>(inblock, outblock);
                break;
            }
            // Tetrahedron
            case LibUtilities::Tet:
            {
                ShapeBlock<LibUtilities::Tet>(inblock, outblock);
                break;
            }
            // Nodal tetrahedron
            case LibUtilities::NodalTet:
            {
                ShapeBlock<LibUtilities::NodalTet>(inblock, outblock);
                break;
            }
            // Pyramid
            case LibUtilities::Pyr:
            {
                ShapeBlock<LibUtilities::Pyr>(inblock, outblock);
                break;
            }
            // Prism
            case LibUtilities::Prism:
            {
                ShapeBlock<LibUtilities::Prism>(inblock, outblock);
                break;
            }
            // Nodal prism
            case LibUtilities::NodalPrism:
            {
                ShapeBlock<LibUtilities::NodalPrism>(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

    // Shape specific block operator, specialised for each shape in
    // LibUtilities/BasicUtils/Switch/BlockOpShapeBlock.cpp.in.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    void ShapeBlock(typename BlockOpBase::InBlock &inblock,
                    typename BlockOpBase::OutBlock &outblock);

    // Number of collapsed coordinate factors used by the kernels in dim
    // dimensions.
    static constexpr unsigned int NumFactor(const unsigned int dim)
    {
        return (dim == 1) ? 0 : (dim == 2) ? 2 : 4;
    }

    // Generic operator. Builds the index sequences from the shape dimension
    // and forwards to OperatorNDImpl. sizeParam describes the native
    // (pre-interpolation) grid only; the over-integrated "fine" grid is a
    // fixed 3/2 scaling of it and is always looked up from m_nqFine, whether
    // or not sizeParam is templated.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter>
    NEK_FORCE_INLINE void OperatorND(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // sizeParam is built by the switch in
        // LibUtilities/BasicUtils/Switch/BlockOpSwitch1DCoordsCode.h.in.
        static_assert(
            (DIM == 1 && IsPhysSizeParameter1D_v<TPhysSizeParameter>) ||
                (DIM == 2 && IsPhysSizeParameter2D_v<TPhysSizeParameter>) ||
                (DIM == 3 && IsPhysSizeParameter3D_v<TPhysSizeParameter>),
            "OperatorND expects a size parameter matching the dimension of the "
            "shape.");

        OperatorNDImpl<SHAPE_TYPE, DEFORMED>(
            inblock, outblock, sizeParam,
            std::make_integer_sequence<unsigned int, DIM>(),
            std::make_integer_sequence<unsigned int, NumFactor(DIM)>());
    }

    // Generic operator implementation. ind0 indexes each direction,
    // ind1 the collapsed coordinate factors used by the kernels.
    template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
              typename TPhysSizeParameter, unsigned int... ind0,
              unsigned int... ind1>
    NEK_FORCE_INLINE void OperatorNDImpl(
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
        LibUtilities::BlockAccessor<TData, FieldState::Phys> &outblock,
        TPhysSizeParameter sizeParam,
        std::integer_sequence<unsigned int, ind0...>,
        std::integer_sequence<unsigned int, ind1...>)
    {
        constexpr unsigned int DIM = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];

        // Reshape advection velocity, if necessary.
        if (this->m_advVel->GetInterleaveWidth() != m_implInterleaveWidth)
        {
            auto advVelPtr =
                this->m_advVel->template GetPtr<MemSpace, ReadWrite>();
            LibUtilities::ReshapeStorage<ExecSpace>(
                m_implInterleaveWidth, this->m_advVel->GetInterleaveWidth(),
                this->m_advVel->GetNumElementsWithPadding() *
                    this->m_advVel->GetNumComponents() *
                    this->m_advVel->GetNumHomoModes(),
                this->m_advVel->GetNumData(), advVelPtr);
            this->m_advVel->template SetInterleaveWidth<TData>(
                m_implInterleaveWidth);
        }

        // Shape size.
        const unsigned int nqTot     = sizeParam.nqTot();
        const unsigned int nqTotFine = NqTot(m_nqFine);

        unsigned int dfsize = m_coordDim * DIM;
        if constexpr (DEFORMED)
        {
            dfsize *= nqTot;
        }

        // Initialize pointers.
        auto inbase     = inblock.template GetPtr<MemSpace, ReadOnly>();
        auto outbase    = (this->m_append)
                              ? outblock.template GetPtr<MemSpace, ReadWrite>()
                              : outblock.template GetPtr<MemSpace, WriteOnly>();
        auto advVelBase = this->m_advVel->template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        const auto inInterleaveWidth  = inblock.GetInterleaveWidth();
        const auto outInterleaveWidth = outblock.GetInterleaveWidth();
        const auto width_ratio =
            (inInterleaveWidth == 1)
                ? 1
                : inInterleaveWidth / m_implInterleaveWidth;
        const auto chunkSize =
            std::max(m_implInterleaveWidth, inInterleaveWidth);

        // Get workspace pointers.
        auto derivNative  = m_wsp.data();
        auto advVelFine   = derivNative + m_coordDim * nqTot;
        auto gradFine     = advVelFine + m_coordDim * nqTotFine;
        auto combinedFine = gradFine + m_coordDim * nqTotFine;

        // Offsets between the components of a block.
        const auto nhomo       = inblock.GetNumHomoModes();
        const auto nComp       = inblock.GetNumComponents();
        const auto inCompSize  = inblock.CompSize();
        const auto outCompSize = outblock.CompSize();
        // The velocity is shared by the variables but not by the planes, so
        // one component of it spans them all and one plane of it steps
        // within that.
        const auto advVelPlaneSize = this->m_advVel->CompSize();
        const auto advVelCompSize =
            advVelPlaneSize * this->m_advVel->GetNumHomoModes();

        // Loop over the planes.
        for (unsigned int p = 0; p < nhomo; ++p)
        {
            // Every plane walks the same elements, so the element pointers
            // restart here.
            auto inptr     = inbase;
            auto outptr    = outbase;
            auto advVelPtr = advVelBase;
            auto dfptr     = m_dfptr;

            // Loop over element groups.
            for (size_t e = 0;
                 e < inblock.GetNumElmtGroups(m_implInterleaveWidth); ++e)
            {
                // Every variable on this plane advects with this plane's
                // velocity, so it is interpolated to the fine grid once here
                // rather than once per variable.
                for (unsigned int d = 0; d < m_coordDim; ++d)
                {
                    InterpKernel<DIM, false>(
                        reinterpret_cast<const simd_t *>((TData *)advVelPtr +
                                                         d * advVelCompSize +
                                                         p * advVelPlaneSize),
                        advVelFine + d * nqTotFine);
                }

                // Loop over components.
                for (unsigned int c = 0; c < nComp; ++c)
                {
                    // The planes of a variable are consecutive.
                    TData *inCompPtr =
                        (TData *)inptr + (c * nhomo + p) * inCompSize;
                    TData *outCompPtr =
                        (TData *)outptr + (c * nhomo + p) * outCompSize;

                    // Reshape, if necessary.
                    if (e % width_ratio == 0)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            m_implInterleaveWidth, inInterleaveWidth, chunkSize,
                            nqTot, inCompPtr);
                        if (this->m_append)
                        {
                            LibUtilities::ReshapeStorage<ExecSpace>(
                                m_implInterleaveWidth, outInterleaveWidth,
                                chunkSize, nqTot, outCompPtr);
                        }
                    }

                    simd_t *derivOut[3] = {nullptr, nullptr, nullptr};
                    for (unsigned int d = 0; d < m_coordDim; ++d)
                    {
                        derivOut[d] = derivNative + d * nqTot;
                    }

                    // PhysDeriv kernel.
                    PhysDerivKernelLauncher<SHAPE_TYPE, DEFORMED>(
                        sizeParam, m_D[ind0]..., m_f[ind1]...,
                        reinterpret_cast<const simd_t *>(dfptr),
                        reinterpret_cast<const simd_t *>(inCompPtr), derivOut);

                    // Interpolate the physical gradient to the fine grid.
                    for (unsigned int d = 0; d < m_coordDim; ++d)
                    {
                        InterpKernel<DIM, false>(derivNative + d * nqTot,
                                                 gradFine + d * nqTotFine);
                    }

                    // Form scale * advVel . grad(u) on the fine grid.
                    AdvectionDealiasCombineKernel(
                        nqTotFine, m_coordDim, advVelFine, nqTotFine, gradFine,
                        nqTotFine, combinedFine, this->m_scale);

                    // Project directly into this component's output.
                    if (this->m_append)
                    {
                        ProjectKernel<DIM, true>(
                            combinedFine,
                            reinterpret_cast<simd_t *>(outCompPtr));
                    }
                    else
                    {
                        ProjectKernel<DIM, false>(
                            combinedFine,
                            reinterpret_cast<simd_t *>(outCompPtr));
                    }

                    // Reshape back, if necessary.
                    if (e % width_ratio == width_ratio - 1)
                    {
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            inCompPtr -
                                (width_ratio - 1) * nqTot * simd_t::width);
                        LibUtilities::ReshapeStorage<ExecSpace>(
                            inInterleaveWidth, m_implInterleaveWidth, chunkSize,
                            nqTot,
                            outCompPtr -
                                (width_ratio - 1) * nqTot * simd_t::width);
                    }
                }

                // Increment pointers for the next elmt group.
                dfptr += dfsize * simd_t::width;
                inptr += nqTot * simd_t::width;
                advVelPtr += nqTot * simd_t::width;
                outptr += nqTot * simd_t::width;
            }
        }

        // Set output block to input interleave.
        outblock.template SetInterleaveWidth<TData>(inInterleaveWidth);
    }

    // Dispatch the native->fine interpolation kernel by dimension. Always
    // overwrites the destination - only ProjectKernel below is ever asked
    // to append.
    template <unsigned int DIM, bool APPEND>
    NEK_FORCE_INLINE void InterpKernel(const simd_t *src, simd_t *dst)
    {
        if constexpr (DIM == 1)
        {
            BwdTransSegKernel<APPEND>(m_nq[0], m_nqFine[0], m_Binterp[0], src,
                                      dst);
        }
        else if constexpr (DIM == 2)
        {
            BwdTransQuadKernel<APPEND>(m_nq[0], m_nq[1], m_nqFine[0],
                                       m_nqFine[1], m_Binterp[0], m_Binterp[1],
                                       m_wspBT0.data(), src, dst);
        }
        else
        {
            BwdTransHexKernel<APPEND>(
                m_nq[0], m_nq[1], m_nq[2], m_nqFine[0], m_nqFine[1],
                m_nqFine[2], m_Binterp[0], m_Binterp[1], m_Binterp[2],
                m_wspBT0.data(), m_wspBT1.data(), src, dst);
        }
    }

    // Dispatch the fine->native Galerkin-projection kernel by dimension.
    template <unsigned int DIM, bool APPEND>
    NEK_FORCE_INLINE void ProjectKernel(const simd_t *src, simd_t *dst)
    {
        if constexpr (DIM == 1)
        {
            BwdTransSegKernel<APPEND>(m_nqFine[0], m_nq[0], m_Bproject[0], src,
                                      dst);
        }
        else if constexpr (DIM == 2)
        {
            BwdTransQuadKernel<APPEND>(m_nqFine[0], m_nqFine[1], m_nq[0],
                                       m_nq[1], m_Bproject[0], m_Bproject[1],
                                       m_wspBT0.data(), src, dst);
        }
        else
        {
            BwdTransHexKernel<APPEND>(
                m_nqFine[0], m_nqFine[1], m_nqFine[2], m_nq[0], m_nq[1],
                m_nq[2], m_Bproject[0], m_Bproject[1], m_Bproject[2],
                m_wspBT0.data(), m_wspBT1.data(), src, dst);
        }
    }
};

} // namespace Nektar::Operators::detail
