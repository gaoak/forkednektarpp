///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledSerialAVXSumFac.hpp
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
// Description: interp in physical space by a scaled number of points
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Common/OperatorHelper.hpp"
#include "ElmtOps/OperatorPhysInterp1DScaled.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

// interpolation is just a bwd trans from a nodal basis so using these kernels
#include "ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

// Matrix-free implementation
template <typename ExecSpace, typename Implementation, typename TData>
class OperatorPhysInterp1DScaledImpl : public OperatorPhysInterp1DScaled<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorPhysInterp1DScaledImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorPhysInterp1DScaled<TData>(expansionList)
    {
    }

    void SetScaleFactor(double scale) final
    {
        if (this->m_scale != scale)
        {
            this->m_scale = scale;

            // Loop over the elements of expansionList.
            size_t nDim = this->m_expansionList->GetShapeDimension();

            for (size_t i = 0; i < this->m_expansionList->GetNumElmts(); ++i)
            {
                const auto expPtr = this->m_expansionList->GetExp(i);

                int npts0 = expPtr->GetBasis(0)->GetNumPoints();

                // Fetch basiskeys of the current element.
                for (size_t d = 0; d < nDim; d++)
                {
                    LibUtilities::BasisKey b =
                        expPtr->GetBasis(d)->GetBasisKey();
                    int npts = b.GetNumPoints();

                    // if delta between npts and npts0 is 1 then keep this delta
                    // for new poitns to capitalise on switch templating
                    npts = (npts0 - npts == 1) ? (int)(scale * npts0) - 1
                                               : (int)(scale * npts);

                    LibUtilities::PointsKey p(npts, b.GetPointsType());

                    // make basis using modified direction with num points as
                    // modes and new quarature points as numpoints
                    LibUtilities::BasisKey bnew(b.GetBasisType(),
                                                b.GetNumPoints(), p);

                    // If necessary initialise this  basis data in  map.
                    if (m_interpMap.find(bnew) == m_interpMap.end())
                    {
                        m_interpMap[bnew] =
                            GetBasisData<MemSpace, double, simd_t>(
                                expPtr->GetBasis(d), eInterp, simd_t::alignment,
                                npts);
                    }
                }
            }
        }
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Phys> &out) override
    {
        // Initialize index.
        size_t exp_idx = 0;

        m_nComps = in.GetNumComponents();
        ASSERTL1(m_nComps == out.GetNumComponents(),
                 "Number of input and output components differ");

        for (size_t blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            m_expPtr = this->m_expansionList->GetExp(exp_idx);

            // Block dependent.
            auto &inblock  = in.GetBlocks()[blk];
            auto &outblock = out.GetBlocks()[blk];

            this->BlockOperator(inblock, outblock);

            // Increment index for next element type.
            exp_idx += inblock.GetNumElements();
        }
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorPhysInterp1DScaledImpl<ExecSpace, Implementation, TData>>(
            expansionList);
    }

    void BlockOperator(BlockAccessor<TData> &inblock,
                       BlockAccessor<TData> &outblock)
    {
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

        // Determine shape and type of the element.
        const auto shapeType = m_expPtr->DetShapeType();

        switch (shapeType)
        {
                // Segment
            case LibUtilities::Seg:
            {
                SegBlock(inblock, outblock);
                break;
            }
                // Quads
            case LibUtilities::Quad:
            {
                QuadBlock(inblock, outblock);
                break;
            }
                // Triangles
            case LibUtilities::Tri:
            {
                TriBlock(inblock, outblock);
                break;
            }
                // Hexes
            case LibUtilities::Hex:
            {
                HexBlock(inblock, outblock);
                break;
            }
                // Tet
            case LibUtilities::Tet:
            {
                TetBlock(inblock, outblock);
                break;
            }
                // Pyr
            case LibUtilities::Pyr:
            {
                PyrBlock(inblock, outblock);
                break;
            }
                // Prism
            case LibUtilities::Prism:
            {
                PrismBlock(inblock, outblock);
                break;
            }
            default:
                std::cout << "shapetype not implemented" << std::endl;
        }
    }

private:
    size_t m_nComps;

    LocalRegions::ExpansionSharedPtr m_expPtr;
    BasisDataMap<simd_t> m_interpMap;

    void SegBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TriBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void QuadBlock(BlockAccessor<TData> &inblock,
                   BlockAccessor<TData> &outblock);

    void HexBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void PrismBlock(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock);

    void PyrBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    void TetBlock(BlockAccessor<TData> &inblock,
                  BlockAccessor<TData> &outblock);

    // templated operator(), which is instantiated by SwitchNodesPoints.h
    // and used in apply().
    // size based template version
    template <int nm0, int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        auto basis0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        BwdTrans1DWorkspace<LibUtilities::Seg>(nm0, nq0);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                }

                // PhysInterp1DScaled kernel.
                BwdTransSegKernel(nm0, nq0, basis0, tmpIn, tmpOut);

                // Increment pointers for the next elmt group.
                tmpIn += nmTot;
                tmpOut += nqTot * simd_t::width;
            }
        }
    }

    // Non-size based operator.
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetNumPoints(0);
        const int nq0  = (int)(this->m_scale * nm0);

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        // Fetch basis data.
        std::vector<LibUtilities::BasisKey> basisKeys;

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        auto basis0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        BwdTrans1DWorkspace<LibUtilities::Seg>(nm0, nq0);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);

        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                }

                // PhysInterp1DScaled kernel.
                BwdTransSegKernel(nm0, nq0, basis0, tmpIn, tmpOut);

                // Increment pointers for the next elmt group.
                tmpIn += nmTot;
                tmpOut += nqTot * simd_t::width;
            }
        }
    }

    // size based template version
    template <int nm0, int nm1, int nq0, int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0 * nm1;
        constexpr auto nqTot = nq0 * nq1;

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        auto basis0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto basis1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<LibUtilities::Quad>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                }

                // PhysInterp1DScaled kernel.
                BwdTransQuadKernel(nm0, nm1, nq0, nq1, basis0, basis1, wsp0,
                                   tmpIn, tmpOut);

                // Increment pointers for the next elmt group.
                tmpIn += nmTot;
                tmpOut += nqTot * simd_t::width;
            }
        }
    }

    // Non-size based operator.
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetNumPoints(0);
        const auto nm1 = m_expPtr->GetNumPoints(1);

        const int nq0 = (int)(this->m_scale * nm0);
        // if delta between nm0 and nm1 is 1 then keep this delta
        // for new poitns to capitalise on switch templating
        const int nq1 = (nm0 - nm1 == 1) ? (int)(this->m_scale * nm0) - 1
                                         : (int)(this->m_scale * nm1);

        // Shape size.
        const auto nmTot = nm0 * nm1;
        const auto nqTot = nq0 * nq1;

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        auto basis0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto basis1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0;
        BwdTrans2DWorkspace<LibUtilities::Quad>(nm0, nm1, nq0, nq1, wsp0Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                }

                // PhysInterp1DScaled kernel.
                BwdTransQuadKernel(nm0, nm1, nq0, nq1, basis0, basis1, wsp0,
                                   tmpIn, tmpOut);

                // Increment pointers for the next elmt group.
                tmpIn += nmTot;
                tmpOut += nqTot * simd_t::width;
            }
        }
    }

    // size based template version
    template <int nm0, int nm1, int nm2, int nq0, int nq1, int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0 * nm1 * nm2;
        constexpr auto nqTot = nq0 * nq1 * nq2;

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        LibUtilities::BasisKey b2 = m_expPtr->GetBasis(2)->GetBasisKey();
        LibUtilities::PointsKey p2(nq2, b2.GetPointsType());
        LibUtilities::BasisKey b2new(b2.GetBasisType(), nm2, p2);

        auto basis0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto basis1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();
        auto basis2 = m_interpMap[b2new].template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<LibUtilities::Hex>(nm0, nm1, nm2, nq0, nq1, nq2,
                                               wsp0Size, wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                }

                // PhysInterp1DScaled kernel.
                BwdTransHexKernel(nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1,
                                  basis2, wsp0, wsp1, tmpIn, tmpOut);

                // Increment pointers for the next elmt group.
                tmpIn += nmTot;
                tmpOut += nqTot * simd_t::width;
            }
        }
    }

    // Non-size based operator.
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = m_expPtr->GetNumPoints(0);
        const auto nm1 = m_expPtr->GetNumPoints(1);
        const auto nm2 = m_expPtr->GetNumPoints(2);

        const int nq0 = (int)(this->m_scale * nm0);
        // if delta between nm0 and nm1 is 1 then keep this delta
        // for new poitns to capitalise on switch templating
        const int nq1 = (nm0 - nm1 == 1) ? (int)(this->m_scale * nm0) - 1
                                         : (int)(this->m_scale * nm1);
        const int nq2 = (nm0 - nm2 == 1) ? (int)(this->m_scale * nm0) - 1
                                         : (int)(this->m_scale * nm2);

        // Shape size.
        const auto nmTot = nm0 * nm1 * nm2;
        const auto nqTot = nq0 * nq1 * nq2;

        LibUtilities::BasisKey b0 = m_expPtr->GetBasis(0)->GetBasisKey();
        LibUtilities::PointsKey p0(nq0, b0.GetPointsType());
        LibUtilities::BasisKey b0new(b0.GetBasisType(), nm0, p0);

        LibUtilities::BasisKey b1 = m_expPtr->GetBasis(1)->GetBasisKey();
        LibUtilities::PointsKey p1(nq1, b1.GetPointsType());
        LibUtilities::BasisKey b1new(b1.GetBasisType(), nm1, p1);

        LibUtilities::BasisKey b2 = m_expPtr->GetBasis(2)->GetBasisKey();
        LibUtilities::PointsKey p2(nq2, b2.GetPointsType());
        LibUtilities::BasisKey b2new(b2.GetBasisType(), nm2, p2);

        auto basis0 = m_interpMap[b0new].template GetPtr<MemSpace, ReadOnly>();
        auto basis1 = m_interpMap[b1new].template GetPtr<MemSpace, ReadOnly>();
        auto basis2 = m_interpMap[b2new].template GetPtr<MemSpace, ReadOnly>();

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        auto chunkSize = std::max(simd_t::width, interleave_width);

        // Set to new interleave width.
        inblock.template SetInterleaveWidth<TData>(simd_t::width);
        outblock.template SetInterleaveWidth<TData>(simd_t::width);

        // Workspace for kernels - also checks preconditions.
        size_t wsp0Size = 0, wsp1Size = 0;
        BwdTrans3DWorkspace<LibUtilities::Hex>(nm0, nm1, nm2, nq0, nq1, nq2,
                                               wsp0Size, wsp1Size);
        std::vector<simd_t, tinysimd::allocator<simd_t>> wsp0(wsp0Size),
            wsp1(wsp1Size);

        // Initialize pointers.
        auto input  = (interleave_width == simd_t::width)
                          ? inblock.template GetPtr<MemSpace, ReadOnly>()
                          : inblock.template GetPtr<MemSpace, ReadWrite>();
        auto output = outblock.template GetPtr<MemSpace, WriteOnly>();
        auto tmpIn =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto tmpOut = reinterpret_cast<typename simd_t::scalarType *>(output);
        for (size_t nc = 0; nc < m_nComps; ++nc)
        {
            for (int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)tmpIn);
                }

                // PhysInterp1DScaled kernel.
                BwdTransHexKernel(nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1,
                                  basis2, wsp0, wsp1, tmpIn, tmpOut);

                // Increment pointers for the next elmt group.
                tmpIn += nmTot;
                tmpOut += nqTot * simd_t::width;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
