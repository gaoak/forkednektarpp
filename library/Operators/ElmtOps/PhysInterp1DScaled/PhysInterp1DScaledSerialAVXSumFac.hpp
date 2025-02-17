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

#include "Operators/ElmtOps/OperatorPhysInterp1DScaled.hpp"
#include "Operators/Utils/UtilsKernels.hpp"

// interpolation is just a bwd trans from a nodal basis so using these kernels
#include "ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename Implementation, typename TData>
class BlockOperatorPhysInterp1DScaledImpl
    : public BlockOperatorPhysInterp1DScaled<TData>
{
    using simd_t =
        typename simd_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TData>::type;
    using MemSpace = typename ExecSpace::memory_space;

public:
    BlockOperatorPhysInterp1DScaledImpl(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperatorPhysInterp1DScaled<TData>(exp, dataWarehouse)
    {
    }

    void apply(BlockAccessor<TData> &inblock,
               BlockAccessor<TData> &outblock) override
    {
        // Check alignment.
        WARNINGL1(inblock.GetAlignment() == simd_t::alignment,
                  "Input Field are not aligned to the required alignment "
                  "for the SIMD vector type.");
        WARNINGL1(outblock.GetAlignment() == simd_t::alignment,
                  "Output Field are not aligned to the required alignment "
                  "for the SIMD vector type.");

        ASSERTL1(this->m_scale != -1.0,
                 "Scale factor has not been initialised");

        // Determine shape and type of the element.
        const auto shapeType = this->m_exp->DetShapeType();

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

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<BlockOperator<TData>> instantiate(
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<BlockOperatorPhysInterp1DScaledImpl<
            ExecSpace, Implementation, TData>>(exp, dataWarehouse);
    }

protected:
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

    // Non-size based operator.
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = this->m_exp->GetNumPoints(0);
        const auto nq0 = (unsigned int)(this->m_scale * nm0);

        const auto nmTot = nm0;
        const auto nqTot = nq0;

        // Fetch basis data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eInterp, nq0));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

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
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransSegKernel(nm0, nq0, B0, inptr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nmTot;
                outptr += nqTot * simd_t::width;
            }
        }
    }

    // size based template version
    template <unsigned int nm0, unsigned int nq0>
    void Operator1D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0;
        constexpr auto nqTot = nq0;

        // Fetch basis data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eInterp, nq0));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

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
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransSegKernel(nm0, nq0, B0, inptr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nmTot;
                outptr += nqTot * simd_t::width;
            }
        }
    }

    // Non-size based operator.
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = this->m_exp->GetNumPoints(0);
        const auto nm1 = this->m_exp->GetNumPoints(1);

        const auto nq0 = (unsigned int)(this->m_scale * nm0);
        // if delta between nm0 and nm1 is 1 then keep this delta
        // for new poitns to capitalise on switch templating
        const auto nq1 = (nm0 - nm1 == 1)
                             ? (unsigned int)(this->m_scale * nm0) - 1
                             : (unsigned int)(this->m_scale * nm1);

        // Shape size.
        const auto nmTot = nm0 * nm1;
        const auto nqTot = nq0 * nq1;

        // Fetch basis data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eInterp, nq0));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eInterp, nq1));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

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
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransQuadKernel(nm0, nm1, nq0, nq1, B0, B1, wsp0, inptr,
                                   outptr);

                // Increment pointers for the next elmt group.
                inptr += nmTot;
                outptr += nqTot * simd_t::width;
            }
        }
    }

    // size based template version
    template <unsigned int nm0, unsigned int nm1, unsigned int nq0,
              unsigned int nq1>
    void Operator2D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0 * nm1;
        constexpr auto nqTot = nq0 * nq1;

        // Fetch basis data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eInterp, nq0));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eInterp, nq1));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

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
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransQuadKernel(nm0, nm1, nq0, nq1, B0, B1, wsp0, inptr,
                                   outptr);

                // Increment pointers for the next elmt group.
                inptr += nmTot;
                outptr += nqTot * simd_t::width;
            }
        }
    }

    // Non-size based operator.
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        const auto nm0 = this->m_exp->GetNumPoints(0);
        const auto nm1 = this->m_exp->GetNumPoints(1);
        const auto nm2 = this->m_exp->GetNumPoints(2);

        const auto nq0 = (unsigned int)(this->m_scale * nm0);
        // if delta between nm0 and nm1 is 1 then keep this delta
        // for new poitns to capitalise on switch templating
        const auto nq1 = (nm0 - nm1 == 1)
                             ? (unsigned int)(this->m_scale * nm0) - 1
                             : (unsigned int)(this->m_scale * nm1);
        const auto nq2 = (nm0 - nm2 == 1)
                             ? (unsigned int)(this->m_scale * nm0) - 1
                             : (unsigned int)(this->m_scale * nm2);

        // Shape size.
        const auto nmTot = nm0 * nm1 * nm2;
        const auto nqTot = nq0 * nq1 * nq2;

        // Fetch basis data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eInterp, nq0));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eInterp, nq1));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                 eInterp, nq2));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

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
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransHexKernel(nm0, nm1, nm2, nq0, nq1, nq2, B0, B1, B2,
                                  wsp0, wsp1, inptr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nmTot;
                outptr += nqTot * simd_t::width;
            }
        }
    }

    // size based template version
    template <unsigned int nm0, unsigned int nm1, unsigned int nm2,
              unsigned int nq0, unsigned int nq1, unsigned int nq2>
    void Operator3D(BlockAccessor<TData> &inblock,
                    BlockAccessor<TData> &outblock)
    {
        // Shape size.
        constexpr auto nmTot = nm0 * nm1 * nm2;
        constexpr auto nqTot = nq0 * nq1 * nq2;

        // Fetch basis data.
        auto B0 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(0)->GetBasisKey(),
                                 eInterp, nq0));
        auto B1 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(1)->GetBasisKey(),
                                 eInterp, nq1));
        auto B2 = this->m_dataWarehouse->template GetData<ExecSpace>(
            BasisDataKey<simd_t>(this->m_exp->GetBasis(2)->GetBasisKey(),
                                 eInterp, nq2));

        // Get interleave parameter.
        unsigned int interleave_width = inblock.GetInterleaveWidth();
        const auto width_ratio =
            (interleave_width == 1) ? 1 : interleave_width / simd_t::width;
        const auto chunkSize = std::max(simd_t::width, interleave_width);

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
        auto inptr =
            reinterpret_cast<const typename simd_t::vectorType *>(input);
        auto outptr = reinterpret_cast<typename simd_t::scalarType *>(output);

        // Loop over components.
        for (unsigned int nc = 0; nc < inblock.GetNumComponents(); ++nc)
        {
            for (unsigned int e = 0; e < inblock.GetNumElmtGroups(); ++e)
            {
                // Reshape, if necessary.
                if (e % width_ratio == 0)
                {
                    ReshapeStorage<ExecSpace, simd_t::width>(
                        interleave_width, chunkSize, nmTot, (TData *)inptr);
                }

                // PhysInterp1DScaled kernel.
                BwdTransHexKernel(nm0, nm1, nm2, nq0, nq1, nq2, B0, B1, B2,
                                  wsp0, wsp1, inptr, outptr);

                // Increment pointers for the next elmt group.
                inptr += nmTot;
                outptr += nqTot * simd_t::width;
            }
        }
    }
};

} // namespace Nektar::Operators::detail
