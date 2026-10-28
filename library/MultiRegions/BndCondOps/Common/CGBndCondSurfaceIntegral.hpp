///////////////////////////////////////////////////////////////////////////////
//
// File: CGBndCondSurfaceIntegral.hpp
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
// Description: Surface integral of a boundary field from its boundary
// inner-product coefficients, summed in the memory space of the execution
// space.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <vector>

#include "LibUtilities/BasicUtils/Field/Field.hpp"
#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include "MultiRegions/ExpList.h"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Adds the surface integral of boundary fields, per component, to a
 * reduction in the memory space of @p ExecSpace, from the inner products of
 * the fields with the basis of their boundary expansion lists.
 *
 * @details
 * The surface integral of a field is the sum of its inner products with the
 * basis, each weighted by the coefficient of the constant function one in
 * that basis: one at every mode of a Lagrange basis, and, for a Modified_A
 * basis, one at the vertex modes, the collapsed vertex of a triangle
 * included, and zero at the others; the value of a point region is its
 * integral. A component of a Lagrange or a point block is summed whole, its
 * padding being zero. A component of a Modified_A block is summed over a
 * mask laid out as one component of the block at its interleave width,
 * holding one at the vertex modes of every element and zero elsewhere, the
 * padding included. One storage index then addresses the same coefficient in
 * both, whatever that width, and the sum is a masked reduction over the
 * component as it is laid out, with no index arithmetic. The mask of a block
 * is built at interleave width 1, the width of the boundary fields, when this
 * is built, and at the interleave width of the execution space, the only
 * other, the first time a field is read at it, and kept. The reduction is
 * moved to the memory space of the execution space on one stream, which the
 * block streams wait for; each block is then summed on its own stream, where
 * it was written, into the reduction, which is not read back.
 */
template <typename ExecSpace, typename TData> class CGBndCondSurfaceIntegral
{
    using MemSpace = typename ExecSpace::memory_space;

    // Interleave width of an interleaved field: 1 (Serial), the SIMD width
    // (AVX) or the warp size (Device).
    static constexpr unsigned int m_interleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;

public:
    CGBndCondSurfaceIntegral() = default;

    /**
     * @brief Build the sum of the surface integrals of the fields on the
     * boundary expansion lists @p bndExpLists, of component nc of field n
     * where isComp[n][nc] holds.
     */
    CGBndCondSurfaceIntegral(
        const std::vector<MultiRegions::ExpListSharedPtr> &bndExpLists,
        const std::vector<std::vector<bool>> &isComp)
        : m_isComp(isComp), m_vertexModes(bndExpLists.size()),
          m_masks(bndExpLists.size())
    {
        for (unsigned int n = 0; n < bndExpLists.size(); ++n)
        {
            const bool isPoint = (bndExpLists[n]->GetShapeDimension() == 0);
            auto collections   = MultiRegions::GetCollections(bndExpLists[n]);
            m_masks[n].resize(collections.size());
            for (auto &collection : collections)
            {
                // Every coefficient of a point or a Lagrange block is summed.
                m_vertexModes[n].emplace_back();
                if (isPoint)
                {
                    continue;
                }

                const auto exp       = collection.GetExpVector()[0];
                const auto basisType = exp->GetBasisType(0);
                if (basisType == LibUtilities::eModified_A)
                {
                    for (int k = 0; k < exp->GetNverts(); ++k)
                    {
                        m_vertexModes[n].back().push_back(exp->GetVertexMap(k));
                    }
                }
                else
                {
                    ASSERTL0(basisType == LibUtilities::eGLL_Lagrange ||
                                 basisType == LibUtilities::eGauss_Lagrange,
                             "The surface integral of a boundary field needs "
                             "a Modified_A or a Lagrange basis.");
                }
            }

            if (isPoint)
            {
                continue;
            }

            // The masks at interleave width 1, the width of the boundary
            // fields.
            const auto blockAttributes =
                MultiRegions::GetBlockAttributes<TData, FieldState::Coeff>(
                    bndExpLists[n]);
            for (unsigned int blk = 0; blk < blockAttributes.size(); ++blk)
            {
                const unsigned int streamID = blk + 1;
                if (!m_vertexModes[n][blk].empty())
                {
                    BuildMask(n, blk, blockAttributes[blk], streamID);
                }
            }
        }
    }

    /**
     * @brief Add the surface integral of component nc of every field
     * coeffs[n], the fields of the boundary expansion lists this was built
     * for, in the same order, to data[offset + nc], where isComp[n][nc]
     * holds.
     */
    void Add(std::vector<LibUtilities::Field<TData, FieldState::Coeff>> &coeffs,
             LibUtilities::MemoryRegion<TData> &data, const size_t offset)
    {
        // data is moved to the memory space of the execution space on
        // m_opStreamID, which the block streams then wait for.
        auto dataPtr = data.template GetPtr<MemSpace, ReadWrite>(m_opStreamID);
        std::vector<unsigned int> blockStreams;
        for (auto &field : coeffs)
        {
            for (unsigned int blk = blockStreams.size();
                 blk < field.GetBlocks().size(); ++blk)
            {
                blockStreams.push_back(blk + 1);
            }
        }
        ReleaseStreams<ExecSpace>(m_opStreamID, blockStreams);

        for (unsigned int n = 0; n < coeffs.size(); ++n)
        {
            for (unsigned int blk = 0; blk < coeffs[n].GetBlocks().size();
                 ++blk)
            {
                const unsigned int streamID = blk + 1;
                auto &block                 = coeffs[n].GetBlocks()[blk];
                const uint8_t *maskPtr      = GetMask(n, blk, block, streamID);
                auto srcPtr =
                    block.template GetPtr<MemSpace, ReadOnly>(streamID);
                const unsigned int nComp = coeffs[n].GetNumComponents();
                for (unsigned int nc = 0; nc < nComp; ++nc)
                {
                    if (!m_isComp[n][nc])
                    {
                        continue;
                    }

                    if (maskPtr != nullptr)
                    {
                        Math::reduceSumKernel<ExecSpace, false>(
                            block.CompSize(), maskPtr,
                            srcPtr + nc * block.CompSize(),
                            dataPtr + offset + nc, streamID);
                    }
                    else
                    {
                        Math::reduceSumKernel<ExecSpace, false>(
                            block.CompSize(), srcPtr + nc * block.CompSize(),
                            dataPtr + offset + nc, streamID);
                    }
                }
            }
        }
    }

private:
    // Stream on which data is moved before the block streams sum into it.
    static constexpr unsigned int m_opStreamID = 1;
    // Per field, whether each component is summed.
    std::vector<std::vector<bool>> m_isComp;
    // Per field and block, the positions within an element of the vertex
    // modes of a Modified_A block; empty for a point or a Lagrange block.
    std::vector<std::vector<std::vector<size_t>>> m_vertexModes;
    // Per field and block, the masks of a Modified_A block at interleave
    // width 1 and at that of the execution space, in that order: one at the
    // vertex modes of every element, zero elsewhere, the padding included;
    // empty until built.
    std::vector<std::vector<std::array<LibUtilities::MemoryRegion<uint8_t>, 2>>>
        m_masks;

    /**
     * @brief The index into #m_masks of interleave width @p interleaveWidth.
     */
    static unsigned int MaskIndex(const unsigned int interleaveWidth)
    {
        ASSERTL0(interleaveWidth == 1u || interleaveWidth == m_interleaveWidth,
                 "The surface integral takes fields at interleave width 1 or "
                 "at the interleave width of its execution space.");
        return (interleaveWidth == 1u) ? 0u : 1u;
    }

    /**
     * @brief The mask of block @p blk of field @p n at the interleave width of
     * @p block, in the memory space of the execution space, built and
     * uploaded on stream @p streamID the first time a block is read at a
     * width other than 1; null for a block summed whole.
     */
    const uint8_t *GetMask(
        const unsigned int n, const unsigned int blk,
        const LibUtilities::BlockAttributes<FieldState::Coeff> &block,
        const unsigned int streamID)
    {
        if (m_vertexModes[n][blk].empty())
        {
            return nullptr;
        }

        auto &mask = m_masks[n][blk][MaskIndex(block.GetInterleaveWidth())];
        if (mask.size() == 0)
        {
            BuildMask(n, blk, block, streamID);
        }

        return mask.template GetPtr<MemSpace, ReadOnly>(streamID);
    }

    /**
     * @brief Build the mask of the Modified_A block @p blk of field @p n at
     * the interleave width of @p block, upload it on stream @p streamID and
     * keep it.
     */
    void BuildMask(
        const unsigned int n, const unsigned int blk,
        const LibUtilities::BlockAttributes<FieldState::Coeff> &block,
        const unsigned int streamID)
    {
        // Element e is lane ilane = e % interleaveWidth of group
        // iwarp = e / interleaveWidth, its coefficient m at storage index
        // (iwarp * nData + m) * interleaveWidth + ilane. The entries of the
        // padded elements stay zero.
        const unsigned int interleaveWidth = block.GetInterleaveWidth();
        const size_t nData                 = block.GetNumData();
        std::vector<uint8_t> hostMask(block.CompSize(), 0);
        for (size_t e = 0; e < block.GetNumElements(); ++e)
        {
            const size_t ilane = e % interleaveWidth;
            const size_t iwarp = e / interleaveWidth;
            for (const size_t mode : m_vertexModes[n][blk])
            {
                hostMask[(iwarp * nData + mode) * interleaveWidth + ilane] = 1;
            }
        }

        m_masks[n][blk][MaskIndex(interleaveWidth)] =
            LibUtilities::MemoryRegion<uint8_t>::template FromVector<MemSpace,
                                                                     uint8_t>(
                hostMask, streamID);
    }
};

} // namespace Nektar::MultiRegions::detail
