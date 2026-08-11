///////////////////////////////////////////////////////////////////////////////
//
// File: Block.hpp
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

#include <Collections/Collection.h>

#include "LibUtilities/BasicUtils/MemoryRegion.hpp"
#include "MultiRegions/MultiRegionsDeclspec.h"

namespace Nektar::MultiRegions
{

class ExpList;

/// Shared pointer to an ExpList object.
typedef std::shared_ptr<ExpList> ExpListSharedPtr;

// Helper function
MULTI_REGIONS_EXPORT Collections::Collection GetCollection(
    MultiRegions::ExpListSharedPtr expansionList, unsigned int block_idx);

/**
 * @brief A block means a group of elements of identical shape,
 * basis and order. BlockAttributes stores the most basic
 * information of a block.
 */
template <FieldState TState> class BlockAttributes
{
public:
    BlockAttributes() = delete;
    BlockAttributes(const size_t num_elements,
                    const size_t num_elements_with_padding,
                    const unsigned int num_data,
                    const unsigned int interleave_width)
        : m_num_elements(num_elements),
          m_num_elements_with_padding(num_elements_with_padding),
          m_num_data(num_data),
          m_compSize(num_elements_with_padding * num_data),
          m_interleave_width(interleave_width)
    {
    }

    BlockAttributes(const BlockAttributes &) = default;

    template <typename TPadding>
    void SetInterleaveWidth(const unsigned interleave_width)
    {
        if (interleave_width != 1)
        {
            ASSERTL0(
                m_num_elements_with_padding % interleave_width == 0,
                "Number of elements is not divisible by interleave width.");
            ASSERTL0(
                interleave_width % tinysimd::simd<TPadding>::width == 0,
                "interleave width should be divisible by AVX vector width.");
        }

        m_interleave_width = interleave_width;
    }

    size_t GetNumElements(void) const
    {
        return m_num_elements;
    }

    size_t GetNumElementsWithPadding(void) const
    {
        return m_num_elements_with_padding;
    }

    unsigned int GetNumData(void) const
    {
        return m_num_data;
    }

    size_t CompSize(void) const
    {
        return m_compSize;
    }

    unsigned int GetInterleaveWidth(void) const
    {
        return m_interleave_width;
    }

    size_t GetNumPaddingElements(void) const
    {
        return m_num_elements_with_padding - m_num_elements;
    }

    size_t GetNumElmtGroups(void) const
    {
        return m_num_elements_with_padding / m_interleave_width;
    }

    size_t GetNumElmtGroups(const unsigned int interleave_width) const
    {
        return m_num_elements_with_padding / interleave_width;
    }

private:
    const size_t m_num_elements;
    const size_t m_num_elements_with_padding;
    const unsigned int m_num_data;
    const size_t m_compSize;
    unsigned int m_interleave_width;
};

/**
 * @brief Get the BlockAttributes for a given field state from an ExpList.
 * This method basically captures identical elements that are contiguously
 * stored in the ExpList and group them into blocks. Padding elements will
 * also be set based on given vector width.
 *
 * @param state     Field state to query.
 * @param explist   Expansion list to query.
 * @param interleave_width Vector width to use for the field.
 * @return std::vector<BlockAttributes>
 */
template <typename TPadding, FieldState TState>
MULTI_REGIONS_EXPORT std::vector<BlockAttributes<TState>> GetBlockAttributes(
    const MultiRegions::ExpListSharedPtr explist,
    const unsigned interleave_width = 1);

template <typename TData, FieldState TState>
class BlockAccessor : public BlockAttributes<TState>
{
    template <typename TDataField, FieldState TStateField> friend class Field;

public:
    BlockAccessor() = delete;
    BlockAccessor(const BlockAttributes<TState> blockAttr,
                  LibUtilities::MemoryRegion<TData> &&memory_region,
                  Field<TData, TState> *field,
                  const unsigned int num_components,
                  const unsigned int num_homo_modes)
        : BlockAttributes<TState>(blockAttr),
          m_memory_region(std::move(memory_region)), m_field(field),
          m_num_components(num_components), m_num_homo_modes(num_homo_modes)
    {
    }

    BlockAccessor(const size_t num_elements,
                  const size_t num_elements_with_padding,
                  const unsigned int num_data, const unsigned interleave_width,
                  LibUtilities::MemoryRegion<TData> &&memory_region,
                  Field<TData, TState> *field,
                  const unsigned int num_components,
                  const unsigned int num_homo_modes)
        : BlockAttributes<TState>(num_elements, num_elements_with_padding,
                                  num_data, interleave_width),
          m_memory_region(std::move(memory_region)), m_field(field),
          m_num_components(num_components), m_num_homo_modes(num_homo_modes)
    {
    }

    /**
     * @brief Constructor methods - move from another BlockAccessor
     *
     * @param rhs - BlockAccessor to move from
     */
    BlockAccessor(BlockAccessor &&rhs)
        : BlockAttributes<TState>(std::move(rhs)),
          m_memory_region(std::move(rhs.m_memory_region)),
          m_field(std::move(rhs.m_field)),
          m_num_components(std::move(rhs.m_num_components)),
          m_num_homo_modes(std::move(rhs.m_num_homo_modes))
    {
        rhs.m_field          = nullptr;
        rhs.m_num_components = 0;
        rhs.m_num_homo_modes = 1;
    }

    /**
     * @brief Constructor methods - No copy methods
     *
     */
    BlockAccessor(const BlockAccessor &)            = delete;
    BlockAccessor &operator=(const BlockAccessor &) = delete;

    /**
     * @brief Move assignment operator.
     *
     * @param rhs - BlockAccessor to move from
     *
     * @return    - BlockAccessor&
     */
    BlockAccessor &operator=(BlockAccessor &&rhs)
    {
        BlockAttributes<TState>::operator=(std::move(rhs));
        m_memory_region  = std::move(rhs.m_memory_region);
        m_field          = std::move(rhs.m_field);
        m_num_components = std::move(rhs.m_num_components);
        m_num_homo_modes = std::move(rhs.m_num_homo_modes);

        rhs.m_field          = nullptr;
        rhs.m_num_components = 0;
        rhs.m_num_homo_modes = 1;

        return *this;
    }

    MULTI_REGIONS_EXPORT void ReshapeStorage(
        const unsigned int &interleaveWidth, const std::string &execSpace,
        const unsigned int streamID = 0);

    /**
     * @brief Get the pointer to the host/device memory.
     *
     * @return    - TData*
     */
    template <typename MemSpace, typename MemAccess>
    MULTI_REGIONS_EXPORT
        typename const_if<std::is_same_v<MemAccess, ReadOnly>, TData>::type *
        GetPtr(const unsigned int streamID = 0);

    /**
     * @brief Gets the alignment of the memory region block.
     *
     * @return size_t
     */
    size_t GetAlignment() const
    {
        return m_memory_region.GetAlignment();
    }

    /**
     * @brief Gets the number of components.
     *
     * @return unsigned int
     */
    unsigned int GetNumComponents() const
    {
        return m_num_components;
    }

    /**
     * @brief Gets the number of homogeneous modes.
     *
     * @return unsigned int
     */
    unsigned int GetNumHomoModes() const
    {
        return m_num_homo_modes;
    }

    /**
     * @brief initialize the storage memory.
     *
     * @param val   - value to set
     * @param count - number of values
     */
    template <typename MemSpace>
    void Initialize(const TData val, const size_t count = 0,
                    const size_t offset = 0, const unsigned int streamID = 0)
    {
        m_memory_region.template Initialize<MemSpace>(val, count, offset,
                                                      streamID);
    }

private:
    // Note: m_field is a pointer to a Field object from which the current
    // BlockAccessor object belong to.
    LibUtilities::MemoryRegion<TData> m_memory_region;
    Field<TData, TState> *m_field;
    unsigned int m_num_components = 0;
    unsigned int m_num_homo_modes = 1;
};

} // namespace Nektar::MultiRegions
