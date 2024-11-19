///////////////////////////////////////////////////////////////////////////////
//
// File: Field.hpp
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

#include <MultiRegions/ExpList.h>

#include "MemoryRegion.hpp"

static constexpr FieldState DefaultState = FieldState::Phys;

/**
 * @brief A block means a group of elements of identical shape,
 * basis and order. BlockAttributes stores the most basic
 * information of a block.
 */
class BlockAttributes
{
public:
    BlockAttributes(const size_t num_elements,
                    const size_t num_elements_with_padding,
                    const size_t num_data, const size_t interleave_width)
        : m_num_elements(num_elements),
          num_elements_with_padding(num_elements_with_padding),
          m_num_data(num_data), m_size(num_elements_with_padding * num_data),
          m_interleave_width(interleave_width)
    {
    }

    BlockAttributes(const BlockAttributes &) = default;

    template <typename TData>
    void SetInterleaveWidth(const size_t interleave_width)
    {
        if (interleave_width != 1)
        {
            ASSERTL0(
                num_elements_with_padding % interleave_width == 0,
                "Number of elements is not divisible by interleave width.");
            ASSERTL0(
                interleave_width % tinysimd::simd<TData>::width == 0,
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
        return num_elements_with_padding;
    }

    size_t GetNumData(void) const
    {
        return m_num_data;
    }

    size_t size(void) const
    {
        return m_size;
    }

    size_t GetInterleaveWidth(void) const
    {
        return m_interleave_width;
    }

    size_t GetNumPaddingElements(void) const
    {
        return num_elements_with_padding - m_num_elements;
    }

    size_t GetNumElmtGroups(void) const
    {
        return num_elements_with_padding / m_interleave_width;
    }

private:
    const size_t m_num_elements;
    const size_t num_elements_with_padding;
    const size_t m_num_data;
    const size_t m_size;
    size_t m_interleave_width;
};

template <typename TData> class BlockAccessor : public BlockAttributes
{
public:
    BlockAccessor(const BlockAttributes blockAttr,
                  MemoryRegion<TData> &memory_region, const size_t offset)
        : BlockAttributes(blockAttr), m_memory_region(memory_region),
          m_offset(offset)
    {
    }

    BlockAccessor(const size_t num_elements,
                  const size_t num_elements_with_padding, const size_t num_data,
                  const size_t interleave_width,
                  MemoryRegion<TData> &memory_region, const size_t offset)
        : BlockAttributes(num_elements, num_elements_with_padding, num_data,
                          interleave_width),
          m_memory_region(memory_region), m_offset(offset)
    {
    }

    /**
     * @brief Get the pointer to the host/device memory.
     *
     * @return    - TData*
     */
    template <typename MemSpace, typename MemAccess>
    typename const_if<std::is_same_v<MemAccess, ReadOnly>, TData>::type *GetPtr()
    {
        return m_memory_region.template GetPtr<MemSpace, MemAccess>() +
               m_offset;
    }

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
     * @brief Gets the device rank of the memory region block.
     *
     * @return size_t
     */
    size_t GetDeviceRank() const
    {
        return m_memory_region.GetDeviceRank();
    }

private:
    MemoryRegion<TData> &m_memory_region;
    size_t m_offset = 0;
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
template <typename TData>
std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const MultiRegions::ExpListSharedPtr explist,
    const size_t interleave_width = 1)
{
    size_t vector_width = NektarSpaces::vector_width<TData>::value;

    std::vector<BlockAttributes> blockAttr;

    // initialize the first block using the first element
    auto expPtr        = explist->GetExp(0);
    int prevIsDeformed = -1, thisIsDeformed = -1;
    std::vector<LibUtilities::BasisKey> prevbasisKeys(
        expPtr->GetNumBases(), LibUtilities::NullBasisKey);
    std::vector<LibUtilities::BasisKey> thisbasisKeys(
        expPtr->GetNumBases(), LibUtilities::NullBasisKey);

    size_t num_elements = 1;
    size_t ndata        = state == FieldState::Phys ? expPtr->GetTotPoints()
                                                    : expPtr->GetNcoeffs();
    for (int d = 0; d < expPtr->GetNumBases(); d++)
    {
        prevbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
    }
    prevIsDeformed = expPtr->GetMetricInfo()->GetGtype();

    // loop over elements
    for (int i = 1; i < explist->GetNumElmts(); i++)
    {
        expPtr = explist->GetExp(i);

        // fetch basiskeys of current element
        for (int d = 0; d < expPtr->GetNumBases(); d++)
        {
            thisbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        thisIsDeformed = expPtr->GetMetricInfo()->GetGtype();

        // if the basis is the same as the previous one, increment the number of
        // elements
        if (thisbasisKeys == prevbasisKeys && thisIsDeformed == prevIsDeformed)
        {
            num_elements++;
        }
        else // if not, create a new block with the number of elements = 1
        {
            size_t num_elements_with_padding =
                ((num_elements + vector_width - 1) / vector_width) *
                vector_width;
            blockAttr.push_back({num_elements, num_elements_with_padding, ndata,
                                 interleave_width});

            // update ndata for a new block
            num_elements   = 1;
            ndata          = state == FieldState::Phys ? expPtr->GetTotPoints()
                                                       : expPtr->GetNcoeffs();
            prevbasisKeys  = thisbasisKeys;
            prevIsDeformed = thisIsDeformed;
        }
    }

    // update the padding elements for the last block
    size_t num_elements_with_padding =
        ((num_elements + vector_width - 1) / vector_width) * vector_width;
    blockAttr.push_back(
        {num_elements, num_elements_with_padding, ndata, interleave_width});

    return blockAttr;
}

/**
 * @brief A Field represents expansion data to be operated on.
 *
 * @tparam TData  The floating-point representation used by the field.
 * @tparam TState A FieldState value representing the state of the field.
 */
template <typename TData, FieldState TState = DefaultState> class Field
{
public:
    Field(){};
    Field(const Field &) = delete;
    ~Field()
    {
        if (m_host)
        {
            operator delete[](m_host, std::align_val_t(m_alignment));
        }
        m_host = nullptr;
    }

    /**
     * @brief Construct a new Field object by moving storage from an existing
     * Field object.
     *
     * @param rhs
     */
    Field(Field &&rhs)
        : m_name(std::move(rhs.m_name)),
          m_var_names(std::move(rhs.m_var_names)),
          m_host(std::move(rhs.m_host)),
          m_alignment(std::move(rhs.m_alignment)),
          m_num_device(std::move(rhs.m_num_device)),
          m_block_accessors(std::move(rhs.m_block_accessors)),
          m_memory_regions(std::move(rhs.m_memory_regions)),
          m_blk_to_mr_offset(std::move(rhs.m_blk_to_mr_offset)),
          m_blk_to_mr_mapping(std::move(rhs.m_blk_to_mr_mapping))
    {
        rhs.m_name = "";
        rhs.m_var_names.clear();
        rhs.m_host       = nullptr;
        rhs.m_alignment  = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_num_device = 1;
        rhs.m_block_accessors.clear();
        rhs.m_memory_regions.clear();
        rhs.m_blk_to_mr_offset.clear();
        rhs.m_blk_to_mr_mapping.clear();
    }

    /**
     * @brief Move assignment operator.
     *
     * @param rhs
     *
     * @return Field&
     */
    Field &operator=(Field &&rhs)
    {
        m_name              = std::move(rhs.m_name);
        m_var_names         = std::move(rhs.m_var_names);
        m_host              = std::move(rhs.m_host);
        m_alignment         = std::move(rhs.m_alignment);
        m_num_device        = std::move(rhs.m_num_device);
        m_block_accessors   = std::move(rhs.m_block_accessors);
        m_memory_regions    = std::move(rhs.m_memory_regions);
        m_blk_to_mr_offset  = std::move(rhs.m_blk_to_mr_offset);
        m_blk_to_mr_mapping = std::move(rhs.m_blk_to_mr_mapping);

        rhs.m_name = "";
        rhs.m_var_names.clear();
        rhs.m_host       = nullptr;
        rhs.m_alignment  = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_num_device = 1;
        rhs.m_block_accessors.clear();
        rhs.m_memory_regions.clear();
        rhs.m_blk_to_mr_offset.clear();
        rhs.m_blk_to_mr_mapping.clear();
        return *this;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the names of the components.
     *
     * @tparam MemSpace   - Type of memory space to use
     *
     * @param name        - Name of the field (memory region)
     * @param blockAttr   - Block attributes.
     * @param components  - Names of components for a vector field.
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> Create(
        const std::string name, const std::vector<BlockAttributes> blockAttr,
        const std::vector<std::string> components, const size_t alignment,
        const bool device_only = false)
    {
        size_t num_device = 1;
        auto field        = Field(name, components, alignment, num_device);

        SetBlockToMemoryRegionMapping<MemSpace>(field, blockAttr, alignment,
                                                device_only);

        return field;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the names of the components.
     *
     * @tparam MemSpace   - Type of memory space to use
     *
     * @param blocks      - Field data specification.
     * @param components  - Names of components for a vector field.
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> Create(
        const std::vector<BlockAttributes> blockAttr,
        const std::vector<std::string> components, const size_t alignment,
        const bool device_only = false)
    {
        return Field<TData, TState>::template Create<MemSpace>(
            "", blockAttr, components, alignment, device_only);
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the number of components.
     *
     * @tparam MemSpace   - Type of memory space to use
     *
     * @param name        - Name of the field (memory region)
     * @param blockAttr   - Block attributes.
     * @param nvar        - Number of components for a vector field.
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> Create(
        const std::string name, const std::vector<BlockAttributes> blockAttr,
        const int nvar, const size_t alignment,
        [[maybe_unused]] const bool device_only = false)
    {
        size_t num_device = 1;
        auto field        = Field(name, nvar, alignment, num_device);

        SetBlockToMemoryRegionMapping<MemSpace>(field, blockAttr, alignment,
                                                device_only);

        return field;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the number of components.
     *
     * @tparam MemSpace   - Type of memory space to use
     *
     * @param blocks      - Field data specification.
     * @param nvar        - Number of components for a vector field.
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> Create(
        const std::vector<BlockAttributes> blockAttr, const int nvar,
        const size_t alignment, const bool device_only = false)
    {
        return Field<TData, TState>::template Create<MemSpace>(
            "", blockAttr, nvar, alignment, device_only);
    }

    template <typename MemSpace>
    static void SetBlockToMemoryRegionMapping(
        Field<TData, TState> &field,
        const std::vector<BlockAttributes> blockAttr, const size_t alignment,
        const bool device_only = false)
    {
#if defined(NEKTAR_USE_SINGLE_MEMORY_REGION_PER_DEVICE)
        std::vector<size_t> offset(field.m_num_device, 0);
        std::vector<size_t> size(field.m_num_device, 0);
        size_t hsize = 0;
        for (size_t blk = 0; blk < blockAttr.size(); ++blk)
        {
            // Set one-to-one MemoryRegion to device mapping.
            auto mr    = blk % field.m_num_device;
            auto nsize = blockAttr[blk].size() * field.GetNumComponents();
            field.m_blk_to_mr_mapping.push_back(mr);
            field.m_blk_to_mr_offset.push_back(offset[mr]);
            offset[mr] += nsize;

            // Compute MemoryRegion memory size.
            size[mr] += nsize;
            hsize += nsize;
        }

        // Allocate contiguous memory on the host.
        if (!device_only)
        {
            field.m_host = static_cast<TData *>(::operator new[](
                hsize * sizeof(TData), std::align_val_t(field.m_alignment)));
        }

        auto hsrc = field.m_host;
        for (size_t mr = 0; mr < field.m_num_device; ++mr)
        {
            // Create memory region.
            auto device_rank = mr;
            if (!device_only)
            {
                field.m_memory_regions.push_back(
                    MemoryRegion<TData>::template CreateFromHostPtr<MemSpace>(
                        field.m_name + std::to_string(mr), hsrc, size[mr],
                        alignment, device_rank));
                hsrc += size;
            }
            else
            {
                field.m_memory_regions.push_back(
                    MemoryRegion<TData>::template Create<MemSpace>(
                        field.m_name + std::to_string(mr), size[mr], alignment,
                        device_only, device_rank));
            }

            // Zero memory.
            field.m_memory_regions[mr].template Initialize<MemSpace>(0);
        }
#else
        size_t hsize = 0;
        for (size_t blk = 0; blk < blockAttr.size(); ++blk)
        {
            // Set one-to-one block to MemoryRegion mapping.
            auto nsize = blockAttr[blk].size() * field.GetNumComponents();
            field.m_blk_to_mr_mapping.push_back(blk);
            field.m_blk_to_mr_offset.push_back(0);
            hsize += nsize;
        }

        // Allocate contiguous memory on the host.
        if (!device_only)
        {
            field.m_host = static_cast<TData *>(::operator new[](
                hsize * sizeof(TData), std::align_val_t(field.m_alignment)));
        }

        auto hsrc = field.m_host;
        for (size_t blk = 0; blk < blockAttr.size(); ++blk)
        {
            // Create memory region.
            auto device_rank = blk % field.m_num_device;
            auto size        = blockAttr[blk].size() * field.GetNumComponents();
            if (!device_only)
            {
                field.m_memory_regions.push_back(
                    MemoryRegion<TData>::template CreateFromHostPtr<MemSpace>(
                        field.m_name + std::to_string(blk), hsrc, size,
                        alignment, device_rank));
                hsrc += size;
            }
            else
            {
                field.m_memory_regions.push_back(
                    MemoryRegion<TData>::template Create<MemSpace>(
                        field.m_name + std::to_string(blk), size, alignment,
                        device_only, device_rank));
            }

            // Zero memory.
            field.m_memory_regions[blk].template Initialize<MemSpace>(0);
        }
#endif
        for (size_t blk = 0; blk < blockAttr.size(); ++blk)
        {
            field.m_block_accessors.push_back(BlockAccessor(
                blockAttr[blk],
                field.m_memory_regions[field.m_blk_to_mr_mapping[blk]],
                field.m_blk_to_mr_offset[blk]));
        }
    }

    /**
     * @brief Templated initialize method.
     *
     */
    template <typename MemSpace> void Initialize(const TData val)
    {
        for (size_t mr = 0; mr < m_memory_regions.size(); ++mr)
        {
            m_memory_regions[mr].template Initialize<MemSpace>(val);
        }
    }

    /**
     * @brief Copy the data to a MemoryRegion
     *
     * @return MemoryRegion
     */
    template <typename MemSpace, typename TDataOut = TData,
              class Alloc = std::allocator<TDataOut>>
    MemoryRegion<TDataOut> ToMemoryRegion(
        const size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        size_t compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        MemoryRegion<TDataOut> mr =
            MemoryRegion<TDataOut>::template Create<MemSpace>(
                compSize * this->GetNumComponents(), alignment, false);

        // Copy the data from the input field
        auto dst = mr.template GetPtr<MemSpace, WriteOnly>();
        for (size_t blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto nSize  = this->GetBlocks()[blk].size();
            auto nElmts = this->GetBlocks()[blk].GetNumElements();
            auto nPts   = this->GetBlocks()[blk].GetNumData();
            auto src =
                this->GetBlocks()[blk].template GetPtr<MemSpace, ReadOnly>();
            auto device_rank = this->GetDeviceRank(blk);
            for (auto n = 0; n < this->GetNumComponents(); n++)
            {
                if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
                {
                    std::copy(src, src + nElmts * nPts, dst + n * compSize);
                }
                else if constexpr (std::is_same_v<MemSpace,
                                                  NektarSpaces::DeviceSpace>)
                {
                    deviceMemcpy<DeviceToDevice>(dst + n * compSize, src,
                                                 nElmts * nPts, device_rank);
                }
                src += nSize;
            }

            dst += nElmts * nPts;
        }

        return mr;
    }

    /**
     * @brief Copy the data to a std::vector
     *
     * @return std::vector
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TDataOut>>
    std::vector<TDataOut, Alloc> ToVector()
    {
        size_t compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        std::vector<TDataOut, Alloc> array(compSize * this->GetNumComponents());

        // Copy the data from the input field.
        auto dst = array.data();
        for (size_t blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto src =
                this->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto nSize  = this->GetBlocks()[blk].size();
            auto nElmts = this->GetBlocks()[blk].GetNumElements();
            auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0; n < this->GetNumComponents(); n++)
            {
                std::copy(src, src + nElmts * nPts, dst + n * compSize);
                src += nSize;
            }

            dst += nElmts * nPts;
        }

        return array;
    }

    /**
     * @brief Copy the data to a Nektar::Array
     *
     * @return Array<Nektar::OneD, TDataOut>
     */
    template <typename TDataOut = TData>
    Nektar::Array<Nektar::OneD, TDataOut> ToArray()
    {
        size_t compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        Nektar::Array<Nektar::OneD, TDataOut> array(compSize *
                                                    this->GetNumComponents());

        // Copy the data from the input field.
        auto dst = array.data();
        for (size_t blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto src =
                this->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto nSize  = this->GetBlocks()[blk].size();
            auto nElmts = this->GetBlocks()[blk].GetNumElements();
            auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0; n < this->GetNumComponents(); n++)
            {
                std::copy(src, src + nElmts * nPts, dst + n * compSize);
                src += nSize;
            }

            dst += nElmts * nPts;
        }

        return array;
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Field
     *
     * @param field - Field to copy from
     *
     */
    template <typename MemSpace> void Copy(Field &field)
    {
        if (this->GetBlocks().size() != field.GetBlocks().size())
        {
            std::stringstream msg;

            msg << "Field::Copy - "
                << "Block number mismatch between (" << field.GetName()
                << ") and (" << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        if (this->size() != field.size())
        {
            std::stringstream msg;

            msg << "Field::Copy - "
                << "Memory size mismatch between (" << field.GetName()
                << ") and (" << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        for (size_t mr = 0; mr < m_memory_regions.size(); ++mr)
        {
            m_memory_regions[mr].template Copy<MemSpace>(
                field.m_memory_regions[mr]);
        }
        for (size_t blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            m_block_accessors[blk].template SetInterleaveWidth<TData>(
                field.m_block_accessors[blk].GetInterleaveWidth());
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        std::vector
     *
     * @param array - std::vector to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              class Alloc = std::allocator<TDataIn>>
    void CopyVector(const std::vector<TDataIn, Alloc> &array)
    {
        size_t nSize = 0;
        for (auto &block : this->GetBlocks())
        {
            nSize += block.GetNumData() * block.GetNumElements();
        }
        nSize *= this->GetNumComponents();

        if (nSize != array.size())
        {
            std::stringstream msg;

            msg << "Field::CopyVector - "
                << "Memory size mismatch between (std::vector) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        this->template CopyFromHostPtr<MemSpace>(array.data());
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn>
    void CopyArray(const Nektar::Array<Nektar::OneD, TDataIn> &array)
    {
        size_t nSize = 0;
        for (auto &block : this->GetBlocks())
        {
            nSize += block.GetNumData() * block.GetNumElements();
        }
        nSize *= this->GetNumComponents();

        if (nSize != array.size())
        {
            std::stringstream msg;

            msg << "Field::CopyArray - "
                << "Memory size mismatch between (Nektar::array) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        this->template CopyFromHostPtr<MemSpace>(array.data());
    }

    /**
     * @brief Get BlockAccessors for the field.
     *
     * @return std::vector<BlockAccessor>
     */
    std::vector<BlockAccessor<TData>> &GetBlocks()
    {
        return m_block_accessors;
    }

    /**
     * @brief Gets the number of the memory region.
     *
     * @return size_t
     */
    size_t GetNumMemorRegion() const
    {
        return m_memory_regions.size();
    }

    /**
     * @brief Gets the size of the field.
     *
     * @return size_t
     */
    size_t size() const
    {
        size_t nSize = 0;
        for (size_t blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            nSize += m_block_accessors[blk].size();
        }
        return nSize * this->GetNumComponents();
    }

    /**
     * @brief Gets the name of the field.
     *
     * @return std::string
     */
    std::string GetName(void) const
    {
        return m_name;
    }

    /**
     * @brief Gets the number of components for a vector field.
     *
     * @return size_t
     */
    size_t GetNumComponents() const
    {
        return m_var_names.size();
    }

private:
    /**
     * @brief Construct a new Field object.
     *
     * @param name       Name of the field object.
     * @param blocks     Field data layout specification.
     * @param components Names of components for vector field.
     */
    Field(const std::string name, const int nvar, const size_t alignment,
          const size_t num_device)
        : m_name(name), m_var_names(nvar), m_alignment(alignment),
          m_num_device(num_device)
    {
    }

    Field(const std::string name, const std::vector<std::string> components,
          const size_t alignment, const size_t num_device)
        : m_name(name), m_var_names(components), m_alignment(alignment),
          m_num_device(num_device)
    {
    }

    /**
     * @brief Copy the data from a host source pointer. This is a helper
     * function to copy the source data from a host container (vector, array,
     * etc.). CopyFromHostPtr should not be otherwise used and has been made
     * private.
     *
     * @param const TDataIn* src
     */
    template <typename MemSpace, typename TDataIn>
    void CopyFromHostPtr(const TDataIn *src)
    {
        auto compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        for (size_t blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto offset = m_blk_to_mr_offset[blk];
            auto nSize  = this->GetBlocks()[blk].size();
            auto nElmts = this->GetBlocks()[blk].GetNumElements();
            auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0; n < this->GetNumComponents(); n++)
            {
                m_memory_regions[m_blk_to_mr_mapping[blk]]
                    .template CopyFromHostPtr<MemSpace>(src + n * compSize,
                                                        nElmts * nPts, offset);
                offset += nSize;
            }
            src += nElmts * nPts;
        }
    }

    // Member variables:
    std::string m_name;
    std::vector<std::string> m_var_names;
    TData *m_host       = nullptr;
    size_t m_alignment  = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    size_t m_num_device = 1;
    std::vector<BlockAccessor<TData>> m_block_accessors;
    std::vector<MemoryRegion<TData>> m_memory_regions;
    std::vector<size_t> m_blk_to_mr_offset;
    std::vector<size_t> m_blk_to_mr_mapping;
};
