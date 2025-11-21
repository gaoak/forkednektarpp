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

namespace Nektar::Operators
{

template <typename MemSpace, typename TData>
void AllocateFieldStorage(FieldBase<TData> *field);

/**
 * @brief A block means a group of elements of identical shape,
 * basis and order. BlockAttributes stores the most basic
 * information of a block.
 */
class BlockAttributes
{
public:
    BlockAttributes(const size_t exp_idx, const size_t num_elements,
                    const size_t num_elements_with_padding,
                    const unsigned int num_data,
                    const unsigned int interleave_width)
        : m_exp_idx(exp_idx), m_num_elements(num_elements),
          m_num_elements_with_padding(num_elements_with_padding),
          m_num_data(num_data), m_size(num_elements_with_padding * num_data),
          m_interleave_width(interleave_width)
    {
    }

    BlockAttributes(const BlockAttributes &) = default;

    template <typename TData>
    void SetInterleaveWidth(const unsigned interleave_width)
    {
        if (interleave_width != 1)
        {
            ASSERTL0(
                m_num_elements_with_padding % interleave_width == 0,
                "Number of elements is not divisible by interleave width.");
            ASSERTL0(
                interleave_width % tinysimd::simd<TData>::width == 0,
                "interleave width should be divisible by AVX vector width.");
        }

        m_interleave_width = interleave_width;
    }

    size_t GetExpIdx(void) const
    {
        return m_exp_idx;
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

    size_t size(void) const
    {
        return m_size;
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
    const size_t m_exp_idx;
    const size_t m_num_elements;
    const size_t m_num_elements_with_padding;
    const unsigned int m_num_data;
    const size_t m_size;
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
template <typename TData>
std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const MultiRegions::ExpListSharedPtr explist,
    const unsigned interleave_width = 1)
{
    const auto vector_width = NektarSpaces::vector_width<TData>::value;

    std::vector<BlockAttributes> blockAttr;

    // initialize the first block using the first element
    auto expPtr        = explist->GetExp(0);
    int prevIsDeformed = -1, thisIsDeformed = -1;
    std::vector<LibUtilities::BasisKey> prevbasisKeys(
        expPtr->GetNumBases(), LibUtilities::NullBasisKey);
    std::vector<LibUtilities::BasisKey> thisbasisKeys(
        expPtr->GetNumBases(), LibUtilities::NullBasisKey);

    size_t exp_idx      = 0;
    size_t num_elements = 1;
    unsigned int ndata  = state == FieldState::Phys ? expPtr->GetTotPoints()
                                                    : expPtr->GetNcoeffs();
    for (unsigned int d = 0; d < expPtr->GetNumBases(); d++)
    {
        prevbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
    }
    prevIsDeformed = expPtr->GetGeomFactors()->GetGtype();

    // loop over elements
    for (size_t i = 1; i < explist->GetNumElmts(); i++)
    {
        expPtr = explist->GetExp(i);

        // fetch basiskeys of current element
        for (unsigned int d = 0; d < expPtr->GetNumBases(); d++)
        {
            thisbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        thisIsDeformed = expPtr->GetGeomFactors()->GetGtype();

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
            blockAttr.push_back({exp_idx, num_elements,
                                 num_elements_with_padding, ndata,
                                 interleave_width});

            // update ndata for a new block
            exp_idx        = i;
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
    blockAttr.push_back({exp_idx, num_elements, num_elements_with_padding,
                         ndata, interleave_width});

    return blockAttr;
}

template <typename TData> class BlockAccessor : public BlockAttributes
{
    template <typename TDataField> friend class FieldBase;
    template <typename TDataField, FieldState TState> friend class Field;
    template <typename MemSpace, typename TDataField>
    friend void AllocateFieldStorage(FieldBase<TDataField> *field);

public:
    BlockAccessor(const BlockAttributes blockAttr,
                  MemoryRegion<TData> &&memory_region, FieldBase<TData> *field,
                  const unsigned int num_components,
                  const unsigned int num_homo_modes)
        : BlockAttributes(blockAttr), m_memory_region(std::move(memory_region)),
          m_field(field), m_num_components(num_components),
          m_num_homo_modes(num_homo_modes)
    {
    }

    BlockAccessor(const size_t exp_idx, const size_t num_elements,
                  const size_t num_elements_with_padding,
                  const unsigned int num_data, const unsigned interleave_width,
                  MemoryRegion<TData> &&memory_region, FieldBase<TData> *field,
                  const unsigned int num_components,
                  const unsigned int num_homo_modes)
        : BlockAttributes(exp_idx, num_elements, num_elements_with_padding,
                          num_data, interleave_width),
          m_memory_region(std::move(memory_region)), m_field(field),
          m_num_components(num_components), m_num_homo_modes(num_homo_modes)
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
        // If not yet allocated, allocate contiguous host OR device memory
        // accross all MemoryRegion objects from m_field. Note: m_field is a
        // pointer to a Field object from which the current BlockAccessor object
        // belong to.
        AllocateFieldStorage<MemSpace>(m_field);

        return m_memory_region.template GetPtr<MemSpace, MemAccess>();
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
                    const size_t offset = 0)
    {
        m_memory_region.template Initialize<MemSpace>(val, count, offset);
    }

private:
    // Note: m_field is a pointer to a Field object from which the current
    // BlockAccessor object belong to.
    MemoryRegion<TData> m_memory_region;
    FieldBase<TData> *m_field;
    unsigned int m_num_components = 0;
    unsigned int m_num_homo_modes = 1;
};

/**
 * @brief A FieldBase represents expansion data to be operated on.
 *
 * @tparam TData  The floating-point representation used by the field.
 */
template <typename TData> class FieldBase
{
    template <typename MemSpace, typename TDataField>
    friend void AllocateFieldStorage(FieldBase<TDataField> *field);

public:
    /**
     * @brief Templated initialize method.
     *
     */
    template <typename MemSpace> void Initialize(const TData val)
    {
        // If not yet allocated, allocate contiguous host OR device memory
        // across all MemoryRegion objects within current Field object before
        // data initialization.
        AllocateFieldStorage<MemSpace>(this);

        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            m_block_accessors[blk]
                .m_memory_region.template Initialize<MemSpace>(val);
        }
    }

    /**
     * @brief Copy the data to a std::vector
     *
     * @return std::vector
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TDataOut>>
    std::vector<TDataOut, Alloc> ToVector()
    {
        // If not yet allocated, allocate contiguous host memory across all
        // MemoryRegion objects within current Field objects before data copy to
        // vector.
        AllocateFieldStorage<NektarSpaces::HostSpace>(this);

        unsigned int compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        std::vector<TDataOut, Alloc> array(compSize * this->GetNumComponents() *
                                           this->GetNumHomoModes());

        // Copy the data from the input field.
        auto dst = array.data();
        for (unsigned int blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto src =
                this->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const auto nSize  = this->GetBlocks()[blk].size();
            const auto nElmts = this->GetBlocks()[blk].GetNumElements();
            const auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
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
        // If not yet allocated, allocate contiguous host memory across all
        // MemoryRegion objects within current Field objects before data copy to
        // NekArray.
        AllocateFieldStorage<NektarSpaces::HostSpace>(this);

        unsigned int compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        Nektar::Array<Nektar::OneD, TDataOut> array(
            compSize * this->GetNumComponents() * this->GetNumHomoModes());

        // Copy the data from the input field.
        auto dst = array.data();
        for (unsigned int blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto src =
                this->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const auto nSize  = this->GetBlocks()[blk].size();
            const auto nElmts = this->GetBlocks()[blk].GetNumElements();
            const auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
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
     *        FieldBase
     *
     * @param field - FieldBase to copy from
     *
     */
    template <typename MemSpace> void Copy(FieldBase &field)
    {
        // If not yet allocated, allocate contiguous host OR device memory
        // across all MemoryRegion objects within the input argument field
        // object before data copy.
        AllocateFieldStorage<MemSpace>(&field);

        // If not yet allocated, allocate contiguous host OR device memory
        // across all MemoryRegion objects within current Field object before
        // data copy from input argument field object.
        AllocateFieldStorage<MemSpace>(this);

        if (this->GetBlocks().size() != field.GetBlocks().size())
        {
            std::stringstream msg;

            msg << "FieldBase::Copy - "
                << "Block number mismatch between (" << field.GetName()
                << ") and (" << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        if (this->size() != field.size())
        {
            std::stringstream msg;

            msg << "FieldBase::Copy - "
                << "Memory size mismatch between (" << field.GetName()
                << ") and (" << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            m_block_accessors[blk].m_memory_region.template Copy<MemSpace>(
                field.m_block_accessors[blk].m_memory_region);
        }
        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
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
        // If not yet allocated, allocate contiguous host memory across all
        // MemoryRegion objects within current Field object before data copy
        // from vector.
        AllocateFieldStorage<MemSpace>(this);

        unsigned int compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        if (compSize * this->GetNumComponents() * this->GetNumHomoModes() !=
            array.size())
        {
            std::stringstream msg;

            msg << "FieldBase::CopyVector - "
                << "Memory size mismatch between (std::vector) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        // Copy the data to the array.
        auto src = array.data();
        for (unsigned int blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto offset       = 0;
            const auto nSize  = this->GetBlocks()[blk].size();
            const auto nElmts = this->GetBlocks()[blk].GetNumElements();
            const auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
            {
                m_block_accessors[blk]
                    .m_memory_region.template CopyFromHostPtr<MemSpace>(
                        src + n * compSize, nElmts * nPts, offset);
                offset += nSize;
            }
            src += nElmts * nPts;
        }
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
        // If not yet allocated, allocate contiguous host memory across all
        // MemoryRegion objects within current Field object before data copy
        // from NekArray.
        AllocateFieldStorage<MemSpace>(this);

        unsigned int compSize = 0;
        for (auto &block : this->GetBlocks())
        {
            compSize += block.GetNumData() * block.GetNumElements();
        }

        if (compSize * this->GetNumComponents() * this->GetNumHomoModes() !=
            array.size())
        {
            std::stringstream msg;

            msg << "FieldBase::CopyArray - "
                << "Memory size mismatch between (Nektar::array) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        // Copy the data to the array.
        auto src = array.data();
        for (unsigned int blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto offset       = 0;
            const auto nSize  = this->GetBlocks()[blk].size();
            const auto nElmts = this->GetBlocks()[blk].GetNumElements();
            const auto nPts   = this->GetBlocks()[blk].GetNumData();
            for (auto n = 0;
                 n < this->GetNumComponents() * this->GetNumHomoModes(); n++)
            {
                m_block_accessors[blk]
                    .m_memory_region.template CopyFromHostPtr<MemSpace>(
                        src + n * compSize, nElmts * nPts, offset);
                offset += nSize;
            }
            src += nElmts * nPts;
        }
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
     * @brief Gets the size of the field.
     *
     * @return size_t
     */
    size_t size() const
    {
        size_t nSize = 0;
        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            nSize += m_block_accessors[blk].m_memory_region.size();
        }
        return nSize;
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
     * @return unsigned int
     */
    unsigned int GetNumComponents() const
    {
        return m_component_names.size();
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

    typedef TData value_type;

protected:
    FieldBase()                  = default;
    FieldBase(const FieldBase &) = delete;

    /**
     * @brief Construct a new FieldBase object.
     *
     * @param name           - Name of the field object.
     * @param num_components - Number of components.
     * @param num_homo_modes - Number of components.
     * @param memAllocType   - [ePageable, ePinned].
     */
    FieldBase(const std::string name, const unsigned int num_components,
              const unsigned int num_homo_modes, const size_t alignment,
              const MemAllocType &memAllocType)
        : m_name(name), m_component_names(num_components),
          m_num_homo_modes(num_homo_modes), m_alignment(alignment),
          m_memAllocType(memAllocType)
    {
    }

    /**
     * @brief Construct a new FieldBase object.
     *
     * @param name           - Name of the field object.
     * @param components     - Names of components for vector field.
     * @param num_homo_modes - Number of components.
     * @param memAllocType   - [ePageable, ePinned].
     */
    FieldBase(const std::string name, const std::vector<std::string> components,
              const unsigned int num_homo_modes, const size_t alignment,
              const MemAllocType &memAllocType)
        : m_name(name), m_component_names(components),
          m_num_homo_modes(num_homo_modes), m_alignment(alignment),
          m_memAllocType(memAllocType)
    {
    }

    ~FieldBase()
    {
        if (m_host)
        {
            if (m_memAllocType == ePageable)
            {
                hostFree(m_host, m_alignment);
            }
            else if (m_memAllocType == ePinned)
            {
                hostFreePinned(m_host, m_alignment);
            }
        }

        if (m_device)
        {
            deviceFree(m_device, this->size(), m_alignment);
        }

        m_host   = nullptr;
        m_device = nullptr;
    }

    /**
     * @brief Construct a new FieldBase object by moving storage from an
     * existing FieldBase object.
     *
     * @param rhs
     */
    FieldBase(FieldBase &&rhs)
        : m_name(std::move(rhs.m_name)),
          m_component_names(std::move(rhs.m_component_names)),
          m_num_homo_modes(std::move(rhs.m_num_homo_modes)),
          m_host(std::move(rhs.m_host)), m_device(std::move(rhs.m_device)),
          m_alignment(std::move(rhs.m_alignment)),
          m_block_accessors(std::move(rhs.m_block_accessors)),
          m_memAllocType(std::move(rhs.m_memAllocType))
    {
        for (auto &blocks : m_block_accessors)
        {
            blocks.m_field = this;
        }
        rhs.m_name = "";
        rhs.m_component_names.clear();
        rhs.m_num_homo_modes = 1;
        rhs.m_host           = nullptr;
        rhs.m_device         = nullptr;
        rhs.m_alignment      = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_block_accessors.clear();
        rhs.m_memAllocType = ePageable;
    }

    /**
     * @brief Move assignment operator.
     *
     * @param rhs
     *
     * @return FieldBase&
     */
    FieldBase &operator=(FieldBase &&rhs)
    {
        m_name            = std::move(rhs.m_name);
        m_component_names = std::move(rhs.m_component_names);
        m_num_homo_modes  = std::move(rhs.m_num_homo_modes);
        m_host            = std::move(rhs.m_host);
        m_device          = std::move(rhs.m_device);
        m_alignment       = std::move(rhs.m_alignment);
        m_block_accessors = std::move(rhs.m_block_accessors);
        m_memAllocType    = std::move(rhs.m_memAllocType);
        for (auto &blocks : m_block_accessors)
        {
            blocks.m_field = this;
        }

        rhs.m_name = "";
        rhs.m_component_names.clear();
        rhs.m_num_homo_modes = 1;
        rhs.m_host           = nullptr;
        rhs.m_device         = nullptr;
        rhs.m_alignment      = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
        rhs.m_block_accessors.clear();
        rhs.m_memAllocType = ePageable;
        return *this;
    }

    // Member variables:
    std::string m_name;
    std::vector<std::string> m_component_names;
    unsigned int m_num_homo_modes = 1;
    TData *m_host                 = nullptr;
    TData *m_device               = nullptr;
    size_t m_alignment            = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
    std::vector<BlockAccessor<TData>> m_block_accessors;
    MemAllocType m_memAllocType;
};

/**
 * @brief A Field represents expansion data to be operated on.
 *
 * @tparam TData  The floating-point representation used by the field.
 * @tparam TState A FieldState value representing the state of the field.
 */
template <typename TData, FieldState TState>
class Field : public FieldBase<TData>
{
public:
    Field()              = default;
    Field(const Field &) = delete;
    ~Field()             = default;

    /**
     * @brief Construct a new Field object.
     *
     * @param name           - Name of the field object.
     * @param blockAttr      - Block attributes.
     * @param num_components - Number of components.
     * @param num_homo_modes - Number of homogeneous modes.
     * @param alignment      - Memory alignment to use.
     * @param memAllocType   - [ePageable, ePinned].
     */
    Field(const std::string name, const std::vector<BlockAttributes> blockAttr,
          const unsigned int num_components, const unsigned int num_homo_modes,
          const size_t alignment, const MemAllocType &memAllocType = ePageable)
        : FieldBase<TData>(name, num_components, num_homo_modes, alignment,
                           memAllocType)
    {
        for (unsigned int blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto nsize =
                blockAttr[blk].size() * num_components * num_homo_modes;
            auto mr = MemoryRegion<TData>(name + std::to_string(blk), nsize,
                                          alignment, memAllocType);
            this->m_block_accessors.push_back(
                BlockAccessor(blockAttr[blk], std::move(mr), this,
                              num_components, num_homo_modes));
        }
    }

    /**
     * @brief Construct a new Field object.
     *
     * @param blockAttr      - Block attributes.
     * @param num_components - Number of components.
     * @param num_homo_modes - Number of homogeneous modes.
     * @param alignment      - Memory alignment to use.
     * @param memAllocType   - [ePageable, ePinned].
     */
    Field(const std::vector<BlockAttributes> blockAttr,
          const unsigned int num_components, const unsigned int num_homo_modes,
          const size_t alignment, const MemAllocType &memAllocType = ePageable)
        : Field<TData, TState>("", blockAttr, num_components, num_homo_modes,
                               alignment, memAllocType)
    {
    }

    /**
     * @brief Construct a new Field object.
     *
     * @param name           - Name of the field object.
     * @param blockAttr      - Block attributes.
     * @param components     - Names of components for vector field.
     * @param num_homo_modes - Number of homogeneous modes.
     * @param alignment      - Memory alignment to use.
     * @param memAllocType   - [ePageable, ePinned].
     */
    Field(const std::string name, const std::vector<BlockAttributes> blockAttr,
          const std::vector<std::string> components,
          const unsigned int num_homo_modes, const size_t alignment,
          const MemAllocType &memAllocType = ePageable)
        : FieldBase<TData>(name, components, num_homo_modes, alignment,
                           memAllocType)
    {
        for (unsigned int blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto nsize =
                blockAttr[blk].size() * components.size() * num_homo_modes;
            auto mr = MemoryRegion<TData>(name + std::to_string(blk), nsize,
                                          alignment, memAllocType);
            this->m_block_accessors.push_back(
                BlockAccessor(blockAttr[blk], std::move(mr), this,
                              components.size(), num_homo_modes));
        }
    }

    /**
     * @brief Construct a new Field object.
     *
     * @param blockAttr      - Block attributes.
     * @param components     - Names of components for vector field.
     * @param num_homo_modes - Number of homogeneous modes.
     * @param alignment      - Memory alignment to use.
     * @param memAllocType   - [ePageable, ePinned].
     */
    Field(const std::vector<BlockAttributes> blockAttr,
          const std::vector<std::string> components,
          const unsigned int num_homo_modes, const size_t alignment,
          const MemAllocType &memAllocType = ePageable)
        : Field<TData, TState>("", blockAttr, components, num_homo_modes,
                               alignment, memAllocType)
    {
    }

    /**
     * @brief Construct a new Field object by moving storage from an
     * existing Field object.
     *
     * @param rhs
     */
    Field(Field &&rhs) : FieldBase<TData>(std::move(rhs))
    {
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
        FieldBase<TData>::operator=(std::move(rhs));
        return *this;
    }
};

/**
 * @brief Allocate contiguous host OR device memory accross all MemoryRegion
 * objects belonging to a FieldBase object pointer.
 *
 * @param field  - FieldBase object pointer.
 */
template <typename MemSpace, typename TData>
void AllocateFieldStorage(FieldBase<TData> *field)
{
    if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
    {
        // Allocate contiguous host memory accross all MemoryRegions of the
        // field object.
        if (!field->m_host)
        {
            if (field->m_memAllocType == ePageable)
            {
                hostMalloc(&field->m_host, field->size() * sizeof(TData),
                           field->m_alignment);
            }
            else if (field->m_memAllocType == ePinned)
            {
                hostMallocPinned(&field->m_host, field->size() * sizeof(TData),
                                 field->m_alignment);
            }
            std::memset((void *)field->m_host, 0,
                        field->size() * sizeof(TData));

            auto src = field->m_host;
            for (unsigned int blk = 0; blk < field->m_block_accessors.size();
                 ++blk)
            {
                field->m_block_accessors[blk].m_memory_region.SetHostStorage(
                    src);
                src += field->m_block_accessors[blk].m_memory_region.size();
            }
        }
    }
    else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
    {
        // Allocate contiguous device memory accross all MemoryRegions of the
        // field object.
        if (!field->m_device)
        {
            deviceMalloc(&field->m_device, field->size() * sizeof(TData),
                         field->m_alignment);
            deviceMemset(field->m_device, 0, field->size() * sizeof(TData));

            auto src = field->m_device;
            for (unsigned int blk = 0; blk < field->m_block_accessors.size();
                 ++blk)
            {
                field->m_block_accessors[blk].m_memory_region.SetDeviceStorage(
                    src);
                src += field->m_block_accessors[blk].m_memory_region.size();
            }
        }
    }
}

} // namespace Nektar::Operators
