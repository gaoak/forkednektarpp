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

#include "Block.hpp"

namespace Nektar::Operators
{

/**
 * @brief A Field represents expansion data to be operated on.
 *
 * @tparam TData  The floating-point representation used by the field.
 * @tparam TState A FieldState value representing the state of the field.
 */
template <typename TData, FieldState TState> class Field
{
    template <typename TDataField, FieldState TStateField>
    friend class BlockAccessor;

public:
    Field() = default;
    ~Field()
    {
        if (m_host)
        {
            if (m_memAllocType == eHostPageable)
            {
                hostFree(m_host, m_alignment);
            }
            else if (m_memAllocType == eHostPinned)
            {
                hostFreePinned(m_host);
            }
        }

        if (m_device)
        {
            deviceFree(m_device, this->size(), m_memAllocType);
        }

        m_instantiated = false;
        m_host         = nullptr;
        m_device       = nullptr;
        m_alignment    = NektarSpaces::host_memory_alignment;
    }

    /**
     * @brief Construct a new Field object.
     *
     * @param name           - Name of the field object.
     * @param blockAttr      - Block attributes.
     * @param num_components - Number of components.
     * @param num_homo_modes - Number of homogeneous modes.
     * @param memAllocType   - [ eHostPageable,  eHostPinned, eDeviceMemoryPool,
     * eHostPinnedDeviceMemoryPool]
     * @param alignment      - Memory alignment to use.
     */
    Field(const std::string name,
          const std::vector<BlockAttributes<TState>> blockAttr,
          const unsigned int num_components, const unsigned int num_homo_modes,
          const MemAllocType &memAllocType = eHostPageable,
          const size_t alignment = NektarSpaces::host_memory_alignment)
        : m_instantiated(true), m_name(name), m_component_names(num_components),
          m_num_homo_modes(num_homo_modes), m_memAllocType(memAllocType),
          m_alignment(alignment)
    {
        for (unsigned int blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto nsize =
                blockAttr[blk].CompSize() * num_components * num_homo_modes;
            auto mr = MemoryRegion<TData>(name + std::to_string(blk), nsize,
                                          memAllocType, alignment);
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
     * @param memAllocType   - [ eHostPageable,  eHostPinned, eDeviceMemoryPool,
     * eHostPinnedDeviceMemoryPool]
     * @param alignment      - Memory alignment to use.
     */
    Field(const std::vector<BlockAttributes<TState>> blockAttr,
          const unsigned int num_components, const unsigned int num_homo_modes,
          const MemAllocType &memAllocType = eHostPageable,
          const size_t alignment = NektarSpaces::host_memory_alignment)
        : Field<TData, TState>("", blockAttr, num_components, num_homo_modes,
                               memAllocType, alignment)
    {
    }

    /**
     * @brief Construct a new Field object.
     *
     * @param name           - Name of the field object.
     * @param blockAttr      - Block attributes.
     * @param components     - Names of components for vector field.
     * @param num_homo_modes - Number of homogeneous modes.
     * @param memAllocType   - [ eHostPageable,  eHostPinned, eDeviceMemoryPool,
     * eHostPinnedDeviceMemoryPool]
     * @param alignment      - Memory alignment to use.
     */
    Field(const std::string name,
          const std::vector<BlockAttributes<TState>> blockAttr,
          const std::vector<std::string> components,
          const unsigned int num_homo_modes,
          const MemAllocType &memAllocType = eHostPageable,
          const size_t alignment = NektarSpaces::host_memory_alignment)
        : m_instantiated(true), m_name(name), m_component_names(components),
          m_num_homo_modes(num_homo_modes), m_memAllocType(memAllocType),
          m_alignment(alignment)
    {
        for (unsigned int blk = 0; blk < blockAttr.size(); ++blk)
        {
            auto nsize =
                blockAttr[blk].CompSize() * components.size() * num_homo_modes;
            auto mr = MemoryRegion<TData>(name + std::to_string(blk), nsize,
                                          memAllocType, alignment);
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
     * @param memAllocType   - [ eHostPageable,  eHostPinned, eDeviceMemoryPool,
     * eHostPinnedDeviceMemoryPool]
     * @param alignment      - Memory alignment to use.
     */
    Field(const std::vector<BlockAttributes<TState>> blockAttr,
          const std::vector<std::string> components,
          const unsigned int num_homo_modes,
          const MemAllocType &memAllocType = eHostPageable,
          const size_t alignment = NektarSpaces::host_memory_alignment)
        : Field<TData, TState>("", blockAttr, components, num_homo_modes,
                               memAllocType, alignment)
    {
    }

    /**
     * @brief Construct a new Field object by moving storage from an
     * existing Field object.
     *
     * @param rhs
     */
    Field(Field &&rhs)
        : m_instantiated(std::move(rhs.m_instantiated)),
          m_name(std::move(rhs.m_name)),
          m_component_names(std::move(rhs.m_component_names)),
          m_num_homo_modes(std::move(rhs.m_num_homo_modes)),
          m_host(std::move(rhs.m_host)), m_device(std::move(rhs.m_device)),
          m_block_accessors(std::move(rhs.m_block_accessors)),
          m_memAllocType(std::move(rhs.m_memAllocType)),
          m_alignment(std::move(rhs.m_alignment))
    {
        for (auto &blocks : m_block_accessors)
        {
            blocks.m_field = this;
        }
        rhs.m_instantiated = false;
        rhs.m_name         = "";
        rhs.m_component_names.clear();
        rhs.m_num_homo_modes = 1;
        rhs.m_host           = nullptr;
        rhs.m_device         = nullptr;
        rhs.m_block_accessors.clear();
        rhs.m_memAllocType = eHostPageable;
        rhs.m_alignment    = NektarSpaces::host_memory_alignment;
    }

    /**
     * @brief Constructor methods - No copy methods
     *
     */
    Field(const Field &)            = delete;
    Field &operator=(const Field &) = delete;

    /**
     * @brief Move assignment operator.
     *
     * @param rhs
     *
     * @return Field&
     */
    Field &operator=(Field &&rhs)
    {
        m_instantiated    = std::move(rhs.m_instantiated);
        m_name            = std::move(rhs.m_name);
        m_component_names = std::move(rhs.m_component_names);
        m_num_homo_modes  = std::move(rhs.m_num_homo_modes);
        m_host            = std::move(rhs.m_host);
        m_device          = std::move(rhs.m_device);
        m_memAllocType    = std::move(rhs.m_memAllocType);
        m_block_accessors = std::move(rhs.m_block_accessors);
        m_alignment       = std::move(rhs.m_alignment);
        for (auto &blocks : m_block_accessors)
        {
            blocks.m_field = this;
        }

        rhs.m_instantiated = false;
        rhs.m_name         = "";
        rhs.m_component_names.clear();
        rhs.m_num_homo_modes = 1;
        rhs.m_host           = nullptr;
        rhs.m_device         = nullptr;
        rhs.m_block_accessors.clear();
        rhs.m_memAllocType = eHostPageable;
        rhs.m_alignment    = NektarSpaces::host_memory_alignment;
        return *this;
    }

    void SetInterleaveWidth(const unsigned int &interleaveWidth)
    {
        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            m_block_accessors[blk].template SetInterleaveWidth<TData>(
                interleaveWidth);
        }
    }

    void SetInterleaveWidth(Field &field)
    {
        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            m_block_accessors[blk].template SetInterleaveWidth<TData>(
                field.m_block_accessors[blk].GetInterleaveWidth());
        }
    }

    /**
     * @brief Templated initialize method.
     *
     */
    template <typename MemSpace> void Initialize(const TData val)
    {
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::Initialize - Object has not been "
                     "instantiated by custom constructor.");
        }

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
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::ToVector - Object has not been "
                     "instantiated by custom constructor.");
        }

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
            const auto nSize  = this->GetBlocks()[blk].CompSize();
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
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::ToArray - Object has not been "
                     "instantiated by custom constructor.");
        }

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
            const auto nSize  = this->GetBlocks()[blk].CompSize();
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
     *        Field
     *
     * @param field - Field to copy from
     *
     */
    template <typename MemSpace> void Copy(Field &field)
    {
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::Copy - Object has not been "
                     "instantiated by custom constructor.");
        }

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

        for (unsigned int blk = 0; blk < m_block_accessors.size(); ++blk)
        {
            m_block_accessors[blk].m_memory_region.template Copy<MemSpace>(
                field.m_block_accessors[blk].m_memory_region);
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
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::CopyVector - Object has not been "
                     "instantiated by custom constructor.");
        }

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

            msg << "Field::CopyVector - "
                << "Memory size mismatch between (std::vector) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        // Copy the data to the array.
        auto src = array.data();
        for (unsigned int blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto offset       = 0;
            const auto nSize  = this->GetBlocks()[blk].CompSize();
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
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::CopyArray - Object has not been "
                     "instantiated by custom constructor.");
        }

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

            msg << "Field::CopyArray - "
                << "Memory size mismatch between (Nektar::array) and ("
                << this->GetName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        // Copy the data to the array.
        auto src = array.data();
        for (unsigned int blk = 0; blk < this->GetBlocks().size(); ++blk)
        {
            auto offset       = 0;
            const auto nSize  = this->GetBlocks()[blk].CompSize();
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
    std::vector<BlockAccessor<TData, TState>> &GetBlocks()
    {
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::GetBlocks - Object has not been "
                     "instantiated by custom constructor.");
        }

        return m_block_accessors;
    }

    /**
     * @brief Gets the size of the field.
     *
     * @return size_t
     */
    size_t size() const
    {
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::size - Object has not been "
                     "instantiated by custom constructor.");
        }

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
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::GetName - Object has not been "
                     "instantiated by custom constructor.");
        }

        return m_name;
    }

    /**
     * @brief Gets the number of components for a vector field.
     *
     * @return unsigned int
     */
    unsigned int GetNumComponents() const
    {
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::GetNumComponents - Object has not been "
                     "instantiated by custom constructor.");
        }

        return m_component_names.size();
    }

    /**
     * @brief Gets the number of homogeneous modes.
     *
     * @return unsigned int
     */
    unsigned int GetNumHomoModes() const
    {
        if (!m_instantiated)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::GetNumHomoModes - Object has not been "
                     "instantiated by custom constructor.");
        }

        return m_num_homo_modes;
    }

    typedef TData value_type;

protected:
    /**
     * @brief Allocate contiguous host OR device memory accross all MemoryRegion
     * objects belonging to a Field object pointer.
     *
     * @param field  - Field object pointer.
     */
    template <typename MemSpace>
    static void AllocateFieldStorage(Field<TData, TState> *field)
    {
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            size_t alignment_offset = 0;

            // Allocate contiguous host memory accross all MemoryRegions of the
            // field object.
            if (!field->m_host)
            {
                if (field->m_memAllocType == eHostPageable)
                {
                    hostMalloc(&field->m_host, field->size() * sizeof(TData),
                               field->m_alignment);
                }
                else if (field->m_memAllocType == eHostPinned)
                {
                    // Add extra bytes for alignment provision
                    size_t aligned_bytes_size =
                        field->size() * sizeof(TData) + field->m_alignment;
                    hostMallocPinned(&field->m_host, aligned_bytes_size);
                    // Compute offset in bytes for non-aligned memory.
                    if ((size_t)field->m_host % field->m_alignment)
                    {
                        alignment_offset =
                            field->m_alignment -
                            (size_t)field->m_host % field->m_alignment;
                    }
                }
                // Get aligned memory pointer.
                auto src = (TData *)((size_t)field->m_host + alignment_offset);
                std::memset((void *)src, 0, field->size() * sizeof(TData));

                for (unsigned int blk = 0;
                     blk < field->m_block_accessors.size(); ++blk)
                {
                    field->m_block_accessors[blk]
                        .m_memory_region.SetHostStorage(src);
                    src += field->m_block_accessors[blk].m_memory_region.size();
                }
            }
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            // Allocate contiguous device memory accross all MemoryRegions of
            // the field object.
            if (!field->m_device)
            {
                deviceMalloc(&field->m_device, field->size() * sizeof(TData),
                             field->m_memAllocType);
                deviceMemset(field->m_device, 0, field->size() * sizeof(TData));

                auto src = field->m_device;
                for (unsigned int blk = 0;
                     blk < field->m_block_accessors.size(); ++blk)
                {
                    field->m_block_accessors[blk]
                        .m_memory_region.SetDeviceStorage(src);
                    src += field->m_block_accessors[blk].m_memory_region.size();
                }
            }
        }
    }

    // Member variables:
    bool m_instantiated = false; ///< Flag indicating if the current object has
                                 ///< been instantiated.
    std::string m_name;
    std::vector<std::string> m_component_names;
    unsigned int m_num_homo_modes = 1;
    TData *m_host                 = nullptr;
    TData *m_device               = nullptr;
    std::vector<BlockAccessor<TData, TState>> m_block_accessors;
    MemAllocType m_memAllocType;
    size_t m_alignment = NektarSpaces::host_memory_alignment;
};

} // namespace Nektar::Operators
