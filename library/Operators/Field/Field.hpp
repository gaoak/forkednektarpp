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

/**
 * @brief Possible states for Field data.
 *
 * These identify the mathematical representation of the field data. The two
 * main states are *Phys*, representing the field at the quadrature points,
 * and *Coeff*, representing the field in terms of its spectral/hp element
 * basis coefficients.
 */
enum class FieldState
{
    Phys,
    Coeff
};

std::string FieldStateString(FieldState);

static constexpr FieldState DefaultState = FieldState::Phys;

using default_fp_type = double;

/**
 * @brief A block means a group of elements of identical shape,
 * basis and order. BlockAttributes stores the most basic
 * information of a block.
 */
struct BlockAttributes
{
    BlockAttributes(const size_t num_elements,
                    const size_t num_elements_with_padding,
                    const size_t num_data, const size_t interleave_width)
        : m_num_elements(num_elements),
          num_elements_with_padding(num_elements_with_padding),
          m_num_data(num_data), m_size(num_elements_with_padding * num_data),
          m_interleave_width(interleave_width)
    {
    }

    template <typename TData = default_fp_type>
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
template <typename TData = default_fp_type, FieldState TState = DefaultState>
class Field : public MemoryRegion<TData>
{
public:
    Field(){};
    Field(const Field &) = delete;
    ~Field() override    = default; // Default removes implicit moves

    /**
     * @brief Construct a new Field object by moving storage from an existing
     * Field object.
     *
     * @param rhs
     */
    Field(Field &&rhs)
        : MemoryRegion<TData>::MemoryRegion(std::move(rhs)),
          m_name(std::move(rhs.m_name)),
          m_block_attributes(std::move(rhs.m_block_attributes)),
          m_var_names(std::move(rhs.m_var_names))
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
        this->m_storage    = std::move(rhs.m_storage);
        m_name             = std::move(rhs.m_name);
        m_block_attributes = std::move(rhs.m_block_attributes);
        m_var_names        = std::move(rhs.m_var_names);

        return *this;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the names of the components.
     *
     * @tparam MemSpace   - Type of memory space to use
     *
     * @param name        - Name of the field (memory region)
     * @param blocks      - Field data specification.
     * @param components  - Names of components for a vector field.
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> Create(
        const std::string name, const std::vector<BlockAttributes> blocks,
        const std::vector<std::string> components, const size_t alignment,
        const bool device_only = false)
    {
        int num_components = components.size();
        auto field         = Field(name, blocks, components);

        size_t storage_size = std::accumulate(
            field.block_attributes.begin(), field.block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.size();
            });

        size_t size = storage_size * num_components;

        // Create new a MemoryRegion and polymorphically store as a
        // MemoryRegionHost.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            field.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment, false);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            field.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment, device_only);
        }
        else
        {
            std::string msg("Field::createN - invaid memory space (");
            msg += field.m_storage->GetName() +
                   "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        // Zero memory
        field.template initialize<MemSpace>(0);

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
        const std::vector<BlockAttributes> blocks,
        const std::vector<std::string> components, const size_t alignment,
        const bool device_only = false)
    {
        return Field<TData, TState>::template Create<MemSpace>(
            "", blocks, components, alignment, device_only);
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the number of components.
     *
     * @tparam MemSpace   - Type of memory space to use
     *
     * @param name        - Name of the field (memory region)
     * @param blocks      - Field data specification.
     * @param nvar        - Number of components for a vector field.
     * @param alignment   - Memory alignment to use.
     * @param device_only - flag to only allocated memory on device
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> Create(
        const std::string name, const std::vector<BlockAttributes> blocks,
        const int nvar, const size_t alignment,
        [[maybe_unused]] const bool device_only = false)
    {
        auto field = Field(name, blocks, nvar);

        size_t storage_size = std::accumulate(
            field.m_block_attributes.begin(), field.m_block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.size();
            });

        size_t size = storage_size * nvar;

        // Create new a MemoryRegion and polymorphically store as a
        // MemoryRegionHost.
        if constexpr (std::is_same_v<MemSpace, NektarSpaces::HostSpace>)
        {
            field.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment, false);
        }
        else if constexpr (std::is_same_v<MemSpace, NektarSpaces::DeviceSpace>)
        {
            field.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment, device_only);
        }
        else
        {
            std::string msg("Field::create - invaid memory space (");
            msg += field.m_storage->GetName() +
                   "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        // Zero memory
        field.template Initialize<MemSpace>(0);

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
        const std::vector<BlockAttributes> blocks, const int nvar,
        const size_t alignment, const bool device_only = false)
    {
        return Field<TData, TState>::template Create<MemSpace>(
            "", blocks, nvar, alignment, device_only);
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
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            compSize += m_block_attributes[blk].GetNumElements() *
                        m_block_attributes[blk].GetNumData();
        }

        std::vector<TDataOut, Alloc> array(compSize * this->GetNumComponents());

        // Copy the data from the input field.
        auto dst = array.data();
        auto src = this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            auto nSize  = m_block_attributes[blk].size();
            auto nElmts = m_block_attributes[blk].GetNumElements();
            auto nPts   = m_block_attributes[blk].GetNumData();
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
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            compSize += m_block_attributes[blk].GetNumElements() *
                        m_block_attributes[blk].GetNumData();
        }

        Nektar::Array<Nektar::OneD, TDataOut> array(compSize *
                                                    this->GetNumComponents());

        // Copy the data from the input field.
        auto dst = array.data();
        auto src = this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            auto nSize  = m_block_attributes[blk].size();
            auto nElmts = m_block_attributes[blk].GetNumElements();
            auto nPts   = m_block_attributes[blk].GetNumData();
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
    template <typename MemSpace, typename MemCopy = DeviceToDevice>
    void Copy(Field &field)
    {
        if (m_block_attributes.size() != field.m_block_attributes.size())
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

        this->MemoryRegion<TData>::template Copy<MemSpace, MemCopy>(field);
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        std::vector
     *
     * @param array - std::vector to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice,
              class Alloc      = std::allocator<TDataIn>>
    void CopyVector(const std::vector<TDataIn, Alloc> &array)
    {
        size_t nSize = 0;
        for (const auto &block : this->GetBlocks())
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

        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::CopyVector - Can only copy std::vector "
                     "from HostToHost or from "
                     "HostToDevice.");
        }
        else
        {
            this->template CopySRC<MemSpace, MemCopy>(array.data());
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Nektar::Array<Nektar::OneD, TDataIn>
     *
     * @param array - Nektar::Array to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice>
    void CopyArray(const Nektar::Array<Nektar::OneD, TDataIn> &array)
    {
        size_t nSize = 0;
        for (const auto &block : this->GetBlocks())
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

        if constexpr (std::is_same_v<MemCopy, DeviceToDevice> ||
                      std::is_same_v<MemCopy, DeviceToHost>)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Field::CopyArray - Can only copy Nektar::Array "
                     "from HostToHost or from "
                     "HostToDevice.");
        }
        else
        {
            this->template CopySRC<MemSpace, MemCopy>(array.data());
        }
    }

    /**
     * @brief Get BlockAttributes for the field.
     *
     * @return std::vector<BlockAttributes>
     */
    std::vector<BlockAttributes> &GetBlocks()
    {
        return m_block_attributes;
    }

    /**
     * @brief Gets the alignment of the field.
     *
     * @return size_t
     */
    size_t GetAlignment()
    {
        return this->m_storage->GetAlignment();
    }

    /**
     * @brief Gets the Field size of a single component.
     *
     * @return size_t
     */
    size_t GetFieldSize()
    {
        return this->m_storage->size() / GetNumComponents();
    }

    /**
     * @brief Gets the size of the field.
     *
     * @return size_t
     */
    size_t size() const
    {
        size_t ans = 0;
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            ans += m_block_attributes[blk].size();
        }
        return ans * this->GetNumComponents();
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
    Field(const std::string name, const std::vector<BlockAttributes> blocks,
          const int nvar)
        : m_name(name), m_block_attributes(blocks), m_var_names(nvar)
    {
    }

    Field(const std::string name, const std::vector<BlockAttributes> blocks,
          const std::vector<std::string> components)
        : m_name(name), m_block_attributes(blocks), m_var_names(components)
    {
    }

    /**
     * @brief Copy the data from a pointer
     *
     * @param const TDataIn*
     */
    template <typename MemSpace, typename MemCopy, typename TDataIn>
    void CopySRC(const TDataIn *src)
    {
        auto compSize = 0;
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            auto nElmts = m_block_attributes[blk].GetNumElements();
            auto nPts   = m_block_attributes[blk].GetNumData();
            compSize += nElmts * nPts;
        }

        auto offset = 0;
        for (size_t blk = 0; blk < m_block_attributes.size(); ++blk)
        {
            auto nSize  = m_block_attributes[blk].size();
            auto nElmts = m_block_attributes[blk].GetNumElements();
            auto nPts   = m_block_attributes[blk].GetNumData();
            for (auto n = 0; n < this->GetNumComponents(); n++)
            {
                this->MemoryRegion<TData>::template CopySRC<MemSpace, MemCopy>(
                    src + n * compSize, nElmts * nPts, offset);
                offset += nSize;
            }
            src += nElmts * nPts;
        }
    }

    // Member variables:
    std::string m_name;
    std::vector<BlockAttributes> m_block_attributes;
    std::vector<std::string> m_var_names;
};
