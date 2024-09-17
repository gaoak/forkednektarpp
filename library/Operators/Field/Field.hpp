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

#include "MemoryRegion.hpp"
#include "Utils.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace Nektar::MultiRegions
{
class ExpList;
typedef std::shared_ptr<ExpList> ExpListSharedPtr;
} // namespace Nektar::MultiRegions

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

/**
 * @brief A block means a group of elements of identical shape,
 * basis and order. BlockAttributes stores the most basic
 * information of a block.
 */
struct BlockAttributes
{
    // default constructor: no padding
    BlockAttributes(size_t num_elements, size_t num_pts)
        : num_elements(num_elements), num_pts(num_pts),
          block_size(num_elements * num_pts), num_elmt_groups(0)
    {
    }

    size_t num_elements;
    size_t num_pts;
    size_t block_size;      // (num_elements + num_padding_elements) * num_pts
    size_t num_elmt_groups; // (num_elements + padding)/width
};

/**
 * @brief Get the BlockAttributes for a given field state from an ExpList.
 * This method basically captures identical elements that are contiguously
 * stored in the ExpList and group them into blocks. Padding elements will
 * also be set based on given vector width.
 *
 * @param state     Field state to query.
 * @param explist   Expansion list to query.
 * @param VectorWidth Vector width to use for the field.
 * @return std::vector<BlockAttributes>
 */
std::vector<BlockAttributes> GetBlockAttributes(
    FieldState state, const Nektar::MultiRegions::ExpListSharedPtr explist,
    size_t VectorWidth);

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
          block_attributes(std::move(rhs.block_attributes)),
          component_names(std::move(rhs.component_names)),
          m_curVecWidth(rhs.m_curVecWidth)
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
        this->m_storage  = std::move(rhs.m_storage);
        block_attributes = std::move(rhs.block_attributes);
        component_names  = std::move(rhs.component_names);
        m_curVecWidth    = std::move(rhs.m_curVecWidth);

        return *this;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the names of the components.
     *
     * @tparam MemSpace      Type of memory space to use
     *
     * @param name           Name of the field (memory region)
     * @param blocks         Field data specification.
     * @param components     Names of components for a vector field.
     * @param alignment      Memory alignment to use.
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> create(std::string name,
                                       std::vector<BlockAttributes> blocks,
                                       std::vector<std::string> components,
                                       size_t alignment)
    {
        int num_components = components.size();
        auto field         = Field(std::move(blocks), components);

        size_t storage_size = std::accumulate(
            field.block_attributes.begin(), field.block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.block_size;
            });

        size_t size = storage_size * num_components;

        // Create new a MemoryRegion and polymorphically store as a
        // MemoryRegionHost.
        if constexpr (std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            field.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment);
        }
        else if constexpr (std::is_same<MemSpace,
                                        NektarSpaces::DeviceSpace>::value)
        {
            field.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment);
        }
        else
        {
            std::string msg("Field::createN - invaid memory space (");
            msg += field.m_storage->GetName() +
                   "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        // Zero memory
        field.template initialize<DeviceOnly>(0);

        return field;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the names of the components.
     *
     * @tparam MemSpace      Type of memory space to use
     *
     * @param blocks         Field data specification.
     * @param components     Names of components for a vector field.
     * @param alignment      Memory alignment to use.
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> create(std::vector<BlockAttributes> blocks,
                                       std::vector<std::string> components,
                                       size_t alignment)
    {
        return Field<TData, TState>::template create<MemSpace>(
            blocks, components, alignment);
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the number of components.
     *
     * @tparam MemSpace      Type of memory space to use
     *
     * @param name           Name of the field (memory region)
     * @param blocks         Field data specification.
     * @param num_components Number of components for a vector field.
     * @param alignment      Memory alignment to use.
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> create(std::string name,
                                       std::vector<BlockAttributes> blocks,
                                       int num_components, size_t alignment)
    {
        auto field = Field(std::move(blocks), num_components);

        size_t storage_size = std::accumulate(
            field.block_attributes.begin(), field.block_attributes.end(), 0,
            [](size_t acc, const BlockAttributes &block) {
                return acc + block.block_size;
            });

        size_t size = storage_size * num_components;

        // Create new a MemoryRegion and polymorphically store as a
        // MemoryRegionHost.
        if constexpr (std::is_same<MemSpace, NektarSpaces::HostSpace>::value)
        {
            field.m_storage = std::make_unique<MemoryRegionHost<TData>>(
                name, size, alignment);
        }
        else if constexpr (std::is_same<MemSpace,
                                        NektarSpaces::DeviceSpace>::value)
        {
            field.m_storage = std::make_unique<MemoryRegionDevice<TData>>(
                name, size, alignment);
        }
        else
        {
            std::string msg("Field::create - invaid memory space (");
            msg += field.m_storage->GetName() +
                   "): " + Nektar::demangleTypeName(typeid(MemSpace));

            NEKERROR(Nektar::ErrorUtil::efatal, msg);
        }

        // Zero memory
        field.template initialize<DeviceOnly>(0);

        return field;
    }

    /**
     * @brief Static templated creation method. This method create
     * new Field by giving the number of components.
     *
     * @tparam MemSpace      Type of memory space to use
     *
     * @param blocks         Field data specification.
     * @param num_components Number of components for a vector field.
     * @param alignment      Memory alignment to use.
     *
     * @return Field<TData, TState>
     */
    template <typename MemSpace>
    static Field<TData, TState> create(std::vector<BlockAttributes> blocks,
                                       int num_components, size_t alignment)
    {
        return Field<TData, TState>::template create<MemSpace>(
            "", blocks, num_components, alignment);
    }

    /**
     * @brief Reshapes the storage to a prescribed vector width.
     *
     * This routine reorders the elemental data to interleave elements to a
     * prescribed vector width VW. This therefore puts the same DOF for groups
     * of VW elements contiguously in memory, enabling the efficient use of
     * vectorised instructions. At the moment this routine reshapes from VW0 to
     * VW1 by first reshaping from VW0 to a vector width of 1, and then to VW1.
     *
     * @tparam  VectorWidth     Target vector width.
     * @tparam  alignment       Memory alignment to use.
     */
    template <typename ExecSpace = NektarSpaces::Serial, size_t VectorWidth>
    void ReshapeStorage()
    {
        using MemSpace = typename ExecSpace::memory_space;

        // No reshape required, early return
        if (m_curVecWidth == VectorWidth)
        {
            return;
        }

        ReshapeToScalar<ExecSpace>();

        // Early return if "scalar" shape is required
        if (VectorWidth == 1)
        {
            return;
        }

        size_t scalar_field_size = GetFieldSize();

        for (int component = 0; component < GetNumComponents(); ++component)
        {
            auto *ptr = this->template GetPtr<MemSpace, ReadWrite>() +
                        component * scalar_field_size;

            for (const auto &block : block_attributes)
            {
                size_t num_padding_elements =
                    block.num_elmt_groups * VectorWidth - block.num_elements;

                const size_t numMetaBlocks =
                    (block.num_elements + num_padding_elements) / VectorWidth;
                const size_t MetaBlockSize = VectorWidth * block.num_pts;

                // Interleave on the device
                interleave<VectorWidth, ExecSpace>(numMetaBlocks, MetaBlockSize,
                                                   block.num_pts, ptr);

                ptr += block.block_size;
            }
        }

        m_curVecWidth = VectorWidth;
    }

    /**
     * @brief Reshapes the current storage interleaving to a non-interleaved
     * arrangement. For multi-component fields, all paddings will be placed
     * at the end of each component after this operation.
     */
    template <typename ExecSpace> void ReshapeToScalar()
    {
        using MemSpace = typename ExecSpace::memory_space;

        if (m_curVecWidth == 1)
        {
            return;
        }

        size_t scalar_field_size = GetFieldSize();

        for (int component = 0; component < GetNumComponents(); ++component)
        {
            auto *ptr = this->template GetPtr<MemSpace, ReadWrite>() +
                        component * scalar_field_size;

            for (const auto &block : block_attributes)
            {
                size_t num_padding_elements =
                    block.num_elmt_groups * m_curVecWidth - block.num_elements;
                const size_t numMetaBlocks =
                    (block.num_elements + num_padding_elements) / m_curVecWidth;
                const size_t MetaBlockSize = m_curVecWidth * block.num_pts;

                deInterleave<ExecSpace>(m_curVecWidth, numMetaBlocks,
                                        MetaBlockSize, block.num_pts, ptr);

                ptr += block.block_size;
            }
        }

        m_curVecWidth = 1;
    }

    /**
     * @brief Copy the data from a pointer
     *
     * @param const TDataIn*
     */
    template <typename MemSpace, typename TDataOut,
              typename MemCopy = HostToDevice>
    void copyTo(TDataOut *dst)
    {
        auto *src = this->template GetPtr<MemSpace, ReadOnly>();
        for (const auto &block : this->GetBlocks())
        {
            auto nSize  = block.block_size;
            auto nElmts = block.num_elements;
            auto nPts   = block.num_pts;

            std::copy(src, src + nElmts * nPts, dst);

            dst += nElmts * nPts;
            src += nSize;
        }
    }

    /**
     * @brief Copy the data to a std::vector
     *
     * @return std::vector
     */
    template <typename TDataOut = TData, class Alloc = std::allocator<TDataOut>>
    std::vector<TDataOut, Alloc> toVector(size_t size = 0)
    {
        if (size == 0)
        {
            for (const auto &block : this->GetBlocks())
            {
                size += block.num_elements * block.num_pts;
            }
        }

        std::vector<TDataOut, Alloc> array(size);

        // Copy the data from the input field
        this->template copyTo<NektarSpaces::HostSpace, TDataOut>(array.data());

        return array;
    }

    /**
     * @brief Copy the data to a Nektar::Array
     *
     * @return Array<Nektar::OneD, TDataOut>
     */
    template <typename TDataOut = TData>
    Nektar::Array<Nektar::OneD, TDataOut> toArray(size_t size = 0)
    {
        if (size == 0)
        {
            for (const auto &block : this->GetBlocks())
            {
                size += block.num_elements * block.num_pts;
            }
        }

        Nektar::Array<Nektar::OneD, TDataOut> array(size);

        // Copy the data from the input field
        this->template copyTo<NektarSpaces::HostSpace, TDataOut>(array.data());

        return array;
    }

    /**
     * @brief Copy the data from a pointer
     *
     * @param const TDataIn*
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice>
    void copyFrom(const TDataIn *src)
    {
        size_t offset = 0;
        for (const auto &block : this->GetBlocks())
        {
            auto nSize  = block.block_size;
            auto nElmts = block.num_elements;
            auto nPts   = block.num_pts;

            this->MemoryRegion<TData>::template copyFrom<MemSpace, TDataIn,
                                                         MemCopy>(
                src, nElmts * nPts, offset);

            src += nElmts * nPts;
            offset += nSize;
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        Field
     *
     * @param region - Field to copy from
     *
     */
    template <typename MemSpace, typename MemCopy = HostToDevice>
    void copyField(Field &field)
    {
        ASSERTL0(field.block_attributes.size() == block_attributes.size(),
                 "Number of blocks are not the same!");
        ASSERTL0(field.m_curVecWidth == m_curVecWidth,
                 "Vector width are not the same!");

        if (this->size() != field.size())
        {
            std::stringstream msg;

            msg << "Field::copyField - "
                << "Memory size mismatch between (" << field.getName()
                << ") and (" << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }

        if constexpr (std::is_same<MemCopy, DeviceToDevice>::value ||
                      std::is_same<MemCopy, DeviceToHost>::value)
        {
            this->MemoryRegion<TData>::template copyFrom<MemSpace, TData,
                                                         MemCopy>(
                field.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>(),
                field.size());
        }
        else if constexpr (std::is_same<MemCopy, HostToDevice>::value ||
                           std::is_same<MemCopy, HostToHost>::value)
        {
            this->MemoryRegion<TData>::template copyFrom<MemSpace, TData,
                                                         MemCopy>(
                field.template GetPtr<NektarSpaces::HostSpace, ReadOnly>(),
                field.size());
        }
    }

    /**
     * @brief Templated copy method. This method copies data from a
     *        MemoryRegion
     *
     * @param region - MemoryRegion to copy from
     *
     */
    template <typename MemSpace, typename TDataIn,
              typename MemCopy = HostToDevice>
    void copyMemoryRegion(MemoryRegion<TDataIn> &region)
    {
        /*if (this->size() != region.size())
        {
            std::stringstream msg;

            msg << "Field::copyMemoryRegion - "
                << "Memory size mismatch between (" << region.getName()
                << ") and (" << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }*/

        if constexpr (std::is_same<MemCopy, DeviceToDevice>::value ||
                      std::is_same<MemCopy, DeviceToHost>::value)
        {
            this->template copyFrom<MemSpace, TDataIn, MemCopy>(
                region.template GetPtr<NektarSpaces::DeviceSpace, ReadOnly>());
        }

        if constexpr (std::is_same<MemCopy, HostToDevice>::value ||
                      std::is_same<MemCopy, HostToHost>::value)
        {
            this->template copyFrom<MemSpace, TDataIn, MemCopy>(
                region.template GetPtr<NektarSpaces::HostSpace, ReadOnly>());
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
              typename MemCopy = HostToDevice,
              class Alloc      = std::allocator<TDataIn>>
    void copyVector(std::vector<TDataIn, Alloc> const &array)
    {
        if constexpr (std::is_same<MemCopy, DeviceToDevice>::value ||
                      std::is_same<MemCopy, DeviceToHost>::value)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::copyVector - Can only copy std::vector "
                     "from HostToHost or from "
                     "HostToDevice.");
        }

        /*if (this->size() != array.size())
        {
            std::stringstream msg;

            msg << "Field::copyVector - "
                << "Memory size mismatch between (std::vector) and ("
                << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }*/

        this->template copyFrom<MemSpace, TDataIn, MemCopy>(array.data());
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
    void copyArray(Nektar::Array<Nektar::OneD, TDataIn> const &array)
    {
        if constexpr (std::is_same<MemCopy, DeviceToDevice>::value ||
                      std::is_same<MemCopy, DeviceToHost>::value)
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "MemoryRegion::copyArray - Can only copy Nektar::Array "
                     "from HostToHost or from "
                     "HostToDevice.");
        }

        /*if (this->size() != array.size())
        {
            std::stringstream msg;

            msg << "Field::copyArray - "
                << "Memory size mismatch between (Nektar::array) and ("
                << this->getName() << ").";
            NEKERROR(Nektar::ErrorUtil::efatal, msg.str());
        }*/

        this->template copyFrom<MemSpace, TDataIn, MemCopy>(array.data());
    }

    /**
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape
     * and same components.
     *
     * @return bool
     */
    bool compare(Field<TData, TState> &rhs, TData tol)
    {
        if (rhs.GetNumComponents() != GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        const std::vector<BlockAttributes> &rhs_blocks = rhs.GetBlocks();

        if (rhs_blocks.size() != block_attributes.size())
        {
            std::cout << "Mismatch of block size." << std::endl;
            return false;
        }

        if (rhs.m_curVecWidth != m_curVecWidth)
        {
            std::cout << "Mismatch of vector width." << std::endl;
            return false;
        }

        bool isMatched = true;

        const TData *store =
            this->template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        const TData *rhs_store =
            rhs.template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

        for (size_t component = 0; component < GetNumComponents(); ++component)
        {
            for (size_t bl = 0; bl < block_attributes.size(); ++bl)
            {
                size_t num_pts        = block_attributes[bl].num_pts;
                size_t num_elements   = block_attributes[bl].num_elements;
                size_t num_metaBlocks = block_attributes[bl].num_elmt_groups;

                // Check that each block have the same structure
                if ((num_elements != rhs_blocks[bl].num_elements) ||
                    (num_pts != rhs_blocks[bl].num_pts))
                {
                    std::cout << "Mismatch of block structure." << std::endl;
                    return false;
                }

                int MisMatchcnt = 0, total = 0;

                for (size_t metaBlock = 0; metaBlock < num_metaBlocks;
                     ++metaBlock)
                {
                    for (size_t coeff = 0; coeff < num_pts; ++coeff)
                    {
                        for (size_t k = 0; k < m_curVecWidth; ++k)
                        {
                            // skip padding elements
                            if (metaBlock * m_curVecWidth + k + 1 <=
                                num_elements)
                            {
                                total++;
                                if (std::abs(*store - *rhs_store) > tol)
                                {
                                    if (MisMatchcnt == 0)
                                    {
                                        isMatched = false;
                                    }

                                    MisMatchcnt++;
                                }
                            }

                            store++;
                            rhs_store++;
                        }
                    }
                }

                if (!isMatched)
                {
                    std::cout << "Number of mismatches in block " << bl
                              << " is " << MisMatchcnt << " out of " << total
                              << std::endl;
                }
            }
        }
        if (isMatched)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
     * @brief Get BlockAttributes for the field.
     *
     * @return std::vector<BlockAttributes>
     */
    std::vector<BlockAttributes> const &GetBlocks() const
    {
        return block_attributes;
    }

    /**
     * @brief Gets the number of components for a vector field.
     *
     * @return size_t
     */
    size_t GetNumComponents()
    {
        return component_names.size();
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
     * @brief Gets the alignment of the field.
     *
     * @return size_t
     */
    size_t GetAlignment()
    {
        return this->m_storage->getAlignment();
    }

    /**
     * @brief Get current vector width.
     *
     * @return size_t
     */
    size_t GetVecWidth()
    {
        return m_curVecWidth;
    }

private:
    /**
     * @brief Construct a new Field object.
     *
     * @param blocks    Field data layout specification
     * @param components Names of components for vector field.
     */
    Field(std::vector<BlockAttributes> blocks, int num_components = 1)
        : block_attributes(std::move(blocks)), component_names(num_components)
    {
    }

    Field(std::vector<BlockAttributes> blocks,
          std::vector<std::string> components = {"u"})
        : block_attributes(std::move(blocks)),
          component_names(std::move(components))
    {
    }

    std::vector<BlockAttributes> block_attributes;
    std::vector<std::string> component_names = {"u"};

    size_t m_curVecWidth = 1;
};
