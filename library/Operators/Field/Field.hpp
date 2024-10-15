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

#include "Operators/LoopExecution/LoopExecution.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/ExpList.h>

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
          block_size(num_elements * num_pts), num_padding_elements(0),
          interleave_width(1)
    {
    }

    size_t num_elements;
    size_t num_pts;
    size_t block_size; // (num_elements + num_padding_elements) * num_pts
    size_t num_padding_elements;
    size_t interleave_width;

    size_t GetNumElmtGroups(void) const
    {
        ASSERTL0((num_elements + num_padding_elements) % interleave_width == 0,
                 "Number of elements is not divisible by interleave width.");
        return (num_elements + num_padding_elements) / interleave_width;
    }
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
    std::vector<BlockAttributes> blockAttr;

    // initialize the first block using the first element
    auto expPtr        = explist->GetExp(0);
    int prevIsDeformed = -1, thisIsDeformed = -1;
    std::vector<LibUtilities::BasisKey> prevbasisKeys(
        expPtr->GetNumBases(), LibUtilities::NullBasisKey);
    std::vector<LibUtilities::BasisKey> thisbasisKeys(
        expPtr->GetNumBases(), LibUtilities::NullBasisKey);
    for (int d = 0; d < expPtr->GetNumBases(); d++)
    {
        prevbasisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
    }
    prevIsDeformed = expPtr->GetMetricInfo()->GetGtype();
    size_t num_pts = state == FieldState::Phys ? expPtr->GetTotPoints()
                                               : expPtr->GetNcoeffs();
    blockAttr.push_back({1, num_pts});

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

        // if the basis is the same as the previous one,
        // increment the number of elements
        if (thisbasisKeys == prevbasisKeys && thisIsDeformed == prevIsDeformed)
        {
            blockAttr.back().num_elements++;
        }
        else // if not, create a new block with the number of elements = 1
        {
            size_t num_elements_with_padding =
                ((blockAttr.back().num_elements +
                  NektarSpaces::vector_width<TData>::value - 1) /
                 NektarSpaces::vector_width<TData>::value) *
                NektarSpaces::vector_width<TData>::value;
            blockAttr.back().num_padding_elements =
                num_elements_with_padding - blockAttr.back().num_elements;
            blockAttr.back().block_size =
                num_elements_with_padding * blockAttr.back().num_pts;
            blockAttr.back().interleave_width = interleave_width;

            // update num_pts for a new block
            num_pts = state == FieldState::Phys ? expPtr->GetTotPoints()
                                                : expPtr->GetNcoeffs();
            blockAttr.push_back({1, num_pts});
            prevbasisKeys  = thisbasisKeys;
            prevIsDeformed = thisIsDeformed;
        }
    }

    // update the padding elements for the last block
    size_t num_elements_with_padding =
        ((blockAttr.back().num_elements +
          NektarSpaces::vector_width<TData>::value - 1) /
         NektarSpaces::vector_width<TData>::value) *
        NektarSpaces::vector_width<TData>::value;
    blockAttr.back().num_padding_elements =
        num_elements_with_padding - blockAttr.back().num_elements;
    blockAttr.back().block_size =
        num_elements_with_padding * blockAttr.back().num_pts;
    blockAttr.back().interleave_width = interleave_width;

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
          block_attributes(std::move(rhs.block_attributes)),
          component_names(std::move(rhs.component_names))
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
                                       size_t alignment,
                                       bool device_only = false)
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
                name, size, alignment, false);
        }
        else if constexpr (std::is_same<MemSpace,
                                        NektarSpaces::DeviceSpace>::value)
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
                                       size_t alignment,
                                       bool device_only = false)
    {
        return Field<TData, TState>::template create<MemSpace>(
            blocks, components, alignment, device_only);
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
    static Field<TData, TState> create(
        std::string name, std::vector<BlockAttributes> blocks,
        int num_components, size_t alignment,
        [[maybe_unused]] bool device_only = false)
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
                name, size, alignment, false);
        }
        else if constexpr (std::is_same<MemSpace,
                                        NektarSpaces::DeviceSpace>::value)
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
                                       int num_components, size_t alignment,
                                       bool device_only = false)
    {
        return Field<TData, TState>::template create<MemSpace>(
            "", blocks, num_components, alignment, device_only);
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
     * @tparam  interleave_width     Target vector width.
     * @tparam  alignment            Memory alignment to use.
     */
    template <typename ExecSpace, size_t interleave_width> void ReshapeStorage()
    {
        using MemSpace = typename ExecSpace::memory_space;

        auto *ptr = this->template GetPtr<MemSpace, ReadWrite>();

        for (auto &block : block_attributes)
        {
            if (block.interleave_width != interleave_width)
            {
                // Reshape block to scalar shape, if necessary
                if (block.interleave_width != 1)
                {
                    for (int component = 0; component < GetNumComponents();
                         ++component)
                    {
                        deInterleave<ExecSpace>(
                            block.interleave_width, block.GetNumElmtGroups(),
                            block.num_pts, ptr + component * block.block_size);
                    }
                }

                // Set new interleave width
                block.interleave_width = interleave_width;

                // Reshape block to required shape, if necessary
                if (block.interleave_width != 1)
                {
                    for (int component = 0; component < GetNumComponents();
                         ++component)
                    {
                        interleave<interleave_width, ExecSpace>(
                            block.GetNumElmtGroups(), block.num_pts,
                            ptr + component * block.block_size);
                    }
                }
            }

            // Increment pointer and index for next block.
            ptr += block.block_size * GetNumComponents();
        }
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

        for (size_t bl = 0; bl < block_attributes.size(); ++bl)
        {
            ASSERTL0(field.block_attributes[bl].interleave_width ==
                         block_attributes[bl].interleave_width,
                     "Vector width are not the same!");
        }

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
    std::vector<std::string> component_names;
};

/// A generic function to reshuffle the map, based on the given interleave or
/// deinterleave map.
template <typename ExecSpace>
void ReshuffleMap(MemoryRegion<int> &deInterleaveMap, MemoryRegion<int> &map)
{
    // assume the map is always in the device memory space
    using MemSpace = typename ExecSpace::memory_space;

    // temporary storage for the map
    MemoryRegion<int> temp = MemoryRegion<int>::template create<MemSpace>(
        map.size(), ExecSpace::alignment, true);
    // copy map to temp
    temp.template copyMemoryRegion<MemSpace>(map);

    // ReMapping using the deinterleave map, temp is used as workspace
    auto *deInterleaveMapPtr =
        deInterleaveMap.template GetPtr<MemSpace, ReadWrite>();
    auto tempPtr = temp.template GetPtr<MemSpace, ReadOnly>();
    auto mapPtr  = map.template GetPtr<MemSpace, WriteOnly>();

    // use deinterleave map to reshuffle the temp
    Nektar::parallel_for<ExecSpace>(
        0, map.size(), NEKTAR_LAMBDA(unsigned int i) {
            mapPtr[i] = deInterleaveMapPtr[tempPtr[i]];
        });
}

/// A generic function to build the interleave map for a given field.
template <typename ExecSpace>
void BuildInterleaveMap(std::vector<BlockAttributes> &blocks,
                        const int new_interleave_width,
                        MemoryRegion<int> &deInterleaveMap,
                        MemoryRegion<int> &InterleaveMap)
{
    // assume the map is always in the device memory space
    using MemSpace = typename ExecSpace::memory_space;

    auto *deInterleaveMapPtr =
        deInterleaveMap.template GetPtr<MemSpace, WriteOnly>();
    auto *InterleaveMapPtr =
        InterleaveMap.template GetPtr<MemSpace, WriteOnly>();

    // Counting the subindex that has been processed so far
    size_t offset = 0;

    for (auto &block : blocks)
    {
        block.interleave_width   = new_interleave_width;
        auto const ncoeff        = block.num_pts;
        const size_t nElmtGroups = block.GetNumElmtGroups();

        // this function fills both InterleaveMap and deInterleaveMap;
        // deInterleaveMap is saved as a member for later use;
        BuildInterleaveMapKernel<ExecSpace>(
            nElmtGroups, ncoeff, new_interleave_width, offset,
            deInterleaveMapPtr, InterleaveMapPtr);

        deInterleaveMapPtr += ncoeff * new_interleave_width * nElmtGroups;
        offset += ncoeff * new_interleave_width * nElmtGroups;
    }
}
