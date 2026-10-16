///////////////////////////////////////////////////////////////////////////////
//
// File: LocalToGlobalDataWarehouse.hpp
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
// Description: Data warehouse keys and creator for the local-to-global
// assembly tables.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp"

#include <cstdint>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>

namespace Nektar::MultiRegions
{

class ExpList;

/// Shared pointer to an ExpList object.
typedef std::shared_ptr<ExpList> ExpListSharedPtr;

class AssemblyMapCG;

typedef std::shared_ptr<AssemblyMapCG> AssemblyMapCGSharedPtr;

class LocalToGlobalDataCreator;

/**
 * @brief Base of the LocalToGlobalDataCreator keys: holds the parameters the
 * tables depend on and hashes them with the key's tag and padding type.
 *
 * All keys of the creator share one map in the data warehouse, so each
 * derived key passes a unique tag.
 */
template <typename TPadding, typename TValue>
class LocalToGlobalKeyBase : public LibUtilities::BaseKey
{
    // The TPadding type is used to determine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "LocalToGlobalKeyBase: Data type must be float or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef TValue value_type;

    ~LocalToGlobalKeyBase() override = default;

protected:
    LocalToGlobalKeyBase(
        const std::string &tag, const std::vector<std::string> &components,
        bool zeroDir = false, bool signChange = false, unsigned width = 1,
        const std::vector<unsigned> &fieldInterleaveWidths = {})
        : m_zeroDir(zeroDir), m_signChange(signChange), m_width(width),
          m_components(components),
          m_fieldInterleaveWidths(fieldInterleaveWidths)
    {
        hash_combine(m_hash, tag, std::type_index(typeid(TPadding)), m_zeroDir,
                     m_signChange, m_width);
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
        for (const auto &fieldInterleaveWidth : m_fieldInterleaveWidths)
        {
            hash_combine(m_hash, fieldInterleaveWidth);
        }
    }

private:
    bool m_zeroDir;
    bool m_signChange;
    unsigned m_width;
    std::vector<std::string> m_components;
    /// Interleave width of each block of the field the tables address.
    std::vector<unsigned> m_fieldInterleaveWidths;
};

/// Number of interior global DOFs to assemble.
template <typename TPadding>
class DeviceLocalToGlobalNumGidsKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceLocalToGlobalNumGidsKey(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width, const std::vector<unsigned> &fieldInterleaveWidths)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceLocalToGlobalNumGidsKey", components, zeroDir, false,
              width, fieldInterleaveWidths)
    {
    }
};

/// Number of local copies of each interior global DOF.
template <typename TPadding>
class DeviceLocalToGlobalNumAssembleKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceLocalToGlobalNumAssembleKey(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width, const std::vector<unsigned> &fieldInterleaveWidths)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceLocalToGlobalNumAssembleKey", components, zeroDir, false,
              width, fieldInterleaveWidths)
    {
    }
};

/// Local indices of the interior global DOFs, packed width-major.
template <typename TPadding>
class DeviceLocalToGlobalIndexKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceLocalToGlobalIndexKey(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width, const std::vector<unsigned> &fieldInterleaveWidths)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceLocalToGlobalIndexKey", components, zeroDir, false, width,
              fieldInterleaveWidths)
    {
    }
};

/// Offset of each interior global DOF into the index and sign tables.
template <typename TPadding>
class DeviceLocalToGlobalIndexOffsetKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceLocalToGlobalIndexOffsetKey(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width, const std::vector<unsigned> &fieldInterleaveWidths)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceLocalToGlobalIndexOffsetKey", components, zeroDir, false,
              width, fieldInterleaveWidths)
    {
    }
};

/// Signs of the interior local copies, packed as the index table.
template <typename TPadding>
class DeviceLocalToGlobalSignKey : public LocalToGlobalKeyBase<TPadding, int8_t>
{
public:
    DeviceLocalToGlobalSignKey(
        bool zeroDir, bool signChange,
        const std::vector<std::string> &components, unsigned width,
        const std::vector<unsigned> &fieldInterleaveWidths)
        : LocalToGlobalKeyBase<TPadding, int8_t>(
              "DeviceLocalToGlobalSignKey", components, zeroDir, signChange,
              width, fieldInterleaveWidths)
    {
    }
};

/// Number of parallel-boundary global DOFs to assemble.
template <typename TPadding>
class DeviceBndLocalToGlobalNumGidsKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceBndLocalToGlobalNumGidsKey(const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceBndLocalToGlobalNumGidsKey", components)
    {
    }
};

/// Number of local copies of each parallel-boundary global DOF.
template <typename TPadding>
class DeviceBndLocalToGlobalNumAssembleKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceBndLocalToGlobalNumAssembleKey(
        const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceBndLocalToGlobalNumAssembleKey", components)
    {
    }
};

/// Number of communication buffer slots of each parallel-boundary global DOF.
template <typename TPadding>
class DeviceBndLocalToGlobalNumBndValsKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceBndLocalToGlobalNumBndValsKey(
        const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceBndLocalToGlobalNumBndValsKey", components)
    {
    }
};

/// Local indices and buffer slots of the parallel-boundary global DOFs.
template <typename TPadding>
class DeviceBndLocalToGlobalIndexKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceBndLocalToGlobalIndexKey(
        const std::vector<std::string> &components,
        const std::vector<unsigned> &fieldInterleaveWidths)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceBndLocalToGlobalIndexKey", components, false, false, 1,
              fieldInterleaveWidths)
    {
    }
};

/// Offset of each parallel-boundary global DOF into the index and sign tables.
template <typename TPadding>
class DeviceBndLocalToGlobalOffsetKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceBndLocalToGlobalOffsetKey(const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceBndLocalToGlobalOffsetKey", components)
    {
    }
};

/// Position of this rank among the ranks sharing each parallel-boundary
/// global DOF, which fixes the order of assembly.
template <typename TPadding>
class DeviceBndLocalToGlobalAssembleOrderKey
    : public LocalToGlobalKeyBase<TPadding, unsigned>
{
public:
    DeviceBndLocalToGlobalAssembleOrderKey(
        const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, unsigned>(
              "DeviceBndLocalToGlobalAssembleOrderKey", components)
    {
    }
};

/// Signs of the parallel-boundary local copies, packed as the index table
/// with zeros in the buffer slots.
template <typename TPadding>
class DeviceBndLocalToGlobalSignKey
    : public LocalToGlobalKeyBase<TPadding, int8_t>
{
public:
    DeviceBndLocalToGlobalSignKey(bool zeroDir, bool signChange,
                                  const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, int8_t>(
              "DeviceBndLocalToGlobalSignKey", components, zeroDir, signChange)
    {
    }
};

/// Mask selecting one local copy of each global DOF this rank owns.
template <typename TPadding>
class LocalToGlobalMaskKey : public LocalToGlobalKeyBase<TPadding, std::uint8_t>
{
public:
    LocalToGlobalMaskKey(const std::vector<std::string> &components)
        : LocalToGlobalKeyBase<TPadding, std::uint8_t>("LocalToGlobalMaskKey",
                                                       components)
    {
    }
};

class LocalToGlobalDataCreator : public LibUtilities::DataCreatorClass
{
public:
    ~LocalToGlobalDataCreator() override = default;
    LocalToGlobalDataCreator(const ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList) {};

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceLocalToGlobalNumGidsKey<TPadding>::value_type>
    Create(const DeviceLocalToGlobalNumGidsKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceLocalToGlobalNumAssembleKey<TPadding>::value_type>
    Create(const DeviceLocalToGlobalNumAssembleKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceLocalToGlobalIndexKey<TPadding>::value_type>
    Create(const DeviceLocalToGlobalIndexKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceLocalToGlobalIndexOffsetKey<TPadding>::value_type>
    Create(const DeviceLocalToGlobalIndexOffsetKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceLocalToGlobalSignKey<TPadding>::value_type>
    Create(const DeviceLocalToGlobalSignKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalNumGidsKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalNumGidsKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalNumAssembleKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalNumAssembleKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalNumBndValsKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalNumBndValsKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalIndexKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalIndexKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalOffsetKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalOffsetKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalAssembleOrderKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalAssembleOrderKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceBndLocalToGlobalSignKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalSignKey<TPadding> &LocToGloKey);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename LocalToGlobalMaskKey<TPadding>::value_type>
    Create(const LocalToGlobalMaskKey<TPadding> &LocToGloKey);

    inline static const std::string m_name = "LocalToGlobalCreator";

private:
    ExpListSharedPtr m_expansionList;
    std::unordered_map<size_t, std::vector<AssemblyMapCGSharedPtr>>
        m_assemblyMaps;
    /// Interior gather-scatter tables the interior keys are built from.
    std::unordered_map<size_t, std::vector<unsigned>> m_gsInfo;
    /// Parallel-boundary gather-scatter tables the boundary keys are built
    /// from.
    std::unordered_map<size_t, std::vector<unsigned>> m_gsBndInfo;

    std::vector<AssemblyMapCGSharedPtr> &GetAssemblyMap(
        const std::vector<std::string> &components);

    template <typename TPadding>
    const std::vector<unsigned> &GetGSInfo(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width, const std::vector<unsigned> &fieldInterleaveWidths);

    template <typename TPadding>
    const std::vector<unsigned> &GetGSBndInfo(
        const std::vector<std::string> &components);

    template <typename TPadding>
    void FillSignArray(std::vector<unsigned> &index,
                       const std::vector<AssemblyMapCGSharedPtr> &loc2glo,
                       bool zeroDir, bool signChange, int8_t *out);
};

} // namespace Nektar::MultiRegions
