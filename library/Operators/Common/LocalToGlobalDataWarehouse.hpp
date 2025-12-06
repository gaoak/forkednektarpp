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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/Common/NekDataWarehouse.hpp"

// Forward declaration
namespace Nektar::MultiRegions
{

class AssemblyMapCG;

typedef std::shared_ptr<AssemblyMapCG> AssemblyMapCGSharedPtr;

} // namespace Nektar::MultiRegions

namespace Nektar::Operators
{

class LocalToGlobalDataCreator;

template <typename TData> class DeviceLocalToGlobalKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalKey() override = default;

    DeviceLocalToGlobalKey(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap,
        bool zeroDir, unsigned width = 1)
        : m_assemblyMap(assemblyMap), m_zeroDir(zeroDir), m_width(width)
    {
        for (auto &e : m_assemblyMap)
        {
            hash_combine(m_hash, e.get());
        }
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobalKey");
    }

private:
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;
    bool m_zeroDir;
    unsigned m_width;
};

template <typename TData>
class DeviceLocalToGlobalNumAssembleKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalNumAssembleKey() override = default;

    DeviceLocalToGlobalNumAssembleKey(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap,
        bool zeroDir, unsigned width)
        : m_assemblyMap(assemblyMap), m_zeroDir(zeroDir), m_width(width)
    {
        for (auto &e : m_assemblyMap)
        {
            hash_combine(m_hash, e.get());
        }
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobaNumAssemblelKey");
    }

private:
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;
    bool m_zeroDir;
    unsigned m_width;
};

template <typename TData> class DeviceLocalToGlobalIndexKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalIndexKey() override = default;

    DeviceLocalToGlobalIndexKey(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap,
        bool zeroDir, unsigned width)
        : m_assemblyMap(assemblyMap), m_zeroDir(zeroDir), m_width(width)
    {
        for (auto &e : m_assemblyMap)
        {
            hash_combine(m_hash, e.get());
        }
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobaIndexlKey");
    }

private:
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;
    bool m_zeroDir;
    unsigned m_width;
};

template <typename TData>
class DeviceLocalToGlobalIndexOffsetKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalIndexOffsetKey() override = default;

    DeviceLocalToGlobalIndexOffsetKey(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap,
        bool zeroDir, unsigned width)
        : m_assemblyMap(assemblyMap), m_zeroDir(zeroDir), m_width(width)
    {
        for (auto &e : m_assemblyMap)
        {
            hash_combine(m_hash, e.get());
        }
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobaIndexOffsetlKey");
    }

private:
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;
    bool m_zeroDir;
    unsigned m_width;
};

template <typename TData> class DeviceLocalToGlobalSignKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    // typedef int8_t value_type;
    typedef int value_type;

    ~DeviceLocalToGlobalSignKey() override = default;

    DeviceLocalToGlobalSignKey(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap,
        bool zeroDir, bool signChange, unsigned width = 1)
        : m_assemblyMap(assemblyMap), m_zeroDir(zeroDir),
          m_signChange(signChange), m_width(width)
    {
        for (auto &e : m_assemblyMap)
        {
            hash_combine(m_hash, e.get());
        }
        hash_combine(m_hash, m_zeroDir, m_signChange, m_width,
                     typeid(value_type).name(), "DeviceLocalToGlobalSignKey");
    }

private:
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;
    bool m_zeroDir;
    bool m_signChange;
    unsigned m_width;
};

template <typename TData> class DeviceBndLocalToGlobalSignKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef int value_type;

    ~DeviceBndLocalToGlobalSignKey() override = default;

    DeviceBndLocalToGlobalSignKey(
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> &assemblyMap,
        bool zeroDir, bool signChange)
        : m_assemblyMap(assemblyMap), m_zeroDir(zeroDir),
          m_signChange(signChange)
    {
        for (auto &e : m_assemblyMap)
        {
            hash_combine(m_hash, e.get());
        }
        hash_combine(m_hash, m_zeroDir, m_signChange, typeid(value_type).name(),
                     "DeviceLocalToGlobalSignKey");
    }

private:
    std::vector<MultiRegions::AssemblyMapCGSharedPtr> m_assemblyMap;
    bool m_zeroDir;
    bool m_signChange;
};

template <typename TData> class DeviceBndLocalToGlobalKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalKey() override = default;

    DeviceBndLocalToGlobalKey(unsigned numComp) : m_numComp(numComp)
    {
        hash_combine(m_hash, m_numComp, typeid(value_type).name(),
                     "DeviceBndLocalToGlobalKey");
    }

private:
    unsigned m_numComp;
};

template <typename TData> class LocalToGlobalMaskKey : public BaseKey
{
    friend class LocalToGlobalDatareator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef std::uint8_t value_type;

    ~LocalToGlobalMaskKey() override = default;

    LocalToGlobalMaskKey()
    {
        hash_combine(m_hash, typeid(value_type).name(), "LocalToGlobalMaskKey");
    }

private:
};

class LocalToGlobalDataCreator : public DataCreatorClass
{
public:
    ~LocalToGlobalDataCreator() override = default;
    LocalToGlobalDataCreator(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceLocalToGlobalKey<TData>::value_type> Create(
        const DeviceLocalToGlobalKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceLocalToGlobalNumAssembleKey<TData>::value_type>
    Create(const DeviceLocalToGlobalNumAssembleKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceLocalToGlobalIndexKey<TData>::value_type> Create(
        const DeviceLocalToGlobalIndexKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceLocalToGlobalIndexOffsetKey<TData>::value_type>
    Create(const DeviceLocalToGlobalIndexOffsetKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceLocalToGlobalSignKey<TData>::value_type> Create(
        const DeviceLocalToGlobalSignKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceBndLocalToGlobalKey<TData>::value_type> Create(
        const DeviceBndLocalToGlobalKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename DeviceBndLocalToGlobalSignKey<TData>::value_type> Create(
        const DeviceBndLocalToGlobalSignKey<TData> &LocToGloKey);

    template <typename MemSpace, typename TData>
    MemoryRegion<typename LocalToGlobalMaskKey<TData>::value_type> Create(
        [[maybe_unused]] const LocalToGlobalMaskKey<TData> &LocToGloKey);

    inline static const std::string m_name = "LocalToGlobalCreator";

private:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
