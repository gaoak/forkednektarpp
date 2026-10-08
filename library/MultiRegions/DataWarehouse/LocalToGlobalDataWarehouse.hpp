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

#include "LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp"

namespace Nektar::MultiRegions
{

class ExpList;

/// Shared pointer to an ExpList object.
typedef std::shared_ptr<ExpList> ExpListSharedPtr;

class AssemblyMapCG;

typedef std::shared_ptr<AssemblyMapCG> AssemblyMapCGSharedPtr;

class LocalToGlobalDataCreator;

template <typename TPadding>
class DeviceLocalToGlobalKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalKey: Data type must be float or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalKey() override = default;

    DeviceLocalToGlobalKey(bool zeroDir,
                           const std::vector<std::string> &components,
                           unsigned width = 1)
        : m_zeroDir(zeroDir), m_width(width), m_components(components)
    {
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobalKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    bool m_zeroDir;
    unsigned m_width;
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceLocalToGlobalNumAssembleKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalNumAssembleKey: Data type must be float "
                  "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalNumAssembleKey() override = default;

    DeviceLocalToGlobalNumAssembleKey(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width)
        : m_zeroDir(zeroDir), m_width(width), m_components(components)
    {
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobaNumAssemblelKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    bool m_zeroDir;
    unsigned m_width;
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceLocalToGlobalIndexKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(
        std::is_floating_point_v<TPadding>,
        "DeviceLocalToGlobalIndexKey: Data type must be float or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalIndexKey() override = default;

    DeviceLocalToGlobalIndexKey(bool zeroDir,
                                const std::vector<std::string> &components,
                                unsigned width)
        : m_zeroDir(zeroDir), m_width(width), m_components(components)
    {
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobaIndexlKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    bool m_zeroDir;
    unsigned m_width;
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceLocalToGlobalIndexOffsetKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalIndexOffsetKey: Data type must be float "
                  "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceLocalToGlobalIndexOffsetKey() override = default;

    DeviceLocalToGlobalIndexOffsetKey(
        bool zeroDir, const std::vector<std::string> &components,
        unsigned width)
        : m_zeroDir(zeroDir), m_width(width), m_components(components)
    {
        hash_combine(m_hash, m_zeroDir, m_width, typeid(value_type).name(),
                     "DeviceLocalToGlobaIndexOffsetlKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    bool m_zeroDir;
    unsigned m_width;
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceLocalToGlobalSignKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(
        std::is_floating_point_v<TPadding>,
        "DeviceLocalToGlobalSignKey: Data type must be float or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    // typedef int8_t value_type;
    typedef int value_type;

    ~DeviceLocalToGlobalSignKey() override = default;

    DeviceLocalToGlobalSignKey(bool zeroDir, bool signChange,
                               const std::vector<std::string> &components,
                               unsigned width = 1)
        : m_zeroDir(zeroDir), m_signChange(signChange), m_width(width),
          m_components(components)
    {
        hash_combine(m_hash, m_zeroDir, m_signChange, m_width,
                     typeid(value_type).name(), "DeviceLocalToGlobalSignKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    bool m_zeroDir;
    bool m_signChange;
    unsigned m_width;
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(
        std::is_floating_point_v<TPadding>,
        "DeviceBndLocalToGlobalKey: Data type must be float or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalKey() override = default;

    DeviceBndLocalToGlobalKey(const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(),
                     "DeviceBndLocalToGlobalKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalNumAssembleKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalNumAssembleKey: Data type must be float "
                  "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalNumAssembleKey() override = default;

    DeviceBndLocalToGlobalNumAssembleKey(
        const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(),
                     "DeviceBndLocalToGlobaNumAssemblelKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalNumBndValsKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalNumBndValsKey: Data type must be float "
                  "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalNumBndValsKey() override = default;

    DeviceBndLocalToGlobalNumBndValsKey(
        const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(),
                     "DeviceBndLocalToGlobaNumBndValslKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalIndexKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalIndexKey: Data type must be float "
                  "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalIndexKey() override = default;

    DeviceBndLocalToGlobalIndexKey(const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(),
                     "DeviceBndLocalToGlobaIndexlKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalOffsetKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "DeviceLocalToGlobalOffsetKey: Data type must be float "
                  "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalOffsetKey() override = default;

    DeviceBndLocalToGlobalOffsetKey(const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(),
                     "DeviceBndLocalToGlobaOffsetlKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalAssembleOrderKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(
        std::is_floating_point_v<TPadding>,
        "DeviceLocalToGlobalAssembleOrderKey: Data type must be float "
        "or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~DeviceBndLocalToGlobalAssembleOrderKey() override = default;

    DeviceBndLocalToGlobalAssembleOrderKey(
        const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(),
                     "DeviceBndLocalToGlobaAssembleOrderlKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

template <typename TPadding>
class DeviceBndLocalToGlobalSignKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(
        std::is_floating_point_v<TPadding>,
        "DeviceBndLocalToGlobalSignKey: Data type must be float or double.");

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef int value_type;

    ~DeviceBndLocalToGlobalSignKey() override = default;

    DeviceBndLocalToGlobalSignKey(bool zeroDir, bool signChange,
                                  const std::vector<std::string> &components)
        : m_zeroDir(zeroDir), m_signChange(signChange), m_components(components)
    {
        hash_combine(m_hash, m_zeroDir, m_signChange, typeid(value_type).name(),
                     "DeviceLocalToGlobalSignKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    bool m_zeroDir;
    bool m_signChange;
    std::vector<std::string> m_components;
};

template <typename TPadding>
class LocalToGlobalMaskKey : public LibUtilities::BaseKey
{
    // The TPadding type is use to dertermine the padding requirement and must
    // be of floating point type.
    static_assert(std::is_floating_point_v<TPadding>,
                  "LocalToGlobalMaskKey: Data type must be float or double.");

    friend class LocalToGlobalDatareator;

    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef std::uint8_t value_type;

    ~LocalToGlobalMaskKey() override = default;

    LocalToGlobalMaskKey(const std::vector<std::string> &components)
        : m_components(components)
    {
        hash_combine(m_hash, typeid(value_type).name(), "LocalToGlobalMaskKey");
        for (const auto &component : m_components)
        {
            hash_combine(m_hash, component);
        }
    }

private:
    std::vector<std::string> m_components;
};

class LocalToGlobalDataCreator : public LibUtilities::DataCreatorClass
{
public:
    ~LocalToGlobalDataCreator() override = default;
    LocalToGlobalDataCreator(const ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList) {};

    std::vector<AssemblyMapCGSharedPtr> &GetAssemblyMap(
        const std::vector<std::string> &components);

    template <typename MemSpace, typename TPadding>
    LibUtilities::MemoryRegion<
        typename DeviceLocalToGlobalKey<TPadding>::value_type>
    Create(const DeviceLocalToGlobalKey<TPadding> &LocToGloKey);

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
        typename DeviceBndLocalToGlobalKey<TPadding>::value_type>
    Create(const DeviceBndLocalToGlobalKey<TPadding> &LocToGloKey);

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
    Create([[maybe_unused]] const LocalToGlobalMaskKey<TPadding> &LocToGloKey);

    template <typename TPadding>
    void FillSignArray(std::vector<unsigned> &index,
                       const std::vector<AssemblyMapCGSharedPtr> &loc2glo,
                       bool zeroDir, bool signChange, int *out);

    inline static const std::string m_name = "LocalToGlobalCreator";

private:
    ExpListSharedPtr m_expansionList;
    std::unordered_map<size_t, std::vector<AssemblyMapCGSharedPtr>>
        m_assemblyMaps;
};

} // namespace Nektar::MultiRegions
