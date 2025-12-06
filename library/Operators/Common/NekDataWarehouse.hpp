///////////////////////////////////////////////////////////////////////////////
//
// File: NekDataWarehouse.hpp
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
// Description: DataWarehouse pattern class for Nektar
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>

#ifdef NEKTAR_USE_THREAD_SAFETY
#include <mutex>
#include <shared_mutex>
#include <thread>
#endif

#include <LibUtilities/BasicUtils/HashUtils.hpp>

#include "Operators/Field/MemoryRegion.hpp"

// Forward declaration
namespace Nektar::MultiRegions
{

class ExpList;

/// Shared pointer to an ExpList object.
typedef std::shared_ptr<ExpList> ExpListSharedPtr;

} // namespace Nektar::MultiRegions

namespace Nektar::Operators
{

// Data creator base class
class DataCreatorClass
{
public:
    virtual ~DataCreatorClass() = default;
};

// Data key base class
class BaseKey
{
public:
    virtual ~BaseKey() = default;

    size_t GetHash(void) const
    {
        return m_hash;
    }

protected:
    size_t m_hash = 0;
};

#ifdef NEKTAR_USE_THREAD_SAFETY
// Generate parameter typenames with default type of 'none'
typedef std::unique_lock<std::shared_mutex> WriteLock;
typedef std::shared_lock<std::shared_mutex> ReadLock;
#endif

class NekDataWarehouse
{
public:
    typedef std::unordered_map<
        std::string,
        std::unordered_map<size_t, std::shared_ptr<MemoryRegionBase>>>
        tMapMemoryRegion;
    typedef std::shared_ptr<DataCreatorClass> tDataCreatorClassSharedPtr;
    typedef std::unordered_map<std::string, tDataCreatorClassSharedPtr>
        tMapDataCreatorClass;

public:
    NekDataWarehouse()  = default;
    ~NekDataWarehouse() = default;

    template <typename MemSpace, typename DataKey>
    const typename DataKey::value_type *GetData(const DataKey &dataKey)
    {
        using TData = typename DataKey::value_type;

#ifdef NEKTAR_USE_THREAD_SAFETY
        ReadLock vReadLock(m_mutex);
#endif

        auto idKey = dataKey.GetHash();

        // Check if creator class is registered.
        auto it = GetMapDataCreatorClass()->find(DataKey::creator::m_name);
        if (it != GetMapDataCreatorClass()->end())
        {
            // Register new data, if necessary.
            auto it2 = GetMapMemoryRegion()->find(DataKey::creator::m_name);
            if (it2->second.find(idKey) == it2->second.end())
            {
                // Fetch creator class.
                auto dataCreator =
                    std::static_pointer_cast<typename DataKey::creator>(
                        it->second);
                auto mr = dataCreator->template Create<MemSpace>(dataKey);
                it2->second.emplace(
                    idKey,
                    std::make_shared<MemoryRegion<TData>>(std::move(mr)));
            }
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "Unknown data creator class " + DataKey::creator::m_name);
        }

        // Fetch data.
        auto it2 = GetMapMemoryRegion()->find(DataKey::creator::m_name);
        auto mr  = std::static_pointer_cast<MemoryRegion<TData>>(
            it2->second.find(idKey)->second);
        return mr->template GetPtr<MemSpace, ReadOnly>();
    }

    // Register data creation class pointer.
    template <typename DataCreator, typename... cParam>
    void RegisterDataCreatorClass(cParam... param)
    {
#ifdef NEKTAR_USE_THREAD_SAFETY
        WriteLock vWriteLock(m_mutex);
#endif

        auto it = GetMapDataCreatorClass()->find(DataCreator::m_name);
        if (it == GetMapDataCreatorClass()->end())
        {
            // Register new DataCreator::m_name in MemoryRegion map.
            GetMapMemoryRegion()->emplace(
                DataCreator::m_name,
                std::unordered_map<size_t,
                                   std::shared_ptr<MemoryRegionBase>>());
        }
        else
        {
            // Clear old Creator class.
            GetMapDataCreatorClass()->erase(it);

            // Clear old MemoryRegion map.
            auto it2 = GetMapMemoryRegion()->find(DataCreator::m_name);
            it2->second.clear();
        }

        GetMapDataCreatorClass()->emplace(
            DataCreator::m_name, std::make_shared<DataCreator>(param...));
    }

    void Clear()
    {
        for (auto it = GetMapMemoryRegion()->begin();
             it != GetMapMemoryRegion()->end(); it++)
        {
            // Clear MemoryRegion object.
            it->second.clear();
        }
    }

private:
    NekDataWarehouse(const NekDataWarehouse &rhs)            = delete;
    NekDataWarehouse &operator=(const NekDataWarehouse &rhs) = delete;

    // Return MemoryRegion map pointer.
    tMapMemoryRegion *GetMapMemoryRegion()
    {
        return &m_mapMemoryRegion;
    }

    // Return data creator class map pointer.
    tMapDataCreatorClass *GetMapDataCreatorClass()
    {
        return &m_mapDataCreatorClass;
    }

    // Data map containing MemoryRegion object.
    tMapMemoryRegion m_mapMemoryRegion;
    // Data map containing data creator class object.
    tMapDataCreatorClass m_mapDataCreatorClass;

#ifdef NEKTAR_USE_THREAD_SAFETY
    std::shared_mutex m_mutex;
#endif
};

typedef std::shared_ptr<NekDataWarehouse> NekDataWarehouseSharedPtr;

} // namespace Nektar::Operators
