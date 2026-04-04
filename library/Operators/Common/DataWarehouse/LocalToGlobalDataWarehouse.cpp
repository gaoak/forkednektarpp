///////////////////////////////////////////////////////////////////////////////
//
// File: LocalToGlobalDataWarehouse.cpp
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

#include "Operators/Common/DataWarehouse/LocalToGlobalDataWarehouseDef.hpp"

namespace Nektar::Operators
{

template MemoryRegion<typename DeviceLocalToGlobalKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceLocalToGlobalKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceLocalToGlobalKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceLocalToGlobalKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<typename DeviceLocalToGlobalKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceLocalToGlobalKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceLocalToGlobalKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceLocalToGlobalKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceLocalToGlobalNumAssembleKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceLocalToGlobalNumAssembleKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceLocalToGlobalNumAssembleKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceLocalToGlobalNumAssembleKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceLocalToGlobalNumAssembleKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceLocalToGlobalNumAssembleKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceLocalToGlobalNumAssembleKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceLocalToGlobalNumAssembleKey<float> &LocToGloKey);
#endif

template MemoryRegion<typename DeviceLocalToGlobalIndexKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceLocalToGlobalIndexKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceLocalToGlobalIndexKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceLocalToGlobalIndexKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<typename DeviceLocalToGlobalIndexKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceLocalToGlobalIndexKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceLocalToGlobalIndexKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceLocalToGlobalIndexKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceLocalToGlobalIndexOffsetKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceLocalToGlobalIndexOffsetKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceLocalToGlobalIndexOffsetKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceLocalToGlobalIndexOffsetKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceLocalToGlobalIndexOffsetKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceLocalToGlobalIndexOffsetKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceLocalToGlobalIndexOffsetKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceLocalToGlobalIndexOffsetKey<float> &LocToGloKey);
#endif

template MemoryRegion<typename DeviceLocalToGlobalSignKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceLocalToGlobalSignKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceLocalToGlobalSignKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceLocalToGlobalSignKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<typename DeviceLocalToGlobalSignKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceLocalToGlobalSignKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceLocalToGlobalSignKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceLocalToGlobalSignKey<float> &LocToGloKey);
#endif

template MemoryRegion<typename DeviceBndLocalToGlobalKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceBndLocalToGlobalKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<typename DeviceBndLocalToGlobalKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceBndLocalToGlobalKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceBndLocalToGlobalNumAssembleKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalNumAssembleKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalNumAssembleKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalNumAssembleKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceBndLocalToGlobalNumAssembleKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalNumAssembleKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalNumAssembleKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalNumAssembleKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceBndLocalToGlobalNumBndValsKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalNumBndValsKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalNumBndValsKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalNumBndValsKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceBndLocalToGlobalNumBndValsKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalNumBndValsKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalNumBndValsKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalNumBndValsKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceBndLocalToGlobalIndexKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalIndexKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalIndexKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalIndexKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceBndLocalToGlobalIndexKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalIndexKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalIndexKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalIndexKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceBndLocalToGlobalOffsetKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalOffsetKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalOffsetKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalOffsetKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceBndLocalToGlobalOffsetKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalOffsetKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalOffsetKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalOffsetKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceBndLocalToGlobalAssembleOrderKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalAssembleOrderKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalAssembleOrderKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalAssembleOrderKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceBndLocalToGlobalAssembleOrderKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalAssembleOrderKey<double> &LocToGloKey);
template MemoryRegion<
    typename DeviceBndLocalToGlobalAssembleOrderKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalAssembleOrderKey<float> &LocToGloKey);
#endif

template MemoryRegion<
    typename DeviceBndLocalToGlobalSignKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const DeviceBndLocalToGlobalSignKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceBndLocalToGlobalSignKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const DeviceBndLocalToGlobalSignKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<
    typename DeviceBndLocalToGlobalSignKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const DeviceBndLocalToGlobalSignKey<double> &LocToGloKey);
template MemoryRegion<typename DeviceBndLocalToGlobalSignKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const DeviceBndLocalToGlobalSignKey<float> &LocToGloKey);
#endif

template MemoryRegion<typename LocalToGlobalMaskKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, double>(
    const LocalToGlobalMaskKey<double> &LocToGloKey);
template MemoryRegion<typename LocalToGlobalMaskKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::HostSpace, float>(
    const LocalToGlobalMaskKey<float> &LocToGloKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<typename LocalToGlobalMaskKey<double>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, double>(
    const LocalToGlobalMaskKey<double> &LocToGloKey);
template MemoryRegion<typename LocalToGlobalMaskKey<float>::value_type>
LocalToGlobalDataCreator::Create<NektarSpaces::DeviceSpace, float>(
    const LocalToGlobalMaskKey<float> &LocToGloKey);
#endif

} // namespace Nektar::Operators
