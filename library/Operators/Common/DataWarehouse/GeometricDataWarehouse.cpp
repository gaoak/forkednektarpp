///////////////////////////////////////////////////////////////////////////////
//
// File: GeometricDataWarehouse.cpp
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

#include <Operators/Common/DataWarehouse/GeometricDataWarehouseDef.hpp>

namespace Nektar::Operators
{

template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(const WeightsKey<double> &jacobianKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(const WeightsKey<float> &jacobianKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(const WeightsKey<double> &jacobianKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(const WeightsKey<float> &jacobianKey);
#endif

template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(const JacobianKey<double> &jacobianKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(const JacobianKey<float> &jacobianKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(const JacobianKey<double> &jacobianKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(const JacobianKey<float> &jacobianKey);
#endif

template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(
    const DerivFactorKey<double> &derivFactorKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(
    const DerivFactorKey<float> &derivFactorKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const DerivFactorKey<double> &derivFactorKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const DerivFactorKey<float> &derivFactorKey);
#endif

template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(const CoordKey<double> &coordKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(const CoordKey<float> &coordKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(const CoordKey<double> &coordKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(const CoordKey<float> &coordKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(const OrientKey<double> &orientKey);
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(const OrientKey<float> &orientKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(const OrientKey<double> &orientKey);
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(const OrientKey<float> &orientKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(
    const TraceToElmtMapKey<double> &traceToElmtMapKey);
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(
    const TraceToElmtMapKey<float> &traceToElmtMapKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const TraceToElmtMapKey<double> &traceToElmtMapKey);
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const TraceToElmtMapKey<float> &traceToElmtMapKey);
#endif

template LibUtilities::MemoryRegion<int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(
    const TraceToElmtSignKey<double> &traceToElmtSignKey);
template LibUtilities::MemoryRegion<int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(
    const TraceToElmtSignKey<float> &traceToElmtSignKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const TraceToElmtSignKey<double> &traceToElmtSignKey);
template LibUtilities::MemoryRegion<int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const TraceToElmtSignKey<float> &traceToElmtSignKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InteriorMapKey<double> &interiorMapKey);
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InteriorMapKey<float> &interiorMapKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InteriorMapKey<double> &interiorMapKey);
template LibUtilities::MemoryRegion<unsigned int> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InteriorMapKey<float> &interiorMapKey);
#endif

template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(
    const JacobianTraceKey<double> &jacobianTraceKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(
    const JacobianTraceKey<float> &jacobianTraceKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const JacobianTraceKey<double> &jacobianTraceKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const JacobianTraceKey<float> &jacobianTraceKey);
#endif

template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, double>(
    const JacobianLocTraceKey<double> &jacobianLocTraceKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::HostSpace, float>(
    const JacobianLocTraceKey<float> &jacobianLocTraceKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const JacobianLocTraceKey<double> &jacobianLocTraceKey);
template LibUtilities::MemoryRegion<float> GeometricDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const JacobianLocTraceKey<float> &jacobianLocTraceKey);
#endif

} // namespace Nektar::Operators
