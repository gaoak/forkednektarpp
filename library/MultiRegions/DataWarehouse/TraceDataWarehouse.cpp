///////////////////////////////////////////////////////////////////////////////
//
// File: TraceDataWarehouse.cpp
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

#include <MultiRegions/DataWarehouse/TraceDataWarehouseDef.hpp>

namespace Nektar::MultiRegions
{
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const IPTraceNormalKey<double> &ipTraceNormalKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const IPTraceNormalKey<float> &ipTraceNormalKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const IPTraceNormalKey<double> &ipTraceNormalKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const IPTraceNormalKey<float> &ipTraceNormalKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const IPTraceScalarKey<double> &ipTraceScalarKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const IPTraceScalarKey<float> &ipTraceScalarKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const IPTraceScalarKey<double> &ipTraceScalarKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const IPTraceScalarKey<float> &ipTraceScalarKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const IPTraceDerivBaseKey<double> &ipTraceDerivBaseKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const IPTraceDerivBaseKey<float> &ipTraceDerivBaseKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const IPTraceDerivBaseKey<double> &ipTraceDerivBaseKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const IPTraceDerivBaseKey<float> &ipTraceDerivBaseKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const LocTracePhysToElmtMapsKey<double> &locTracePhysToElmtMapsKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const LocTracePhysToElmtMapsKey<float> &locTracePhysToElmtMapsKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const LocTracePhysToElmtMapsKey<double> &locTracePhysToElmtMapsKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const LocTracePhysToElmtMapsKey<float> &locTracePhysToElmtMapsKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const OrientationMapsKey<double> &orientationMapsKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const OrientationMapsKey<float> &orientationMapsKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const OrientationMapsKey<double> &orientationMapsKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const OrientationMapsKey<float> &orientationMapsKey);
#endif

template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const OrientationMapsOffsetKey<double> &orientationMapsOffsetKey);
template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const OrientationMapsOffsetKey<float> &orientationMapsOffsetKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const OrientationMapsOffsetKey<double> &orientationMapsOffsetKey);
template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const OrientationMapsOffsetKey<float> &orientationMapsOffsetKey);
#endif

template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const LocToTracePhysOffsetKey<double> &locToTracePhysOffsetKey);
template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const LocToTracePhysOffsetKey<float> &locToTracePhysOffsetKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const LocToTracePhysOffsetKey<double> &locToTracePhysOffsetKey);
template LibUtilities::MemoryRegion<size_t> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const LocToTracePhysOffsetKey<float> &locToTracePhysOffsetKey);
#endif

template LibUtilities::MemoryRegion<bool> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const IsLocTraceLeftAdjacentKey<double> &isLocTraceLeftAdjacentKey);
template LibUtilities::MemoryRegion<bool> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const IsLocTraceLeftAdjacentKey<float> &isLocTraceLeftAdjacentKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<bool> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const IsLocTraceLeftAdjacentKey<double> &isLocTraceLeftAdjacentKey);
template LibUtilities::MemoryRegion<bool> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const IsLocTraceLeftAdjacentKey<float> &isLocTraceLeftAdjacentKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpTraceIndexKey<double> &interpTraceIndexKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpTraceIndexKey<float> &interpTraceIndexKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpTraceIndexKey<double> &interpTraceIndexKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpTraceIndexKey<float> &interpTraceIndexKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpPointsKey<double> &interpPointsKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpPointsKey<float> &interpPointsKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpPointsKey<double> &interpPointsKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpPointsKey<float> &interpPointsKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpTypesKey<double> &interpTypesKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpTypesKey<float> &interpTypesKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpTypesKey<double> &interpTypesKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpTypesKey<float> &interpTypesKey);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(const QuadRangeKey<double> &quadRangeKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(const QuadRangeKey<float> &quadRangeKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const QuadRangeKey<double> &quadRangeKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(const QuadRangeKey<float> &quadRangeKey);
#endif

template LibUtilities::MemoryRegion<InterpLocTraceToTrace> TraceEssentialCreator::
    Create<NektarSpaces::HostSpace, double>(
        const InterpTraceKey<double> &interpTraceKey);
template LibUtilities::MemoryRegion<InterpLocTraceToTrace> TraceEssentialCreator::
    Create<NektarSpaces::HostSpace, float>(
        const InterpTraceKey<float> &interpTraceKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<InterpLocTraceToTrace> TraceEssentialCreator::
    Create<NektarSpaces::DeviceSpace, double>(
        const InterpTraceKey<double> &interpTraceKey);
template LibUtilities::MemoryRegion<InterpLocTraceToTrace> TraceEssentialCreator::
    Create<NektarSpaces::DeviceSpace, float>(
        const InterpTraceKey<float> &interpTraceKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpTraceI0Key<double> &interpTraceI0Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpTraceI0Key<float> &interpTraceI0Key);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpTraceI0Key<double> &interpTraceI0Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpTraceI0Key<float> &interpTraceI0Key);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpTraceI0OffsetKey<double> &interpTraceI0OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpTraceI0OffsetKey<float> &interpTraceI0OffsetKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpTraceI0OffsetKey<double> &interpTraceI0OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpTraceI0OffsetKey<float> &interpTraceI0OffsetKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpTraceI1Key<double> &interpTraceI1Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpTraceI1Key<float> &interpTraceI1Key);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpTraceI1Key<double> &interpTraceI1Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpTraceI1Key<float> &interpTraceI1Key);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpTraceI1OffsetKey<double> &interpTraceI1OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpTraceI1OffsetKey<float> &interpTraceI1OffsetKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpTraceI1OffsetKey<double> &interpTraceI1OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpTraceI1OffsetKey<float> &interpTraceI1OffsetKey);
#endif

template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::HostSpace, double>(
        const InterpFromTraceI0Key<double> &interpFromTraceI0Key);
template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::HostSpace, float>(
        const InterpFromTraceI0Key<float> &interpFromTraceI0Key);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::DeviceSpace, double>(
        const InterpFromTraceI0Key<double> &interpFromTraceI0Key);
template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::DeviceSpace, float>(
        const InterpFromTraceI0Key<float> &interpFromTraceI0Key);
#endif

template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::HostSpace, double>(
        const InterpFromTraceI1Key<double> &interpFromTraceI1Key);
template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::HostSpace, float>(
        const InterpFromTraceI1Key<float> &interpFromTraceI1Key);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::DeviceSpace, double>(
        const InterpFromTraceI1Key<double> &interpFromTraceI1Key);
template LibUtilities::MemoryRegion<DNekMatSharedPtr> TraceEssentialCreator::
    Create<NektarSpaces::DeviceSpace, float>(
        const InterpFromTraceI1Key<float> &interpFromTraceI1Key);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpEndPtI0Key<double> &interpEndPtI0Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpEndPtI0Key<float> &interpEndPtI0Key);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpEndPtI0Key<double> &interpEndPtI0Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpEndPtI0Key<float> &interpEndPtI0Key);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpEndPtI0OffsetKey<double> &interpEndPtI0OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpEndPtI0OffsetKey<float> &interpEndPtI0OffsetKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpEndPtI0OffsetKey<double> &interpEndPtI0OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpEndPtI0OffsetKey<float> &interpEndPtI0OffsetKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpEndPtI1Key<double> &interpEndPtI1Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpEndPtI1Key<float> &interpEndPtI1Key);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpEndPtI1Key<double> &interpEndPtI1Key);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpEndPtI1Key<float> &interpEndPtI1Key);
#endif

template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(
    const InterpEndPtI1OffsetKey<double> &interpEndPtI1OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(
    const InterpEndPtI1OffsetKey<float> &interpEndPtI1OffsetKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const InterpEndPtI1OffsetKey<double> &interpEndPtI1OffsetKey);
template LibUtilities::MemoryRegion<unsigned int> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(
    const InterpEndPtI1OffsetKey<float> &interpEndPtI1OffsetKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(const Interp1DKey<double> &interp1DKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(const Interp1DKey<float> &interp1DKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(const Interp1DKey<double> &interp1DKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(const Interp1DKey<float> &interp1DKey);
#endif

template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, double>(const Interp2DKey<double> &interp2DKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::HostSpace, float>(const Interp2DKey<float> &interp2DKey);
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, double>(const Interp2DKey<double> &interp2DKey);
template LibUtilities::MemoryRegion<float> TraceEssentialCreator::Create<
    NektarSpaces::DeviceSpace, float>(const Interp2DKey<float> &interp2DKey);
#endif

} // namespace Nektar::MultiRegions
