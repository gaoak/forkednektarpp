///////////////////////////////////////////////////////////////////////////////
//
// File: StdMatDataWarehouse.cpp
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

#include "Operators/Common/DataWarehouse/StdMatDataWarehouseDef.hpp"

namespace Nektar::Operators
{

template LibUtilities::MemoryRegion<double> StdMatDataCreator::Create<
    NektarSpaces::HostSpace, double>(const StdMatKey<double> &stdMatKey);
template LibUtilities::MemoryRegion<float> StdMatDataCreator::Create<
    NektarSpaces::HostSpace, float>(const StdMatKey<float> &stdMatKey);
template LibUtilities::MemoryRegion<tinysimd::scalarT<double>> StdMatDataCreator::
    Create<NektarSpaces::HostSpace, tinysimd::scalarT<double>>(
        const StdMatKey<tinysimd::scalarT<double>> &stdMatKey);
template LibUtilities::MemoryRegion<tinysimd::scalarT<float>> StdMatDataCreator::
    Create<NektarSpaces::HostSpace, tinysimd::scalarT<float>>(
        const StdMatKey<tinysimd::scalarT<float>> &stdMatKey);
#if (NEKTAR_ENABLE_SIMD)
template LibUtilities::MemoryRegion<tinysimd::simd<double>> StdMatDataCreator::
    Create<NektarSpaces::HostSpace, tinysimd::simd<double>>(
        const StdMatKey<tinysimd::simd<double>> &stdMatKey);
template LibUtilities::MemoryRegion<tinysimd::simd<float>> StdMatDataCreator::
    Create<NektarSpaces::HostSpace, tinysimd::simd<float>>(
        const StdMatKey<tinysimd::simd<float>> &stdMatKey);
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
template LibUtilities::MemoryRegion<double> StdMatDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(const StdMatKey<double> &stdMatKey);
template LibUtilities::MemoryRegion<float> StdMatDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(const StdMatKey<float> &stdMatKey);
template LibUtilities::MemoryRegion<tinysimd::scalarT<double>> StdMatDataCreator::
    Create<NektarSpaces::DeviceSpace, tinysimd::scalarT<double>>(
        const StdMatKey<tinysimd::scalarT<double>> &stdMatKey);
template LibUtilities::MemoryRegion<tinysimd::scalarT<float>> StdMatDataCreator::
    Create<NektarSpaces::DeviceSpace, tinysimd::scalarT<float>>(
        const StdMatKey<tinysimd::scalarT<float>> &stdMatKey);
#if (NEKTAR_ENABLE_SIMD)
template LibUtilities::MemoryRegion<tinysimd::simd<double>> StdMatDataCreator::
    Create<NektarSpaces::DeviceSpace, tinysimd::simd<double>>(
        const StdMatKey<tinysimd::simd<double>> &stdMatKey);
template LibUtilities::MemoryRegion<tinysimd::simd<float>> StdMatDataCreator::
    Create<NektarSpaces::DeviceSpace, tinysimd::simd<float>>(
        const StdMatKey<tinysimd::simd<float>> &stdMatKey);
#endif
#endif

} // namespace Nektar::Operators
