///////////////////////////////////////////////////////////////////////////////
//
// File: BasisDataWarehouse.cpp
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

#include "Operators/Common/BasisDataWarehouseDef.hpp"

namespace Nektar::Operators
{

template MemoryRegion<double> BasisDataCreator::Create<
    NektarSpaces::HostSpace, double>(const BasisDataKey<double> &basisDataKey);
template MemoryRegion<float> BasisDataCreator::Create<
    NektarSpaces::HostSpace, float>(const BasisDataKey<float> &basisDataKey);
template MemoryRegion<tinysimd::scalarT<double>> BasisDataCreator::Create<
    NektarSpaces::HostSpace, tinysimd::scalarT<double>>(
    const BasisDataKey<tinysimd::scalarT<double>> &basisDataKey);
template MemoryRegion<tinysimd::scalarT<float>> BasisDataCreator::Create<
    NektarSpaces::HostSpace, tinysimd::scalarT<float>>(
    const BasisDataKey<tinysimd::scalarT<float>> &basisDataKey);
#if (NEKTAR_ENABLE_SIMD)
template MemoryRegion<tinysimd::simd<double>> BasisDataCreator::Create<
    NektarSpaces::HostSpace, tinysimd::simd<double>>(
    const BasisDataKey<tinysimd::simd<double>> &basisDataKey);
template MemoryRegion<tinysimd::simd<float>> BasisDataCreator::Create<
    NektarSpaces::HostSpace, tinysimd::simd<float>>(
    const BasisDataKey<tinysimd::simd<float>> &basisDataKey);
#endif
#if defined(NEKTAR_ENABLE_DEVICE)
template MemoryRegion<double> BasisDataCreator::Create<
    NektarSpaces::DeviceSpace, double>(
    const BasisDataKey<double> &basisDataKey);
template MemoryRegion<float> BasisDataCreator::Create<
    NektarSpaces::DeviceSpace, float>(const BasisDataKey<float> &basisDataKey);
template MemoryRegion<tinysimd::scalarT<double>> BasisDataCreator::Create<
    NektarSpaces::DeviceSpace, tinysimd::scalarT<double>>(
    const BasisDataKey<tinysimd::scalarT<double>> &basisDataKey);
template MemoryRegion<tinysimd::scalarT<float>> BasisDataCreator::Create<
    NektarSpaces::DeviceSpace, tinysimd::scalarT<float>>(
    const BasisDataKey<tinysimd::scalarT<float>> &basisDataKey);
#if (NEKTAR_ENABLE_SIMD)
template MemoryRegion<tinysimd::simd<double>> BasisDataCreator::Create<
    NektarSpaces::DeviceSpace, tinysimd::simd<double>>(
    const BasisDataKey<tinysimd::simd<double>> &basisDataKey);
template MemoryRegion<tinysimd::simd<float>> BasisDataCreator::Create<
    NektarSpaces::DeviceSpace, tinysimd::simd<float>>(
    const BasisDataKey<tinysimd::simd<float>> &basisDataKey);
#endif
#endif

} // namespace Nektar::Operators
