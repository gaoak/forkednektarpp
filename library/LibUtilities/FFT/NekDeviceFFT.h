///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFT.h
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
// Description: Portable device-FFT wrapper for Nektar++.
//
// This is the public-facing header. Include this file rather than any
// backend header directly. The actual implementation class is selected at
// compile time by the active device backend:
//
//   NEKTAR_ENABLE_CUDA  -> NekCuFFTImpl   (library/LibUtilities/FFT/NekCuFFT.h)
//   NEKTAR_ENABLE_HIP   -> NekHipFFTImpl  (TODO: not yet implemented)
//   NEKTAR_ENABLE_SYCL  -> NekSyclFFTImpl (TODO: not yet implemented)
//
// Public type aliases exposed by this header:
//   NekDeviceFFTImpl<T>  -- templated implementation class
//   NekDeviceFFT         -- double-precision alias (NekDeviceFFTImpl<double>)
//   NekDeviceFFTFloat    -- single-precision alias  (NekDeviceFFTImpl<float>)
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_FFT_NEKDEVICEFFT_H
#define NEKTAR_LIB_UTILITIES_FFT_NEKDEVICEFFT_H

#if defined(NEKTAR_ENABLE_CUDA)

#include <LibUtilities/FFT/NekCuFFT.h>

namespace Nektar::LibUtilities
{

template <typename TData> using NekDeviceFFTImpl = NekCuFFTImpl<TData>;

using NekDeviceFFT      = NekCuFFT;      // NekCuFFTImpl<double>
using NekDeviceFFTFloat = NekCuFFTFloat; // NekCuFFTImpl<float>

} // namespace Nektar::LibUtilities

#elif defined(NEKTAR_ENABLE_HIP)

// TODO: implement NekHipFFTImpl in NekHipFFT.h / NekHipFFT.hip and include it
// here. The CMake block in library/LibUtilities/CMakeLists.txt should add:
//   enable_language(HIP)
//   SET(FFTHeaders ... ./FFT/NekDeviceFFT.h ./FFT/NekHipFFT.h)
//   SET(FFTSources ... ./FFT/NekHipFFT.hip)
#error "NekDeviceFFT: HIP backend is not yet implemented. \
See library/LibUtilities/FFT/NekDeviceFFT.h for the TODO."

#elif defined(NEKTAR_ENABLE_SYCL)

// TODO: implement NekSyclFFTImpl in NekSyclFFT.h and include it here.
#error "NekDeviceFFT: SYCL backend is not yet implemented. \
See library/LibUtilities/FFT/NekDeviceFFT.h for the TODO."

#else

#error "NekDeviceFFT.h requires a device backend. \
Configure with NEKTAR_ENABLE_DEVICE=CUDA (or HIP/SYCL once implemented)."

#endif

#endif // NEKTAR_LIB_UTILITIES_FFT_NEKDEVICEFFT_H
