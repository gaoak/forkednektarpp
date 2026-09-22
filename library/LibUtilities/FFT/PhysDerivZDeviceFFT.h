///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZDeviceFFT.h
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
// Description: Device z-derivative (D2Z + wavenumber multiply + Z2D) for the
//              Nektar++ homogeneous-1D PhysDeriv pipeline.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>

namespace Nektar::LibUtilities
{

/// \brief Compute the z-derivative of a real field in the homogeneous
/// direction on the device (D2Z + wavenumber multiply + Z2D).
///
/// The three stages run as cuFFT, hipFFT or oneMath calls, or fused into a
/// single cuFFTDx kernel when the build defines NEKTAR_USE_CUFFTDX.
/// Explicitly instantiated for double and float in PhysDerivZDeviceFFT.cu
/// (CUDA), .hip (HIP) and .cpp (SYCL); both precisions take the same path in
/// any build.
///
/// The field is in plane-major layout: \c d_in[plane * compStride + xy].
/// \c d_in and \c d_out may alias the same buffer (in-place is safe).
///
/// \param d_in       Device pointer to the input real field.
/// \param d_out      Device pointer to the output real field.
/// \param nhomo      Number of homogeneous planes (must be even).
/// \param NXY        Number of points per plane (FFT batch size).
/// \param compStride Stride between successive planes in the buffers
///                   (>= NXY; equals NXY when there is no padding).
/// \param beta       Wavenumber factor \f$2\pi/L_z\f$.
/// \param streamID   Id of the stream on which all work is submitted. The
///                   id keeps this header free of the CUDA, HIP and SYCL
///                   headers: each backend resolves it through its own
///                   CUDAStream / HIPStream / SYCLQueue registry, so the
///                   same id always names the same stream or queue.
template <typename TData>
void PhysDerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                      size_t NXY, size_t compStride, TData beta,
                      unsigned int streamID);

/// \brief Create and cache everything PhysDerivZDirect() needs for this
/// problem size on this stream, without submitting any of the pipeline.
///
/// PhysDerivZDirect() does this itself on a cache miss, but the setup
/// allocates, and under cuFFT and hipFFT it also runs a warm-up transform and
/// synchronises the stream. None of that is legal inside a device graph
/// capture, so a caller that captures its very first call has to prepare each
/// stream beforehand. A call for an entry that already exists does nothing.
/// The fused cuFFTDx pipeline holds no such state, so there it is a no-op.
///
/// \param nhomo      Number of homogeneous planes (must be even).
/// \param NXY        Number of points per plane (FFT batch size).
/// \param compStride Stride between successive planes in the buffers.
/// \param streamID   Id of the stream the prepared state belongs to; it has
///                   to match the one passed to PhysDerivZDirect() later.
template <typename TData>
void PhysDerivZPrepare(unsigned int nhomo, size_t NXY, size_t compStride,
                       unsigned int streamID);

} // namespace Nektar::LibUtilities
