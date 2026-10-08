///////////////////////////////////////////////////////////////////////////////
//
// File: DerivZDeviceFFT.h
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
//              Nektar++ homogeneous-1D DerivZ operators.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>

namespace Nektar::LibUtilities
{

/// \brief Which z-derivative the pipeline takes.
///
/// The two share their plans and differ only in the wavenumber multiply, so
/// one entry point covers both and the second derivative costs one transform
/// pair rather than the two that applying the first twice would need.
enum class DerivZOrder
{
    /// Multiplier \f$ik\beta\f$, giving \f$\partial_z u\f$. Imaginary and
    /// odd, so a mode moves to its conjugate partner plane.
    First,
    /// Multiplier \f$-(k\beta)^2\f$, giving \f$\partial_z^2 u\f$. Real, so
    /// a mode stays in its own plane.
    Second,
};

/// \brief Compute a z-derivative of a real field in the homogeneous
/// direction on the device (D2Z + wavenumber multiply + Z2D).
///
/// The three stages run as cuFFT, hipFFT or oneMath calls, or fused into a
/// single cuFFTDx kernel when the build defines NEKTAR_USE_CUFFTDX.
/// Explicitly instantiated for both precisions, both orders and both APPEND
/// modes in each backend's implementation file; every instantiation takes the
/// same path in any one build.
///
/// The field is in plane-major layout: \c d_in[plane * compStride + xy].
/// \c d_in and \c d_out may alias the same buffer when overwriting; that is
/// not meaningful when appending, which reads \c d_out as well.
///
/// \tparam DERIVORDER
///                   Which derivative to take; see DerivZOrder. Both orders
///                   zero the mean and the mode at \f$k = N/2\f$, which
///                   Nektar's eFourier basis cannot represent.
/// \tparam APPEND    false overwrites \c d_out, true adds the result on top
///                   of what is already there, for a caller whose xy pass has
///                   written a partial result. The fused cuFFTDx kernel
///                   accumulates in its own final store and costs nothing
///                   extra; over cuFFT, hipFFT and oneMath the library owns
///                   that store, so the inverse transform lands in a buffer
///                   the plan holds for the purpose and is summed in
///                   afterwards. Either way the caller needs no scratch of
///                   its own.
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
template <typename TData, DerivZOrder DERIVORDER, bool APPEND>
void DerivZDirect(const TData *d_in, TData *d_out, unsigned int nhomo,
                  size_t NXY, size_t compStride, TData beta,
                  unsigned int streamID);

/// \brief Create and cache everything DerivZDirect() needs for this
/// problem size on this stream, without submitting any of the pipeline.
///
/// DerivZDirect() does this itself on a cache miss, but the setup
/// allocates, and under cuFFT and hipFFT it also runs a warm-up transform and
/// synchronises the stream. None of that is legal inside a device graph
/// capture, so a caller that captures its very first call has to prepare each
/// stream beforehand. A call for an entry that already has everything it
/// needs does nothing. The fused cuFFTDx pipeline holds no such state, so
/// there it is a no-op.
///
/// \tparam APPEND    Must match the DerivZDirect() calls that follow, which
///                   dereference the accumulate buffer without checking it.
///                   When true that buffer is allocated here, since
///                   allocating inside a graph capture is not allowed; an
///                   entry first created by an overwriting caller gains one
///                   on the first appending prepare.
///
/// \param nhomo      Number of homogeneous planes (must be even).
/// \param NXY        Number of points per plane (FFT batch size).
/// \param compStride Stride between successive planes in the buffers.
/// \param streamID   Id of the stream the prepared state belongs to; it has
///                   to match the one passed to DerivZDirect() later.
template <typename TData, bool APPEND>
void DerivZPrepare(unsigned int nhomo, size_t NXY, size_t compStride,
                   unsigned int streamID);

} // namespace Nektar::LibUtilities
