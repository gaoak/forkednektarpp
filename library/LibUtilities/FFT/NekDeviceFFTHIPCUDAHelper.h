///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFTHIPCUDAHelper.h
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
// Description: Shared helpers for the CUDA and HIP device FFT backends, over
//              cuFFT and hipFFT respectively. NekDeviceFFTSYCLHelper.h is the
//              SYCL counterpart.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

// This header defines __global__ kernels and must only be included from
// translation units compiled by the device compiler (i.e. .cu files under
// nvcc and .hip files under hipcc), never from a .cpp.

#include <cstdlib>
#include <iostream>
#include <type_traits>

#if defined(NEKTAR_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cufft.h>

#include <LibUtilities/Backends/CUDA_Host_API.hpp> // CHECK_HIPCUDA_ERROR
#else
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <LibUtilities/Backends/HIP_Host_API.hpp> // CHECK_HIPCUDA_ERROR
#endif

namespace Nektar::LibUtilities
{

// The backends differ only in the names below; everything after this block is
// written once against these aliases. The kernels need no mapping at all.
#if defined(NEKTAR_ENABLE_CUDA)

using DeviceFFTResult = cufftResult;

inline constexpr DeviceFFTResult kDeviceFFTSuccess = CUFFT_SUCCESS;
inline constexpr const char *kDeviceFFTResultName  = "cufftResult";

/// \brief Maps a real scalar type to its device FFT complex element type.
template <typename TReal>
using DeviceFFTCmplx = std::conditional_t<std::is_same_v<TReal, double>,
                                          cufftDoubleComplex, cufftComplex>;

#else // NEKTAR_ENABLE_HIP

using DeviceFFTResult = hipfftResult;

inline constexpr DeviceFFTResult kDeviceFFTSuccess = HIPFFT_SUCCESS;
inline constexpr const char *kDeviceFFTResultName  = "hipfftResult";

/// \brief Maps a real scalar type to its device FFT complex element type.
template <typename TReal>
using DeviceFFTCmplx = std::conditional_t<std::is_same_v<TReal, double>,
                                          hipfftDoubleComplex, hipfftComplex>;

#endif

// Counterpart of CHECK_HIPCUDA_ERROR (LibUtilities/Backends) for FFT status
// codes, which that macro cannot take: it compares against cudaSuccess and
// formats with cudaGetErrorString. Evaluates its argument once.
#define CHECK_HIPCUDA_FFT_ERROR(err)                                           \
    {                                                                          \
        const Nektar::LibUtilities::DeviceFFTResult nekFFTErr = (err);         \
        if (nekFFTErr != Nektar::LibUtilities::kDeviceFFTSuccess)              \
        {                                                                      \
            std::cerr << "Device FFT Error at: " << __FILE__ << ":"            \
                      << __LINE__ << std::endl;                                \
            std::cerr << Nektar::LibUtilities::kDeviceFFTResultName << "="     \
                      << static_cast<int>(nekFFTErr) << std::endl;             \
            exit(0);                                                           \
        }                                                                      \
    }

/// \brief Multiply each Fourier mode by \f$i k \beta\f$ times \p normScale, in
///        place; the DC and Nyquist modes are zeroed.
///
/// The half-spectrum is laid out as \c d_cmplx[b * (halfN + 1) + k] for batch
/// entry \c b and wavenumber \c k. The batch index runs in the grid x
/// dimension and the wavenumber range in y.
///
/// \param d_cmplx   Device pointer to the half-spectrum, modified in place.
/// \param halfN     Index of the Nyquist mode (the spectrum holds halfN + 1).
/// \param beta      Wavenumber factor \f$2\pi/L_z\f$.
/// \param normScale Normalisation applied alongside the wavenumber multiply
///                  (typically \f$1/N\f$ to fold in the inverse-transform
///                  scaling).
template <typename TReal, typename TComplex = DeviceFFTCmplx<TReal>>
__global__ static void WavenumberMultiplyKernel(TComplex *__restrict__ d_cmplx,
                                                int halfN, TReal beta,
                                                TReal normScale)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    if (k == 0 || k == halfN)
    {
        d_cmplx[b * (halfN + 1) + k] = {TReal(0), TReal(0)};
        return;
    }

    const TComplex cx            = d_cmplx[b * (halfN + 1) + k];
    const TReal scale            = static_cast<TReal>(k) * beta * normScale;
    d_cmplx[b * (halfN + 1) + k] = {-cx.y * scale, cx.x * scale};
}

/// \brief Multiply each Fourier mode by \f$-(k\beta)^2\f$ times \p normScale,
///        in place; the DC and Nyquist modes are zeroed.
///
/// The half-spectrum is laid out as \c d_cmplx[b * (halfN + 1) + k] for batch
/// entry \c b and wavenumber \c k. The batch index runs in the grid x
/// dimension and the wavenumber range in y.
///
/// The Nyquist mode goes the same way as in the first derivative, for a
/// reason particular to this basis rather than to the derivative: Nektar's
/// eFourier basis spans \f$k = 0\f$ to \f$N/2 - 1\f$ only -- its second slot
/// is a structural zero, not a Nyquist mode -- so \f$k = N/2\f$ is not
/// representable and the host transform drops it either way.
///
/// \param d_cmplx   Device pointer to the half-spectrum, modified in place.
/// \param halfN     Index of the Nyquist mode (the spectrum holds halfN + 1).
/// \param beta      Wavenumber factor \f$2\pi/L_z\f$.
/// \param normScale Normalisation applied alongside the wavenumber multiply
///                  (typically \f$1/N\f$ to fold in the inverse-transform
///                  scaling).
template <typename TReal, typename TComplex = DeviceFFTCmplx<TReal>>
__global__ static void WavenumberMultiply2Kernel(TComplex *__restrict__ d_cmplx,
                                                 int halfN, TReal beta,
                                                 TReal normScale)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    if (k == 0 || k == halfN)
    {
        d_cmplx[b * (halfN + 1) + k] = {TReal(0), TReal(0)};
        return;
    }

    const TComplex cx            = d_cmplx[b * (halfN + 1) + k];
    const TReal betaK            = static_cast<TReal>(k) * beta;
    const TReal scale            = -betaK * betaK * normScale;
    d_cmplx[b * (halfN + 1) + k] = {cx.x * scale, cx.y * scale};
}

/// \brief Scale every complex element of \p d by \p alpha, in place.
///
/// \param d        Device pointer to \p nComplex complex elements.
/// \param nComplex Number of complex elements.
/// \param alpha    Scale factor.
template <typename TReal, typename TComplex>
__global__ static void ScaleComplexKernel(TComplex *__restrict__ d,
                                          int nComplex, TReal alpha)
{
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < nComplex)
    {
        TComplex val = d[i];
        val.x *= alpha;
        val.y *= alpha;
        d[i] = val;
    }
}

/// \brief Convert device FFT half-complex output to the Nektar++
///        coefficient layout.
///
/// With \c Scaled true the \f$1/N\f$ normalisation is folded into the
/// output, which is what the plan needs when no store callback is registered.
template <typename TReal, typename TComplex, bool Scaled>
__global__ static void ComplexToCoefKernel(const TComplex *__restrict__ d_cmplx,
                                           TReal *__restrict__ d_coef, int N,
                                           int halfN, TReal invN)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    const TComplex cx = d_cmplx[b * (halfN + 1) + k];
    TReal *coef       = d_coef + b * N;

    if (k == 0)
    {
        coef[0] = Scaled ? cx.x * invN : cx.x;
        coef[1] = TReal(0);
    }
    else if (k < halfN)
    {
        const TReal factor = Scaled ? TReal(2) * invN : TReal(2);
        coef[2 * k]        = cx.x * factor;
        coef[2 * k + 1]    = cx.y * factor;
    }
    // k == halfN: Nyquist bin has no Nektar++ slot, left unwritten.
}

/// \brief Convert the Nektar++ coefficient layout to device FFT
///        half-complex input.
template <typename TReal, typename TComplex>
__global__ static void CoefToComplexKernel(const TReal *__restrict__ d_coef,
                                           TComplex *__restrict__ d_cmplx,
                                           int N, int halfN)
{
    const int b = static_cast<int>(blockIdx.x);
    const int k = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (k > halfN)
    {
        return;
    }

    const TReal *coef = d_coef + b * N;
    TComplex *cx      = d_cmplx + b * (halfN + 1);

    if (k == 0)
    {
        cx[0] = {__ldg(&coef[0]), TReal(0)};
    }
    else if (k == halfN)
    {
        cx[k] = {TReal(0), TReal(0)};
    }
    else
    {
        cx[k] = {__ldg(&coef[2 * k]) * TReal(0.5),
                 __ldg(&coef[2 * k + 1]) * TReal(0.5)};
    }
}

} // namespace Nektar::LibUtilities
