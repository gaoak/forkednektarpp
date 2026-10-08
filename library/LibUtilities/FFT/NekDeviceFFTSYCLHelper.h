///////////////////////////////////////////////////////////////////////////////
//
// File: NekDeviceFFTSYCLHelper.h
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
// Description: SYCL counterpart of NekDeviceFFTHIPCUDAHelper.h. Holds the type
// aliases and the kernels the SYCL FFT sources share, and the plan they run a
// transform through: a descriptor where oneMath or oneMKL is configured, and
// a naive DFT kernel where neither is.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(NEKTAR_ENABLE_SYCL)

#include <complex>
#include <cstdint>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#include <LibUtilities/Backends/SYCLQueue.hpp>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#if defined(NEKTAR_ENABLE_ONEMATH)
#include "oneapi/math.hpp"
#elif defined(NEKTAR_ENABLE_ONEMKL)
#include "oneapi/mkl.hpp"
#endif

namespace Nektar::LibUtilities
{

#if defined(NEKTAR_ENABLE_ONEMATH) || defined(NEKTAR_ENABLE_ONEMKL)
#define NEKTAR_SYCL_HAVE_DFT_LIBRARY

// The two libraries differ only in the namespace below; everything after this
// block is written once against the alias.
#if defined(NEKTAR_ENABLE_ONEMATH)
namespace dft = oneapi::math::dft;
#else
namespace dft = oneapi::mkl::dft;
#endif

/// \brief Maps a real scalar type to its DFT descriptor type.
///
/// The precision has to track TReal: a descriptor committed at one precision
/// rejects pointers of the other at compute time.
template <typename TReal>
using DFTDescriptor =
    dft::descriptor<std::is_same_v<TReal, double> ? dft::precision::DOUBLE
                                                  : dft::precision::SINGLE,
                    dft::domain::REAL>;

#endif // NEKTAR_ENABLE_ONEMATH || NEKTAR_ENABLE_ONEMKL

/// \brief Maps a real scalar type to its backward-domain element type.
template <typename TReal> using DFTCmplx = std::complex<TReal>;

/// \brief View a half-spectrum buffer as the interleaved real/imaginary pairs
///        the kernels below index.
///
/// std::complex is not guaranteed to be usable in device code, while its
/// layout as two adjacent reals is, so every kernel here works on that view.
template <typename TReal> inline TReal *AsReIm(DFTCmplx<TReal> *d_cmplx)
{
    return reinterpret_cast<TReal *>(d_cmplx);
}

/// \brief Const overload of AsReIm().
template <typename TReal>
inline const TReal *AsReIm(const DFTCmplx<TReal> *d_cmplx)
{
    return reinterpret_cast<const TReal *>(d_cmplx);
}

/// One full turn, at more digits than either precision can hold, so the
/// naive kernels below round it once on conversion to TReal.
inline constexpr double kTwoPi = 6.283185307179586476925286766559;

/// \brief Real-to-half-complex transform, computed term by term.
///
/// Stands in for the library transform where no oneMath or oneMKL build is
/// configured, and matches what that transform produces: an unnormalised
///
///   X[b][k] = sum_n x[b][n] exp(-2 pi i k n / N),   k = 0 .. N/2
///
/// with the half spectrum contiguous as d_cmplx[b * (N/2 + 1) + k]. The real
/// field is addressed as d_real[b * realDist + n * realStride], which covers
/// both the contiguous batches NekDeviceFFT uses and the strided planes of
/// the homogeneous z-derivative.
///
/// One work-item per output mode, each summing over N points: O(N) per mode
/// against the O(log N) a real FFT would take. It exists so that a build
/// without a DFT library still computes the right answer, not to be fast.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d_real       Device pointer to the real input.
/// \param d_cmplx      Device pointer to the half-spectrum output.
/// \param batch        Number of transforms.
/// \param N            Transform size (number of real points).
/// \param realStride   Distance between successive points of one transform.
/// \param realDist     Distance between successive transforms.
/// \param dependencies Events the kernel waits on.
template <typename TReal>
sycl::event NaiveForwardKernel(
    sycl::queue &Q, const TReal *d_real, DFTCmplx<TReal> *d_cmplx, size_t batch,
    std::int64_t N, std::int64_t realStride, std::int64_t realDist,
    const std::vector<sycl::event> &dependencies = {})
{
    const std::int64_t nModes = N / 2 + 1;
    TReal *d_reim             = AsReIm(d_cmplx);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(
            sycl::range<2>(batch, static_cast<size_t>(nModes)),
            [=](sycl::id<2> indx) {
                const std::int64_t b = static_cast<std::int64_t>(indx[0]);
                const std::int64_t k = static_cast<std::int64_t>(indx[1]);
                const TReal *x       = d_real + b * realDist;

                TReal re = TReal(0);
                TReal im = TReal(0);
                for (std::int64_t n = 0; n < N; ++n)
                {
                    // Reduced before scaling: k * n runs to N^2 / 2, and the
                    // angle it stands for repeats every N.
                    const TReal angle = static_cast<TReal>(kTwoPi) *
                                        static_cast<TReal>((k * n) % N) /
                                        static_cast<TReal>(N);
                    const TReal v = x[n * realStride];
                    re += v * sycl::cos(angle);
                    im -= v * sycl::sin(angle);
                }

                const size_t out = 2 * static_cast<size_t>(b * nModes + k);
                d_reim[out]      = re;
                d_reim[out + 1]  = im;
            });
    });
}

/// \brief Half-complex-to-real transform, computed term by term.
///
/// The inverse of NaiveForwardKernel() and, like it, unnormalised, so a round
/// trip scales by N exactly as the library transform does. The spectrum is
/// Hermitian, so mode k and mode N - k contribute the same amount and the
/// sum below doubles the interior modes rather than storing their conjugates.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d_cmplx      Device pointer to the half-spectrum input.
/// \param d_real       Device pointer to the real output.
/// \param batch        Number of transforms.
/// \param N            Transform size (number of real points).
/// \param realStride   Distance between successive points of one transform.
/// \param realDist     Distance between successive transforms.
/// \param dependencies Events the kernel waits on.
template <typename TReal>
sycl::event NaiveBackwardKernel(
    sycl::queue &Q, const DFTCmplx<TReal> *d_cmplx, TReal *d_real, size_t batch,
    std::int64_t N, std::int64_t realStride, std::int64_t realDist,
    const std::vector<sycl::event> &dependencies = {})
{
    const std::int64_t halfN  = N / 2;
    const std::int64_t nModes = halfN + 1;
    const TReal *d_reim       = AsReIm(d_cmplx);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(
            sycl::range<2>(batch, static_cast<size_t>(N)),
            [=](sycl::id<2> indx) {
                const std::int64_t b = static_cast<std::int64_t>(indx[0]);
                const std::int64_t n = static_cast<std::int64_t>(indx[1]);
                const TReal *cx      = d_reim + 2 * b * nModes;

                // k == 0 is real and appears once. With N even so is the
                // Nyquist mode at halfN, handled after the loop; with N odd
                // there is no Nyquist mode and halfN is an ordinary pair.
                const bool hasNyquist   = (N % 2 == 0);
                const std::int64_t last = hasNyquist ? halfN - 1 : halfN;

                TReal sum = cx[0];

                for (std::int64_t k = 1; k <= last; ++k)
                {
                    const TReal angle = static_cast<TReal>(kTwoPi) *
                                        static_cast<TReal>((k * n) % N) /
                                        static_cast<TReal>(N);
                    sum += TReal(2) * (cx[2 * k] * sycl::cos(angle) -
                                       cx[2 * k + 1] * sycl::sin(angle));
                }

                if (hasNyquist)
                {
                    // Its exponential alternates in sign with n.
                    sum +=
                        cx[2 * halfN] * ((n % 2 == 0) ? TReal(1) : TReal(-1));
                }

                d_real[b * realDist + n * realStride] = sum;
            });
    });
}

/// \brief Multiply each Fourier mode by \f$i k \beta\f$ times \p normScale, in
///        place; the DC and Nyquist modes are zeroed.
///
/// Named for the __global__ kernel it replaces, which also keeps it clear of
/// NekDeviceFFTImpl::WavenumberMultiply().
///
/// The half-spectrum is laid out as \c d_cmplx[b * (halfN + 1) + k] for batch
/// entry \c b and wavenumber \c k.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d_cmplx      Device pointer to the half-spectrum, modified in place.
/// \param batch        Number of batch entries.
/// \param halfN        Index of the Nyquist mode (the spectrum holds
///                     halfN + 1).
/// \param beta         Wavenumber factor \f$2\pi/L_z\f$.
/// \param normScale    Normalisation applied alongside the wavenumber
///                     multiply (typically \f$1/N\f$ to fold in the
///                     inverse-transform scaling).
/// \param dependencies Events the kernel waits on.
template <typename TReal>
sycl::event WavenumberMultiplyKernel(
    sycl::queue &Q, DFTCmplx<TReal> *d_cmplx, size_t batch, std::int64_t halfN,
    TReal beta, TReal normScale,
    const std::vector<sycl::event> &dependencies = {})
{
    const std::int64_t nModes = halfN + 1;
    TReal *d_reim             = AsReIm(d_cmplx);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(
            sycl::range<2>(batch, static_cast<size_t>(nModes)),
            [=](sycl::id<2> indx) {
                const std::int64_t b = static_cast<std::int64_t>(indx[0]);
                const std::int64_t k = static_cast<std::int64_t>(indx[1]);
                const size_t re      = 2 * static_cast<size_t>(b * nModes + k);

                if (k == 0 || k == halfN)
                {
                    d_reim[re]     = TReal(0);
                    d_reim[re + 1] = TReal(0);
                    return;
                }

                const TReal scale = static_cast<TReal>(k) * beta * normScale;
                const TReal cx    = d_reim[re];
                const TReal cy    = d_reim[re + 1];
                d_reim[re]        = -cy * scale;
                d_reim[re + 1]    = cx * scale;
            });
    });
}

/// \brief Multiply each Fourier mode by \f$-(k\beta)^2\f$ times \p normScale,
///        in place; only the DC mode is zeroed.
///
/// The half-spectrum is laid out as \c d_cmplx[b * (halfN + 1) + k] for batch
/// entry \c b and wavenumber \c k.
///
/// The Nyquist mode goes the same way as in the first derivative, for a
/// reason particular to this basis rather than to the derivative: Nektar's
/// eFourier basis spans \f$k = 0\f$ to \f$N/2 - 1\f$ only -- its second slot
/// is a structural zero, not a Nyquist mode -- so \f$k = N/2\f$ is not
/// representable and the host transform drops it either way.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d_cmplx      Device pointer to the half-spectrum, modified in place.
/// \param batch        Number of batch entries.
/// \param halfN        Index of the Nyquist mode (the spectrum holds
///                     halfN + 1).
/// \param beta         Wavenumber factor \f$2\pi/L_z\f$.
/// \param normScale    Normalisation applied alongside the wavenumber
///                     multiply (typically \f$1/N\f$ to fold in the
///                     inverse-transform scaling).
/// \param dependencies Events the kernel waits on.
template <typename TReal>
sycl::event WavenumberMultiply2Kernel(
    sycl::queue &Q, DFTCmplx<TReal> *d_cmplx, size_t batch, std::int64_t halfN,
    TReal beta, TReal normScale,
    const std::vector<sycl::event> &dependencies = {})
{
    const std::int64_t nModes = halfN + 1;
    TReal *d_reim             = AsReIm(d_cmplx);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(
            sycl::range<2>(batch, static_cast<size_t>(nModes)),
            [=](sycl::id<2> indx) {
                const std::int64_t b = static_cast<std::int64_t>(indx[0]);
                const std::int64_t k = static_cast<std::int64_t>(indx[1]);
                const size_t re      = 2 * static_cast<size_t>(b * nModes + k);

                if (k == 0 || k == halfN)
                {
                    d_reim[re]     = TReal(0);
                    d_reim[re + 1] = TReal(0);
                    return;
                }

                const TReal betaK = static_cast<TReal>(k) * beta;
                const TReal scale = -betaK * betaK * normScale;
                d_reim[re]        = d_reim[re] * scale;
                d_reim[re + 1]    = d_reim[re + 1] * scale;
            });
    });
}

/// \brief Scale every complex element of \p d by \p alpha, in place.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d            Device pointer to \p nComplex complex elements.
/// \param nComplex     Number of complex elements.
/// \param alpha        Scale factor.
/// \param dependencies Events the kernel waits on.
template <typename TReal>
sycl::event ScaleComplexKernel(
    sycl::queue &Q, DFTCmplx<TReal> *d, size_t nComplex, TReal alpha,
    const std::vector<sycl::event> &dependencies = {})
{
    TReal *d_reim = AsReIm(d);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(sycl::range<1>(2 * nComplex),
                         [=](sycl::id<1> indx) { d_reim[indx] *= alpha; });
    });
}

/// \brief Convert DFT half-complex output to the Nektar++ coefficient layout.
///
/// With \c Scaled true the \f$1/N\f$ normalisation is folded into the output.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d_cmplx      Device pointer to the half-spectrum.
/// \param d_coef       Device pointer to the coefficient output.
/// \param batch        Number of batch entries.
/// \param N            Transform size (number of real points).
/// \param halfN        Index of the Nyquist mode.
/// \param invN         \f$1/N\f$, used only when \c Scaled is true.
/// \param dependencies Events the kernel waits on.
template <typename TReal, bool Scaled>
sycl::event ComplexToCoefKernel(
    sycl::queue &Q, const DFTCmplx<TReal> *d_cmplx, TReal *d_coef, size_t batch,
    std::int64_t N, std::int64_t halfN, TReal invN,
    const std::vector<sycl::event> &dependencies = {})
{
    const std::int64_t nModes = halfN + 1;
    const TReal *d_reim       = AsReIm(d_cmplx);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(
            sycl::range<2>(batch, static_cast<size_t>(nModes)),
            [=](sycl::id<2> indx) {
                const std::int64_t b = static_cast<std::int64_t>(indx[0]);
                const std::int64_t k = static_cast<std::int64_t>(indx[1]);
                const size_t re      = 2 * static_cast<size_t>(b * nModes + k);
                TReal *coef          = d_coef + b * N;

                if (k == 0)
                {
                    coef[0] = Scaled ? d_reim[re] * invN : d_reim[re];
                    coef[1] = TReal(0);
                }
                else if (k < halfN)
                {
                    const TReal factor = Scaled ? TReal(2) * invN : TReal(2);
                    coef[2 * k]        = d_reim[re] * factor;
                    coef[2 * k + 1]    = d_reim[re + 1] * factor;
                }
                // k == halfN: Nyquist bin has no Nektar++ slot, left
                // unwritten.
            });
    });
}

/// \brief Convert the Nektar++ coefficient layout to DFT half-complex input.
///
/// \param Q            Queue the kernel is submitted to.
/// \param d_coef       Device pointer to the coefficients.
/// \param d_cmplx      Device pointer to the half-spectrum output.
/// \param batch        Number of batch entries.
/// \param N            Transform size (number of real points).
/// \param halfN        Index of the Nyquist mode.
/// \param dependencies Events the kernel waits on.
template <typename TReal>
sycl::event CoefToComplexKernel(
    sycl::queue &Q, const TReal *d_coef, DFTCmplx<TReal> *d_cmplx, size_t batch,
    std::int64_t N, std::int64_t halfN,
    const std::vector<sycl::event> &dependencies = {})
{
    const std::int64_t nModes = halfN + 1;
    TReal *d_reim             = AsReIm(d_cmplx);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(
            sycl::range<2>(batch, static_cast<size_t>(nModes)),
            [=](sycl::id<2> indx) {
                const std::int64_t b = static_cast<std::int64_t>(indx[0]);
                const std::int64_t k = static_cast<std::int64_t>(indx[1]);
                const size_t re      = 2 * static_cast<size_t>(b * nModes + k);
                const TReal *coef    = d_coef + b * N;

                if (k == 0)
                {
                    d_reim[re]     = coef[0];
                    d_reim[re + 1] = TReal(0);
                }
                else if (k == halfN)
                {
                    d_reim[re]     = TReal(0);
                    d_reim[re + 1] = TReal(0);
                }
                else
                {
                    d_reim[re]     = coef[2 * k] * TReal(0.5);
                    d_reim[re + 1] = coef[2 * k + 1] * TReal(0.5);
                }
            });
    });
}

/// \brief A committed transform, and the shape it runs over.
///
/// The descriptor is null where no library is configured, and where the one
/// that is cannot take the requested layout; ComputeForward() and
/// ComputeBackward() then run the kernels above instead, so callers never
/// branch on the build.
///
/// The real field is addressed as d_real[b * realDist + n * realStride] and
/// the half spectrum as d_cmplx[b * (N / 2 + 1) + k]. Both transforms are
/// unnormalised, so a round trip scales by N and the caller folds in whatever
/// normalisation it wants.
template <typename TReal> struct DFTPlan
{
#if defined(NEKTAR_SYCL_HAVE_DFT_LIBRARY)
    /// Committed descriptor, owned by whoever made the plan, driving both
    /// directions.
    DFTDescriptor<TReal> *desc = nullptr;
#endif
    size_t batch            = 0;
    std::int64_t N          = 0;
    std::int64_t halfN      = 0;
    std::int64_t realStride = 1;
    std::int64_t realDist   = 0;
};

/// \brief Say once that the device FFTs are running term by term.
///
/// \param reason Why no library transform is used.
inline void WarnNaiveTransform(const std::string &reason)
{
    static bool warned = false;
    if (!warned)
    {
        warned = true;
        NEKERROR(ErrorUtil::ewarning,
                 "Device FFTs run term by term, at O(N) per mode, because " +
                     reason +
                     ". Results are correct; performance is not "
                     "representative.");
    }
}

/// \brief Build and commit a plan for \p batch transforms of size \p N.
///
/// \param Q          Queue the transforms will run on.
/// \param batch      Number of transforms.
/// \param N          Transform size (number of real points).
/// \param realStride Distance between successive points of one transform.
/// \param realDist   Distance between successive transforms.
template <typename TReal>
DFTPlan<TReal> MakeDFTPlan([[maybe_unused]] sycl::queue &Q, size_t batch,
                           std::int64_t N, std::int64_t realStride,
                           std::int64_t realDist)
{
    DFTPlan<TReal> plan;
    plan.batch      = batch;
    plan.N          = N;
    plan.halfN      = N / 2;
    plan.realStride = realStride;
    plan.realDist   = realDist;

#if !defined(NEKTAR_SYCL_HAVE_DFT_LIBRARY)
    WarnNaiveTransform("this build configures no oneMath or oneMKL DFT "
                       "backend");
#else

#if defined(NEKTAR_ENABLE_ONEMKL)
    if (realStride != 1)
    {
        // oneMKL will not transform a real side the caller has strided. It
        // rejects such a descriptor at commit; an equivalent one built on a
        // packed copy commits but is then rejected at compute, as is the
        // same descriptor committed against a queue built here rather than
        // the caller's.
        WarnNaiveTransform("oneMKL does not transform a strided real field");
        return plan;
    }
#endif

    // set_value() is variadic, so the strides have to arrive as an array of
    // {offset, stride}: a std::vector cannot be passed through an ellipsis.
    // Both scales are left at 1, which is what makes the pair unnormalised.
    std::int64_t realStrides[2]  = {0, realStride};
    std::int64_t cmplxStrides[2] = {0, 1};

    plan.desc = new DFTDescriptor<TReal>(N);
    plan.desc->set_value(dft::config_param::NUMBER_OF_TRANSFORMS,
                         static_cast<std::int64_t>(batch));
    plan.desc->set_value(dft::config_param::FWD_DISTANCE, realDist);
    plan.desc->set_value(dft::config_param::BWD_DISTANCE, plan.halfN + 1);

#if defined(NEKTAR_ENABLE_ONEMATH)
    // Pinned to the domains, so they hold for both directions.
    plan.desc->set_value(dft::config_param::FWD_STRIDES, realStrides);
    plan.desc->set_value(dft::config_param::BWD_STRIDES, cmplxStrides);
    plan.desc->set_value(dft::config_param::PLACEMENT,
                         dft::config_value::NOT_INPLACE);
#else
    // Relative to the direction rather than the domain, so they would have to
    // be exchanged between the two transforms -- except that only a unit real
    // stride reaches here, which makes both sets the same. oneMKL exposes
    // only the workspace values through config_value, so placement takes the
    // DFTI enumerator.
    plan.desc->set_value(dft::config_param::INPUT_STRIDES, realStrides);
    plan.desc->set_value(dft::config_param::OUTPUT_STRIDES, cmplxStrides);
    plan.desc->set_value(dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
#endif

    try
    {
        plan.desc->commit(Q);
    }
    catch (const std::exception &e)
    {
        NEKERROR(ErrorUtil::efatal,
                 "The DFT library refused a descriptor for N=" +
                     std::to_string(N) + ", batch=" + std::to_string(batch) +
                     ", real-side stride=" + std::to_string(realStride) +
                     ", real-side distance=" + std::to_string(realDist) + ": " +
                     e.what());
    }
#endif

    return plan;
}

/// \brief Release whatever MakeDFTPlan() allocated.
///
/// \param plan Plan to release.
template <typename TReal>
void DestroyDFTPlan([[maybe_unused]] DFTPlan<TReal> &plan)
{
#if defined(NEKTAR_SYCL_HAVE_DFT_LIBRARY)
    delete plan.desc;
    plan.desc = nullptr;
#endif
}

/// \brief Run \p plan forward, real to half-complex.
///
/// \param Q            Queue the transform is submitted to.
/// \param plan         Plan from MakeDFTPlan().
/// \param d_real       Device pointer to the real input.
/// \param d_cmplx      Device pointer to the half-spectrum output.
/// \param dependencies Events the transform waits on.
template <typename TReal>
sycl::event ComputeForward(sycl::queue &Q, const DFTPlan<TReal> &plan,
                           const TReal *d_real, DFTCmplx<TReal> *d_cmplx,
                           const std::vector<sycl::event> &dependencies = {})
{
#if defined(NEKTAR_SYCL_HAVE_DFT_LIBRARY)
    if (plan.desc != nullptr)
    {
        return dft::compute_forward(*plan.desc, const_cast<TReal *>(d_real),
                                    d_cmplx, dependencies);
    }
#endif

    return NaiveForwardKernel(Q, d_real, d_cmplx, plan.batch, plan.N,
                              plan.realStride, plan.realDist, dependencies);
}

/// \brief Run \p plan backward, half-complex to real.
///
/// \param Q            Queue the transform is submitted to.
/// \param plan         Plan from MakeDFTPlan().
/// \param d_cmplx      Device pointer to the half-spectrum input.
/// \param d_real       Device pointer to the real output.
/// \param dependencies Events the transform waits on.
template <typename TReal>
sycl::event ComputeBackward(sycl::queue &Q, const DFTPlan<TReal> &plan,
                            const DFTCmplx<TReal> *d_cmplx, TReal *d_real,
                            const std::vector<sycl::event> &dependencies = {})
{
#if defined(NEKTAR_SYCL_HAVE_DFT_LIBRARY)
    if (plan.desc != nullptr)
    {
        return dft::compute_backward(*plan.desc,
                                     const_cast<DFTCmplx<TReal> *>(d_cmplx),
                                     d_real, dependencies);
    }
#endif

    return NaiveBackwardKernel(Q, d_cmplx, d_real, plan.batch, plan.N,
                               plan.realStride, plan.realDist, dependencies);
}

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_SYCL
