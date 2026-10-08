///////////////////////////////////////////////////////////////////////////////
//
// File: TestDeviceFFT.cpp
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

#include <cmath>

#include <stdexcept>

#include <boost/test/unit_test.hpp>

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/FFT/NekDeviceFFT.h>
#include <LibUtilities/FFT/NektarFFT.h>

namespace Nektar::DeviceFFTUnitTests
{

using namespace Nektar::LibUtilities;

static constexpr double TOL = 1.0e-12;

#if !defined(NEKTAR_ENABLE_SYCL)
// Graph capture needs a non-default stream: stream 0 is the legacy default
// stream, which cannot be captured.
static constexpr unsigned int GRAPH_STREAM_ID = 1;
#endif // !NEKTAR_ENABLE_SYCL

// Signal helpers

static void FillSine(Array<OneD, double> &phys, int N, int k)
{
    const double omega = 2.0 * M_PI * k / N;
    for (int n = 0; n < N; ++n)
        phys[n] = std::sin(omega * n);
}

static void FillCosine(Array<OneD, double> &phys, int N, int k)
{
    const double omega = 2.0 * M_PI * k / N;
    for (int n = 0; n < N; ++n)
        phys[n] = std::cos(omega * n);
}

static void FillSignal(Array<OneD, double> &arr, int N)
{
    const double omega = 2.0 * M_PI / N;
    for (int i = 0; i < static_cast<int>(arr.size()); ++i)
    {
        const int n = i % N;
        arr[i] =
            std::sin(3.0 * omega * n) + 0.5 * std::cos(7.0 * omega * n) + 1.0;
    }
}

static void CheckOtherModesZero(const Array<OneD, double> &coef, int N,
                                int skipReal, int skipImag)
{
    for (int i = 0; i < N; ++i)
    {
        if (i == skipReal || i == skipImag)
        {
            continue;
        }
        BOOST_CHECK_SMALL(coef[i], TOL);
    }
}

// Correctness tests

BOOST_AUTO_TEST_CASE(TestFactoryRegistration)
{
    BOOST_CHECK_NO_THROW(
        GetNektarFFTFactory().CreateInstance("NekDeviceFFT", 16));
}

BOOST_AUTO_TEST_CASE(TestDCSignal)
{
    const int N = 16;
    auto fft    = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, 1);
    Array<OneD, double> phys(N, 3.5), coef(N, 0.0);
    fft->FFTFwdTrans(phys, coef);
    BOOST_CHECK_SMALL(coef[0] - 3.5, TOL);
    CheckOtherModesZero(coef, N, 0, -1);
}

BOOST_AUTO_TEST_CASE(TestSineCosineMode)
{
    for (const int k : {1, 3})
    {
        const int N = 16;
        auto fft    = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, 1);
        Array<OneD, double> phys(N, 0.0), coef(N, 0.0);
        FillSine(phys, N, k);
        fft->FFTFwdTrans(phys, coef);
        BOOST_CHECK_SMALL(coef[2 * k], TOL);
        BOOST_CHECK_SMALL(coef[2 * k + 1] + 1.0, TOL);
        CheckOtherModesZero(coef, N, 2 * k, 2 * k + 1);
    }
    {
        const int N = 16, k = 1;
        auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, 1);
        Array<OneD, double> phys(N, 0.0), coef(N, 0.0);
        FillCosine(phys, N, k);
        fft->FFTFwdTrans(phys, coef);
        BOOST_CHECK_SMALL(coef[2 * k] - 1.0, TOL);
        BOOST_CHECK_SMALL(coef[2 * k + 1], TOL);
        CheckOtherModesZero(coef, N, 2 * k, 2 * k + 1);
    }
}

BOOST_AUTO_TEST_CASE(TestRoundTrip)
{
    for (const int N : {16, 64, 256})
    {
        auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, 1);
        Array<OneD, double> phys(N, 0.0), coef(N, 0.0), recv(N, 0.0);
        FillSignal(phys, N);
        fft->FFTFwdTrans(phys, coef);
        fft->FFTBwdTrans(coef, recv);
        for (int n = 0; n < N; ++n)
            BOOST_CHECK_SMALL(recv[n] - phys[n], TOL);
    }
}

BOOST_AUTO_TEST_CASE(TestRoundTripDeviceResident)
{
    const int N = 64;
    auto fft    = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, 1);
    Array<OneD, double> phys(N, 0.0), recv(N, 0.0);
    FillSignal(phys, N);
    fft->UploadPhys(phys.data());
    fft->FFTFwdTransDevice();
    fft->FFTBwdTransDevice();
    fft->DownloadPhys(recv.data());
    for (int n = 0; n < N; ++n)
        BOOST_CHECK_SMALL(recv[n] - phys[n], TOL);
}

BOOST_AUTO_TEST_CASE(TestBatchedRoundTrip)
{
    for (const int M : {2, 8, 32})
    {
        const int N = 64;
        auto fft    = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, M);
        Array<OneD, double> phys(M * N, 0.0), coef(M * N, 0.0),
            recv(M * N, 0.0);
        FillSignal(phys, N);
        fft->FFTFwdTrans(phys, coef);
        fft->FFTBwdTrans(coef, recv);
        for (int i = 0; i < M * N; ++i)
            BOOST_CHECK_SMALL(recv[i] - phys[i], TOL);
    }
}

BOOST_AUTO_TEST_CASE(TestPlanCacheIsPopulated)
{
    const int N = 64, M = 2;
    {
        auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(N, M);
    }
    int deviceId = Nektar::nekGetDevice();
    BOOST_CHECK(NekDeviceFFT::IsCacheWarmed(deviceId, N, M));
}

// Graph capture is a CUDA and HIP facility. The SYCL backend has no
// equivalent and resubmits the pipeline instead, so the round-trip cases below
// are replaced there by one that pins that behaviour.
#if defined(NEKTAR_ENABLE_SYCL)

BOOST_AUTO_TEST_CASE(TestGraphCaptureUnsupported)
{
    auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(16, 1);
    BOOST_CHECK(!fft->HasGraph());
    BOOST_CHECK_THROW(fft->BeginGraphCapture(), std::runtime_error);
    BOOST_CHECK(!fft->HasGraph());
}

#else

BOOST_AUTO_TEST_CASE(TestGraphCaptureDefaultStream)
{
    auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(16, 1);
    BOOST_CHECK_THROW(fft->BeginGraphCapture(), std::runtime_error);
    BOOST_CHECK(!fft->HasGraph());
}

BOOST_AUTO_TEST_CASE(TestGraphRoundTrip)
{
    for (const int N : {16, 64, 256})
    {
        auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(
            N, 1, GRAPH_STREAM_ID);
        Array<OneD, double> phys(N, 0.0), ref(N, 0.0), recv(N, 0.0);
        FillSignal(phys, N);
        fft->UploadPhys(phys.data());
        fft->FFTFwdTransDevice();
        fft->FFTBwdTransDevice();
        fft->DownloadPhys(ref.data());
        fft->BeginGraphCapture();
        fft->FFTFwdTransDevice();
        fft->FFTBwdTransDevice();
        fft->EndGraphCapture();
        for (int rep = 0; rep < 3; ++rep)
        {
            fft->UploadPhys(phys.data());
            fft->LaunchGraph();
            fft->DownloadPhys(recv.data());
            for (int n = 0; n < N; ++n)
                BOOST_CHECK_SMALL(recv[n] - ref[n], TOL);
        }
    }
}

#endif // NEKTAR_ENABLE_SYCL

// LaunchGraph() without a capture raises the same error on every backend; on
// SYCL that is the only outcome, there being no capture to make.
BOOST_AUTO_TEST_CASE(TestLaunchGraphWithoutCapture)
{
    auto fft = MemoryManager<NekDeviceFFT>::AllocateSharedPtr(16, 1);
    BOOST_CHECK_THROW(fft->LaunchGraph(), std::runtime_error);
}

// Float-precision tests

BOOST_AUTO_TEST_CASE(TestFloatFactoryRegistration)
{
    BOOST_CHECK_NO_THROW(
        GetNektarFFTFloatFactory().CreateInstance("NekDeviceFFT", 16));
}

BOOST_AUTO_TEST_CASE(TestFloatDCSignal)
{
    const int N = 16;
    auto fft    = MemoryManager<NekDeviceFFTFloat>::AllocateSharedPtr(N, 1);
    Array<OneD, float> phys(N, 3.5f), coef(N, 0.0f);
    fft->FFTFwdTrans(phys, coef);
    BOOST_CHECK_SMALL(coef[0] - 3.5f, 1.0e-5f);
    for (int i = 1; i < N; ++i)
        BOOST_CHECK_SMALL(coef[i], 1.0e-5f);
}

BOOST_AUTO_TEST_CASE(TestFloatRoundTrip)
{
    for (const int N : {16, 64, 256})
    {
        auto fft = MemoryManager<NekDeviceFFTFloat>::AllocateSharedPtr(N, 1);
        Array<OneD, float> phys(N, 0.0f), coef(N, 0.0f), recv(N, 0.0f);
        const float omega = 2.0f * static_cast<float>(M_PI) / N;
        for (int i = 0; i < N; ++i)
            phys[i] = std::sin(3.0f * omega * i) +
                      0.5f * std::cos(7.0f * omega * i) + 1.0f;
        fft->FFTFwdTrans(phys, coef);
        fft->FFTBwdTrans(coef, recv);
        for (int n = 0; n < N; ++n)
            BOOST_CHECK_SMALL(recv[n] - phys[n], 1.0e-5f);
    }
}

BOOST_AUTO_TEST_CASE(TestFloatBatchedRoundTrip)
{
    const int N = 64, M = 8;
    auto fft = MemoryManager<NekDeviceFFTFloat>::AllocateSharedPtr(N, M);
    Array<OneD, float> phys(M * N, 0.0f), coef(M * N, 0.0f), recv(M * N, 0.0f);
    const float omega = 2.0f * static_cast<float>(M_PI) / N;
    for (int i = 0; i < M * N; ++i)
    {
        const int n = i % N;
        phys[i]     = std::sin(3.0f * omega * n) +
                  0.5f * std::cos(7.0f * omega * n) + 1.0f;
    }
    fft->FFTFwdTrans(phys, coef);
    fft->FFTBwdTrans(coef, recv);
    for (int i = 0; i < M * N; ++i)
        BOOST_CHECK_SMALL(recv[i] - phys[i], 1.0e-5f);
}

BOOST_AUTO_TEST_CASE(TestFloatSineCosineMode)
{
    const int N = 16, k = 3;
    auto fft = MemoryManager<NekDeviceFFTFloat>::AllocateSharedPtr(N, 1);
    Array<OneD, float> phys(N, 0.0f), coef(N, 0.0f);
    const float omega = 2.0f * static_cast<float>(M_PI) * k / N;
    for (int n = 0; n < N; ++n)
        phys[n] = std::sin(omega * n);
    fft->FFTFwdTrans(phys, coef);
    BOOST_CHECK_SMALL(coef[2 * k], 1.0e-5f);
    BOOST_CHECK_SMALL(coef[2 * k + 1] + 1.0f, 1.0e-5f);
    for (int i = 0; i < N; ++i)
        if (i != 2 * k && i != 2 * k + 1)
            BOOST_CHECK_SMALL(coef[i], 1.0e-5f);
}

BOOST_AUTO_TEST_CASE(TestFloatRoundTripDeviceResident)
{
    const int N = 64;
    auto fft    = MemoryManager<NekDeviceFFTFloat>::AllocateSharedPtr(N, 1);
    Array<OneD, float> phys(N, 0.0f), recv(N, 0.0f);
    const float omega = 2.0f * static_cast<float>(M_PI) / N;
    for (int i = 0; i < N; ++i)
        phys[i] = std::sin(3.0f * omega * i) + 1.0f;
    fft->UploadPhys(phys.data());
    fft->FFTFwdTransDevice();
    fft->FFTBwdTransDevice();
    fft->DownloadPhys(recv.data());
    for (int n = 0; n < N; ++n)
        BOOST_CHECK_SMALL(recv[n] - phys[n], 1.0e-5f);
}

#if !defined(NEKTAR_ENABLE_SYCL)

BOOST_AUTO_TEST_CASE(TestFloatGraphRoundTrip)
{
    const int N = 64;
    auto fft    = MemoryManager<NekDeviceFFTFloat>::AllocateSharedPtr(
        N, 1, GRAPH_STREAM_ID);
    Array<OneD, float> phys(N, 0.0f), ref(N, 0.0f), recv(N, 0.0f);
    const float omega = 2.0f * static_cast<float>(M_PI) / N;
    for (int i = 0; i < N; ++i)
        phys[i] = std::sin(3.0f * omega * i) + 1.0f;
    fft->UploadPhys(phys.data());
    fft->FFTFwdTransDevice();
    fft->FFTBwdTransDevice();
    fft->DownloadPhys(ref.data());
    fft->BeginGraphCapture();
    fft->FFTFwdTransDevice();
    fft->FFTBwdTransDevice();
    fft->EndGraphCapture();
    for (int rep = 0; rep < 3; ++rep)
    {
        fft->UploadPhys(phys.data());
        fft->LaunchGraph();
        fft->DownloadPhys(recv.data());
        for (int n = 0; n < N; ++n)
            BOOST_CHECK_SMALL(recv[n] - ref[n], 1.0e-5f);
    }
}

#endif // !NEKTAR_ENABLE_SYCL

} // namespace Nektar::DeviceFFTUnitTests
