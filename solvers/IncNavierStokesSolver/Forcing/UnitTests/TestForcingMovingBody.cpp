///////////////////////////////////////////////////////////////////////////////
//
// File: TestForcingMovingBody.cpp
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
// Description: Unit tests for the cable model of ForcingMovingBody
//
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <vector>

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>

#include <LibUtilities/BasicUtils/Vmath.hpp>
#include <LibUtilities/FFT/NektarFFT.h>

#include "../ForcingMovingBody.h"

namespace Nektar::ForcingMovingBodyTests
{
static const NekDouble tol = 1e-12;

/// Arbitrary, non-symmetric data to transform
static Array<OneD, NekDouble> TestData(size_t N)
{
    Array<OneD, NekDouble> u(N);
    for (size_t k = 0; k < N; ++k)
    {
        u[k] = 0.3 + std::sin(1.7 * k) + 0.1 * k * k;
    }
    return u;
}

// The Pinned-Pinned forward and backward sine transforms are inverses of
// each other, in both orders.
BOOST_AUTO_TEST_CASE(TestPinnedSineRoundTrip)
{
    for (size_t N : {1, 2, 5, 8, 16})
    {
        Array<OneD, NekDouble> u = TestData(N), uhat(N), v(N);

        ForcingMovingBody::PinnedSineFwdTrans(u, uhat);
        ForcingMovingBody::PinnedSineBwdTrans(uhat, v);
        for (size_t k = 0; k < N; ++k)
        {
            BOOST_CHECK_SMALL(v[k] - u[k], tol);
        }

        ForcingMovingBody::PinnedSineBwdTrans(u, v);
        ForcingMovingBody::PinnedSineFwdTrans(v, uhat);
        for (size_t k = 0; k < N; ++k)
        {
            BOOST_CHECK_SMALL(uhat[k] - u[k], tol);
        }
    }
}

// The cable sampled at z_k = (k + 1/2) L / N with the sine mode of
// wavenumber CableWavenumber(m) transforms to slot m only.
BOOST_AUTO_TEST_CASE(TestPinnedSineModes)
{
    size_t N = 8;
    Array<OneD, NekDouble> u(N), uhat(N);

    for (size_t m = 0; m < N; ++m)
    {
        unsigned int K = ForcingMovingBody::CableWavenumber("Pinned-Pinned", m);
        BOOST_CHECK_EQUAL(K, m + 1);

        for (size_t k = 0; k < N; ++k)
        {
            u[k] = std::sin(M_PI * K * (k + 0.5) / N);
        }

        ForcingMovingBody::PinnedSineFwdTrans(u, uhat);
        for (size_t i = 0; i < N; ++i)
        {
            NekDouble expected = 0.0;
            if (i == m)
            {
                expected = (m == N - 1) ? N : 0.5 * N;
            }
            BOOST_CHECK_SMALL(uhat[i] - expected, tol);
        }
    }
}

// Free-Free modes are stored in the layout of the Fourier basis: the cosine
// and sine coefficients of wavenumber k in slots 2k and 2k + 1.
BOOST_AUTO_TEST_CASE(TestFreeFreeWavenumber)
{
    size_t N                           = 8;
    std::vector<unsigned int> expected = {0, 0, 1, 1, 2, 2, 3, 3};

    for (size_t plane = 0; plane < N; ++plane)
    {
        BOOST_CHECK_EQUAL(
            ForcingMovingBody::CableWavenumber("Free-Free", plane),
            expected[plane]);
    }
}

// CableWavenumber matches the layout of the NekFFTW transform that the
// Free-Free model is solved with: cos(2 pi K z / L) and sin(2 pi K z / L)
// transform only to slots whose wavenumber is K, and the Nyquist mode
// K = N / 2 is not represented. Skipped when Nektar++ is built without FFTW,
// in which case the MovingBody forcing cannot be used either.
BOOST_AUTO_TEST_CASE(TestFreeFreeWavenumberFFTW)
{
    if (!LibUtilities::GetNektarFFTFactory().ModuleExists("NekFFTW"))
    {
        BOOST_TEST_MESSAGE("NekFFTW is not available, skipping");
        return;
    }

    size_t N = 8;
    LibUtilities::NektarFFTSharedPtr fft =
        LibUtilities::GetNektarFFTFactory().CreateInstance("NekFFTW", N);

    Array<OneD, NekDouble> u(N), uhat(N);

    for (unsigned int K = 0; K < N / 2; ++K)
    {
        for (NekDouble phase : {0.0, 0.5 * M_PI})
        {
            for (size_t k = 0; k < N; ++k)
            {
                u[k] = std::cos(2.0 * M_PI * K * k / N - phase);
            }

            fft->FFTFwdTrans(u, uhat);

            NekDouble maxInK = 0.0;
            for (size_t plane = 0; plane < N; ++plane)
            {
                if (ForcingMovingBody::CableWavenumber("Free-Free", plane) == K)
                {
                    maxInK = std::max(maxInK, std::abs(uhat[plane]));
                }
                else
                {
                    BOOST_CHECK_SMALL(uhat[plane], tol);
                }
            }

            // sin(0) vanishes at every plane; any other signal must appear
            // in a slot of wavenumber K
            if (Vmath::Vamax(N, u, 1) > 0.5)
            {
                BOOST_CHECK_GT(maxInK, 0.1);
            }
        }
    }

    // The Nyquist mode cos(pi k) transforms to zero in every slot
    for (size_t k = 0; k < N; ++k)
    {
        u[k] = std::cos(M_PI * k);
    }
    fft->FFTFwdTrans(u, uhat);
    for (size_t plane = 0; plane < N; ++plane)
    {
        BOOST_CHECK_SMALL(uhat[plane], tol);
    }
}

} // namespace Nektar::ForcingMovingBodyTests
