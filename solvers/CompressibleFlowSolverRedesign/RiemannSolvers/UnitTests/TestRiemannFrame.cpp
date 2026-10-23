///////////////////////////////////////////////////////////////////////////////
//
// File: TestRiemannFrame.cpp
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
// Description: The Riemann solvers in Cartesian components: rotating the
// states and the normal together rotates the flux, and equal states give the
// physical flux through the normal.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Backends/Backends.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "EquationOfState/IdealGasEoS.hpp"
#include "RiemannSolvers/AUSM0SolverKernels.hpp"
#include "RiemannSolvers/AUSM1SolverKernels.hpp"
#include "RiemannSolvers/AUSM2SolverKernels.hpp"
#include "RiemannSolvers/AUSM3SolverKernels.hpp"
#include "RiemannSolvers/AverageSolverKernels.hpp"
#include "RiemannSolvers/HLLCSolverKernels.hpp"
#include "RiemannSolvers/HLLSolverKernels.hpp"
#include "RiemannSolvers/LaxFriedrichsSolverKernels.hpp"
#include "RiemannSolvers/RoeSolverKernels.hpp"

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <random>

namespace Nektar::RiemannFrameUnitTests
{

using Serial = NektarSpaces::Serial;
using EoS    = detail::IdealGasEoS<double>;

static std::mt19937_64 rng(2026);

static double Uniform(const double a, const double b)
{
    return std::uniform_real_distribution<double>(a, b)(rng);
}

/// A random conserved state of density, velocity and pressure in a range
/// spanning subsonic to supersonic flow.
template <unsigned NDIM> static void State(const EoS &eos, double *q)
{
    const double rho = Uniform(0.5, 2.0), p = Uniform(0.5, 2.0);
    const double c = std::sqrt(eos.gamma() * p / rho);
    double ke      = 0.0;
    q[0]           = rho;
    for (unsigned d = 0; d < NDIM; ++d)
    {
        const double u = Uniform(-2.0, 2.0) * c;
        q[1 + d]       = rho * u;
        ke += 0.5 * rho * u * u;
    }
    q[1 + NDIM] = p / (eos.gamma() - 1.0) + ke;
}

/// A random rotation: Gram-Schmidt on a random matrix, each row
/// orthogonalised twice, since one pass loses orthogonality in proportion to
/// the matrix's condition number and a test rotation off by that much would
/// read as a lack of invariance.
template <unsigned NDIM> static void Rotation(double R[3][3])
{
    for (unsigned i = 0; i < NDIM; ++i)
    {
        for (unsigned j = 0; j < NDIM; ++j)
        {
            R[i][j] = Uniform(-1.0, 1.0);
        }
        for (unsigned k = 0; k < 2 * i; ++k)
        {
            double dot = 0.0;
            for (unsigned j = 0; j < NDIM; ++j)
            {
                dot += R[i][j] * R[k % i][j];
            }
            for (unsigned j = 0; j < NDIM; ++j)
            {
                R[i][j] -= dot * R[k % i][j];
            }
        }
        double s = 0.0;
        for (unsigned j = 0; j < NDIM; ++j)
        {
            s += R[i][j] * R[i][j];
        }
        for (unsigned j = 0; j < NDIM; ++j)
        {
            R[i][j] /= std::sqrt(s);
        }
    }
}

/// @p q with its momentum rotated by @p R.
template <unsigned NDIM>
static void Rotate(const double R[3][3], const double *q, double *qr)
{
    qr[0]        = q[0];
    qr[1 + NDIM] = q[1 + NDIM];
    for (unsigned i = 0; i < NDIM; ++i)
    {
        qr[1 + i] = 0.0;
        for (unsigned j = 0; j < NDIM; ++j)
        {
            qr[1 + i] += R[i][j] * q[1 + j];
        }
    }
}

/**
 * For random states and normals: the flux of the rotated states through the
 * rotated normal is the rotated flux, and the flux of a state against itself
 * is its physical flux through the normal.
 */
template <template <typename, typename, unsigned int> typename Kernel,
          unsigned NDIM>
static void CheckFrame()
{
    const EoS eos(1.4, 287.058);
    constexpr unsigned NV = NDIM + 2;
    double worstRot = 0.0, worstCons = 0.0;
    for (int trial = 0; trial < 500; ++trial)
    {
        double l[5], r[5], n[3] = {0, 0, 0}, R[3][3], f[5];
        State<NDIM>(eos, l);
        State<NDIM>(eos, r);
        double s = 0.0;
        for (unsigned d = 0; d < NDIM; ++d)
        {
            n[d] = Uniform(-1.0, 1.0);
            s += n[d] * n[d];
        }
        for (unsigned d = 0; d < NDIM; ++d)
        {
            n[d] /= std::sqrt(s);
        }
        Kernel<Serial, EoS, NDIM>()(eos, 1, l, r, n, f);

        // Rotation invariance.
        if (NDIM > 1)
        {
            Rotation<NDIM>(R);
            double lr[5], rr[5], nr[3] = {0, 0, 0}, fr[5], fRot[5];
            Rotate<NDIM>(R, l, lr);
            Rotate<NDIM>(R, r, rr);
            for (unsigned i = 0; i < NDIM; ++i)
            {
                for (unsigned j = 0; j < NDIM; ++j)
                {
                    nr[i] += R[i][j] * n[j];
                }
            }
            Kernel<Serial, EoS, NDIM>()(eos, 1, lr, rr, nr, fr);
            Rotate<NDIM>(R, f, fRot);
            double scale = 1.0;
            for (unsigned c = 0; c < NV; ++c)
            {
                scale = std::max(scale, std::abs(fRot[c]));
            }
            for (unsigned c = 0; c < NV; ++c)
            {
                worstRot =
                    std::max(worstRot, std::abs(fr[c] - fRot[c]) / scale);
            }
        }

        // Consistency: F(l, l) = F(l) . n.
        double fll[5], fPhys[5];
        Kernel<Serial, EoS, NDIM>()(eos, 1, l, l, n, fll);
        double un = 0.0, ke = 0.0;
        for (unsigned d = 0; d < NDIM; ++d)
        {
            un += l[1 + d] / l[0] * n[d];
            ke += 0.5 * l[1 + d] * l[1 + d] / l[0];
        }
        const double p = (eos.gamma() - 1.0) * (l[1 + NDIM] - ke);
        fPhys[0]       = l[0] * un;
        for (unsigned d = 0; d < NDIM; ++d)
        {
            fPhys[1 + d] = l[1 + d] * un + p * n[d];
        }
        fPhys[1 + NDIM] = (l[1 + NDIM] + p) * un;
        double scale    = 1.0;
        for (unsigned c = 0; c < NV; ++c)
        {
            scale = std::max(scale, std::abs(fPhys[c]));
        }
        for (unsigned c = 0; c < NV; ++c)
        {
            worstCons =
                std::max(worstCons, std::abs(fll[c] - fPhys[c]) / scale);
        }
    }
    BOOST_CHECK_SMALL(worstRot, 1.0e-12);
    BOOST_CHECK_SMALL(worstCons, 1.0e-12);
}

template <template <typename, typename, unsigned int> typename Kernel>
static void CheckFrameAllDims()
{
    CheckFrame<Kernel, 1>();
    CheckFrame<Kernel, 2>();
    CheckFrame<Kernel, 3>();
}

BOOST_AUTO_TEST_CASE(RiemannFrame_AUSM0)
{
    CheckFrameAllDims<detail::AUSM0SolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_AUSM1)
{
    CheckFrameAllDims<detail::AUSM1SolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_AUSM2)
{
    CheckFrameAllDims<detail::AUSM2SolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_AUSM3)
{
    CheckFrameAllDims<detail::AUSM3SolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_Average)
{
    CheckFrameAllDims<detail::AverageSolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_HLL)
{
    CheckFrameAllDims<detail::HLLSolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_HLLC)
{
    CheckFrameAllDims<detail::HLLCSolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_LaxFriedrichs)
{
    CheckFrameAllDims<detail::LaxFriedrichsSolverKernel>();
}
BOOST_AUTO_TEST_CASE(RiemannFrame_Roe)
{
    CheckFrameAllDims<detail::RoeSolverKernel>();
}

} // namespace Nektar::RiemannFrameUnitTests
