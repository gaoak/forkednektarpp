///////////////////////////////////////////////////////////////////////////////
//
// File: TestPyrExactMass.cpp
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
// Description: The orthogonal pyramid on its default rule: a diagonal mass
// matrix, the polynomials of total degree P and the modified pyramid in its
// span, and polynomial traces of degree P on its triangular faces.
//
///////////////////////////////////////////////////////////////////////////////

#include <StdRegions/StdPyrExp.h>
#include <array>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>

namespace Nektar::StdRegionsUnitTests
{

using namespace LibUtilities;

/// The orthogonal pyramid of order @p P on @p np points per direction of
/// the default orthogonal rule: Gauss-Legendre, Gauss-Legendre and
/// Gauss-Jacobi(1,0) in the collapsed direction.
static StdRegions::StdPyrExpSharedPtr MakeOrthoPyr(const unsigned int P,
                                                   const unsigned int np)
{
    const BasisKey b0(eOrtho_A, P + 1, PointsKey(np, eGaussGaussLegendre));
    const BasisKey b2(eOrthoPyr_C, P + 1,
                      PointsKey(np, eGaussGaussAlpha1Beta0));
    return MemoryManager<StdRegions::StdPyrExp>::AllocateSharedPtr(b0, b0, b2);
}

/// Solve the dense system @p A x = @p b in place by Gaussian elimination
/// with partial pivoting; @p A is row major, @p n by @p n.
static std::vector<NekDouble> Solve(std::vector<NekDouble> A,
                                    std::vector<NekDouble> b,
                                    const unsigned int n)
{
    for (unsigned int k = 0; k < n; ++k)
    {
        unsigned int piv = k;
        for (unsigned int i = k + 1; i < n; ++i)
        {
            if (std::abs(A[i * n + k]) > std::abs(A[piv * n + k]))
            {
                piv = i;
            }
        }
        for (unsigned int j = 0; j < n; ++j)
        {
            std::swap(A[k * n + j], A[piv * n + j]);
        }
        std::swap(b[k], b[piv]);
        for (unsigned int i = k + 1; i < n; ++i)
        {
            const NekDouble f = A[i * n + k] / A[k * n + k];
            for (unsigned int j = k; j < n; ++j)
            {
                A[i * n + j] -= f * A[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    for (unsigned int k = n; k-- > 0;)
    {
        for (unsigned int j = k + 1; j < n; ++j)
        {
            b[k] -= A[k * n + j] * b[j];
        }
        b[k] /= A[k * n + k];
    }
    return b;
}

static DNekMatSharedPtr Mass(const StdRegions::StdPyrExpSharedPtr &e)
{
    return e->GetStdMatrix(
        StdRegions::StdMatrixKey(StdRegions::eMass, ePyramid, *e));
}

/**
 * With the default NUMPOINTS = NUMMODES the mass matrix is diagonal, and its
 * diagonal is that of the mass matrix on an over-integrated rule: every mode
 * has degree at most P in the collapsed direction, so the products the rule
 * integrates have degree at most 2P + 1, its exactness.
 */
BOOST_AUTO_TEST_CASE(TestPyrOrthoMassDiagonal)
{
    for (unsigned int P = 1; P <= 8; ++P)
    {
        const auto m         = Mass(MakeOrthoPyr(P, P + 1));
        const auto ref       = Mass(MakeOrthoPyr(P, 3 * P + 6));
        const unsigned int n = m->GetRows();
        NekDouble dmax       = 0.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            dmax = std::max(dmax, std::abs(ref->GetValue(i, i)));
        }
        NekDouble off = 0.0, diag = 0.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            for (unsigned int j = 0; j < n; ++j)
            {
                if (i == j)
                {
                    diag = std::max(diag, std::abs(m->GetValue(i, i) -
                                                   ref->GetValue(i, i)));
                }
                else
                {
                    off = std::max(off, std::abs(m->GetValue(i, j)));
                }
            }
        }
        BOOST_TEST_INFO("P = " << P);
        BOOST_CHECK_SMALL(off / dmax, 1.0e-13);
        BOOST_CHECK_SMALL(diag / dmax, 1.0e-13);
    }
}

/// Every monomial x^a y^b z^c of total degree at most P is in the span: its
/// projection reproduces it at the quadrature points.
BOOST_AUTO_TEST_CASE(TestPyrOrthoPolynomials)
{
    for (unsigned int P = 1; P <= 6; ++P)
    {
        const auto e          = MakeOrthoPyr(P, P + 3);
        const unsigned int nq = e->GetTotPoints();
        Array<OneD, NekDouble> x(nq), y(nq), z(nq), f(nq), c(e->GetNcoeffs()),
            g(nq);
        e->GetCoords(x, y, z);
        for (unsigned int a = 0; a <= P; ++a)
        {
            for (unsigned int b = 0; a + b <= P; ++b)
            {
                for (unsigned int k = 0; a + b + k <= P; ++k)
                {
                    for (unsigned int i = 0; i < nq; ++i)
                    {
                        f[i] = std::pow(x[i], a) * std::pow(y[i], b) *
                               std::pow(z[i], k);
                    }
                    e->FwdTrans(f, c);
                    e->BwdTrans(c, g);
                    NekDouble err = 0.0;
                    for (unsigned int i = 0; i < nq; ++i)
                    {
                        err = std::max(err, std::abs(g[i] - f[i]));
                    }
                    BOOST_TEST_INFO("P = " << P << ", x^" << a << " y^" << b
                                           << " z^" << k);
                    BOOST_CHECK_SMALL(err, 1.0e-11);
                }
            }
        }
    }
}

/**
 * On the triangular face y = -1 every mode is a polynomial of total degree at
 * most P in (x, z), as on a tetrahedron's face. Each mode is interpolated
 * from a degree-P fit on a lattice of the face and checked at other face
 * points.
 */
BOOST_AUTO_TEST_CASE(TestPyrOrthoTriFaceTrace)
{
    for (unsigned int P = 1; P <= 6; ++P)
    {
        // Enough points that evaluating a mode from its values is exact.
        const auto e          = MakeOrthoPyr(P, 2 * P + 3);
        const unsigned int nq = e->GetTotPoints();
        const unsigned int nm = e->GetNcoeffs();

        // A lattice of the face shrunk about its centroid, so the apex is
        // not sampled, and the degree-P monomials in (x, z).
        const NekDouble shrink = 0.8, cx = -1.0 / 3.0, cz = -1.0 / 3.0;
        std::vector<std::array<NekDouble, 2>> lattice, check;
        for (unsigned int i = 0; i <= P; ++i)
        {
            for (unsigned int k = 0; i + k <= P; ++k)
            {
                const NekDouble lx = -1.0 + 2.0 * i / P;
                const NekDouble lz = -1.0 + 2.0 * k / P;
                lattice.push_back(
                    {cx + shrink * (lx - cx), cz + shrink * (lz - cz)});
            }
        }
        for (const auto &st : std::vector<std::array<NekDouble, 2>>{
                 {0.13, 0.21}, {0.37, 0.11}, {0.05, 0.60}, {0.41, 0.33}})
        {
            check.push_back({-1.0 + 2.0 * st[0], -1.0 + 2.0 * st[1]});
        }
        const unsigned int nl = lattice.size();
        auto monomial         = [P](const std::array<NekDouble, 2> &pt,
                            const unsigned int j) {
            unsigned int cnt = 0;
            for (unsigned int a = 0; a <= P; ++a)
            {
                for (unsigned int k = 0; a + k <= P; ++k, ++cnt)
                {
                    if (cnt == j)
                    {
                        return std::pow(pt[0], a) * std::pow(pt[1], k);
                    }
                }
            }
            return 0.0;
        };
        std::vector<NekDouble> V(nl * nl);
        for (unsigned int i = 0; i < nl; ++i)
        {
            for (unsigned int j = 0; j < nl; ++j)
            {
                V[i * nl + j] = monomial(lattice[i], j);
            }
        }

        Array<OneD, NekDouble> phys(nq), coord(3);
        for (unsigned int m = 0; m < nm; ++m)
        {
            e->FillMode(m, phys);
            auto eval = [&](const std::array<NekDouble, 2> &pt) {
                coord[0] = pt[0];
                coord[1] = -1.0;
                coord[2] = pt[1];
                return e->PhysEvaluate(coord, phys);
            };
            std::vector<NekDouble> vals(nl);
            for (unsigned int i = 0; i < nl; ++i)
            {
                vals[i] = eval(lattice[i]);
            }
            const std::vector<NekDouble> coef = Solve(V, vals, nl);
            NekDouble err = 0.0, scale = 1.0;
            for (const auto &pt : check)
            {
                NekDouble fit = 0.0;
                for (unsigned int j = 0; j < nl; ++j)
                {
                    fit += coef[j] * monomial(pt, j);
                }
                const NekDouble v = eval(pt);
                err               = std::max(err, std::abs(fit - v));
                scale             = std::max(scale, std::abs(v));
            }
            BOOST_TEST_INFO("P = " << P << ", mode " << m);
            BOOST_CHECK_SMALL(err / scale, 1.0e-9);
        }
    }
}

/**
 * The modified (C0) pyramid against the orthogonal span, which spectral
 * vanishing viscosity and other modal operations transform to. Its five
 * vertex modes, rational functions of (x, y, z), lie in the span for every
 * order, where with the old p+q exponent four of them did not even at P = 1.
 * Up to P = 2 every modified mode lies in it. From P = 3 the modified
 * quadrilateral-face and interior modes whose exponent p+q-2 exceeds
 * max(p,q) do not; they carry the same mismatch the orthogonal pyramid had.
 */
BOOST_AUTO_TEST_CASE(TestPyrModifiedInOrthoSpan)
{
    for (unsigned int P = 1; P <= 6; ++P)
    {
        // One grid for both, rich enough for either space.
        const unsigned int np = P + 3;
        const PointsKey gll(np, eGaussLobattoLegendre);
        const PointsKey grj(np, eGaussRadauMAlpha2Beta0);
        const auto mod =
            MemoryManager<StdRegions::StdPyrExp>::AllocateSharedPtr(
                BasisKey(eModified_A, P + 1, gll),
                BasisKey(eModified_A, P + 1, gll),
                BasisKey(eModifiedPyr_C, P + 1, grj));
        const auto orth =
            MemoryManager<StdRegions::StdPyrExp>::AllocateSharedPtr(
                BasisKey(eOrtho_A, P + 1, gll), BasisKey(eOrtho_A, P + 1, gll),
                BasisKey(eOrthoPyr_C, P + 1, grj));
        BOOST_REQUIRE_EQUAL(mod->GetNcoeffs(), orth->GetNcoeffs());

        std::vector<unsigned int> modes;
        if (P <= 2)
        {
            for (unsigned int m = 0; m < mod->GetNcoeffs(); ++m)
            {
                modes.push_back(m);
            }
        }
        else
        {
            for (unsigned int v = 0; v < 5; ++v)
            {
                modes.push_back(mod->GetVertexMap(v));
            }
        }

        const unsigned int nq = mod->GetTotPoints();
        Array<OneD, NekDouble> f(nq), c(orth->GetNcoeffs()), g(nq);
        for (const unsigned int m : modes)
        {
            mod->FillMode(m, f);
            orth->FwdTrans(f, c);
            orth->BwdTrans(c, g);
            NekDouble err = 0.0, scale = 0.0;
            for (unsigned int i = 0; i < nq; ++i)
            {
                err   = std::max(err, std::abs(g[i] - f[i]));
                scale = std::max(scale, std::abs(f[i]));
            }
            BOOST_TEST_INFO("P = " << P << ", modified mode " << m);
            BOOST_CHECK_SMALL(err / scale, 1.0e-11);
        }
    }
}

} // namespace Nektar::StdRegionsUnitTests
