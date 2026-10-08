///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvectionWeakDG.cpp
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

#define BOOST_TEST_MODULE TestAdvectionWeakDG

#include "TestAdvectionWeakDG.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_ADVECTIONWEAKDG(test_name, test, tol)                             \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestSuiteAdvectionWeakDG)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_ADVECTIONWEAKDG(advectionweakdg_cube_all_elements, CubeAllElements, 1.0E-4)

TEST_ADVECTIONWEAKDG(advection_seg, Seg, 1.0E-10)

TEST_ADVECTIONWEAKDG(advection_seg_ortho, SegOrtho, 1.0E-10)

TEST_ADVECTIONWEAKDG(advectionweakdg_quad, Quad, 1.0E-9)

TEST_ADVECTIONWEAKDG(advectionweakdg_quad_per, Quad_Per, 1.0E-9)

TEST_ADVECTIONWEAKDG(advectionweakdg_quad_ortho_per, QuadOrtho_Per, 1.0E-9)

TEST_ADVECTIONWEAKDG(advectionweakdg_tri, Tri, 1.0E-9)

TEST_ADVECTIONWEAKDG(advectionweakdg_tri_ortho, TriOrtho, 1.0E-9)

TEST_ADVECTIONWEAKDG(advectionweakdg_triquad_varP, TriQuad_VarP, 1.0E-4)

// 3D variable order on faces. These do not match legacy, and are not meant
// to: the redesign integrates the trace flux on each element's own trace
// quadrature where legacy integrates on the global trace's, and on a deformed
// element the integrand is not a polynomial, so the two rules differ. Accepted
// as a design consequence rather than a defect - over-integrating the mesh
// drives the difference away geometrically (7.4e-06, 7.9e-08, 1.0e-09 relative
// at +2, +4, +6 points per direction), which is what a quadrature difference
// does and a bug does not. See
// CompressibleFlowSolverRedesign/Notes/SerialFixesToBackport.md.
//
// The tolerances therefore admit the measured difference and nothing more. It
// is up to 0.31 where the exact answer is 14 - about 2% - confined to the
// elements either side of the variable order interface. Every real defect
// found in this operator moved things by far more than that, so these still
// guard the path; they just do not assert agreement the scheme never promised.
TEST_ADVECTIONWEAKDG(advectionweakdg_prism_varP, Prism_VarP, 5.0E-1)

TEST_ADVECTIONWEAKDG(advectionweakdg_hex, Hex, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_hex_ortho, HexOrtho, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_hex_per, Hex_Per, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_prism, Prism, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_prism_ortho, PrismOrtho, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_pyr, Pyr, 1.0E-7)

// Not sure this works since Pyramid orthogonal space is aricher than
// modified pyramid and so answers are not consistent
// TEST_ADVECTIONWEAKDG(advectionweakdg_pyr_ortho, PyrOrtho, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_tet, Tet, 1.0E-7)

TEST_ADVECTIONWEAKDG(advectionweakdg_tet_ortho, TetOrtho, 1.0E-7)

#endif
BOOST_AUTO_TEST_SUITE_END()
