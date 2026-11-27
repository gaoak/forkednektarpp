///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvectionFusedElmt.cpp
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
// Description: Unit test of the fused element-local advection operator against
// the separate operators it replaces, for model Euler policies in one, two and
// three dimensions.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestAdvectionFusedElmt

#include <UnitTests/SolverCore/TestFusedElmt.hpp>

#include <iostream>

TEST_FUSEDELMT_FIXTURE(EulerSeg, EulerModel<1>, false, "run/segment.xml")
TEST_FUSEDELMT_FIXTURE(EulerSegCurved, EulerModel<1>, false,
                       "run/segment_curved.xml")
TEST_FUSEDELMT_FIXTURE(EulerQuadTri, EulerModel<2>, false,
                       "run/square_all_elements.xml")
TEST_FUSEDELMT_FIXTURE(EulerQuadTriCurved, EulerModel<2>, false,
                       "run/square_all_elements_curved.xml")
TEST_FUSEDELMT_FIXTURE(EulerHex, EulerModel<3>, false, "run/hex.xml")
TEST_FUSEDELMT_FIXTURE(EulerPrism, EulerModel<3>, false, "run/prism.xml")
TEST_FUSEDELMT_FIXTURE(EulerPrismCurved, EulerModel<3>, false,
                       "run/prism_curved.xml")
TEST_FUSEDELMT_FIXTURE(EulerPyr, EulerModel<3>, false, "run/pyr.xml")
TEST_FUSEDELMT_FIXTURE(EulerPyrCurved, EulerModel<3>, false,
                       "run/pyr_curved.xml")
TEST_FUSEDELMT_FIXTURE(EulerTet, EulerModel<3>, false, "run/tet.xml")
TEST_FUSEDELMT_FIXTURE(EulerTetCurved, EulerModel<3>, false,
                       "run/tet_curved.xml")

BOOST_AUTO_TEST_SUITE(TestSuiteAdvectionFusedElmt)

TEST_FUSEDELMT_CASE(euler_seg, EulerSeg, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_seg_curved, EulerSegCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_quad_tri, EulerQuadTri, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_quad_tri_curved, EulerQuadTriCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_hex, EulerHex, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_prism, EulerPrism, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_prism_curved, EulerPrismCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_pyr, EulerPyr, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_pyr_curved, EulerPyrCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_tet, EulerTet, 1.0E-10)
TEST_FUSEDELMT_CASE(euler_tet_curved, EulerTetCurved, 1.0E-10)

TEST_FUSEDELMT_DEFAULT_CASE(euler_hex_default, EulerHex)

BOOST_AUTO_TEST_SUITE_END()
