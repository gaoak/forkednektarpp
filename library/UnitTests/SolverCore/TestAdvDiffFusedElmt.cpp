///////////////////////////////////////////////////////////////////////////////
//
// File: TestAdvDiffFusedElmt.cpp
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
// Description: Unit test of the fused element-local advection-diffusion
// operator against the separate operators it replaces, for model Navier-Stokes
// policies in one, two and three dimensions.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestAdvDiffFusedElmt

#include <UnitTests/SolverCore/TestFusedElmt.hpp>

#include <iostream>

TEST_FUSEDELMT_FIXTURE(NSSeg, NavierStokesModel<1>, true, "run/segment.xml")
TEST_FUSEDELMT_FIXTURE(NSSegCurved, NavierStokesModel<1>, true,
                       "run/segment_curved.xml")
TEST_FUSEDELMT_FIXTURE(NSQuadTri, NavierStokesModel<2>, true,
                       "run/square_all_elements.xml")
TEST_FUSEDELMT_FIXTURE(NSQuadTriCurved, NavierStokesModel<2>, true,
                       "run/square_all_elements_curved.xml")
TEST_FUSEDELMT_FIXTURE(NSHex, NavierStokesModel<3>, true, "run/hex.xml")
TEST_FUSEDELMT_FIXTURE(NSPrism, NavierStokesModel<3>, true, "run/prism.xml")
TEST_FUSEDELMT_FIXTURE(NSPrismCurved, NavierStokesModel<3>, true,
                       "run/prism_curved.xml")
TEST_FUSEDELMT_FIXTURE(NSPyr, NavierStokesModel<3>, true, "run/pyr.xml")
TEST_FUSEDELMT_FIXTURE(NSPyrCurved, NavierStokesModel<3>, true,
                       "run/pyr_curved.xml")
TEST_FUSEDELMT_FIXTURE(NSTet, NavierStokesModel<3>, true, "run/tet.xml")
TEST_FUSEDELMT_FIXTURE(NSTetCurved, NavierStokesModel<3>, true,
                       "run/tet_curved.xml")

BOOST_AUTO_TEST_SUITE(TestSuiteAdvDiffFusedElmt)

TEST_FUSEDELMT_CASE(ns_seg, NSSeg, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_seg_curved, NSSegCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_quad_tri, NSQuadTri, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_quad_tri_curved, NSQuadTriCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_hex, NSHex, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_prism, NSPrism, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_prism_curved, NSPrismCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_pyr, NSPyr, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_pyr_curved, NSPyrCurved, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_tet, NSTet, 1.0E-10)
TEST_FUSEDELMT_CASE(ns_tet_curved, NSTetCurved, 1.0E-10)

TEST_FUSEDELMT_DEFAULT_CASE(ns_quad_tri_default, NSQuadTri)
TEST_FUSEDELMT_DEFAULT_CASE(ns_hex_default, NSHex)

BOOST_AUTO_TEST_SUITE_END()
