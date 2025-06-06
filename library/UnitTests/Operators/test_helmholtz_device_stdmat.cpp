///////////////////////////////////////////////////////////////////////////////
//
// File: test_helmholtz_device_stdmat.cpp
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

#define BOOST_TEST_MODULE TestHelmholtzDeviceStdMat

#include "init_helmholtzfields.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_HELMHOLTZ(test_name, test, tol)                                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure("Device", "StdMat", 2, 2);                                   \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestHelmholtzDeviceStdMat)

TEST_HELMHOLTZ(helmholtz_device_stdmat_seg, Seg, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_seg_sem, SegSEM, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_quad, Quad, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_quad_sem, QuadSEM, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_quad_varp, QuadVarP, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_tri, Tri, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_tri_varp, TriVarP, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_tri_nodal, TriNodal, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_square_all_elements, SquareAllElements,
               1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_hex, Hex, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_hex_sem, HexSEM, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_hex_varp, HexVarP, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_prism, Prism, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_prism_varp, PrismVarP, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_prism_nodal, PrismNodal, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_pyr, Pyr, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_pyr_varp, PyrVarP, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_tet, Tet, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_tet_varp, TetVarP, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_tet_nodal, TetNodal, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_cube_prism_hex, CubePrismHex, 1.0E-12)

TEST_HELMHOLTZ(helmholtz_device_stdmat_cube_all_elements, CubeAllElements,
               1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
