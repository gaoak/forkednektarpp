///////////////////////////////////////////////////////////////////////////////
//
// File: test_fwdtrans.cpp
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

#define BOOST_TEST_MODULE TestFwdTrans

#include "init_fwdtransfields.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_FWDTRANS(test_name, test, tol)                                    \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon = 1;                                              \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        SetTestCase();                                                         \
        RunTestCase("ConjGrad");                                               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestFwdTrans)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_FWDTRANS(fwdtrans_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-08)
TEST_FWDTRANS(fwdtrans_tri_quad_3c, Helmholtz2D_Tri_Quad_3C, 1.0E-08)

TEST_FWDTRANS(fwdtrans_seg, Helmholtz1D_Seg, 2.0E-12)
TEST_FWDTRANS(fwdtrans_all_bcs, Helmholtz2D_AllBCs, 1.0E-08)

TEST_FWDTRANS(fwdtrans_hex, Helmholtz3D_Hex, 5.0E-08)
TEST_FWDTRANS(fwdtrans_hex_3c, Helmholtz3D_Hex_3C, 6.0E-08)

TEST_FWDTRANS(fwdtrans_prism, Helmholtz3D_Prism, 1.0E-08)
TEST_FWDTRANS(fwdtrans_prism_3c, Helmholtz3D_Prism_3C, 1.0E-08)

TEST_FWDTRANS(fwdtrans_pyr, Helmholtz3D_Pyr, 1.0E-08)

TEST_FWDTRANS(fwdtrans_tet, Helmholtz3D_Tet, 5.0E-08)
TEST_FWDTRANS(fwdtrans_tet_3c, Helmholtz3D_Tet_3C, 5.0E-08)

TEST_FWDTRANS(fwdtrans_dg_seg, DGSeg, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_seg_sem, DGSegSEM, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_seg_3d, DGSeg3D, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_quad, DGQuad, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_quad_3d, DGQuad3D, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_quad_varp, DGQuadVarP, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_quad_sem, DGQuadSEM, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_tri, DGTri, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_tri_3d, DGTri3D, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_tri_varp, DGTriVarP, 1.0E-08)
// TEST_FWDTRANS(fwdtrans_dg_tri_nodal, DGTriNodal, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_square_all_elements, DGSquareAllElements, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_hex, DGHex, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_hex_varp, DGHexVarP, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_hex_sem, DGHexSEM, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_prism, DGPrism, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_prism_varp, DGPrismVarP, 1.0E-08)
// TEST_FWDTRANS(fwdtrans_dg_prism_nodal, DGPrismNodal, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_pyr, DGPyr, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_pyr_varp, DGPyrVarP, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_tet, DGTet, 1.0E-08)
TEST_FWDTRANS(fwdtrans_dg_tet_varp, DGTetVarP, 1.0E-08)
// TEST_FWDTRANS(fwdtrans_dg_tet_nodal, DGTetNodal, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_cube_prism_hex, DGCubePrismHex, 1.0E-08)

TEST_FWDTRANS(fwdtrans_dg_cube_all_elements, DGCubeAllElements, 1.0E-08)
#endif

BOOST_AUTO_TEST_SUITE_END()
