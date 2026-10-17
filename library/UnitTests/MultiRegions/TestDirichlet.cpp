///////////////////////////////////////////////////////////////////////////////
//
// File: TestDirichlet.cpp
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

#define BOOST_TEST_MODULE TestDirichlet

#include "TestDirichlet.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_DIRICHLET(test_name, test, tol)                                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        if (this->session->GetComm()->GetRank() == 0)                          \
        {                                                                      \
            std::cout << std::string("Run: ") + std::string(#test_name)        \
                      << std::endl;                                            \
        }                                                                      \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_DIRICHLET_UPDATE(test_name, test, tol, time)                      \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        if (this->session->GetComm()->GetRank() == 0)                          \
        {                                                                      \
            std::cout << std::string("Run: ") + std::string(#test_name)        \
                      << std::endl;                                            \
        }                                                                      \
        SetTestCase(time);                                                     \
        RunTestCase(time);                                                     \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestSuiteDirichlet)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_DIRICHLET(dirichlet1d_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_DIRICHLET_UPDATE(dirichlet1d_seg_time_update,
                      Helmholtz1D_Seg_TimeDependentDirichlet, 1.0E-12, 0.75)

TEST_DIRICHLET(dirichlet1d_seg_3c, Helmholtz1D_Seg_3C, 1.0E-12)

TEST_DIRICHLET(dirichlet1d_seg_3c_mixedbc, Helmholtz1D_Seg_3C_mixedBC, 1.0E-12)

TEST_DIRICHLET(dirichlet2d_quad, Helmholtz2D_Quad, 1.0E-12)

TEST_DIRICHLET(dirichlet2d_quad_3c, Helmholtz2D_Quad_3C, 1.0E-12)

TEST_DIRICHLET(dirichlet2d_tri, Helmholtz2D_Tri, 1.0E-12)

TEST_DIRICHLET(dirichlet2d_tri_3c, Helmholtz2D_Tri_3C, 1.0E-12)

TEST_DIRICHLET(dirichlet2d_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-12)

TEST_DIRICHLET_UPDATE(dirichlet2d_tri_quad_time_update,
                      Helmholtz2D_Tri_Quad_TimeDependentDirichlet, 1.0E-12,
                      0.75)

TEST_DIRICHLET(dirichlet2d_tri_quad_3c, Helmholtz2D_Tri_Quad_3C, 1.0E-12)

// For the np=2 partition, vertex 20 is shared by rank 0's boundary region 2
// (Dirichlet for v and w) and rank 1's all-Neumann boundary region 3. This
// exercises resolving both components' shared Dirichlet values across the
// partition boundary, including on the rank without a local Dirichlet edge.
TEST_DIRICHLET(dirichlet2d_tri_quad_3c_mixedbc, Helmholtz2D_Tri_Quad_3C_mixedBC,
               1.0E-12)

TEST_DIRICHLET(dirichlet3d_hex, Helmholtz3D_Hex, 1.0E-12)

TEST_DIRICHLET_UPDATE(dirichlet3d_hex_time_update,
                      Helmholtz3D_Hex_TimeDependentDirichlet, 1.0E-12, 0.75)

TEST_DIRICHLET(dirichlet3d_prism, Helmholtz3D_Prism, 1.0E-12)

TEST_DIRICHLET_UPDATE(dirichlet3d_prism_time_update,
                      Helmholtz3D_Prism_TimeDependentDirichlet, 1.0E-12, 0.75)
TEST_DIRICHLET(dirichlet3d_couette_prism, CouetteFlow3D_Prism, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_pyr, Helmholtz3D_Pyr, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_tet, Helmholtz3D_Tet, 1.0E-12)

TEST_DIRICHLET_UPDATE(dirichlet3d_tet_time_update,
                      Helmholtz3D_Tet_TimeDependentDirichlet, 1.0E-12, 0.75)
TEST_DIRICHLET(dirichlet3d_uniform_tet, UniformFlow3D_Tet, 1.0E-12)
TEST_DIRICHLET(dirichlet3d_channel_tet, ChannelFlow3D_Tet, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_hex_allbcs, Helmholtz3D_Hex_AllBCs, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_hex_3c, Helmholtz3D_Hex_3C, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_hex_3c_mixedbc, Helmholtz3D_Hex_3C_mixedBC, 1.0E-12)
#endif

BOOST_AUTO_TEST_SUITE_END()
