///////////////////////////////////////////////////////////////////////////////
//
// File: test_helmsolve.cpp
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

#define BOOST_TEST_MODULE TestHelmSolve

#include "init_helmsolve_fields.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_HELMSOLVE_RICH(test_name, test, tol)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("Richardson");                                             \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_CG(test_name, test, tol)                                \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("ConjGrad");                                               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_PCG(test_name, test, tol)                               \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("PipeConjGrad");                                           \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_PCG2(test_name, test, tol)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("PipeConjGrad2");                                          \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_CR(test_name, test, tol)                                \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("ConjRes");                                                \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_PCR(test_name, test, tol)                               \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("PipeConjRes");                                            \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_MINRES(test_name, test, tol)                            \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("MINRES");                                                 \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_GMRES(test_name, test, tol)                             \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysRightPrecon   = 1;                                           \
        int GMRESDeltaDirection = 3;                                           \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        this->session->SetParameter("GMRESDeltaDirection",                     \
                                    GMRESDeltaDirection);                      \
        SetTestCase();                                                         \
        RunTestCase("GMRES");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_HELMSOLVE_FGMRES(test_name, test, tol)                            \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysRightPrecon   = 1;                                           \
        int FlexibleGMRES       = 1;                                           \
        int GMRESDeltaDirection = 0;                                           \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        this->session->SetParameter("FlexibleGMRES", FlexibleGMRES);           \
        this->session->SetParameter("GMRESDeltaDirection",                     \
                                    GMRESDeltaDirection);                      \
        SetTestCase();                                                         \
        RunTestCase("GMRES");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestHelmSolve)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_HELMSOLVE_RICH(helmsolve_rich_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_CG(helmsolve_cg_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_CG(helmsolve_cg_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_CG(helmsolve_cg_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_CG(helmsolve_cg_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_CG(helmsolve_cg_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_PCG(helmsolve_pipe_cg_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_PCG(helmsolve_pipe_cg_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_PCG(helmsolve_pipe_cg_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_PCG(helmsolve_pipe_cg_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_PCG(helmsolve_pipe_cg_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_PCG2(helmsolve_pipe_cg2_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_PCG2(helmsolve_pipe_cg2_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_PCG2(helmsolve_pipe_cg2_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_PCG2(helmsolve_pipe_cg2_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_PCG2(helmsolve_pipe_cg2_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_CR(helmsolve_cr_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_CR(helmsolve_cr_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_CR(helmsolve_cr_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_CR(helmsolve_cr_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_CR(helmsolve_cr_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_PCR(helmsolve_pipe_cr_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_PCR(helmsolve_pipe_cr_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_PCR(helmsolve_pipe_cr_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_PCR(helmsolve_pipe_cr_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_PCR(helmsolve_pipe_cr_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_MINRES(helmsolve_minres_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_MINRES(helmsolve_minres_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_MINRES(helmsolve_minres_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_MINRES(helmsolve_minres_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_MINRES(helmsolve_minres_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_GMRES(helmsolve_gmres_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_GMRES(helmsolve_gmres_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_GMRES(helmsolve_gmres_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_GMRES(helmsolve_gmres_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_GMRES(helmsolve_gmres_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_HELMSOLVE_FGMRES(helmsolve_fgmres_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_HELMSOLVE_FGMRES(helmsolve_fgmres_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-10)

TEST_HELMSOLVE_FGMRES(helmsolve_fgmres_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

TEST_HELMSOLVE_FGMRES(helmsolve_fgmres_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_HELMSOLVE_FGMRES(helmsolve_fgmres_tet, Helmholtz3D_Tet, 1.0E-10)
#endif

BOOST_AUTO_TEST_SUITE_END()
