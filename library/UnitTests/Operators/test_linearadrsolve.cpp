///////////////////////////////////////////////////////////////////////////////
//
// File: test_linearadrsolve.cpp
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

#define BOOST_TEST_MODULE TestLinearADRSolve

#include "init_linearadrsolve_fields.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_LINEARADRSOLVE_CGS(test_name, test, tol)                          \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("CGS");                                                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_GCR(test_name, test, tol)                          \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("GCR");                                                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_BICGSTAB(test_name, test, tol)                     \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("BICGSTAB");                                               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_BICGSTABR(test_name, test, tol)                    \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("BICGSTABR");                                              \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_BICGSTABL(test_name, test, tol)                    \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        int BICGSTABLLeftPrecon = 0;                                           \
        this->session->SetParameter("BICGSTABLLeftPrecon",                     \
                                    BICGSTABLLeftPrecon);                      \
        SetTestCase();                                                         \
        RunTestCase("BICGSTABL");                                              \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_BICGSTABL2(test_name, test, tol)                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        int BICGSTABLLeftPrecon = 1;                                           \
        this->session->SetParameter("BICGSTABLLeftPrecon",                     \
                                    BICGSTABLLeftPrecon);                      \
        SetTestCase();                                                         \
        RunTestCase("BICGSTABL");                                              \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_GMRES(test_name, test, tol)                        \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("GMRES");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_FGMRES(test_name, test, tol)                       \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        int FlexibleGMRES = 0;                                                 \
        this->session->SetParameter("FlexibleGMRES", FlexibleGMRES);           \
        SetTestCase();                                                         \
        RunTestCase("GMRES");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_MGMRES(test_name, test, tol)                       \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        int ModifiedGramSchmidt = 0;                                           \
        this->session->SetParameter("ModifiedGramSchmidt",                     \
                                    ModifiedGramSchmidt);                      \
        SetTestCase();                                                         \
        RunTestCase("GMRES");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_MFGMRES(test_name, test, tol)                      \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        int FlexibleGMRES       = 0;                                           \
        int ModifiedGramSchmidt = 0;                                           \
        this->session->SetParameter("FlexibleGMRES", FlexibleGMRES);           \
        this->session->SetParameter("ModifiedGramSchmidt",                     \
                                    ModifiedGramSchmidt);                      \
        SetTestCase();                                                         \
        RunTestCase("GMRES");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_TFQMR(test_name, test, tol)                        \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("TFQMR");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_IDRS(test_name, test, tol)                         \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCase("IDRS");                                                   \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestLinearADRSolve)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_seg, Helmholtz1D_Seg, 4.0E-12)

TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_tri_quad, Helmholtz2D_Tri_Quad,
                        2.0E-09)

TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_all_bcs, Helmholtz2D_AllBCs, 4.0E-10)

TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_seg, Helmholtz1D_Seg, 4.0E-12)

TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_tri_quad, Helmholtz2D_Tri_Quad,
                        2.0E-09)

TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)

// TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_seg, Helmholtz1D_Seg,
                             4.0E-12)

TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_tri_quad,
                             Helmholtz2D_Tri_Quad, 2.0E-09)

TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_all_bcs,
                             Helmholtz2D_AllBCs, 4.0E-10)

TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_hex, Helmholtz3D_Hex,
                             1.0E-10)

TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_tet, Helmholtz3D_Tet,
                             1.0E-10)

TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_seg, Helmholtz1D_Seg,
                              4.0E-12)

TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_tri_quad,
                              Helmholtz2D_Tri_Quad, 2.0E-09)

TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_all_bcs,
                              Helmholtz2D_AllBCs, 4.0E-10)

TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_hex, Helmholtz3D_Hex,
                              1.0E-10)

TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_tet, Helmholtz3D_Tet,
                              1.0E-10)

TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_seg, Helmholtz1D_Seg,
                              4.0E-12)

TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_tri_quad,
                              Helmholtz2D_Tri_Quad, 2.0E-09)

TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_all_bcs,
                              Helmholtz2D_AllBCs, 4.0E-10)

TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_hex, Helmholtz3D_Hex,
                              1.0E-10)

TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_tet, Helmholtz3D_Tet,
                              1.0E-10)

TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_seg,
                               Helmholtz1D_Seg, 4.0E-12)

TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_tri_quad,
                               Helmholtz2D_Tri_Quad, 2.0E-09)

TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_all_bcs,
                               Helmholtz2D_AllBCs, 4.0E-10)

TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_hex,
                               Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_tet,
                               Helmholtz3D_Tet, 1.0E-10)

TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_tri_quad, Helmholtz2D_Tri_Quad,
                          1.0E-10)

TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_all_bcs, Helmholtz2D_AllBCs,
                          1.0E-10)

TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_all_bcs, Helmholtz2D_AllBCs,
                           1.0E-10)

TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_all_bcs, Helmholtz2D_AllBCs,
                           1.0E-10)

TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_all_bcs, Helmholtz2D_AllBCs,
                            1.0E-10)

TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_hex, Helmholtz3D_Hex,
                            1.0E-10)

TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqm_rtri_quad, Helmholtz2D_Tri_Quad,
                          1.0E-10)

TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_all_bcs, Helmholtz2D_AllBCs,
                          4.0E-09)

TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_hex, Helmholtz3D_Hex, 1.0E-10)

TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_tet, Helmholtz3D_Tet, 1.0E-10)

TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_tri_quad, Helmholtz2D_Tri_Quad,
                         2.0E-09)

TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_all_bcs, Helmholtz2D_AllBCs,
                         5.0E-10)

TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_hex, Helmholtz3D_Hex, 5.0E-10)

TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_tet, Helmholtz3D_Tet, 5.0E-10)
#endif

BOOST_AUTO_TEST_SUITE_END()
