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

#include <LibUtilities/LinearAlgebra/MatrixOperations.hpp>

#include <boost/test/tools/output_test_stream.hpp>
#include <cmath>
#include <iostream>
#include <memory>

namespace
{
struct MatrixDiagnostics
{
    double frobeniusNorm      = 0.0;
    double asymmetryRatio     = 0.0;
    double nonNormalityRatio  = 0.0;
    double conditionEstimateF = 0.0;
};

DNekMat CopyMatrix(const DNekMat &input)
{
    DNekMat result(input.GetRows(), input.GetColumns());

    for (unsigned int i = 0; i < input.GetRows(); ++i)
    {
        for (unsigned int j = 0; j < input.GetColumns(); ++j)
        {
            result(i, j) = input(i, j);
        }
    }

    return result;
}

DNekMat TransposeMatrix(const DNekMat &input)
{
    DNekMat result(input.GetColumns(), input.GetRows());

    for (unsigned int i = 0; i < input.GetRows(); ++i)
    {
        for (unsigned int j = 0; j < input.GetColumns(); ++j)
        {
            result(j, i) = input(i, j);
        }
    }

    return result;
}

double FrobeniusNorm(const DNekMat &input)
{
    double sum = 0.0;

    for (unsigned int i = 0; i < input.GetRows(); ++i)
    {
        for (unsigned int j = 0; j < input.GetColumns(); ++j)
        {
            sum += input(i, j) * input(i, j);
        }
    }

    return std::sqrt(sum);
}

double FrobeniusNormDiff(const DNekMat &lhs, const DNekMat &rhs)
{
    double sum = 0.0;

    for (unsigned int i = 0; i < lhs.GetRows(); ++i)
    {
        for (unsigned int j = 0; j < lhs.GetColumns(); ++j)
        {
            double diff = lhs(i, j) - rhs(i, j);
            sum += diff * diff;
        }
    }

    return std::sqrt(sum);
}

[[maybe_unused]] MatrixDiagnostics DiagnoseMatrix(
    const DNekMatSharedPtr &matrix)
{
    MatrixDiagnostics diagnostics;
    const DNekMat &A = *matrix;
    DNekMat AT       = TransposeMatrix(A);

    diagnostics.frobeniusNorm = FrobeniusNorm(A);
    diagnostics.asymmetryRatio =
        FrobeniusNormDiff(A, AT) / diagnostics.frobeniusNorm;

    DNekMat AAT(A.GetRows(), A.GetRows());
    DNekMat ATA(A.GetRows(), A.GetRows());
    Multiply(AAT, A, AT);
    Multiply(ATA, AT, A);

    diagnostics.nonNormalityRatio =
        FrobeniusNormDiff(AAT, ATA) /
        (diagnostics.frobeniusNorm * diagnostics.frobeniusNorm);

    DNekMat invA = CopyMatrix(A);
    invA.Invert();
    diagnostics.conditionEstimateF =
        diagnostics.frobeniusNorm * FrobeniusNorm(invA);

    return diagnostics;
}
} // namespace

#define TEST_LINEARADRSOLVE_CGS(test_name, test, tol)                          \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 0;                                             \
        int LinSysRightPrecon = 1;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        SetTestCase();                                                         \
        RunTestCase("CGS");                                                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_CGS2(test_name, test, tol)                         \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 1;                                             \
        int LinSysRightPrecon = 0;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon = 1;                                              \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 0;                                             \
        int LinSysRightPrecon = 1;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        SetTestCase();                                                         \
        RunTestCase("BICGSTAB");                                               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_BICGSTAB2(test_name, test, tol)                    \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 1;                                             \
        int LinSysRightPrecon = 0;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 0;                                             \
        int LinSysRightPrecon = 1;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        SetTestCase();                                                         \
        RunTestCase("BICGSTABR");                                              \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_BICGSTABR2(test_name, test, tol)                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 1;                                             \
        int LinSysRightPrecon = 0;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 0;                                             \
        int LinSysRightPrecon = 1;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 1;                                             \
        int LinSysRightPrecon = 0;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon    = 0;                                           \
        int LinSysRightPrecon   = 1;                                           \
        int GMRESDeltaDirection = 3;                                           \
        int ModifiedGramSchmidt = 0;                                           \
        int LinSysMaxStorage    = 10;                                          \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        this->session->SetParameter("GMRESDeltaDirection",                     \
                                    GMRESDeltaDirection);                      \
        this->session->SetParameter("ModifiedGramSchmidt",                     \
                                    ModifiedGramSchmidt);                      \
        this->session->SetParameter("LinSysMaxStorage", LinSysMaxStorage);     \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon    = 0;                                           \
        int LinSysRightPrecon   = 1;                                           \
        int GMRESDeltaDirection = 0;                                           \
        int FlexibleGMRES       = 1;                                           \
        int ModifiedGramSchmidt = 0;                                           \
        int LinSysMaxStorage    = 10;                                          \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        this->session->SetParameter("GMRESDeltaDirection",                     \
                                    GMRESDeltaDirection);                      \
        this->session->SetParameter("FlexibleGMRES", FlexibleGMRES);           \
        this->session->SetParameter("ModifiedGramSchmidt",                     \
                                    ModifiedGramSchmidt);                      \
        this->session->SetParameter("LinSysMaxStorage", LinSysMaxStorage);     \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon    = 0;                                           \
        int LinSysRightPrecon   = 1;                                           \
        int GMRESDeltaDirection = 3;                                           \
        int ModifiedGramSchmidt = 1;                                           \
        int LinSysMaxStorage    = 10;                                          \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        this->session->SetParameter("GMRESDeltaDirection",                     \
                                    GMRESDeltaDirection);                      \
        this->session->SetParameter("ModifiedGramSchmidt",                     \
                                    ModifiedGramSchmidt);                      \
        this->session->SetParameter("LinSysMaxStorage", LinSysMaxStorage);     \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon    = 0;                                           \
        int LinSysRightPrecon   = 1;                                           \
        int GMRESDeltaDirection = 0;                                           \
        int FlexibleGMRES       = 1;                                           \
        int ModifiedGramSchmidt = 1;                                           \
        int LinSysMaxStorage    = 10;                                          \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        this->session->SetParameter("GMRESDeltaDirection",                     \
                                    GMRESDeltaDirection);                      \
        this->session->SetParameter("FlexibleGMRES", FlexibleGMRES);           \
        this->session->SetParameter("ModifiedGramSchmidt",                     \
                                    ModifiedGramSchmidt);                      \
        this->session->SetParameter("LinSysMaxStorage", LinSysMaxStorage);     \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        SetTestCase();                                                         \
        int LinSysLeftPrecon  = 0;                                             \
        int LinSysRightPrecon = 1;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        RunTestCase("TFQMR");                                                  \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_TFQMR2(test_name, test, tol)                       \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 1;                                             \
        int LinSysRightPrecon = 0;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
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
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 0;                                             \
        int LinSysRightPrecon = 1;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        SetTestCase();                                                         \
        RunTestCase("IDRS");                                                   \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#define TEST_LINEARADRSOLVE_IDRS2(test_name, test, tol)                        \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        Configure();                                                           \
        int LinSysLeftPrecon  = 1;                                             \
        int LinSysRightPrecon = 0;                                             \
        this->session->SetParameter("LinSysLeftPrecon", LinSysLeftPrecon);     \
        this->session->SetParameter("LinSysRightPrecon", LinSysRightPrecon);   \
        SetTestCase();                                                         \
        RunTestCase("IDRS");                                                   \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestLinearADRSolve)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_seg_3c, Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_tri_quad_3c, Helmholtz2D_Tri_Quad_3C,
                        2.0E-09)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_all_bcs, Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_hex_3c, Helmholtz3D_Hex_3C, 7.0E-08)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_seg, Helmholtz1D_Seg, 4.0E-12)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_tri_quad, Helmholtz2D_Tri_Quad,
                        2.0E-09)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_hex, Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_CGS(linearadrsolve_cgs_tet, Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_seg_3c,
                         Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_tri_quad_3c,
                         Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_all_bcs,
                         Helmholtz2D_AllBCs, 2.0E-09)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_hex_3c,
                         Helmholtz3D_Hex_3C, 7.0E-08)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_seg, Helmholtz1D_Seg,
                         4.0E-12)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_tri_quad,
                         Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_hex, Helmholtz3D_Hex,
                         4.0E-10)
TEST_LINEARADRSOLVE_CGS2(linearadrsolve_cgs_left_precon_tet, Helmholtz3D_Tet,
                         1.0E-10)
#endif

TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_seg_3c, Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_tri_quad_3c, Helmholtz2D_Tri_Quad_3C,
                        2.0E-09)
TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_all_bcs, Helmholtz2D_AllBCs, 1.0E-10)
// TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_hex_3c,
// Helmholtz3D_Hex_3C, 1.0E-10)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_seg, Helmholtz1D_Seg, 4.0E-12)
TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_tri_quad, Helmholtz2D_Tri_Quad,
                        2.0E-09)
// TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_hex, Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_GCR(linearadrsolve_gcr_tet, Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_seg_3c, Helmholtz1D_Seg_3C,
                             4.0E-12)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_tri_quad_3c,
                             Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_all_bcs,
                             Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_hex_3c, Helmholtz3D_Hex_3C,
                             2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_seg, Helmholtz1D_Seg,
                             4.0E-12)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_tri_quad,
                             Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_hex, Helmholtz3D_Hex,
                             1.0E-10)
TEST_LINEARADRSOLVE_BICGSTAB(linearadrsolve_bicgstab_tet, Helmholtz3D_Tet,
                             1.0E-10)
#endif

BOOST_FIXTURE_TEST_CASE(linearadrsolve_hex_3c_single_matrix_diagnostics,
                        Helmholtz3D_Hex_3C_Single)
{
    std::cout << "Run: linearadrsolve_hex_3c_single_matrix_diagnostics"
              << std::endl;
    Configure();
    SetTestCase();

    auto advectedMatrix = BuildGlobalADRMatrix("u", m_vel);
    Array<OneD, double> zeroVel(m_vel.size(), 0.0);
    auto diffusiveMatrix = BuildGlobalADRMatrix("u", zeroVel);

    auto advectedDiagnostics  = DiagnoseMatrix(advectedMatrix);
    auto diffusiveDiagnostics = DiagnoseMatrix(diffusiveMatrix);

    std::cout << "single-hex ADR matrix diagnostics: asym="
              << advectedDiagnostics.asymmetryRatio
              << ", non-normal=" << advectedDiagnostics.nonNormalityRatio
              << ", cond_F=" << advectedDiagnostics.conditionEstimateF
              << ", diffusion-only cond_F="
              << diffusiveDiagnostics.conditionEstimateF << std::endl;

    BOOST_TEST(diffusiveDiagnostics.asymmetryRatio < 1.0e-12);
    BOOST_TEST(diffusiveDiagnostics.nonNormalityRatio < 1.0e-12);
    BOOST_TEST(advectedDiagnostics.asymmetryRatio > 1.0e-2);
    BOOST_TEST(advectedDiagnostics.nonNormalityRatio > 1.0e-2);
    BOOST_TEST(std::isfinite(advectedDiagnostics.conditionEstimateF));
    BOOST_TEST(std::isfinite(diffusiveDiagnostics.conditionEstimateF));
}

TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_seg_3c,
                              Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_tri_quad_3c,
                              Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_all_bcs,
                              Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_hex_3c,
                              Helmholtz3D_Hex_3C, 2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_seg,
                              Helmholtz1D_Seg, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_tri_quad,
                              Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_hex,
                              Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_BICGSTAB2(linearadrsolve_bicgstab_left_precon_tet,
                              Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_seg_3c,
                              Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_tri_quad_3c,
                              Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_all_bcs,
                              Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_hex_3c,
                              Helmholtz3D_Hex_3C, 2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_seg, Helmholtz1D_Seg,
                              4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_tri_quad,
                              Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_hex, Helmholtz3D_Hex,
                              1.0E-10)
TEST_LINEARADRSOLVE_BICGSTABR(linearadrsolve_bicgstabr_tet, Helmholtz3D_Tet,
                              1.0E-10)
#endif

TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_seg_3c,
                               Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_tri_quad_3c,
                               Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_all_bcs,
                               Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_hex_3c,
                               Helmholtz3D_Hex_3C, 2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_seg,
                               Helmholtz1D_Seg, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_tri_quad,
                               Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_hex,
                               Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_BICGSTABR2(linearadrsolve_bicgstabr_left_precon_tet,
                               Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_seg_3c,
                              Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_tri_quad_3c,
                              Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_all_bcs,
                              Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_hex_3c,
                              Helmholtz3D_Hex_3C, 2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_seg, Helmholtz1D_Seg,
                              4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_tri_quad,
                              Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_hex, Helmholtz3D_Hex,
                              1.0E-10)
TEST_LINEARADRSOLVE_BICGSTABL(linearadrsolve_bicgstabl_tet, Helmholtz3D_Tet,
                              1.0E-10)
#endif

TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_seg_3c,
                               Helmholtz1D_Seg_3C, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_tri_quad_3c,
                               Helmholtz2D_Tri_Quad_3C, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_all_bcs,
                               Helmholtz2D_AllBCs, 4.0E-10)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_hex_3c,
                               Helmholtz3D_Hex_3C, 2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_seg,
                               Helmholtz1D_Seg, 4.0E-12)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_tri_quad,
                               Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_hex,
                               Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_BICGSTABL2(linearadrsolve_bicgstabl_left_precon_tet,
                               Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_seg_3c, Helmholtz1D_Seg_3C,
                          1.0E-12)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_tri_quad_3c,
                          Helmholtz2D_Tri_Quad_3C, 1.0E-10)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_all_bcs, Helmholtz2D_AllBCs,
                          1.0E-10)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_hex_3c, Helmholtz3D_Hex_3C,
                          2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_seg, Helmholtz1D_Seg, 1.0E-12)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_tri_quad, Helmholtz2D_Tri_Quad,
                          1.0E-10)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_hex, Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_GMRES(linearadrsolve_gmres_tet, Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_all_bcs, Helmholtz2D_AllBCs,
                           1.0E-10)
TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_tri_quad_3c,
                           Helmholtz2D_Tri_Quad_3C, 1.0E-10)
TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_hex_3c, Helmholtz3D_Hex_3C,
                           2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_tri_quad, Helmholtz2D_Tri_Quad,
                           1.0E-10)
TEST_LINEARADRSOLVE_FGMRES(linearadrsolve_fgmres_hex, Helmholtz3D_Hex, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_all_bcs, Helmholtz2D_AllBCs,
                           1.0E-10)
TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_tri_quad_3c,
                           Helmholtz2D_Tri_Quad_3C, 1.0E-10)
TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_hex_3c, Helmholtz3D_Hex_3C,
                           2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_tri_quad, Helmholtz2D_Tri_Quad,
                           1.0E-10)
TEST_LINEARADRSOLVE_MGMRES(linearadrsolve_mgmres_hex, Helmholtz3D_Hex, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_all_bcs, Helmholtz2D_AllBCs,
                            1.0E-10)
TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_tri_quad_3c,
                            Helmholtz2D_Tri_Quad_3C, 1.0E-10)
TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_hex_3c, Helmholtz3D_Hex_3C,
                            2.0E-09)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_tri_quad,
                            Helmholtz2D_Tri_Quad, 1.0E-10)
TEST_LINEARADRSOLVE_MFGMRES(linearadrsolve_mfgmres_hex, Helmholtz3D_Hex,
                            1.0E-10)
#endif

TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_seg_3c, Helmholtz1D_Seg_3C,
                          1.0E-12)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_tri_quad_3c,
                          Helmholtz2D_Tri_Quad_3C, 2.0E-10)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_all_bcs, Helmholtz2D_AllBCs,
                          4.0E-09)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_hex_3c, Helmholtz3D_Hex_3C,
                          2.0E-08)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_seg, Helmholtz1D_Seg, 1.0E-12)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_tri_quad, Helmholtz2D_Tri_Quad,
                          3.0E-10)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_hex, Helmholtz3D_Hex, 1.0E-10)
TEST_LINEARADRSOLVE_TFQMR(linearadrsolve_tfqmr_tet, Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_seg_3c,
                           Helmholtz1D_Seg_3C, 1.0E-12)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_tri_quad_3c,
                           Helmholtz2D_Tri_Quad_3C, 2.0E-08)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_all_bcs,
                           Helmholtz2D_AllBCs, 4.0E-09)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_hex_3c,
                           Helmholtz3D_Hex_3C, 2.0E-08)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_seg,
                           Helmholtz1D_Seg, 1.0E-12)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_tri_quad,
                           Helmholtz2D_Tri_Quad, 2.0E-08)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_hex,
                           Helmholtz3D_Hex, 5.0E-10)
TEST_LINEARADRSOLVE_TFQMR2(linearadrsolve_tfqmr_left_precon_tet,
                           Helmholtz3D_Tet, 1.0E-10)
#endif

TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_seg_3c, Helmholtz1D_Seg_3C,
                         5.0E-11)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_tri_quad_3c,
                         Helmholtz2D_Tri_Quad_3C, 4.0E-09)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_all_bcs, Helmholtz2D_AllBCs,
                         1.0E-09)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_hex_3c, Helmholtz3D_Hex_3C,
                         2.0E-08)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_seg, Helmholtz1D_Seg, 1.0E-12)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_tri_quad, Helmholtz2D_Tri_Quad,
                         2.0E-09)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_hex, Helmholtz3D_Hex, 5.0E-10)
TEST_LINEARADRSOLVE_IDRS(linearadrsolve_idrs_tet, Helmholtz3D_Tet, 4.0E-08)
#endif

TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_seg_3c,
                          Helmholtz1D_Seg_3C, 1.0E-11)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_tri_quad_3c,
                          Helmholtz2D_Tri_Quad_3C, 4.0E-09)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_all_bcs,
                          Helmholtz2D_AllBCs, 1.0E-09)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_hex_3c,
                          Helmholtz3D_Hex_3C, 2.0E-08)
#if defined(NEKTAR_TEST_DEBUG)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_seg, Helmholtz1D_Seg,
                          1.0E-12)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_tri_quad,
                          Helmholtz2D_Tri_Quad, 2.0E-09)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_hex, Helmholtz3D_Hex,
                          5.0E-10)
TEST_LINEARADRSOLVE_IDRS2(linearadrsolve_idrs_left_precon_tet, Helmholtz3D_Tet,
                          5.0E-10)
#endif
#endif

BOOST_AUTO_TEST_SUITE_END()
