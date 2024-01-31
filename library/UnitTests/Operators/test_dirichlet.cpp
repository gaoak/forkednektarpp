#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDirichlet
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDirBndCond.hpp"
#include "init_dirichletfields.hpp"

BOOST_AUTO_TEST_SUITE(TestDirichlet)

BOOST_FIXTURE_TEST_CASE(dirichlet1d_seg, Helmholtz1D_Seg)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "");
    DirBndCondOp->apply(*fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet2d_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "");
    DirBndCondOp->apply(*fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3d_hex, Helmholtz3D_Hex)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "");
    DirBndCondOp->apply(*fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3d_prism, Helmholtz3D_Prism)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "");
    DirBndCondOp->apply(*fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3d_pyr, Helmholtz3D_Pyr)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "");
    DirBndCondOp->apply(*fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3d_tet, Helmholtz3D_Tet)
{
    Configure();
    auto DirBndCondOp = DirBndCond<>::create(fixt_explist, "");
    DirBndCondOp->apply(*fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_AUTO_TEST_SUITE_END()
