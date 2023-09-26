#define BOOST_TEST_MODULE TestPhysDeriv
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorPhysDeriv.hpp"
#include "init_physderivfields.hpp"

BOOST_FIXTURE_TEST_CASE(physderiv_seg, Seg)
{
    Configure(1, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_quad, Quad)
{
    Configure(1, 2);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_tri, Tri)
{
    Configure(1, 2);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_square_all_elements, SquareAllElements)
{
    Configure(1, 2);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_hex, Hex)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_prism, Prism)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_pyr, Pyr)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    NektarSolution(fixt_expected->GetBlocks(),
                   fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_tet, Tet)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_cube_prism_hex, CubePrismHex)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderiv_cube_all_elements, CubeAllElements)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}
