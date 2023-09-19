#define BOOST_TEST_MODULE example
#include <boost/test/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorPhysDeriv.hpp"
#include "init_physderivfields.hpp"

BOOST_FIXTURE_TEST_CASE(physderivseg, Seg)
{
    Configure(1, 1);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(physderivquad, Quad)
{
    Configure(1, 2);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(1.0E-12);
    }
}

/*BOOST_FIXTURE_TEST_CASE(physderivtri, Tri)
{
    Configure(1, 2);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(1.0E-12);
    }
}*/

/*BOOST_FIXTURE_TEST_CASE(physderivhex, Hex)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-10));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(1.0E-12);
    }
}*/

/*BOOST_FIXTURE_TEST_CASE(physderivprism, Prism)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-06));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(1.0E-12);
    }
}*/

/*BOOST_FIXTURE_TEST_CASE(physderivpyr, Pyr)
{
    Configure(1, 3);
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    PhysDeriv<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-06));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(1.0E-12);
    }
}*/
