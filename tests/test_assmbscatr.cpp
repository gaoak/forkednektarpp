#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestAssmbScatr
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorAssmbScatr.hpp"
#include "init_assmbscatrfields.hpp"

BOOST_AUTO_TEST_SUITE(TestAssmbScatr)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

BOOST_FIXTURE_TEST_CASE(assmbscatr_seg, Seg)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_quad, Quad)
{
    Configure();
    boost::test_tools::output_test_stream output;
    {
        SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
        AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
        ExpectedSolution(fixt_expected->GetBlocks(),
                         fixt_expected->GetStorage().GetCPUPtr());
        BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_tri, Tri)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_square_all_elements, SquareAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_hex, Hex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_prism, Prism)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_pyr, Pyr)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_tet, Tet)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_cube_prism_hex, CubePrismHex)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
    ExpectedSolution(fixt_expected->GetBlocks(),
                     fixt_expected->GetStorage().GetCPUPtr());
    BOOST_TEST(fixt_out->compare(*fixt_expected, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixt_out->GetStorage().GetCPUPtr(),
                         fixt_expected->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(assmbscatr_cube_all_elements, CubeAllElements)
{
    Configure();
    SetTestCase(fixt_in->GetBlocks(), fixt_in->GetStorage().GetCPUPtr());
    AssmbScatr<>::create(fixt_explist, "")->apply(*fixt_in, *fixt_out);
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
