#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestDirichlet
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorDirBndCond.hpp"
#include "init_dirichletfields.hpp"

BOOST_AUTO_TEST_SUITE(TestDirichlet)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

BOOST_FIXTURE_TEST_CASE(dirichlet2d_p7_allbcs, Helmholtz2D_P7_AllBCs)
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

BOOST_FIXTURE_TEST_CASE(dirichlet3d_hex_allbcs_p6, Helmholtz3D_Hex_AllBCs_P6)
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
