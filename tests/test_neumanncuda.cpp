#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestNeumannCUDA
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorNeuBndCond.hpp"
#include "init_dirichletfields.hpp"

BOOST_AUTO_TEST_SUITE(TestNeumann)

BOOST_FIXTURE_TEST_CASE(dirichlet1dcuda_seg, Helmholtz1D_Seg)
{
    Configure();
    NeuBndCond<>::create(fixt_explist)->apply(*fixt_out);
    NeuBndCond<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_out->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet2dcuda_tri_quad, Helmholtz2D_Tri_Quad)
{
    Configure();
    NeuBndCond<>::create(fixt_explist)->apply(*fixt_out);
    NeuBndCond<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_out->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_hex, Helmholtz3D_Hex)
{
    Configure();
    NeuBndCond<>::create(fixt_explist)->apply(*fixt_out);
    NeuBndCond<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_out->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_prism, Helmholtz3D_Prism)
{
    Configure();
    NeuBndCond<>::create(fixt_explist)->apply(*fixt_out);
    NeuBndCond<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_out->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_pyr, Helmholtz3D_Pyr)
{
    Configure();
    NeuBndCond<>::create(fixt_explist)->apply(*fixt_out);
    NeuBndCond<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_out->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_FIXTURE_TEST_CASE(dirichlet3dcuda_tet, Helmholtz3D_Tet)
{
    Configure();
    NeuBndCond<>::create(fixt_explist)->apply(*fixt_out);
    NeuBndCond<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_out);
    BOOST_TEST(fixtcuda_out->compare(*fixt_out, 1.0E-12));
    boost::test_tools::output_test_stream output;
    {
        OutputIfNotMatch(fixtcuda_out->GetStorage().GetCPUPtr(),
                         fixt_out->GetStorage().GetCPUPtr(), 1.0E-12);
    }
}

BOOST_AUTO_TEST_SUITE_END()
