#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class Line : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
protected:
    virtual std::string GetMeshName() override
    {
        return "line.xml";
    }
};

class Square : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
protected:
    virtual std::string GetMeshName() override
    {
        return "square.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(ipwrtbase, Line)
{
    Configure();

    static double *x = fixt_in->GetStorage().GetCPUPtr();

    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t phys = 0; phys < block.num_pts; ++phys)
            {
                *(x++) = phys;
            }
        }
    }

    IProductWRTBase<>::create(fixt_explist, "StdMat")
        ->apply(*fixt_in, *fixt_out);

    static double *y = fixt_out->GetStorage().GetCPUPtr();
    double TOL       = 1e-12;
    BOOST_CHECK_CLOSE(y[0], 1.09753217840612, TOL);
    BOOST_CHECK_CLOSE(y[1], 1.90246782159389, TOL);
    BOOST_CHECK_CLOSE(y[2], 0.5, TOL);
    BOOST_CHECK_CLOSE(y[3], 0.150377322580158, TOL);
    BOOST_TEST(std::abs(y[4] - 9.36750677027476e-17) < TOL);
    BOOST_CHECK_CLOSE(y[5], 0.00746847423694819, TOL);
}
