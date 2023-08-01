#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestBwdTrans
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

BOOST_AUTO_TEST_SUITE(TestBwdTrans)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;


class Line : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    Line() : InitFields<double, FieldState::Coeff, FieldState::Phys>() {
        meshName = "line.xml";
    }
};

class Square : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    Square() : InitFields<double, FieldState::Coeff, FieldState::Phys>() {
        meshName = "square.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(bwdtrans, Line)
{

    Configure();

    double *x = fixt_in->GetStorage().GetCPUPtr();

    // For each element, initialise first coefficient to zero and rest
    // to 1.
    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                if (coeff == 0)
                {
                    *(x++) = 1.0;
                }
                else
                {
                    *(x++) = 0.0;
                }
            }
        }
    }

    // TODO: Initialise expected solution
    x = fixt_expected->GetStorage().GetCPUPtr();

    // For each element, initialise first coefficient to zero and rest
    // to 1.
    for (auto const &block : fixt_expected->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                if (coeff == 0)
                {
                    *(x++) = 1.0;
                }
                else
                {
                    *(x++) = 0.0;
                }
            }
        }
    }

    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);
    double TOL{0.01};
    BOOST_TEST(fixt_out->compare(*fixt_expected, TOL));
}

BOOST_AUTO_TEST_SUITE_END()
