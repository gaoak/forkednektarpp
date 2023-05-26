#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SpatialDomains/MeshGraph.h>
#include <MultiRegions/ExpList.h>

#include "init_fields.hpp"

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

BOOST_FIXTURE_TEST_CASE( bwdtrans, InitFields )
{
    double *x = fixt_in->GetStorage().GetCPUPtr();

    // For each element, initialise first coefficient to zero and rest
    // to 1.
    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                if (coeff == 0) {
                    *(x++) = 1.0;
                }
                else {
                    *(x++) = 0.0;
                }
            }
        }
    }

    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);

    double *y = fixt_out->GetStorage().GetCPUPtr();
    BOOST_TEST( y[0] == 1.0 );
}
