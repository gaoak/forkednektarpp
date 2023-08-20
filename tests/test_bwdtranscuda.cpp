#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestBwdTransCUDA
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

BOOST_AUTO_TEST_SUITE(TestBwdTransCUDA)

using namespace std;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class Line : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    Line() : InitFields<double, FieldState::Coeff, FieldState::Phys>()
    {
        meshName = "line.xml";
    }
};

class Square : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    Square() : InitFields<double, FieldState::Coeff, FieldState::Phys>()
    {
        meshName = "square.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(bwdtranscuda, Line)
{
    Configure();

    static double *x =
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();

    for (auto const &block : fixtcuda_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                *(x++) = coeff + 1;
            }
        }
    }

    BwdTrans<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);

    static double *y =
        fixtcuda_out->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();
    double TOL = 1e-12;
    BOOST_CHECK_CLOSE(y[0], 1.000000000000000, TOL);
    BOOST_CHECK_CLOSE(y[1], 0.808463389187877, TOL);
    BOOST_CHECK_CLOSE(y[2], 1.993385866728399, TOL);
    BOOST_CHECK_CLOSE(y[3], 1.312500000000000, TOL);
    BOOST_CHECK_CLOSE(y[4], 2.321846776942122, TOL);
    BOOST_CHECK_CLOSE(y[5], 4.082915537389534, TOL);
    BOOST_CHECK_CLOSE(y[6], 2.000000000000000, TOL);
}

BOOST_AUTO_TEST_SUITE_END()
