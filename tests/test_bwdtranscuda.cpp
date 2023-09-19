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

BOOST_FIXTURE_TEST_CASE(bwdtranscuda_line, Line)
{
    Configure();

    // Set pointers to initialize menory chunks on both CPU and GPU
    double *x = fixt_in->GetStorage().GetCPUPtr();
    double *y = 
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();

    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                *(x++) = coeff + 1;
                *(y++) = coeff + 1;
            }
        }
    }

    BwdTrans<>::create(fixt_explist, "CUDA")
        ->apply(*fixtcuda_in, *fixtcuda_out);

    // Generate the expected results as reference to be compared
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);

    double TOL = 1e-12;

    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, TOL));
}



BOOST_FIXTURE_TEST_CASE(bwdtranscuda_sqaure, Square)
{
    Configure();

    // Set pointers to initialize menory chunks on both CPU and GPU
    double *x = fixt_in->GetStorage().GetCPUPtr();
    double *y = 
        fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr();

    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                *(x++) = coeff + 1;
                *(y++) = coeff + 1;
            }
        }
    }


    BwdTrans<>::create(fixt_explist, "CUDA")->apply(*fixtcuda_in, *fixtcuda_out);

    // Generate the expected results as reference to be compared
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);

    double TOL = 1e-12;

    BOOST_TEST(fixtcuda_out->compare(*fixt_expected, TOL));
}



BOOST_AUTO_TEST_SUITE_END()
