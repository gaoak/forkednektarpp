#define BOOST_TEST_MODULE example
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <memory>

#include "Field.hpp"
#include "Operators/OperatorBwdTrans.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraph.h>

#include "init_fields.hpp"

BOOST_AUTO_TEST_SUITE(TestBwdTransMatFree)

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

class SquareAllElements
    : public InitFields<double, FieldState::Coeff, FieldState::Phys>
{
public:
    SquareAllElements()
        : InitFields<double, FieldState::Coeff, FieldState::Phys>()
    {
        meshName = "square_all_elements.xml";
    }
};

BOOST_FIXTURE_TEST_CASE(line, Line)
{

    Configure();
    // initialize the expected output
    double *exptr = fixt_expected->GetStorage().GetCPUPtr();

    // initialize input to the operator
    double *x = fixt_in->GetStorage().GetCPUPtr();
    x[0]      = 36;
    x[1]      = -6;
    x[2]      = -84;
    x[3]      = 90;
    x[4]      = -42;
    x[5]      = 49.5;

    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // apply MatFree implementation of the BwdTrans operator
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    fixt_out->ReshapeStorage<1>();

    double TOL = 1e-12;
    BOOST_TEST(fixt_out->compare(*fixt_expected, TOL));
}

BOOST_FIXTURE_TEST_CASE(square, Square)
{

    Configure();

    // Set up the input field (fixt_in, fixt_explist)
    double *x = fixt_in->GetStorage().GetCPUPtr();
    int id    = 0;
    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++id)
            {
                *(x++)                           = coeff;
                fixt_explist->UpdateCoeffs()[id] = coeff;
            }
        }
        for (size_t el = 0; el < block.num_padding_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                x++;
            }
        }
    }

    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();

    // compare
    double TOL = 1e-12;
    BOOST_TEST(fixt_out->compare(*fixt_expected, TOL));
}

BOOST_FIXTURE_TEST_CASE(square_all_elements, SquareAllElements)
{

    Configure();

    // Set up the input field (fixt_in, fixt_explist)
    double *x = fixt_in->GetStorage().GetCPUPtr();
    int id    = 0;
    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++id)
            {
                *(x++)                           = coeff;
                fixt_explist->UpdateCoeffs()[id] = coeff;
            }
        }
        for (size_t el = 0; el < block.num_padding_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                x++;
            }
        }
    }

    // apply StdMat implementation of the BwdTrans operator to define the
    // expected output
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "MatFree")->apply(*fixt_in, *fixt_out);
    // reshape fixt_out to scalar
    fixt_out->ReshapeStorage<1>();

    // compare
    double TOL = 1e-12;
    BOOST_TEST(fixt_out->compare(*fixt_expected, TOL));
}

BOOST_AUTO_TEST_SUITE_END()