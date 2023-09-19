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

    double *x = fixt_in->GetStorage().GetCPUPtr();
    // initialize the input to the operator
    x[0] = 36;
    x[1] = -6;
    x[2] = -84;
    x[3] = 90;
    x[4] = -42;
    x[5] = 49.5;

    // apply the StdMat implementation of the BwdTrans
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);

    // compare the output of StdMat with analytic evaluation of BwdTrans
    static double *y = fixt_out->GetStorage().GetCPUPtr();
    double TOL       = 1e-8;
    BOOST_CHECK_CLOSE(y[0], 36.00000000, TOL);
    BOOST_CHECK_CLOSE(y[1], 2.4885027623, TOL);
    BOOST_CHECK_CLOSE(y[2], -1.9926349937, TOL);
    BOOST_CHECK_CLOSE(y[3], 1.875000000, TOL);
    BOOST_CHECK_CLOSE(y[4], -1.9926349937, TOL);
    BOOST_CHECK_CLOSE(y[5], 2.4885027623, TOL);
    BOOST_CHECK_CLOSE(y[6], -6.000000000, TOL);
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
    }

    // create Array to store expected result from Nektar++
    int totQP{fixt_explist->GetTotPoints()};
    Array<Nektar::OneD, Nektar::NekDouble> expected_result(totQP);

    // calculate expected result from Nektar++
    fixt_explist->BwdTrans(fixt_explist->GetCoeffs(), expected_result);
    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);

    // Copy expected result from Array to fixt_expected
    x         = fixt_expected->GetStorage().GetCPUPtr();
    double *z = expected_result.data();
    for (auto const &block : fixt_expected->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                (*x++) = (*z++);
            }
        }
    }

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
    }

    // create Array to store expected result from Nektar++
    int totQP{fixt_explist->GetTotPoints()};
    Array<Nektar::OneD, Nektar::NekDouble> expected_result(totQP);
    // calculate expected result from Nektar++
    fixt_explist->BwdTrans(fixt_explist->GetCoeffs(), expected_result);

    // calculate result from current implementation
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_out);

    x         = fixt_expected->GetStorage().GetCPUPtr();
    double *z = expected_result.data();
    for (auto const &block : fixt_expected->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                (*x++) = (*z++);
            }
        }
    }

    // compare
    double TOL = 1e-12;
    BOOST_TEST(fixt_out->compare(*fixt_expected, TOL));
}*/

BOOST_FIXTURE_TEST_CASE(bwdtrans_line, Line)
{
    Configure();

    static double *x =
        fixt_in->GetStorage().GetCPUPtr();

    for (auto const &block : fixt_in->GetBlocks())
    {
        for (size_t el = 0; el < block.num_elements; ++el)
        {
            for (size_t coeff = 0; coeff < block.num_pts; ++coeff)
            {
                *(x++) = coeff + 1;
            }
        }
    }

    // TODO:
    // SumFac implmentation not updated yet
    BwdTrans<>::create(fixt_explist, "SumFac")->apply(*fixt_in, *fixt_out);

    // Generate the expected results as reference to be compared
    BwdTrans<>::create(fixt_explist, "StdMat")->apply(*fixt_in, *fixt_expected);

    double TOL = 1e-12;

    BOOST_TEST(fixt_out->compare(*fixt_expected, TOL));
}

BOOST_AUTO_TEST_SUITE_END()
