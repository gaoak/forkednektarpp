///////////////////////////////////////////////////////////////////////////////
//
// File: TestAvgAssmbScatr.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Unit tests of the AvgAssmbScatr operator.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestAvgAssmbScatr

#include "TestAvgAssmbScatr.hpp"

#include <iostream>
#include <memory>

// Averaging against AvgAssemble() and GlobalToLocal() of the legacy assembly
// map, keeping the interleave width of the input.
#define TEST_AVGASSMBSCATR(test_name, test, tol)                               \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure();                                                           \
        if (this->session->GetComm()->GetRank() == 0)                          \
        {                                                                      \
            std::cout << std::string("Run: ") + std::string(#test_name)        \
                      << std::endl;                                            \
        }                                                                      \
        SetTestCase();                                                         \
        BOOST_TEST(RunTestCase());                                             \
        BOOST_TEST(Compare(tol));                                              \
    }

// Properties of the averaging: applied twice it equals applied once, it
// leaves a continuous field unchanged, and one operator applied to two
// different inputs, at different interleave widths where the execution space
// has one of its own, averages both.
#define TEST_AVGASSMBSCATR_PROPERTIES(test_name, test, tol)                    \
    BOOST_FIXTURE_TEST_CASE(test_name##_twice, test)                           \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCaseTwice();                                                    \
        BOOST_TEST(Compare(tol));                                              \
    }                                                                          \
    BOOST_FIXTURE_TEST_CASE(test_name##_continuous, test)                      \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCaseContinuous();                                               \
        BOOST_TEST(Compare(tol));                                              \
    }                                                                          \
    BOOST_FIXTURE_TEST_CASE(test_name##_reuse, test)                           \
    {                                                                          \
        Configure();                                                           \
        SetTestCase();                                                         \
        RunTestCaseReuseFirst();                                               \
        BOOST_TEST(Compare(tol));                                              \
        RunTestCaseReuseSecond();                                              \
        BOOST_TEST(Compare(tol));                                              \
    }

BOOST_AUTO_TEST_SUITE(TestSuiteAvgAssmbScatr)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

TEST_AVGASSMBSCATR(avgassmbscatr_seg, Seg, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_quad, Quad, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_quad_sem, QuadSEM, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_quad_varp, QuadVarP, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_quad_per, QuadPer, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_tri, Tri, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_tri_varp, TriVarP, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_square_all_elements, SquareAllElements,
                   1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_hex, Hex, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_hex_sem, HexSEM, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_hex_varp, HexVarP, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_hex_per, HexPer, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_prism, Prism, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_prism_varp, PrismVarP, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_pyr, Pyr, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_pyr_varp, PyrVarP, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_tet, Tet, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_tet_varp, TetVarP, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_cube_prism_hex, CubePrismHex, 1.0E-12)

TEST_AVGASSMBSCATR(avgassmbscatr_cube_all_elements, CubeAllElements, 1.0E-12)

TEST_AVGASSMBSCATR_PROPERTIES(avgassmbscatr_quad_varp, QuadVarP, 1.0E-12)

TEST_AVGASSMBSCATR_PROPERTIES(avgassmbscatr_quad_per, QuadPer, 1.0E-12)

TEST_AVGASSMBSCATR_PROPERTIES(avgassmbscatr_square_all_elements,
                              SquareAllElements, 1.0E-12)

TEST_AVGASSMBSCATR_PROPERTIES(avgassmbscatr_hex_varp, HexVarP, 1.0E-12)

TEST_AVGASSMBSCATR_PROPERTIES(avgassmbscatr_hex_per, HexPer, 1.0E-12)

TEST_AVGASSMBSCATR_PROPERTIES(avgassmbscatr_cube_all_elements, CubeAllElements,
                              1.0E-12)

#endif

BOOST_AUTO_TEST_SUITE_END()
