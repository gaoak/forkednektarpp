///////////////////////////////////////////////////////////////////////////////
//
// File: test_physderivcuda.cpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TestPhysDerivCUDA
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorPhysDeriv.hpp"
#include "init_physderivfields.hpp"

#define TEST_PHYSDERIV(test_name, test, dim, tol)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        Configure(1, dim);                                                     \
        SetTestCase(                                                           \
            fixtcuda_in->GetBlocks(),                                          \
            fixtcuda_in->template GetStorage<MemoryRegionCUDA>().GetCPUPtr()); \
        NektarSolution(fixt_expected->GetBlocks(),                             \
                       fixt_expected->GetStorage().GetCPUPtr());               \
        PhysDeriv<>::create(fixt_explist, "CUDA")                              \
            ->apply(*fixtcuda_in, *fixtcuda_out);                              \
        BOOST_TEST(fixtcuda_out->compare(*fixt_expected, tol));                \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(                                                  \
                fixtcuda_out->template GetStorage<MemoryRegionCUDA>()          \
                    .GetCPUPtr(),                                              \
                fixt_expected->GetStorage().GetCPUPtr(), tol);                 \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestPhysDerivCUDA)

TEST_PHYSDERIV(physderivcuda_seg, Seg, 1, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_quad, Quad, 2, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_tri, Tri, 2, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_square_all_elements, SquareAllElements, 2, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_hex, Hex, 3, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_prism, Prism, 3, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_pyr, Pyr, 3, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_tet, Tet, 3, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_cube_prism_hex, CubePrismHex, 3, 1.0E-12)

TEST_PHYSDERIV(physderivcuda_cube_all_elements, CubeAllElements, 3, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
