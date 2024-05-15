///////////////////////////////////////////////////////////////////////////////
//
// File: test_helmholtz_kokkos_stdmat.cpp
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

#define BOOST_TEST_MODULE TestHelmholtzKokkos
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/OperatorHelmholtz.hpp"
#include "init_helmholtzfields.hpp"

#define TEST_HELMSOLVE(test_name, test, tol)                                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace1 = NektarSpaces::Serial;                               \
        using ExecSpace2 = Kokkos::DefaultExecutionSpace;                      \
        using Impl1      = Operators::StdMat;                                  \
        using Impl2      = Operators::StdMat;                                  \
        Configure();                                                           \
        SetTestCase(fixt_in->GetBlocks(),                                      \
                    fixt_in->GetPtr<NektarSpaces::HostSpace>());               \
        SetTestCase(fixt_kokkos_in->GetBlocks(),                               \
                    fixt_kokkos_in->GetPtr<NektarSpaces::HostSpace>());        \
        Helmholtz<>::template create<ExecSpace1, Impl1>(fixt_explist)          \
            ->apply(*fixt_in, *fixt_expected);                                 \
        Helmholtz<>::template create<ExecSpace2, Impl2>(fixt_explist)          \
            ->apply(*fixt_kokkos_in, *fixt_kokkos_out);                        \
        BOOST_TEST(fixt_kokkos_out->compare(*fixt_expected, tol));             \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(                                                  \
                fixt_kokkos_out->GetPtr<NektarSpaces::HostSpace>(),            \
                fixt_expected->GetPtr<NektarSpaces::HostSpace>(), tol);        \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestHelmholtzKokkos)

TEST_HELMSOLVE(helmholtz_kokkos_seg, Seg, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_quad, Quad, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_tri, Tri, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_square_all_elements, SquareAllElements, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_hex, Hex, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_prism, Prism, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_pyr, Pyr, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_tet, Tet, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_cube_prism_hex, CubePrismHex, 1.0E-12)

TEST_HELMSOLVE(helmholtz_kokkos_cube_all_elements, CubeAllElements, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
