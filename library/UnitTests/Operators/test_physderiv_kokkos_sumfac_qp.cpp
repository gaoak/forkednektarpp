///////////////////////////////////////////////////////////////////////////////
//
// File: test_physderiv_kokkos_sumfac_qp.cpp
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

#define BOOST_TEST_MODULE TestPhysDerivKokkos

#include "init_physderivfields.hpp"

#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#define TEST_PHYSDERIV(test_name, test, dim, tol)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::KOKKOS;                                \
        using Impl      = Operators::SumFacQP;                                 \
        Configure(1, dim);                                                     \
        SetTestCase(                                                           \
            fixt_kokkos_in->GetBlocks(),                                       \
            fixt_kokkos_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());     \
        PhysDeriv<>::template create<ExecSpace, Impl>(fixt_explist)            \
            ->apply(*fixt_kokkos_in, *fixt_kokkos_out);                        \
        NektarSolution(                                                        \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(*fixt_kokkos_out, *fixt_expected, tol));        \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestPhysDerivKokkos)

TEST_PHYSDERIV(physderiv_kokkos_seg, Seg, 1, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_seg_sem, SegSEM, 1, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_quad, Quad, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_quad_sem, QuadSEM, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_quad_varp, QuadVarP, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_tri, Tri, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_tri_varp, TriVarP, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_square_all_elements, SquareAllElements, 2,
               1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_hex, Hex, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_hex_sem, HexSEM, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_hex_varp, HexVarP, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_prism, Prism, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_prism_varp, PrismVarP, 3, 2.5E-12)

TEST_PHYSDERIV(physderiv_kokkos_pyr, Pyr, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_pyr_varp, PyrVarP, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_tet, Tet, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_tet_varp, TetVarP, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_cube_prism_hex, CubePrismHex, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_kokkos_cube_all_elements, CubeAllElements, 3, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
