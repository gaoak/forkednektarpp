///////////////////////////////////////////////////////////////////////////////
//
// File: test_physderiv_avx_sumfac.cpp
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

#define BOOST_TEST_MODULE TestPhysDerivAVX

#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"
#include "init_physderivfields.hpp"

#define TEST_PHYSDERIV(test_name, test, dim, tol)                              \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::AVX;                                   \
        using Impl      = Operators::SumFac;                                   \
        Configure(1, dim);                                                     \
        SetTestCase(fixt_in->GetBlocks(),                                      \
                    fixt_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());    \
        NektarSolution(                                                        \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        PhysDeriv<>::template create<ExecSpace, Impl>(fixt_explist)            \
            ->apply(*fixt_in, *fixt_out);                                      \
        fixt_out->ReshapeStorage<ExecSpace, 1>();                              \
        BOOST_TEST(fixt_out->compare(*fixt_expected, tol));                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(                                                  \
                fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),         \
                fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),    \
                tol);                                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestPhysDerivAVX)

TEST_PHYSDERIV(physderiv_avx_seg, Seg, 1, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_seg_sem, SegSEM, 1, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_quad, Quad, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_quad_sem, QuadSEM, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_quad_varp, QuadVarP, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_tri, Tri, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_tri_varp, TriVarP, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_square_all_elements, SquareAllElements, 2, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_hex, Hex, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_hex_sem, HexSEM, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_hex_varp, HexVarP, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_prism, Prism, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_prism_varp, PrismVarP, 3, 2.5E-12)

TEST_PHYSDERIV(physderiv_avx_pyr, Pyr, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_pyr_varp, PyrVarP, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_tet, Tet, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_tet_varp, TetVarP, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_cube_prism_hex, CubePrismHex, 3, 1.0E-12)

TEST_PHYSDERIV(physderiv_avx_cube_all_elements, CubeAllElements, 3, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
