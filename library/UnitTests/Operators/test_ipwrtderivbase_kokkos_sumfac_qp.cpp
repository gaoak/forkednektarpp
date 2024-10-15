///////////////////////////////////////////////////////////////////////////////
//
// File: test_ipwrtderivbase_kokkos_sumfac_qp.cpp
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

#define BOOST_TEST_MODULE TestIProductWRTDerivBaseKokkos

#include "init_ipwrtderivbasefields.hpp"

#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#define TEST_IPWRTDERIVBASE(test_name, test, dim, tol)                         \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::KOKKOS;                                \
        using Impl      = Operators::SumFacQP;                                 \
        Configure(dim, 1);                                                     \
        SetTestCase(                                                           \
            fixt_kokkos_in->GetBlocks(),                                       \
            fixt_kokkos_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());     \
        IProductWRTDerivBase<>::template create<ExecSpace, Impl>(fixt_explist) \
            ->apply(*fixt_kokkos_in, *fixt_kokkos_out);                        \
        ExpectedSolution(                                                      \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(*fixt_kokkos_out, *fixt_expected, tol));        \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestIProductWRTDerivBaseKokkos)
#if 0 
TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_seg, Seg, 1, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_seg_sem, SegSEM, 1, 1.0E-12)
#endif
TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_quad, Quad, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_quad_sem, QuadSEM, 2,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_quad_varp, QuadVarP, 2,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_tri, Tri, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_tri_varp, TriVarP, 2,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_square_all_elements,
                    SquareAllElements, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_hex, Hex, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_hex_sem, HexSEM, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_hex_varp, HexVarP, 3,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_prism, Prism, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_prism_varp, PrismVarP, 3,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_pyr, Pyr, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_pyr_varp, PyrVarP, 3,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_tet, Tet, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_tet_varp, TetVarP, 3,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_cube_prism_hex,
                    CubePrismHex, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_kokkos_sumfac_qp_cube_all_elements,
                    CubeAllElements, 3, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
