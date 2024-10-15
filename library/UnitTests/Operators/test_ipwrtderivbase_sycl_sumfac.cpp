///////////////////////////////////////////////////////////////////////////////
//
// File: test_ipwrtderivbase_sycl_sumfac.cpp
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

#define BOOST_TEST_MODULE TestIProductWRTDerivBaseSYCL

#include "init_ipwrtderivbasefields.hpp"

#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#define TEST_IPWRTDERIVBASE(test_name, test, dim, tol)                         \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::SYCL;                                  \
        using Impl      = Operators::SumFac;                                   \
        Configure(dim, 1);                                                     \
        SetTestCase(                                                           \
            fixt_sycl_in->GetBlocks(),                                         \
            fixt_sycl_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());       \
        IProductWRTDerivBase<>::template create<ExecSpace, Impl>(fixt_explist) \
            ->apply(*fixt_sycl_in, *fixt_sycl_out);                            \
        ExpectedSolution(                                                      \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        fixt_sycl_out->ReshapeStorage<ExecSpace, 1>();                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(*fixt_sycl_out, *fixt_expected, tol));          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestIProductWRTDerivBaseSYCL)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_seg, Seg, 1, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_seg_sem, SegSEM, 1, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_quad, Quad, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_quad_sem, QuadSEM, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_quad_varp, QuadVarP, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_tri, Tri, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_tri_varp, TriVarP, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_square_all_elements,
                    SquareAllElements, 2, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_hex, Hex, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_hex_sem, HexSEM, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_hex_varp, HexVarP, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_prism, Prism, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_prism_varp, PrismVarP, 3,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_pyr, Pyr, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_pyr_varp, PyrVarP, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_tet, Tet, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_tet_varp, TetVarP, 3, 1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_cube_prism_hex, CubePrismHex, 3,
                    1.0E-12)

TEST_IPWRTDERIVBASE(ipwrtderivbase_sycl_sumfac_cube_all_elements,
                    CubeAllElements, 3, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
