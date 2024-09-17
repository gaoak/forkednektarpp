///////////////////////////////////////////////////////////////////////////////
//
// File: test_ipwrtbase_serial_sumfac.cpp
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

#define BOOST_TEST_MODULE TestIProductWRTBase
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/ElmtOps/OperatorIProductWRTBase.hpp"
#include "init_ipwrtbasefields.hpp"

//#define TIMER
#ifdef TIMER
#include <LibUtilities/BasicUtils/Timer.h>
#define TEST_IPWRTBASE(test_name, test, tol, printTimer)                       \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::Serial;                                \
        using Impl      = Operators::SumFac;                                   \
        Configure();                                                           \
        SetTestCase(fixt_in->GetBlocks(),                                      \
                    fixt_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());    \
        LibUtilities::Timer timer;                                             \
        timer.Start();                                                         \
        for (int i = 0; i < 100000; ++i)                                       \
            IProductWRTBase<>::template create<ExecSpace, Impl>(fixt_explist)  \
                ->apply(*fixt_in, *fixt_out);                                  \
        timer.Stop();                                                          \
        std::string outstr = boost::lexical_cast<std::string>(#test);          \
        timer.AccumulateRegion(outstr.c_str());                                \
        if (printTimer)                                                        \
            timer.PrintElapsedRegions();                                       \
        ExpectedSolution(                                                      \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        BOOST_TEST(fixt_out->compare(*fixt_expected, tol));                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(                                                  \
                fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),         \
                fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),    \
                tol);                                                          \
        }                                                                      \
    }
#else
#define TEST_IPWRTBASE(test_name, test, tol, printTimer)                       \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::Serial;                                \
        using Impl      = Operators::SumFac;                                   \
        Configure();                                                           \
        SetTestCase(fixt_in->GetBlocks(),                                      \
                    fixt_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());    \
        IProductWRTBase<>::template create<ExecSpace, Impl>(fixt_explist)      \
            ->apply(*fixt_in, *fixt_out);                                      \
        ExpectedSolution(                                                      \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        BOOST_TEST(fixt_out->compare(*fixt_expected, tol));                    \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(                                                  \
                fixt_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),         \
                fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),    \
                tol);                                                          \
        }                                                                      \
    }
#endif

BOOST_AUTO_TEST_SUITE(TestIProductWRTBase)

TEST_IPWRTBASE(ipwrtbase_seg, Seg, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_seg_sem, SegSEM, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_tri, Tri, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_tri_varp, TriVarP, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_quad_sem, QuadSEM, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_quad, Quad, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_quad_varp, QuadVarP, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_square_all_elements, SquareAllElements, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_hex, Hex, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_hex_varP, HexVarP, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_hex_sem, HexSEM, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_prism, Prism, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_prism_varp, PrismVarP, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_pyr, Pyr, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_pyr_varp, PyrVarP, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_tet, Tet, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_tet_varp, TetVarP, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_cube_prism_hex, CubePrismHex, 1.0E-12, false)

TEST_IPWRTBASE(ipwrtbase_cube_all_elements, CubeAllElements, 1.0E-10, true)

BOOST_AUTO_TEST_SUITE_END()
