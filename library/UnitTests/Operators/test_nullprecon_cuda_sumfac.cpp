///////////////////////////////////////////////////////////////////////////////
//
// File: test_nullprecon_cuda_sumfac.cpp
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

#define BOOST_TEST_MODULE TestNullPreconCUDA
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/PreconOps/OperatorNullPrecon.hpp"
#include "init_nullpreconfields.hpp"

#define TEST_NULLPRECON(test_name, test, tol)                                  \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::CUDA;                                  \
        using Impl      = Operators::SumFac;                                   \
        Configure();                                                           \
        SetTestCase(                                                           \
            fixt_cuda_in->GetBlocks(),                                         \
            fixt_cuda_in->GetPtr<NektarSpaces::HostSpace, WriteOnly>());       \
        NullPrecon<>::template create<ExecSpace, Impl>(fixt_explist)           \
            ->apply(*fixt_cuda_in, *fixt_cuda_out);                            \
        ExpectedSolution(                                                      \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        BOOST_TEST(fixt_cuda_out->compare(*fixt_expected, tol));               \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            OutputIfNotMatch(                                                  \
                fixt_cuda_out->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),    \
                fixt_expected->GetPtr<NektarSpaces::HostSpace, ReadOnly>(),    \
                tol);                                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestNullPreconCUDA)

TEST_NULLPRECON(nullprecon_cuda_sumfac_seg, Helmholtz1D_Seg, 1.0E-15)

TEST_NULLPRECON(nullprecon_cuda_sumfac_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-15)

TEST_NULLPRECON(nullprecon_cuda_sumfac_hex, Helmholtz3D_Hex, 1.0E-15)

TEST_NULLPRECON(nullprecon_cuda_sumfac_prism, Helmholtz3D_Prism, 1.0E-15)

TEST_NULLPRECON(nullprecon_cuda_sumfac_pyr, Helmholtz3D_Pyr, 1.0E-15)

TEST_NULLPRECON(nullprecon_cuda_sumfac_tet, Helmholtz3D_Tet, 1.0E-15)

BOOST_AUTO_TEST_SUITE_END()
