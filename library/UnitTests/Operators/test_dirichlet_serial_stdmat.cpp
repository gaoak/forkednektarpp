///////////////////////////////////////////////////////////////////////////////
//
// File: test_dirichlet_serial_stdmat.cpp
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

#define BOOST_TEST_MODULE TestDirichlet
#include <boost/test/tools/output_test_stream.hpp>

#include <iostream>
#include <memory>

#include "Operators/BndCondOps/OperatorDirBndCond.hpp"
#include "init_dirichletfields.hpp"

#define TEST_DIRICHLET(test_name, test, tol)                                   \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        using ExecSpace = NektarSpaces::Serial;                                \
        using Impl      = Operators::StdMat;                                   \
        Configure();                                                           \
        SetTestCase(fixt_out->GetBlocks(),                                     \
                    fixt_out->GetPtr<NektarSpaces::HostSpace, WriteOnly>());   \
        DirBndCond<>::template create<ExecSpace, Impl>(fixt_explist)           \
            ->apply(*fixt_out);                                                \
        ExpectedSolution(                                                      \
            fixt_expected->GetBlocks(),                                        \
            fixt_expected->GetPtr<NektarSpaces::HostSpace, WriteOnly>());      \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(*fixt_out, *fixt_expected, tol));               \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestDirichlet)

TEST_DIRICHLET(dirichlet1d_seg, Helmholtz1D_Seg, 1.0E-12)

TEST_DIRICHLET(dirichlet2d_tri_quad, Helmholtz2D_Tri_Quad, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_hex, Helmholtz3D_Hex, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_prism, Helmholtz3D_Prism, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_pyr, Helmholtz3D_Pyr, 1.0E-12)

TEST_DIRICHLET(dirichlet3d_tet, Helmholtz3D_Tet, 1.0E-12)

BOOST_AUTO_TEST_SUITE_END()
