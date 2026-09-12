///////////////////////////////////////////////////////////////////////////////
//
// File: test_physgalerkinproject1dscaled.cpp
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

#define BOOST_TEST_MODULE TestPhysGalerkinProject1DScaled

#include "init_physgalerkinproject1dscaled.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>
#include <memory>

#define TEST_PHYSGALERKINPROJECT1DSCALED(test_name, test, tol)                 \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        double scale = 1.5;                                                    \
        Configure(scale);                                                      \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

BOOST_AUTO_TEST_SUITE(TestPhysGalerkinProject1DScaled)

// NOTE: ExpList::PhysGalerkinProjection1DScaled only implements its 2D and
// 3D switch cases (ExpList.cpp:6748), so for the Seg/SegSEM/Seg3D (1D)
// cases below, ExpectedSolution() in init_physgalerkinproject1dscaled.hpp
// drives LibUtilities::PhysGalerkinProject1D directly per element instead
// - the same dimension-general, PointsManager()-cached primitive the 2D/3D
// legacy path (and this new operator) are built on - rather than the
// incomplete ExpList wrapper.
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
// Exercises SetAppend(true): seeds fixt_out with a known offset before
// Apply() and expects out == offset + (the same projected result the
// non-append case produces), validating the ReadWrite/BwdTransXxxKernel<true>
// accumulate path added to PhysGalerkinProject1DScaledOp (previously always
// overwrote; nothing exercised append=true before this test).
BOOST_FIXTURE_TEST_CASE(physgalerkinproject1d_quad_append, Quad)
{
    double scale = 1.5;
    Configure(scale);
    SetTestCase();

    // fixt_out has not been written to yet, so ToArray() would trip
    // MemoryRegion's uninitialized-read guard - size it from metadata
    // instead (matches how SetFixture() built it: nout components at
    // native size).
    const double offset = 3.0;
    auto outSize = fixt_out->GetNumComponents() * fixt_explist->GetTotPoints();
    Array<OneD, double> offsetArr(outSize, offset);
    fixt_out->template CopyArray<NektarSpaces::HostSpace>(offsetArr);

    auto expectedArr = fixt_expected->ToArray();
    Vmath::Sadd(expectedArr.size(), offset, expectedArr, 1, expectedArr, 1);
    fixt_expected->template CopyArray<NektarSpaces::HostSpace>(expectedArr);

    auto op = PhysGalerkinProject1DScaledOp<double>::Create(
        fixt_explist, session->GetVariables());
    op->SetScaleFactor(scale);
    op->SetAppend(true);
    op->Apply(*fixt_in, *fixt_out);

    BOOST_TEST(Compare(1.0E-12));
}

// Same as physgalerkinproject1d_quad_append, but for a 3D (Hex) mesh, to
// also cover the Operator3D append path with a shape/dimension distinct
// from the 2D Quad case above.
BOOST_FIXTURE_TEST_CASE(physgalerkinproject1d_hex_append, Hex)
{
    double scale = 1.5;
    Configure(scale);
    SetTestCase();

    const double offset = 3.0;
    auto outSize = fixt_out->GetNumComponents() * fixt_explist->GetTotPoints();
    Array<OneD, double> offsetArr(outSize, offset);
    fixt_out->template CopyArray<NektarSpaces::HostSpace>(offsetArr);

    auto expectedArr = fixt_expected->ToArray();
    Vmath::Sadd(expectedArr.size(), offset, expectedArr, 1, expectedArr, 1);
    fixt_expected->template CopyArray<NektarSpaces::HostSpace>(expectedArr);

    auto op = PhysGalerkinProject1DScaledOp<double>::Create(
        fixt_explist, session->GetVariables());
    op->SetScaleFactor(scale);
    op->SetAppend(true);
    op->Apply(*fixt_in, *fixt_out);

    BOOST_TEST(Compare(1.0E-12));
}

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_seg, Seg, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_seg_sem, SegSEM, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_seg_3d, Seg3D, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_quad, Quad, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_quad_3d, Quad3D, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_quad_varp, QuadVarP,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_quad_sem, QuadSEM,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tri, Tri, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tri_3d, Tri3D, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tri_varp, TriVarP,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tri_nodal, TriNodal,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_square_all_elements,
                                 SquareAllElements, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tet, Tet, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tet_varp, TetVarP,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_tet_nodal, TetNodal,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_pyr, Pyr, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_pyr_varp, PyrVarP,
                                 2.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_prism, Prism, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_prism_varp, PrismVarP,
                                 2.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_prism_nodal, PrismNodal,
                                 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_hex, Hex, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_hex_varp, HexVarP,
                                 2.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_hex_sem, HexSEM, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_cube_prism_hex,
                                 CubePrismHex, 1.0E-12)

TEST_PHYSGALERKINPROJECT1DSCALED(physgalerkinproject1d_cube_all_elements,
                                 CubeAllElements, 1.0E-12)
#endif

BOOST_AUTO_TEST_SUITE_END()
