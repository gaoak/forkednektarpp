///////////////////////////////////////////////////////////////////////////////
//
// File: TestPhysTraceExtractBndCond.cpp
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
// Description: Unit tests for detail::PhysTraceExtractBndCond.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestPhysTraceExtractBndCond

#include "TestPhysTraceExtractBndCond.hpp"

#include <boost/test/tools/output_test_stream.hpp>
#include <iostream>

#define TEST_PHYSTRACEEXTRACTBNDCOND(test_name, test, meshfile, bndRegion,     \
                                     tol)                                      \
    BOOST_FIXTURE_TEST_CASE(test_name, test)                                   \
    {                                                                          \
        std::cout << std::string("Run: ") + std::string(#test_name)            \
                  << std::endl;                                                \
        meshName = meshfile;                                                   \
        Configure(bndRegion);                                                  \
        SetTestCase();                                                         \
        RunTestCase();                                                         \
        boost::test_tools::output_test_stream output;                          \
        {                                                                      \
            BOOST_TEST(Compare(tol));                                          \
        }                                                                      \
    }

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
class PhysTraceExtractBndCondFixture
    : public TestPhysTraceExtractBndCond<double>
{
};
#endif

BOOST_AUTO_TEST_SUITE(TestSuitePhysTraceExtractBndCond)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)

// One face of hexahedra: every element exposes the same local trace.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_hex_allbcs,
                             PhysTraceExtractBndCondFixture,
                             "run/Helmholtz3D_Hex_AllBCs_P6.xml", 1u, 1.0E-10)

// All six faces of a cube of hexahedra of anisotropic order in one region:
// several buckets and face orientations at once.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_hex_heterogeneous,
                             PhysTraceExtractBndCondFixture,
                             "run/Helmholtz3D_Hex_Heterogeneous.xml", 0u,
                             1.0E-10)
// Boundary faces in seven of the eight orientations, including both rotations
// by 90 degrees, with a different number of points in each direction.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_hex_orient_varq,
                             PhysTraceExtractBndCondFixture,
                             "run/hex_orient_varq_bndregions.xml", 0u, 1.0E-10)

// Faces and edges on a collapsed direction, extracted onto the Gauss-Lobatto
// trace points: tetrahedra of variable order, prisms, and triangles and
// quadrilaterals.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_tet_varp,
                             PhysTraceExtractBndCondFixture,
                             "run/Helmholtz3D_Tet_VarP.xml", 0u, 1.0E-10)
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_couette_prism,
                             PhysTraceExtractBndCondFixture,
                             "run/CouetteFlow3D_Prism.xml", 0u, 1.0E-10)
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_tri_quad,
                             PhysTraceExtractBndCondFixture,
                             "run/Helmholtz2D_P7_AllBCs.xml", 8u, 1.0E-10)

// Prisms of variable order: quadrilateral faces on the collapsed direction.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_prism_varp,
                             PhysTraceExtractBndCondFixture,
                             "run/Helmholtz3D_Prism_VarP.xml", 0u, 1.0E-10)

// Two-dimensional meshes in three-dimensional space: triangles, and
// quadrilaterals on a warped surface.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_tri_3d,
                             PhysTraceExtractBndCondFixture,
                             "run/tri_3D_bndregions.xml", 0u, 1.0E-10)
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_square_3d,
                             PhysTraceExtractBndCondFixture,
                             "run/square_3D_bndregions.xml", 0u, 1.0E-10)

// One region whose faces come from both tetrahedra and prisms.
TEST_PHYSTRACEEXTRACTBNDCOND(phystraceextractbndcond_tetprism_mixed,
                             PhysTraceExtractBndCondFixture,
                             "run/ChannelFlow3D_TetPrism.xml", 0u, 1.0E-10)

// The volume field at the largest vector width rather than at width 1.
BOOST_FIXTURE_TEST_CASE(phystraceextractbndcond_tetprism_interleaved,
                        PhysTraceExtractBndCondFixture)
{
    meshName = "run/ChannelFlow3D_TetPrism.xml";
    Configure(0u);
    SetTestCase();
    m_inputInterleaveWidth = NektarSpaces::max_vector_width<double>::value;
    RunTestCase();
    BOOST_TEST(Compare(1.0E-10));
}

#endif

BOOST_AUTO_TEST_SUITE_END()
