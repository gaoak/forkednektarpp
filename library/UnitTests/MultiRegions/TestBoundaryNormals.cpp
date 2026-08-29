///////////////////////////////////////////////////////////////////////////////
//
// File: TestBoundaryNormals.cpp
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
// Description: Unit tests for boundary normal interpolation.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <fstream>
#include <string>

namespace Nektar::BoundaryNormalsTest
{
namespace
{
const std::string sessionFile = R"END(<?xml version="1.0" encoding="UTF-8"?>
<NEKTAR>
  <GEOMETRY DIM="3" SPACE="3">
    <VERTEX>
      <V ID="0"> 0 0 0 </V><V ID="1"> 1 0 0 </V>
      <V ID="2"> 1 1 0 </V><V ID="3"> 0 1 0 </V>
      <V ID="4"> 0 0 1 </V><V ID="5"> 0 1 1 </V>
    </VERTEX>
    <EDGE>
      <E ID="0">0 1</E><E ID="1">1 2</E><E ID="2">2 3</E>
      <E ID="3">3 0</E><E ID="4">0 4</E><E ID="5">1 4</E>
      <E ID="6">2 5</E><E ID="7">3 5</E><E ID="8">4 5</E>
    </EDGE>
    <FACE>
      <Q ID="0">3 0 1 2</Q><T ID="1">0 5 4</T>
      <Q ID="2">1 6 8 5</Q><T ID="3">2 6 7</T>
      <Q ID="4">3 7 8 4</Q>
    </FACE>
    <ELEMENT><R ID="0">0 1 2 3 4</R></ELEMENT>
    <COMPOSITE>
      <C ID="0">R[0]</C><C ID="1">F[0-4]</C>
    </COMPOSITE>
    <DOMAIN>C[0]</DOMAIN>
  </GEOMETRY>
  <EXPANSIONS>
    <E COMPOSITE="C[0]" NUMMODES="4" TYPE="MODIFIED" FIELDS="u" />
  </EXPANSIONS>
  <CONDITIONS>
    <VARIABLES><V ID="0">u</V></VARIABLES>
    <BOUNDARYREGIONS><B ID="0">C[1]</B></BOUNDARYREGIONS>
    <BOUNDARYCONDITIONS>
      <REGION REF="0"><D VAR="u" VALUE="0" /></REGION>
    </BOUNDARYCONDITIONS>
  </CONDITIONS>
</NEKTAR>
)END";
}

BOOST_AUTO_TEST_CASE(TestBoundaryNormalsUseLocalTraceExpansion)
{
    fs::path path =
        fs::temp_directory_path() / LibUtilities::UniquePath("bndnormals");
    fs::create_directories(path);
    path /= "TestBoundaryNormals.xml";

    std::ofstream stream(path);
    stream << sessionFile;
    stream.close();

    std::string pathString = path.string();
    char executable[]      = "MultiRegionsUnitTests";
    char *argv[]           = {executable, pathString.data(), nullptr};
    auto session = LibUtilities::SessionReader::CreateInstance(2, argv);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto field   = MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
        session, graph, session->GetVariable(0));

    Array<OneD, int> elmtId, faceId;
    field->GetBoundaryToElmtMap(elmtId, faceId);

    // Guard the fixture: using the parent volume expansion as the source of
    // PhysInterp must require values beyond at least one trace-normal array.
    bool exercisesMismatchedTrace = false;
    for (int n = 0; n < field->GetBndCondExpansions()[0]->GetExpSize(); ++n)
    {
        auto elmt         = field->GetExp(elmtId[n]);
        auto localNormals = elmt->GetTraceNormal(faceId[n]);
        int volume01      = elmt->GetBasis(0)->GetNumPoints() *
                       elmt->GetBasis(1)->GetNumPoints();
        exercisesMismatchedTrace |= localNormals[0].size() != volume01;
    }
    BOOST_REQUIRE(exercisesMismatchedTrace);

    Array<OneD, Array<OneD, NekDouble>> normals;
    field->GetBoundaryNormals(0, normals);

    int npoints = field->GetBndCondExpansions()[0]->GetTotPoints();
    Array<OneD, Array<OneD, NekDouble>> expected(3);
    for (int d = 0; d < 3; ++d)
    {
        expected[d] = Array<OneD, NekDouble>(npoints, 0.0);
    }

    for (int n = 0; n < field->GetBndCondExpansions()[0]->GetExpSize(); ++n)
    {
        auto elmt         = field->GetExp(elmtId[n]);
        auto localNormals = elmt->GetTraceNormal(faceId[n]);
        auto localTrace   = elmt->GetLocTraceExp(faceId[n]);
        int offset        = field->GetBndCondExpansions()[0]->GetPhys_Offset(n);
        Array<OneD, NekDouble> tmp;

        for (int d = 0; d < 3; ++d)
        {
            field->GetBndCondExpansions()[0]->GetExp(n)->PhysInterp(
                localTrace, localNormals[d], tmp = expected[d] + offset);
        }
    }

    BOOST_REQUIRE_EQUAL(normals.size(), 3);
    for (int d = 0; d < 3; ++d)
    {
        BOOST_REQUIRE_EQUAL(normals[d].size(), npoints);
    }

    for (int i = 0; i < npoints; ++i)
    {
        NekDouble magnitude = 0.0;
        for (int d = 0; d < 3; ++d)
        {
            BOOST_REQUIRE(std::isfinite(normals[d][i]));
            BOOST_CHECK_SMALL(normals[d][i] - expected[d][i], 1.0e-12);
            magnitude += normals[d][i] * normals[d][i];
        }
        BOOST_CHECK_SMALL(std::sqrt(magnitude) - 1.0, 1.0e-12);
    }

    fs::remove_all(path.parent_path());
}
} // namespace Nektar::BoundaryNormalsTest
