///////////////////////////////////////////////////////////////////////////////
//
// File: TestContFieldLocalToGlobalMap.cpp
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
// Description: Unit tests for the assembly map of another variable on a
// continuous field, ContField::GetLocalToGlobalMap(variable).
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/AssemblyMap/AssemblyMapCG.h>
#include <MultiRegions/ContField.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <string>

namespace Nektar::ContFieldLocalToGlobalMapTest
{
namespace
{
/**
 * Four quadrilaterals over the unit square, with the boundary split into its
 * four sides: region 0 at y = 0, 1 at x = 1, 2 at y = 1 and 3 at x = 0. All
 * variables share one expansion, so that the field of one can be copied for
 * another.
 *
 * The global system is solved iteratively on the full matrix, as the
 * redesign solvers do, so the maps number the degrees of freedom without
 * reordering. A reordering for static condensation draws on the random state
 * of the graph library, which earlier maps in the same process advance, and
 * two maps built at different times could then be numbered differently.
 */
const std::string sessionTemplate =
    R"END(<?xml version="1.0" encoding="UTF-8"?>
<NEKTAR>
    <GEOMETRY DIM="2" SPACE="2">
        <VERTEX>
            <V ID="0"> 0.0 0.0 0.0 </V>
            <V ID="1"> 0.5 0.0 0.0 </V>
            <V ID="2"> 1.0 0.0 0.0 </V>
            <V ID="3"> 0.0 0.5 0.0 </V>
            <V ID="4"> 0.5 0.5 0.0 </V>
            <V ID="5"> 1.0 0.5 0.0 </V>
            <V ID="6"> 0.0 1.0 0.0 </V>
            <V ID="7"> 0.5 1.0 0.0 </V>
            <V ID="8"> 1.0 1.0 0.0 </V>
        </VERTEX>
        <EDGE>
            <E ID="0"> 0 1 </E>
            <E ID="1"> 1 2 </E>
            <E ID="2"> 0 3 </E>
            <E ID="3"> 1 4 </E>
            <E ID="4"> 2 5 </E>
            <E ID="5"> 3 4 </E>
            <E ID="6"> 4 5 </E>
            <E ID="7"> 3 6 </E>
            <E ID="8"> 4 7 </E>
            <E ID="9"> 5 8 </E>
            <E ID="10"> 6 7 </E>
            <E ID="11"> 7 8 </E>
        </EDGE>
        <ELEMENT>
            <Q ID="0"> 0 3 5 2 </Q>
            <Q ID="1"> 1 4 6 3 </Q>
            <Q ID="2"> 5 8 10 7 </Q>
            <Q ID="3"> 6 9 11 8 </Q>
        </ELEMENT>
        <COMPOSITE>
            <C ID="0"> Q[0-3] </C>
            <C ID="1"> E[0-1] </C>
            <C ID="2"> E[4,9] </C>
            <C ID="3"> E[10-11] </C>
            <C ID="4"> E[2,7] </C>
        </COMPOSITE>
        <DOMAIN> C[0] </DOMAIN>
    </GEOMETRY>

    <EXPANSIONS>
        <E COMPOSITE="C[0]" NUMMODES="4" TYPE="MODIFIED" FIELDS="@FIELDS@" />
    </EXPANSIONS>

    <CONDITIONS>
        <SOLVERINFO>
            <I PROPERTY="GlobalSysSoln" VALUE="IterativeFull" />
        </SOLVERINFO>
        <VARIABLES>
@VARIABLES@
        </VARIABLES>
        <BOUNDARYREGIONS>
            <B ID="0"> C[1] </B>
            <B ID="1"> C[2] </B>
            <B ID="2"> C[3] </B>
            <B ID="3"> C[4] </B>
        </BOUNDARYREGIONS>
        <BOUNDARYCONDITIONS>
@CONDITIONS@
        </BOUNDARYCONDITIONS>
    </CONDITIONS>
</NEKTAR>
)END";

/**
 * u and v are Dirichlet everywhere; w is Dirichlet at y = 0 and y = 1 only,
 * so its map differs from u's; p and q are Neumann everywhere, so a field
 * that checks for a singular system pins a degree of freedom.
 */
const std::string variables = R"END(
            <V ID="0"> u </V>
            <V ID="1"> v </V>
            <V ID="2"> w </V>
            <V ID="3"> p </V>
            <V ID="4"> q </V>)END";

const std::string conditions = R"END(
            <REGION REF="0">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <D VAR="w" VALUE="0" />
                <N VAR="p" VALUE="0" />
                <N VAR="q" VALUE="0" />
            </REGION>
            <REGION REF="1">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <N VAR="w" VALUE="0" />
                <N VAR="p" VALUE="0" />
                <N VAR="q" VALUE="0" />
            </REGION>
            <REGION REF="2">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <D VAR="w" VALUE="0" />
                <N VAR="p" VALUE="0" />
                <N VAR="q" VALUE="0" />
            </REGION>
            <REGION REF="3">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <N VAR="w" VALUE="0" />
                <N VAR="p" VALUE="0" />
                <N VAR="q" VALUE="0" />
            </REGION>)END";

/**
 * Periodic in x. u and v are Dirichlet at y = 0 and y = 1; w is Dirichlet
 * at y = 0 only.
 */
const std::string periodicVariables = R"END(
            <V ID="0"> u </V>
            <V ID="1"> v </V>
            <V ID="2"> w </V>)END";

const std::string periodicConditions = R"END(
            <REGION REF="0">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <D VAR="w" VALUE="0" />
            </REGION>
            <REGION REF="1">
                <P VAR="u" VALUE="[3]" />
                <P VAR="v" VALUE="[3]" />
                <P VAR="w" VALUE="[3]" />
            </REGION>
            <REGION REF="2">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <N VAR="w" VALUE="0" />
            </REGION>
            <REGION REF="3">
                <P VAR="u" VALUE="[1]" />
                <P VAR="v" VALUE="[1]" />
                <P VAR="w" VALUE="[1]" />
            </REGION>)END";

/// Substitute every occurrence of @p token in @p text with @p value.
void Substitute(std::string &text, const std::string &token,
                const std::string &value)
{
    for (std::size_t at = text.find(token); at != std::string::npos;
         at             = text.find(token))
    {
        text.replace(at, token.size(), value);
    }
}

/**
 * Writes the session for @p fields with the given variables and boundary
 * conditions to a directory of its own, and reads it and its mesh.
 */
struct Fixture
{
    fs::path directory;
    LibUtilities::SessionReaderSharedPtr session;
    SpatialDomains::MeshGraphSharedPtr graph;

    Fixture(const std::string &fields, const std::string &variableBlock,
            const std::string &conditionBlock)
    {
        std::string sessionFile = sessionTemplate;
        Substitute(sessionFile, "@FIELDS@", fields);
        Substitute(sessionFile, "@VARIABLES@", variableBlock);
        Substitute(sessionFile, "@CONDITIONS@", conditionBlock);

        directory = fs::temp_directory_path() /
                    LibUtilities::UniquePath("contfieldmaps");
        fs::create_directories(directory);
        fs::path path = directory / "TestContFieldLocalToGlobalMap.xml";

        std::ofstream stream(path);
        stream << sessionFile;
        stream.close();

        std::string pathString = path.string();
        char executable[]      = "MultiRegionsUnitTests";
        char *argv[]           = {executable, pathString.data(), nullptr};
        session = LibUtilities::SessionReader::CreateInstance(2, argv);
        graph   = SpatialDomains::MeshGraphIO::Read(session);
    }

    ~Fixture()
    {
        fs::remove_all(directory);
    }

    /// A continuous field for @p variable, built from the session.
    MultiRegions::ContFieldSharedPtr Field(const std::string &variable,
                                           const bool checkIfSingular = false)
    {
        return MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
            session, graph, variable, true, checkIfSingular);
    }
};

/**
 * Checks that @p actual holds the same numbering, boundary condition maps and
 * Dirichlet data as @p expected, which a field built from the session for the
 * same variable owns.
 */
void CheckSameMap(const MultiRegions::AssemblyMapCGSharedPtr &actual,
                  const MultiRegions::AssemblyMapCGSharedPtr &expected)
{
    BOOST_CHECK_EQUAL(actual->GetVariable(), expected->GetVariable());
    BOOST_CHECK_EQUAL(actual->GetNumGlobalCoeffs(),
                      expected->GetNumGlobalCoeffs());
    BOOST_CHECK_EQUAL(actual->GetNumGlobalDirBndCoeffs(),
                      expected->GetNumGlobalDirBndCoeffs());
    BOOST_CHECK_EQUAL(actual->GetNumLocalDirBndCoeffs(),
                      expected->GetNumLocalDirBndCoeffs());
    BOOST_CHECK_EQUAL(actual->GetSingularSystem(),
                      expected->GetSingularSystem());

    const auto &l2g         = actual->GetLocalToGlobalMap();
    const auto &expectedL2g = expected->GetLocalToGlobalMap();
    BOOST_CHECK_EQUAL_COLLECTIONS(l2g.begin(), l2g.end(), expectedL2g.begin(),
                                  expectedL2g.end());

    const auto &bndMap         = actual->GetBndCondCoeffsToLocalCoeffsMap();
    const auto &expectedBndMap = expected->GetBndCondCoeffsToLocalCoeffsMap();
    BOOST_CHECK_EQUAL_COLLECTIONS(bndMap.begin(), bndMap.end(),
                                  expectedBndMap.begin(), expectedBndMap.end());

    const auto &bndSign         = actual->GetBndCondCoeffsToLocalCoeffsSign();
    const auto &expectedBndSign = expected->GetBndCondCoeffsToLocalCoeffsSign();
    BOOST_CHECK_EQUAL_COLLECTIONS(bndSign.begin(), bndSign.end(),
                                  expectedBndSign.begin(),
                                  expectedBndSign.end());

    BOOST_CHECK_EQUAL(actual->GetSignChange(), expected->GetSignChange());

    BOOST_CHECK(actual->GetCopyLocalDirDofs() ==
                expected->GetCopyLocalDirDofs());
    BOOST_CHECK(actual->GetParallelDirBndSign() ==
                expected->GetParallelDirBndSign());

    const auto &paraDirBnd         = actual->GetParaDirBnd();
    const auto &expectedParaDirBnd = expected->GetParaDirBnd();
    BOOST_CHECK_EQUAL_COLLECTIONS(paraDirBnd.begin(), paraDirBnd.end(),
                                  expectedParaDirBnd.begin(),
                                  expectedParaDirBnd.end());
}
} // namespace

/**
 * A variable with the same types of boundary conditions shares the field's
 * own map, and the field's own variable returns it directly.
 */
BOOST_AUTO_TEST_CASE(TestContFieldLocalToGlobalMapSameTypes)
{
    Fixture fixture("u,v,w,p,q", variables, conditions);
    BOOST_REQUIRE(fixture.graph->SameExpansionInfo("u", "v"));

    auto field = fixture.Field("u");
    const MultiRegions::AssemblyMapCGSharedPtr &own =
        field->GetLocalToGlobalMap();

    BOOST_CHECK(field->GetLocalToGlobalMap("u") == own);

    const MultiRegions::AssemblyMapCGSharedPtr &mapV =
        field->GetLocalToGlobalMap("v");
    BOOST_CHECK(mapV == own);
    BOOST_CHECK(&field->GetLocalToGlobalMap("v") == &mapV);
}

/**
 * A variable with other types of boundary conditions gets a map of its own,
 * kept for later calls and equal to the map of a field built for it.
 */
BOOST_AUTO_TEST_CASE(TestContFieldLocalToGlobalMapOtherTypes)
{
    Fixture fixture("u,v,w,p,q", variables, conditions);
    BOOST_REQUIRE(fixture.graph->SameExpansionInfo("u", "w"));

    auto field = fixture.Field("u");
    const MultiRegions::AssemblyMapCGSharedPtr &mapW =
        field->GetLocalToGlobalMap("w");
    BOOST_REQUIRE(mapW);
    BOOST_CHECK(mapW != field->GetLocalToGlobalMap());
    BOOST_CHECK(field->GetLocalToGlobalMap("w") == mapW);

    // w is Neumann on two sides, so it has fewer Dirichlet dofs than u.
    BOOST_CHECK_LT(mapW->GetNumGlobalDirBndCoeffs(),
                   field->GetLocalToGlobalMap()->GetNumGlobalDirBndCoeffs());

    CheckSameMap(mapW, fixture.Field("w")->GetLocalToGlobalMap());
}

/**
 * A map that pins a degree of freedom of a singular system is not shared,
 * even with a variable whose boundary conditions are of the same types.
 */
BOOST_AUTO_TEST_CASE(TestContFieldLocalToGlobalMapSingular)
{
    Fixture fixture("u,v,w,p,q", variables, conditions);

    auto field = fixture.Field("p", true);
    const MultiRegions::AssemblyMapCGSharedPtr &own =
        field->GetLocalToGlobalMap();
    BOOST_REQUIRE(own->GetSingularSystem());
    BOOST_REQUIRE_GT(own->GetNumGlobalDirBndCoeffs(), 0);

    const MultiRegions::AssemblyMapCGSharedPtr &mapQ =
        field->GetLocalToGlobalMap("q");
    BOOST_REQUIRE(mapQ);
    BOOST_CHECK(mapQ != own);
    BOOST_CHECK(!mapQ->GetSingularSystem());
    BOOST_CHECK_EQUAL(mapQ->GetNumGlobalDirBndCoeffs(), 0);

    CheckSameMap(mapQ, fixture.Field("q")->GetLocalToGlobalMap());
}

/**
 * On a periodic mesh, a variable with the same types of boundary conditions
 * shares the field's map and one with other types gets a map equal to that
 * of a field built for it.
 */
BOOST_AUTO_TEST_CASE(TestContFieldLocalToGlobalMapPeriodic)
{
    Fixture fixture("u,v,w", periodicVariables, periodicConditions);

    auto field = fixture.Field("u");
    MultiRegions::PeriodicMap periodicVerts, periodicEdges;
    field->GetPeriodicEntities(periodicVerts, periodicEdges);
    BOOST_REQUIRE(!periodicEdges.empty());

    BOOST_CHECK(field->GetLocalToGlobalMap("v") ==
                field->GetLocalToGlobalMap());

    const MultiRegions::AssemblyMapCGSharedPtr &mapW =
        field->GetLocalToGlobalMap("w");
    BOOST_REQUIRE(mapW);
    BOOST_CHECK(mapW != field->GetLocalToGlobalMap());

    CheckSameMap(mapW, fixture.Field("w")->GetLocalToGlobalMap());
}

/**
 * A copy for another variable adds its map to those the field shares with it,
 * and the lookup then returns that map instead of building one.
 */
BOOST_AUTO_TEST_CASE(TestContFieldLocalToGlobalMapSharedCopy)
{
    Fixture fixture("u,v,w,p,q", variables, conditions);

    auto field = fixture.Field("u");
    auto copyW = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        *field, fixture.graph, "w", true, false);
    BOOST_REQUIRE(copyW->GetLocalToGlobalMap() != field->GetLocalToGlobalMap());
    BOOST_CHECK(field->GetLocalToGlobalMap("w") ==
                copyW->GetLocalToGlobalMap());

    // A copy sharing the field's map adds it for its own variable and leaves
    // the field's own map in place.
    auto copyV = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        *field, fixture.graph, "v", true, false);
    BOOST_CHECK(field->GetLocalToGlobalMap("u") ==
                field->GetLocalToGlobalMap());
    BOOST_CHECK(field->GetLocalToGlobalMap("v") ==
                field->GetLocalToGlobalMap());

    // The copies look up the maps there too, also once the copy has gone.
    BOOST_CHECK(copyV->GetLocalToGlobalMap("w") ==
                copyW->GetLocalToGlobalMap());
    const MultiRegions::AssemblyMapCGSharedPtr mapW =
        copyW->GetLocalToGlobalMap();
    copyW.reset();
    BOOST_CHECK(field->GetLocalToGlobalMap("w") == mapW);
}

/**
 * The map of a copy that pins a degree of freedom of a singular system is not
 * shared, and the field's own lookup still returns it. A field with element
 * expansions of its own shares nothing with the field.
 */
BOOST_AUTO_TEST_CASE(TestContFieldLocalToGlobalMapNotShared)
{
    Fixture fixture("u,v,w,p,q", variables, conditions);

    auto field     = fixture.Field("u");
    auto singularQ = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        *field, fixture.graph, "q", true, true);
    BOOST_REQUIRE(singularQ->GetLocalToGlobalMap()->GetSingularSystem());

    const MultiRegions::AssemblyMapCGSharedPtr &mapQ =
        field->GetLocalToGlobalMap("q");
    BOOST_CHECK(mapQ != singularQ->GetLocalToGlobalMap());
    BOOST_CHECK(!mapQ->GetSingularSystem());
    BOOST_CHECK(singularQ->GetLocalToGlobalMap("q") ==
                singularQ->GetLocalToGlobalMap());

    auto fieldW = fixture.Field("w");
    BOOST_REQUIRE(fieldW->GetExp() != field->GetExp());
    const MultiRegions::AssemblyMapCGSharedPtr &mapW =
        field->GetLocalToGlobalMap("w");
    BOOST_CHECK(mapW != fieldW->GetLocalToGlobalMap());
    CheckSameMap(mapW, fieldW->GetLocalToGlobalMap());
}
} // namespace Nektar::ContFieldLocalToGlobalMapTest
