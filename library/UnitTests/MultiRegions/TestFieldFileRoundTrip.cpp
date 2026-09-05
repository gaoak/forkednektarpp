///////////////////////////////////////////////////////////////////////////////
//
// File: TestFieldFileRoundTrip.cpp
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
// Description: Field file write/read round trip, the path a restart takes.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <boost/test/unit_test.hpp>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace Nektar::FieldFileRoundTripTest
{

/**
 * A mesh of two quadrilaterals and four triangles over the unit square. The
 * two shapes are the point of it: ExpList::GetFieldDefinitions() emits one
 * field definition per shape, and a reader that handles only the last of them
 * is the failure this test exists to catch. A single shape mesh cannot see it.
 */
static const std::string sessionTemplate =
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
            <E ID="12"> 3 7 </E>
            <E ID="13"> 4 8 </E>
        </EDGE>

        <ELEMENT>
            <Q ID="0"> 0 3 5 2 </Q>
            <Q ID="1"> 1 4 6 3 </Q>
            <T ID="2"> 5 8 12 </T>
            <T ID="3"> 12 10 7 </T>
            <T ID="4"> 6 9 13 </T>
            <T ID="5"> 13 11 8 </T>
        </ELEMENT>

        <COMPOSITE>
            <C ID="0"> Q[0-1] </C>
            <C ID="1"> T[2-5] </C>
        </COMPOSITE>

        <DOMAIN> C[0-1] </DOMAIN>

    </GEOMETRY>

    <EXPANSIONS>
        <E COMPOSITE="C[0]" FIELDS="u" TYPE="@TYPE@" NUMMODES="@NMODES@"/>
        <E COMPOSITE="C[1]" FIELDS="u" TYPE="@TYPE@" NUMMODES="@NMODES@"/>
    </EXPANSIONS>

    <CONDITIONS>
        <PARAMETERS>
            <P> TimeStep = 0.01 </P>
        </PARAMETERS>

        <VARIABLES>
            <V ID="0"> u </V>
        </VARIABLES>

        <BOUNDARYREGIONS />
        <BOUNDARYCONDITIONS />
    </CONDITIONS>

</NEKTAR>
)END";

/**
 * @brief Write a coefficient vector to a field file and read it back.
 *
 * This is the path a restart takes: the solver projects its state to
 * coefficients, ExpList::AppendFieldData() lays them out per field definition,
 * and on resumption ExpList::ExtractDataToCoeffs() puts them back. The
 * coefficients are the state, so anything the round trip loses is state the
 * resumed run does not have.
 *
 * The reader is called once per field definition and accumulates into one
 * vector, which is what makes a mesh of more than one element shape the
 * interesting case: a reader that starts from zero each call returns only the
 * shape it was given last and silently zeros the others. On this mesh that
 * would leave the two quadrilaterals - the first 32 coefficients - at zero
 * while the triangles survived.
 */
/// Substitute every occurrence of @p token in @p text with @p value.
static void Substitute(std::string &text, const std::string &token,
                       const std::string &value)
{
    for (std::size_t at = text.find(token); at != std::string::npos;
         at             = text.find(token))
    {
        text.replace(at, token.size(), value);
    }
}

static void RoundTrip(const std::string &expansionType)
{
    std::string sessionFile = sessionTemplate;
    Substitute(sessionFile, "@TYPE@", expansionType);
    Substitute(sessionFile, "@NMODES@", "4");

    // Write the session out so the reader has a file to open.
    fs::path sessionPath =
        fs::temp_directory_path() /
        fs::path("FieldFileRoundTrip" + expansionType + ".xml");
    fs::path fieldPath =
        fs::temp_directory_path() /
        fs::path("FieldFileRoundTrip" + expansionType + ".fld");

    std::ofstream sessionOut(sessionPath.string());
    sessionOut << sessionFile;
    sessionOut.close();

    int argc         = 2;
    std::string exe  = "FieldFileRoundTrip";
    std::string file = sessionPath.string();
    char *argv[]     = {exe.data(), file.data(), nullptr};

    LibUtilities::SessionReaderSharedPtr session =
        LibUtilities::SessionReader::CreateInstance(argc, argv);
    SpatialDomains::MeshGraphSharedPtr graph =
        SpatialDomains::MeshGraphIO::Read(session);
    MultiRegions::ExpListSharedPtr expList =
        MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(session, graph);

    // Both shapes must be present, or the test cannot see the failure it is
    // written for.
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDefs =
        expList->GetFieldDefinitions();
    BOOST_REQUIRE_MESSAGE(fieldDefs.size() > 1,
                          "Expected one field definition per element shape.");

    // A distinct value per coefficient, so a misplaced one is visible rather
    // than coincidentally right.
    const int nCoeffs = expList->GetNcoeffs();
    std::vector<NekDouble> written(nCoeffs);
    for (int i = 0; i < nCoeffs; ++i)
    {
        written[i] = 1.0 + 0.25 * i;
    }

    std::vector<std::vector<NekDouble>> fieldData(fieldDefs.size());
    for (std::size_t i = 0; i < fieldDefs.size(); ++i)
    {
        fieldDefs[i]->m_fields.push_back("u");
        expList->AppendFieldData(fieldDefs[i], fieldData[i], written);
    }

    LibUtilities::FieldIOSharedPtr fieldIO =
        LibUtilities::FieldIO::CreateDefault(session);
    fieldIO->Write(fieldPath.string(), fieldDefs, fieldData);

    // Read it back the way a restart does.
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> readDefs;
    std::vector<std::vector<NekDouble>> readData;
    fieldIO->Import(fieldPath.string(), readDefs, readData);

    std::string variable = "u";
    std::vector<NekDouble> read(nCoeffs, 0.0);
    for (std::size_t i = 0; i < readDefs.size(); ++i)
    {
        expList->ExtractDataToCoeffs(readDefs[i], readData[i], variable, read);
    }

    for (int i = 0; i < nCoeffs; ++i)
    {
        BOOST_CHECK_CLOSE(read[i], written[i], 1.0e-8);
    }

    session->Finalise();

    fs::remove(sessionPath);
    fs::remove(fieldPath);
}

/**
 * @brief Round trip a coefficient vector through a different polynomial order
 * and back.
 *
 * Restarting at the order the file was written at is the easy case:
 * ExpList::ExtractDataToCoeffs() sees a matching mode count and basis and
 * copies the element's coefficients straight across, never consulting the
 * expansion. Restarting at a *different* order is what sends it into the per
 * element routine, which switches on the basis type and knows how to place
 * mode (i,j) of one order among the modes of another. A basis that switch does
 * not name cannot be read at all.
 *
 * Writing at one order, reading at a higher one and coming back is exact for a
 * hierarchical basis - the added modes are zero and are dropped again - so the
 * original coefficients are the expected answer, without the test needing to
 * know how the modes are laid out.
 */
static void RoundTripOrderChange(const std::string &expansionType)
{
    auto build = [&expansionType](const std::string &numModes,
                                  const std::string &tag) {
        std::string text = sessionTemplate;
        Substitute(text, "@TYPE@", expansionType);
        Substitute(text, "@NMODES@", numModes);

        fs::path path =
            fs::temp_directory_path() /
            fs::path("FieldOrderChange" + expansionType + tag + ".xml");
        std::ofstream out(path.string());
        out << text;
        out.close();
        return path;
    };

    fs::path lowPath  = build("4", "Low");
    fs::path highPath = build("5", "High");
    fs::path fieldLow =
        fs::temp_directory_path() /
        fs::path("FieldOrderChange" + expansionType + "Low.fld");
    fs::path fieldHigh =
        fs::temp_directory_path() /
        fs::path("FieldOrderChange" + expansionType + "High.fld");

    std::string exe      = "FieldOrderChange";
    std::string lowFile  = lowPath.string();
    std::string highFile = highPath.string();

    int argc        = 2;
    char *argvLow[] = {exe.data(), lowFile.data(), nullptr};
    LibUtilities::SessionReaderSharedPtr sessionLow =
        LibUtilities::SessionReader::CreateInstance(argc, argvLow);
    SpatialDomains::MeshGraphSharedPtr graphLow =
        SpatialDomains::MeshGraphIO::Read(sessionLow);
    MultiRegions::ExpListSharedPtr expLow =
        MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(sessionLow,
                                                                graphLow);

    char *argvHigh[] = {exe.data(), highFile.data(), nullptr};
    LibUtilities::SessionReaderSharedPtr sessionHigh =
        LibUtilities::SessionReader::CreateInstance(argc, argvHigh);
    SpatialDomains::MeshGraphSharedPtr graphHigh =
        SpatialDomains::MeshGraphIO::Read(sessionHigh);
    MultiRegions::ExpListSharedPtr expHigh =
        MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(sessionHigh,
                                                                graphHigh);

    BOOST_REQUIRE(expLow->GetNcoeffs() != expHigh->GetNcoeffs());

    // Writes @p coeffs from @p exp into a field file.
    auto writeField = [](const MultiRegions::ExpListSharedPtr &exp,
                         const LibUtilities::SessionReaderSharedPtr &session,
                         std::vector<NekDouble> &coeffs, const fs::path &path) {
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> defs =
            exp->GetFieldDefinitions();
        std::vector<std::vector<NekDouble>> data(defs.size());
        for (std::size_t i = 0; i < defs.size(); ++i)
        {
            defs[i]->m_fields.push_back("u");
            exp->AppendFieldData(defs[i], data[i], coeffs);
        }
        LibUtilities::FieldIO::CreateDefault(session)->Write(path.string(),
                                                             defs, data);
    };

    // Reads a field file back into a coefficient vector sized for @p exp.
    auto readField = [](const MultiRegions::ExpListSharedPtr &exp,
                        const LibUtilities::SessionReaderSharedPtr &session,
                        const fs::path &path) {
        std::vector<LibUtilities::FieldDefinitionsSharedPtr> defs;
        std::vector<std::vector<NekDouble>> data;
        LibUtilities::FieldIO::CreateDefault(session)->Import(path.string(),
                                                              defs, data);
        std::string variable = "u";
        std::vector<NekDouble> coeffs(exp->GetNcoeffs(), 0.0);
        for (std::size_t i = 0; i < defs.size(); ++i)
        {
            exp->ExtractDataToCoeffs(defs[i], data[i], variable, coeffs);
        }
        return coeffs;
    };

    std::vector<NekDouble> written(expLow->GetNcoeffs());
    for (std::size_t i = 0; i < written.size(); ++i)
    {
        written[i] = 1.0 + 0.25 * static_cast<NekDouble>(i);
    }

    writeField(expLow, sessionLow, written, fieldLow);

    // Up to the higher order, then back down.
    std::vector<NekDouble> raised = readField(expHigh, sessionHigh, fieldLow);
    writeField(expHigh, sessionHigh, raised, fieldHigh);
    std::vector<NekDouble> read = readField(expLow, sessionLow, fieldHigh);

    BOOST_REQUIRE_EQUAL(read.size(), written.size());
    for (std::size_t i = 0; i < written.size(); ++i)
    {
        BOOST_CHECK_CLOSE(read[i], written[i], 1.0e-8);
    }

    sessionLow->Finalise();
    sessionHigh->Finalise();

    fs::remove(lowPath);
    fs::remove(highPath);
    fs::remove(fieldLow);
    fs::remove(fieldHigh);
}

/// The modified basis, which is what almost every session uses.
BOOST_AUTO_TEST_CASE(TestFieldFileRoundTripHybridMeshModified)
{
    RoundTrip("MODIFIED");
}

/**
 * The orthogonal basis, whose second direction is eOrtho_B rather than
 * eModified_B. The per-element extraction asserts on the basis it is given,
 * so a basis it does not name cannot be read back at all - the failure is an
 * assertion rather than wrong numbers, and no modified-basis session reaches
 * it.
 */
BOOST_AUTO_TEST_CASE(TestFieldFileRoundTripHybridMeshOrthogonal)
{
    RoundTrip("ORTHOGONAL");
}

/// Restarting at a different order, which is what reaches the per element
/// extraction, for each basis.
BOOST_AUTO_TEST_CASE(TestFieldFileOrderChangeModified)
{
    RoundTripOrderChange("MODIFIED");
}

BOOST_AUTO_TEST_CASE(TestFieldFileOrderChangeOrthogonal)
{
    RoundTripOrderChange("ORTHOGONAL");
}

} // namespace Nektar::FieldFileRoundTripTest
