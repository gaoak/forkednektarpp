///////////////////////////////////////////////////////////////////////////////
//
// File: TestFieldFileIO.cpp
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
// Description: Unit test for reading and writing solver fields as field
// files at the interleave width of the solver fields.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestFieldFileIO

#include <LibUtilities/BasicUtils/FieldIO.h>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <SolverCore/Core/SessionFunction.h>
#include <SolverCore/EquationSystems/EquationSystem.h>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

#include <fstream>

using namespace Nektar;
using namespace Nektar::SolverCore;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

#include <UnitTests/TestBoostTeardown.hpp>

namespace
{

// The files are named after the execution space, so that the tests of the
// execution spaces can run concurrently in one directory.
std::string FileName(const std::string &execStr, const std::string &suffix)
{
    return "field_file_io_" + execStr + suffix;
}

// Writes the session conditions: a continuous projection, the variables u and
// v of the meshes and initial conditions read from the input file.
void WriteConditions(const std::string &execStr)
{
    std::ofstream conditions(FileName(execStr, ".xml"));
    conditions << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
               << "<NEKTAR>\n"
               << "  <CONDITIONS>\n"
               << "    <SOLVERINFO>\n"
               << "      <I PROPERTY=\"Projection\" VALUE=\"Continuous\"/>\n"
               << "    </SOLVERINFO>\n"
               << "    <VARIABLES>\n"
               << "      <V ID=\"0\">u</V>\n"
               << "      <V ID=\"1\">v</V>\n"
               << "    </VARIABLES>\n"
               << "    <FUNCTION NAME=\"InitialConditions\">\n"
               << R"(      <F VAR="u,v" FILE=")" << FileName(execStr, "_in.fld")
               << "\"/>\n"
               << "    </FUNCTION>\n"
               << "  </CONDITIONS>\n"
               << "</NEKTAR>\n";
}

// Gives the test access to the solver fields of an equation system.
class FieldFileSystem : public EquationSystem
{
public:
    FieldFileSystem(const LibUtilities::SessionReaderSharedPtr &session,
                    const SpatialDomains::MeshGraphSharedPtr &graph)
        : EquationSystem(session, graph)
    {
    }

    LibUtilities::Field<double, FieldState::Coeff> &Coeffs()
    {
        return m_fields_coeff;
    }

    const MultiRegions::ExpListSharedPtr &Expansion() const
    {
        return m_expansionLists[0];
    }

    // Sets the physical solution from the coefficients, which WriteFld()
    // projects back to coefficients before writing them.
    void SetPhysFromCoeffs()
    {
        m_bwdTransOp->Apply(m_fields_coeff, m_fields);
    }
};

LibUtilities::SessionReaderSharedPtr SetSession(const std::string &xml,
                                                const std::string &exec)
{
    int argc    = 5;
    char **argv = new char *[argc];
    argv[0]     = strdup("TestFieldFileIO");
    argv[1]     = strdup(xml.c_str());
    argv[2]     = strdup(FileName(exec, ".xml").c_str());
    argv[3]     = strdup(("--opExecSpace=" + exec).c_str());
    argv[4]     = strdup("--opImpl=SumFac");

    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);

    for (int i = 0; i < argc; ++i)
    {
        free(argv[i]);
    }
    delete[] argv;

    return session;
}

// Reads the given variables of a field file in the element order of the
// expansion list, component-major.
std::vector<double> ReadFieldFile(
    const LibUtilities::SessionReaderSharedPtr &session,
    const MultiRegions::ExpListSharedPtr &explist,
    const std::vector<std::string> &variables, const std::string &filename)
{
    std::vector<LibUtilities::FieldDefinitionsSharedPtr> fieldDef;
    std::vector<std::vector<double>> fieldData;
    LibUtilities::FieldIO::CreateForFile(session, filename)
        ->Import(filename, fieldDef, fieldData);

    const size_t ncoeffs = explist->GetNcoeffs();
    std::vector<double> coeffs(variables.size() * ncoeffs, 0.0);
    for (size_t comp = 0; comp < variables.size(); ++comp)
    {
        std::string variable = variables[comp];
        std::vector<double> component(ncoeffs, 0.0);
        for (size_t i = 0; i < fieldDef.size(); ++i)
        {
            explist->ExtractDataToCoeffs(fieldDef[i], fieldData[i], variable,
                                         component);
        }
        std::copy(component.begin(), component.end(),
                  coeffs.begin() + comp * ncoeffs);
    }
    return coeffs;
}

// Reads the coefficients of a field in the element order of the expansion
// list, component-major, through a reshape to interleave width 1 rather than
// through the de-interleaving copy of ToVector(), and leaves the field at the
// given interleave width.
std::vector<double> FieldCoeffs(
    LibUtilities::Field<double, FieldState::Coeff> &field,
    const unsigned int interleaveWidth, const std::string &execStr)
{
    field.ReshapeStorage(1, execStr);
    auto coeffs = field.ToVector<double>();
    field.ReshapeStorage(interleaveWidth, execStr);
    return coeffs;
}

void CheckInterleaveWidth(LibUtilities::Field<double, FieldState::Coeff> &field,
                          const unsigned int interleaveWidth)
{
    for (auto &block : field.GetBlocks())
    {
        BOOST_CHECK_EQUAL(block.GetInterleaveWidth(), interleaveWidth);
    }
}

void CheckEqual(const std::vector<double> &result,
                const std::vector<double> &expected)
{
    BOOST_REQUIRE_EQUAL(result.size(), expected.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        mismatches += result[i] != expected[i];
    }
    BOOST_CHECK_EQUAL(mismatches, 0u);
}

// Loads distinct coefficients from a field file into the solver fields, which
// have the interleave width of the solver, and writes the solver fields to a
// field file. The loaded field must hold the coefficients of the input file,
// and the written file those of the solver fields, element by element.
[[maybe_unused]] void RunFieldFileTest(const std::string &xml,
                                       const std::string &execStr)
{
    const auto inputFile  = FileName(execStr, "_in.fld");
    const auto outputFile = FileName(execStr, "_out.fld");
    WriteConditions(execStr);
    auto session = SetSession(xml, execStr);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);
    auto system  = std::make_shared<FieldFileSystem>(session, graph);
    system->InitObject();
    system->InitialiseFields();
    system->InitialiseOperators();

    const auto &explist  = system->Expansion();
    const auto variables = session->GetVariables();
    const size_t ncoeffs = explist->GetNcoeffs();
    const auto interleaveWidth =
        Operators::Operator<double>::GetDefaultInterleaveWidth(session);
    auto &coeffs = system->Coeffs();
    CheckInterleaveWidth(coeffs, interleaveWidth);

    // Distinct values for every coefficient of every component.
    std::vector<double> expected(variables.size() * ncoeffs);
    for (size_t i = 0; i < expected.size(); ++i)
    {
        expected[i] = 1.0 + 1.0e-3 * static_cast<double>(i);
    }

    auto fieldDef = explist->GetFieldDefinitions();
    std::vector<std::vector<double>> fieldData(fieldDef.size());
    for (size_t comp = 0; comp < variables.size(); ++comp)
    {
        Array<OneD, double> component(ncoeffs,
                                      expected.data() + comp * ncoeffs);
        for (size_t i = 0; i < fieldDef.size(); ++i)
        {
            fieldDef[i]->m_fields.push_back(variables[comp]);
            explist->AppendFieldData(fieldDef[i], fieldData[i], component);
        }
    }
    LibUtilities::FieldIO::CreateDefault(session)->Write(inputFile, fieldDef,
                                                         fieldData);

    // Load.
    SessionFunction initialConditions(session, explist, "InitialConditions");
    initialConditions.EvaluateFld(variables, coeffs, 0.0);
    CheckInterleaveWidth(coeffs, interleaveWidth);
    BOOST_TEST_CONTEXT("load")
    {
        CheckEqual(FieldCoeffs(coeffs, interleaveWidth, execStr), expected);
    }

    // ToVector() de-interleaves the field without changing it.
    BOOST_TEST_CONTEXT("ToVector")
    {
        CheckEqual(coeffs.ToVector<double>(), expected);
        CheckInterleaveWidth(coeffs, interleaveWidth);
    }

    // Write. The file must hold the coefficients WriteFld() projected the
    // solution to, and the field must keep its interleave width.
    system->SetPhysFromCoeffs();
    system->WriteFld(outputFile);
    CheckInterleaveWidth(coeffs, interleaveWidth);
    BOOST_TEST_CONTEXT("write")
    {
        CheckEqual(ReadFieldFile(session, explist, variables, outputFile),
                   FieldCoeffs(coeffs, interleaveWidth, execStr));
    }

    fs::remove(FileName(execStr, ".xml"));
    fs::remove(inputFile);
    fs::remove(outputFile);
}

} // namespace

BOOST_AUTO_TEST_SUITE(TestSuiteFieldFileIO)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
// Blocks with fewer elements than the interleave width, so that every block
// is padded, in 2D and 3D.
BOOST_AUTO_TEST_CASE(FieldFileIO_LoadAndWrite)
{
    const std::string execStr(
        boost::unit_test::framework::master_test_suite().argv[1]);
    for (const std::string xml :
         {"run/square_all_elements.xml", "run/cube_all_elements.xml"})
    {
        BOOST_TEST_CONTEXT("exec=" << execStr << " xml=" << xml)
        {
            RunFieldFileTest(xml, execStr);
        }
    }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
