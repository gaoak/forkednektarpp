///////////////////////////////////////////////////////////////////////////////
//
// File: TestRootTableWriter.cpp
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
// Description: Unit test for the table writer of the SolverCore filters:
// every row ended with EndRow() is in the file while the writer is still
// open, so a run that terminates abnormally keeps its completed rows.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestRootTableWriter

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/Communication/Comm.h>

#include <SolverCore/Filters/FilterUtils.h>

#include <UnitTests/TestBoostSetup.hpp>
#include <UnitTests/TestGlobalConfiguration.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>

using namespace Nektar;
using namespace Nektar::SolverCore;

NEKTAR_TEST_GLOBAL_CONFIGURATION(Nektar::UnitTests::TestArgs::None);

#include <UnitTests/TestBoostTeardown.hpp>

namespace
{

const std::string tablePath = "root_table_writer.tbl";

// Reads the table through a stream of its own, as another process reading
// the file would, without closing or flushing the writer.
std::string ReadTable()
{
    std::ifstream file(tablePath);
    BOOST_REQUIRE(file.good());
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

} // namespace

BOOST_AUTO_TEST_CASE(RootTableWriter_RowsReachFileBeforeClose)
{
    fs::remove(tablePath);

    auto comm =
        LibUtilities::GetCommFactory().CreateInstance("Serial", 0, nullptr);
    RootTableWriter writer(comm);
    BOOST_REQUIRE(writer.IsWriter());
    writer.Open(tablePath);

    // The expected contents, written with the formatting Open() sets up.
    std::ostringstream expected;
    expected << std::scientific << std::setprecision(16);

    writer.Metadata("Table of test rows");
    expected << "# Table of test rows\n";
    BOOST_CHECK_EQUAL(ReadTable(), expected.str());

    writer.Columns({"Time", "Value"});
    expected << "Time Value\n";
    BOOST_CHECK_EQUAL(ReadTable(), expected.str());

    // Each row is far shorter than the stream buffer, so it is only in the
    // file if EndRow() flushed it.
    for (int row = 0; row < 3; ++row)
    {
        const double time  = 0.25 * row;
        const double value = 1.0 / (row + 3.0);
        writer.Stream() << time << " " << value;
        writer.EndRow();
        expected << time << " " << value << "\n";
        BOOST_CHECK_EQUAL(ReadTable(), expected.str());
    }

    // Rows ended with a newline reach the file together at Flush().
    for (int row = 3; row < 5; ++row)
    {
        const double time  = 0.25 * row;
        const double value = 1.0 / (row + 3.0);
        writer.Stream() << time << " " << value << '\n';
        expected << time << " " << value << "\n";
    }
    writer.Flush();
    BOOST_CHECK_EQUAL(ReadTable(), expected.str());

    writer.Close();
    BOOST_CHECK_EQUAL(ReadTable(), expected.str());

    fs::remove(tablePath);
}
