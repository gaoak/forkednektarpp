///////////////////////////////////////////////////////////////////////////////
//
// File: TestStatisticsFilters.cpp
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
// Description: Unit tests of the values written by the AverageFields filter.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestStatisticsFilters

#include <UnitTests/SolverCore/TestStatisticsFilters.hpp>

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
namespace
{

// Samples at steps 1 to 4 and output every two steps: the numbered files
// hold the averages of samples 1-2 and 1-4, and the final file that of 1-4.
void CheckAverageValues(const std::string &mesh)
{
    FilterRun run(ExecSpace(), mesh);
    run.CheckRepresentable();
    const std::string base = run.FileName("average");
    auto filter            = run.CreateFilter(
        "AverageFields", {{"OutputFile", base}, {"OutputFrequency", "2"}});
    run.Run(*filter, 1, 4);

    BOOST_CHECK(WrittenNames(run) ==
                (std::set<std::string>{base + "_0_avg.fld", base + "_1_avg.fld",
                                       base + "_avg.fld"}));
    run.CheckAverages(GetFile(run, base + "_0_avg.fld"), {1, 2});
    run.CheckAverages(GetFile(run, base + "_1_avg.fld"), {1, 2, 3, 4});
    run.CheckAverages(GetFile(run, base + "_avg.fld"), {1, 2, 3, 4});
}

// A run of samples 1 to 4 writes its average, and a second filter restarting
// from that file adds samples 5 and 6: its average is that of samples 1 to 6
// and keeps the initial time of the first run.
void CheckAverageRestart(const std::string &mesh)
{
    FilterRun run(ExecSpace(), mesh);
    const std::string first  = run.FileName("average_first");
    const std::string second = run.FileName("average_second");
    auto firstFilter         = run.CreateFilter(
        "AverageFields", {{"OutputFile", first}, {"OutputFrequency", "100"}});
    run.Run(*firstFilter, 1, 4);
    auto secondFilter = run.CreateFilter("AverageFields",
                                         {{"OutputFile", second},
                                          {"OutputFrequency", "100"},
                                          {"RestartFile", first + "_avg.fld"}});
    run.Run(*secondFilter, 5, 6);

    const std::string restartedName = second + "_avg.fld";
    const auto &restarted           = GetFile(run, restartedName);
    run.CheckAverages(restarted, {1, 2, 3, 4, 5, 6});
    BOOST_CHECK_EQUAL(restarted.metadata.at("InitialTime"),
                      std::to_string(0.0));
}

} // namespace
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteStatisticsFilters)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(AverageFields_Values)
{
    CheckAverageValues(triQuadMesh);
}

// Samples at steps 2, 4 and 6, and output steps 3 and 6: only step 6 is due
// for both, so one numbered file is written.
BOOST_AUTO_TEST_CASE(AverageFields_OutputOnSampledSteps)
{
    FilterRun run(ExecSpace(), triQuadMesh);
    const std::string base = run.FileName("average_sampled");
    auto filter = run.CreateFilter("AverageFields", {{"OutputFile", base},
                                                     {"OutputFrequency", "3"},
                                                     {"SampleFrequency", "2"}});
    run.Run(*filter, 1, 6);

    BOOST_CHECK(
        WrittenNames(run) ==
        (std::set<std::string>{base + "_0_avg.fld", base + "_avg.fld"}));
    run.CheckAverages(GetFile(run, base + "_0_avg.fld"), {2, 4, 6});
    run.CheckAverages(GetFile(run, base + "_avg.fld"), {2, 4, 6});
}

// Phase 0.5 of period 1 at time step 0.25 samples the times 0.5 and 1.5, at
// steps 2 and 6. Step 8 is due for sampling and output and writes the
// numbered file, although it adds no sample.
BOOST_AUTO_TEST_CASE(AverageFields_PhaseAverage)
{
    FilterRun run(ExecSpace(), triQuadMesh);
    const std::string base = run.FileName("average_phase");
    auto filter =
        run.CreateFilter("AverageFields", {{"OutputFile", base},
                                           {"OutputFrequency", "8"},
                                           {"PhaseAverage", "true"},
                                           {"PhaseAveragePeriod", "1.0"},
                                           {"PhaseAveragePhase", "0.5"}});
    run.Run(*filter, 1, 8);

    BOOST_CHECK(
        WrittenNames(run) ==
        (std::set<std::string>{base + "_0_avg.fld", base + "_avg.fld"}));
    run.CheckAverages(GetFile(run, base + "_0_avg.fld"), {2, 6});
    run.CheckAverages(GetFile(run, base + "_avg.fld"), {2, 6});
}

BOOST_AUTO_TEST_CASE(AverageFields_Restart)
{
    CheckAverageRestart(triQuadMesh);
}

BOOST_AUTO_TEST_CASE(CurvedPrismTet_Mesh)
{
    FilterRun run(ExecSpace(), curvedPrismTetMesh);
    run.CheckCurvedPrismTetMesh();
}

BOOST_AUTO_TEST_CASE(AverageFields_CurvedPrismTet_Values)
{
    CheckAverageValues(curvedPrismTetMesh);
}

BOOST_AUTO_TEST_CASE(AverageFields_CurvedPrismTet_Restart)
{
    CheckAverageRestart(curvedPrismTetMesh);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
