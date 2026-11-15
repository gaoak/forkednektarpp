///////////////////////////////////////////////////////////////////////////////
//
// File: TestFilterReynoldsStresses.cpp
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
// Description: Unit tests of the values written by the ReynoldsStresses
// filter.
//
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_MODULE TestFilterReynoldsStresses

#include <UnitTests/SolverCore/TestStatisticsFilters.hpp>

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
namespace
{

// Samples at steps 1 to 4 and output every two steps: each file holds the
// averages of the velocity and the scalar followed by the covariances of the
// velocity.
void CheckStressValues(const std::string &mesh)
{
    FilterRun run(ExecSpace(), mesh);
    run.CheckRepresentable();
    const std::string base = run.FileName("stresses");
    auto filter            = run.CreateFilter(
        "ReynoldsStresses", {{"OutputFile", base}, {"OutputFrequency", "2"}});
    run.Run(*filter, 1, 4);

    BOOST_CHECK(
        WrittenNames(run) ==
        (std::set<std::string>{base + "_0_stress.fld", base + "_1_stress.fld",
                               base + "_stress.fld"}));
    for (const auto &[name, samples] :
         std::vector<std::pair<std::string, std::vector<unsigned int>>>{
             {base + "_0_stress.fld", {1, 2}},
             {base + "_stress.fld", {1, 2, 3, 4}}})
    {
        BOOST_TEST_CONTEXT("file=" << name)
        {
            run.CheckAverages(GetFile(run, name), samples);
            run.CheckStresses(GetFile(run, name), samples);
        }
    }
}

} // namespace
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteFilterReynoldsStresses)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
BOOST_AUTO_TEST_CASE(ReynoldsStresses_Values)
{
    CheckStressValues(triQuadMesh);
}

BOOST_AUTO_TEST_CASE(CurvedPrismTet_Mesh)
{
    FilterRun run(ExecSpace(), curvedPrismTetMesh);
    run.CheckCurvedPrismTetMesh();
}

BOOST_AUTO_TEST_CASE(ReynoldsStresses_CurvedPrismTet_Values)
{
    CheckStressValues(curvedPrismTetMesh);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
