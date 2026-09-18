///////////////////////////////////////////////////////////////////////////////
//
// File: MetricFilesMatch.h
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
// Description: Check that two output files hold the same thing.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_TESTS_METRICFILESMATCH_H
#define NEKTAR_TESTS_METRICFILESMATCH_H

#include <Metric.h>

#include <string>
#include <vector>

namespace Nektar
{

/**
 * @brief Checks that two files written by the same test hold the same thing.
 *
 * Unlike the file metric this compares against nothing recorded in the test
 * file, so there is no hash to regenerate when output legitimately changes, and
 * nothing to be fragile across platforms: it only asks whether the run agrees
 * with itself.
 *
 * Its use is a test that runs the same command twice, into two output files,
 * to assert that a tool is deterministic. NekMesh was not: iterating containers
 * keyed on the address of a geometry object walks them in whatever order the
 * allocator happened to lay them out, which changes from run to run, and the
 * mesh that came out changed with it.
 *
 * Anything recording where and when the file was written is skipped, since that
 * is expected to differ between two runs -- the Metadata element of a Nektar++
 * mesh holds a timestamp and the command line that produced it.
 */
class MetricFilesMatch : public Metric
{
public:
    ~MetricFilesMatch() override
    {
    }

    static MetricSharedPtr create(TiXmlElement *metric, bool generate)
    {
        return MetricSharedPtr(new MetricFilesMatch(metric, generate));
    }

    static std::string type;

protected:
    /// The files to compare, in pairs.
    std::vector<std::pair<std::string, std::string>> m_filePairs;

    MetricFilesMatch(TiXmlElement *metric, bool generate);

    bool v_Test(std::istream &pStdout, std::istream &pStderr) override;
    void v_Generate(std::istream &pStdout, std::istream &pStderr) override;

private:
    /// Contents of @p filename with the volatile parts taken out.
    static std::string Read(const std::string &filename);
};

} // namespace Nektar

#endif
