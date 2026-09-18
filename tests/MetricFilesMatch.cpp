///////////////////////////////////////////////////////////////////////////////
//
// File: MetricFilesMatch.cpp
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

#include <MetricFilesMatch.h>

#include <fstream>
#include <iterator>

namespace Nektar
{

std::string MetricFilesMatch::type = GetMetricFactory().RegisterCreatorFunction(
    "FILESMATCH", MetricFilesMatch::create);

MetricFilesMatch::MetricFilesMatch(TiXmlElement *metric, bool generate)
    : Metric(metric, generate)
{
    TiXmlElement *cmp = metric->FirstChildElement("compare");
    ASSERTL0(cmp, "Missing compare tag for FilesMatch metric!");

    while (cmp)
    {
        TiXmlElement *file = cmp->FirstChildElement("file");
        ASSERTL0(file, "Missing file tag inside compare for FilesMatch "
                       "metric!");

        std::vector<std::string> names;
        while (file)
        {
            ASSERTL0(file->GetText(), "Missing filename for file tag!");
            names.push_back(file->GetText());
            file = file->NextSiblingElement("file");
        }

        ASSERTL0(names.size() == 2,
                 "A compare tag needs exactly two file tags, the two outputs "
                 "that should hold the same thing.");

        m_filePairs.push_back(std::make_pair(names[0], names[1]));
        cmp = cmp->NextSiblingElement("compare");
    }
}

/**
 * @brief Read @p filename, dropping the parts that are expected to differ.
 *
 * A Nektar++ mesh records where it came from in a Metadata element -- the time
 * it was written and the command line that wrote it -- and two runs differ
 * there for reasons that have nothing to do with the mesh. Everything else is
 * compared as it stands.
 */
std::string MetricFilesMatch::Read(const std::string &filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    ASSERTL0(file.is_open(), "Unable to open file: " + filename);

    std::string contents((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    const std::string open = "<Metadata>", close = "</Metadata>";
    size_t start = contents.find(open);
    while (start != std::string::npos)
    {
        size_t finish = contents.find(close, start);
        if (finish == std::string::npos)
        {
            break;
        }

        contents.erase(start, finish + close.size() - start);
        start = contents.find(open, start);
    }

    return contents;
}

bool MetricFilesMatch::v_Test([[maybe_unused]] std::istream &pStdout,
                              [[maybe_unused]] std::istream &pStderr)
{
    bool success = true;

    for (auto &pair : m_filePairs)
    {
        std::string first = Read(pair.first), second = Read(pair.second);

        if (first == second)
        {
            continue;
        }

        std::cerr << "Failed test." << std::endl;
        std::cerr << "  Files should hold the same thing but do not:"
                  << std::endl;
        std::cerr << "    " << pair.first << " (" << first.size() << " bytes)"
                  << std::endl;
        std::cerr << "    " << pair.second << " (" << second.size() << " bytes)"
                  << std::endl;

        if (first.size() == second.size())
        {
            size_t i = 0;
            while (i < first.size() && first[i] == second[i])
            {
                ++i;
            }
            std::cerr << "  First difference at byte " << i << "." << std::endl;
        }

        std::cerr << "  If this is NekMesh, the usual cause is a container "
                     "keyed on a geometry"
                  << std::endl
                  << "  pointer being iterated: the order follows the "
                     "addresses the allocator"
                  << std::endl
                  << "  handed out, which differ from one run to the next."
                  << std::endl;

        success = false;
    }

    return success;
}

void MetricFilesMatch::v_Generate([[maybe_unused]] std::istream &pStdout,
                                  [[maybe_unused]] std::istream &pStderr)
{
    // Nothing to record: the metric compares the run against itself, so there
    // is no expected value to put back into the test file.
}

} // namespace Nektar
