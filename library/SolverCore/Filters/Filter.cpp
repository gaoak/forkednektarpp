///////////////////////////////////////////////////////////////////////////////
//
// File: Filter.cpp
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
// Description: Base class for SolverCore filters.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <SolverCore/Filters/Filter.h>

#include <iostream>
#include <system_error>

namespace Nektar::SolverCore
{

FilterFactory &GetFilterFactory()
{
    static FilterFactory instance;
    return instance;
}

Filter::Filter(const LibUtilities::SessionReaderSharedPtr &session,
               const EquationSystemSharedPtr &equation)
    : m_session(session), m_equ(equation)
{
}

std::string Filter::SetupOutput(const std::string &ext, const ParamMap &params)
{
    std::string outname = m_session->GetSessionName();

    const auto it = params.find("OutputFile");
    if (it != params.end())
    {
        ASSERTL0(!it->second.empty(), "Missing parameter 'OutputFile'.");
        outname = it->second;
    }

    return SetupOutput(ext, outname);
}

std::string Filter::SetupOutput(const std::string &ext,
                                const std::string &inname)
{
    fs::path specPath = fs::path(inname).replace_extension(ext);

    if (m_session->GetBackups() && m_session->GetComm()->TreatAsRankZero() &&
        fs::exists(specPath))
    {
        fs::path bakPath = specPath;
        int count        = 0;
        while (fs::exists(bakPath))
        {
            bakPath = specPath.parent_path();
            bakPath += specPath.stem();
            bakPath += fs::path(".bak" + std::to_string(count++));
            bakPath += specPath.extension();
        }

        std::cout << "renaming " << specPath << " -> " << bakPath << std::endl;
        try
        {
            fs::rename(specPath, bakPath);
        }
        catch (fs::filesystem_error &e)
        {
            ASSERTL0(e.code() == std::errc::no_such_file_or_directory,
                     "Filesystem error: " + std::string(e.what()));
        }
    }

    return LibUtilities::PortablePath(specPath);
}

} // namespace Nektar::SolverCore
