///////////////////////////////////////////////////////////////////////////////
//
// File: MiscUtils.cpp
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
// Description: error related utilities
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/MiscUtils.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#elif defined(_MSC_VER)
#include <windows.h>

#include <dbghelp.h>
#endif

namespace Nektar
{

#if defined(__GNUC__) || defined(__clang__)
std::string demangleTypeName(const std::type_info &info)
{
    int status;
    char *name(abi::__cxa_demangle(info.name(), nullptr, nullptr, &status));

    if (name)
    {
        std::string ret(name);

        free(name);

        return ret;
    }
    else
    {
        // WARNINGL0(false, std::string("Cannot demangle: '") + info.name() +
        // "'");

        std::string ret(info.name());

        return ret;
    }
}

#elif defined(_MSC_VER)
std::string demangleTypeName(const std::type_info &info)
{
    char *name = (char *)malloc(1024 * sizeof(char));

    UnDecorateSymbolName(info.name(), name, 1024, 0);

    std::string ret(name);

    free(name);

    return ret;
}
#endif

bool stripString(std::string &inStr, const std::string baseStr)
{
    size_t found = inStr.find(baseStr);

    if (found != std::string::npos)
    {
        inStr.erase(found, baseStr.length());
        return true;
    }

    return false;
}

} // namespace Nektar
