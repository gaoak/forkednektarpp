////////////////////////////////////////////////////////////////////////////////
//
// File: Misc.h
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////
///
#pragma once

#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <MultiRegions/MultiRegions.hpp>
#include <SolverCore/SolverCore.hpp>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Nektar::SolverCore
{

using SummaryList = std::vector<std::pair<std::string, std::string>>;

namespace detail
{
template <typename T> std::string SummaryValueToString(const T &value)
{
    using ValueType = std::decay_t<T>;

    if constexpr (std::is_same_v<ValueType, std::string>)
    {
        return value;
    }
    else if constexpr (std::is_convertible_v<T, std::string>)
    {
        return std::string(value);
    }
    else if constexpr (std::is_same_v<ValueType, NekDouble>)
    {
        std::ostringstream valueStream;
        valueStream << std::setprecision(8) << value;
        return valueStream.str();
    }
    else
    {
        return std::to_string(value);
    }
}

} // namespace detail

template <typename T>
void AddSummaryItem(SummaryList &summary, const std::string &name,
                    const T &value)
{
    summary.emplace_back(name, detail::SummaryValueToString(value));
}

std::string GetProjectionString(MultiRegions::ProjectionType projectionType);
MultiRegions::ProjectionType GetProjectionType(std::string projectionString);

} // namespace Nektar::SolverCore
