///////////////////////////////////////////////////////////////////////////////
//
// File: StringUtils.hpp
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
// Description: String utilities. Semantics follow the Boost.StringAlgo
// functions they replace; case handling is ASCII.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_BASIC_UTILS_STRINGUTILS_HPP
#define NEKTAR_LIB_UTILITIES_BASIC_UTILS_STRINGUTILS_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

namespace Nektar::StringUtils
{

namespace detail
{
inline char ToUpper(char c)
{
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

inline char ToLower(char c)
{
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

inline bool IEqualChar(char a, char b)
{
    return ToUpper(a) == ToUpper(b);
}

inline bool IsSpace(char c)
{
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}
} // namespace detail

/// Predicate matching any of a set of characters.
class IsAnyOf
{
public:
    explicit IsAnyOf(std::string_view chars) : m_chars(chars)
    {
    }

    bool operator()(char c) const
    {
        return m_chars.find(c) != std::string::npos;
    }

private:
    std::string m_chars;
};

/// Whether Split() merges adjacent separators.
enum TokenCompressMode
{
    eTokenCompressOff,
    eTokenCompressOn
};

/// Splits @p input at characters matching @p pred into @p result, which is
/// cleared first. An empty input gives a single empty token; leading and
/// trailing separators give empty tokens even when compressing.
template <typename Container, typename Pred>
Container &Split(Container &result, std::string_view input, Pred pred,
                 TokenCompressMode mode = eTokenCompressOff)
{
    result.clear();
    const std::size_t n = input.size();
    std::size_t start   = 0;
    for (std::size_t i = 0; i <= n; ++i)
    {
        if (i == n || pred(input[i]))
        {
            result.insert(result.end(),
                          std::string(input.substr(start, i - start)));
            if (mode == eTokenCompressOn)
            {
                while (i + 1 < n && pred(input[i + 1]))
                {
                    ++i;
                }
            }
            start = i + 1;
        }
    }
    return result;
}

/// Concatenates the strings in @p input separated by @p sep.
template <typename Container>
std::string Join(const Container &input, std::string_view sep)
{
    std::string result;
    bool first = true;
    for (const auto &item : input)
    {
        if (!first)
        {
            result += sep;
        }
        result += item;
        first = false;
    }
    return result;
}

inline bool IEquals(std::string_view a, std::string_view b)
{
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), detail::IEqualChar);
}

inline bool StartsWith(std::string_view input, std::string_view test)
{
    return input.substr(0, test.size()) == test;
}

inline bool EndsWith(std::string_view input, std::string_view test)
{
    return input.size() >= test.size() &&
           input.substr(input.size() - test.size()) == test;
}

inline bool IStartsWith(std::string_view input, std::string_view test)
{
    return input.size() >= test.size() &&
           IEquals(input.substr(0, test.size()), test);
}

inline bool IContains(std::string_view input, std::string_view test)
{
    return std::search(input.begin(), input.end(), test.begin(), test.end(),
                       detail::IEqualChar) != input.end() ||
           test.empty();
}

inline void ToUpper(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), detail::ToUpper);
}

inline void ToLower(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), detail::ToLower);
}

inline std::string ToUpperCopy(std::string str)
{
    ToUpper(str);
    return str;
}

inline std::string ToLowerCopy(std::string str)
{
    ToLower(str);
    return str;
}

/// Removes leading and trailing characters matching @p pred.
template <typename Pred> void TrimIf(std::string &str, Pred pred)
{
    auto end = std::find_if_not(str.rbegin(), str.rend(), pred).base();
    str.erase(end, str.end());
    str.erase(str.begin(), std::find_if_not(str.begin(), str.end(), pred));
}

inline void Trim(std::string &str)
{
    TrimIf(str, detail::IsSpace);
}

inline std::string TrimCopy(std::string str)
{
    Trim(str);
    return str;
}

inline void ReplaceAll(std::string &str, std::string_view search,
                       std::string_view format)
{
    if (search.empty())
    {
        return;
    }
    for (std::size_t pos = str.find(search); pos != std::string::npos;
         pos             = str.find(search, pos + format.size()))
    {
        str.replace(pos, search.size(), format);
    }
}

inline void EraseFirst(std::string &str, std::string_view search)
{
    std::size_t pos = str.find(search);
    if (!search.empty() && pos != std::string::npos)
    {
        str.erase(pos, search.size());
    }
}

} // namespace Nektar::StringUtils

#endif
