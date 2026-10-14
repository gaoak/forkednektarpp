///////////////////////////////////////////////////////////////////////////////
//
//  File: CompositeNames.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Resolution of composite names in composite lists.
//  other I/O routiens in SpatialDomains
//
////////////////////////////////////////////////////////////////////////////////
#include <SpatialDomains/CompositeNames.h>

#include <LibUtilities/BasicUtils/ParseUtils.h>

#include <algorithm>
#include <cctype>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

namespace Nektar::SpatialDomains
{

std::string SanitiseCompositeName(const std::string &name)
{
    std::string out(name);

    auto isJunk = [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) || c == '"' ||
               c == '<' || c == '>' || c == '&' || c == ',' || c == '[' ||
               c == ']';
    };

    std::replace_if(out.begin(), out.end(), isJunk, ' ');
    boost::trim(out);

    return out;
}

bool ParseCompositeList(const std::string &str,
                        const std::map<int, std::string> &labels,
                        std::vector<unsigned int> &ids,
                        std::vector<std::string> *unresolved)
{
    // The whole list is tried as a sequence first, so that a list of numbers
    // -- every list that could be written before names existed -- is parsed
    // exactly as it always was rather than being reassembled entry by entry
    // below. The parse fills its output as it goes and only then reports
    // whether it reached the end, so it is given a vector of its own and the
    // result is taken only once it has succeeded; appending straight to @p ids
    // would leave the leading numbers of a list like "1-2,WALL" behind when
    // the parse stopped at the name.
    std::vector<unsigned int> numeric;
    if (ParseUtils::GenerateSeqVector(str, numeric))
    {
        ids.insert(ids.end(), numeric.begin(), numeric.end());
        return true;
    }

    std::vector<std::string> entries;
    boost::split(entries, str, boost::is_any_of(","));

    // Reverse the labels once per list rather than searching them per entry.
    // A name may be shared, so this is not a one-to-one mapping.
    std::multimap<std::string, int> byName;
    for (const auto &l : labels)
    {
        if (!l.second.empty())
        {
            byName.emplace(l.second, l.first);
        }
    }

    bool allResolved = true;
    size_t nEntries  = 0;

    for (auto &entry : entries)
    {
        boost::trim(entry);

        if (entry.empty())
        {
            continue;
        }

        ++nEntries;

        numeric.clear();
        if (ParseUtils::GenerateSeqVector(entry, numeric))
        {
            ids.insert(ids.end(), numeric.begin(), numeric.end());
            continue;
        }

        auto range = byName.equal_range(entry);
        if (range.first == range.second)
        {
            allResolved = false;
            if (unresolved)
            {
                unresolved->push_back(entry);
            }
            continue;
        }

        for (auto it = range.first; it != range.second; ++it)
        {
            ids.push_back(static_cast<unsigned int>(it->second));
        }
    }

    // A list with nothing in it resolves to nothing, which is not the same as
    // resolving successfully: callers rely on a false here to report a list
    // they could not read, as they did before names were understood.
    return allResolved && nEntries > 0;
}

} // namespace Nektar::SpatialDomains
