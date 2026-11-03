///////////////////////////////////////////////////////////////////////////////
//
// File: PointSet.cpp
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
// Description: Point sets of the HistoryPoints filter, from a list or a line,
// plane or box.
//
///////////////////////////////////////////////////////////////////////////////

#include <SolverCore/Filters/PointSet.h>

#include <cmath>
#include <limits>
#include <sstream>

#include <LibUtilities/BasicUtils/ParseUtils.h>

namespace Nektar::SolverCore
{

namespace
{
size_t PointCount(const double value, const std::string &name)
{
    double integer = 0.0;
    ASSERTL0(std::isfinite(value) && value > 1.0 &&
                 std::modf(value, &integer) == 0.0 &&
                 value <=
                     static_cast<double>(std::numeric_limits<size_t>::max()),
             "HistoryPoints " + name + " must be an integer greater than one.");
    return static_cast<size_t>(integer);
}

void ValidateFinite(const std::vector<double> &values,
                    const std::string &definition)
{
    for (const double value : values)
    {
        ASSERTL0(std::isfinite(value),
                 "HistoryPoints " + definition + " values must be finite.");
    }
}

void CheckProduct(const size_t left, const size_t right)
{
    ASSERTL0(left == 0 || right <= std::numeric_limits<size_t>::max() / left,
             "HistoryPoints point count overflows size_t.");
}
} // namespace

PointSet PointSet::Parse(FilterParameters &parameters)
{
    const bool hasPoints = parameters.Has("Points");
    const bool hasLine   = parameters.Has("line");
    const bool hasPlane  = parameters.Has("plane");
    const bool hasBox    = parameters.Has("box");
    ASSERTL0(static_cast<unsigned int>(hasPoints) +
                     static_cast<unsigned int>(hasLine) +
                     static_cast<unsigned int>(hasPlane) +
                     static_cast<unsigned int>(hasBox) ==
                 1,
             "HistoryPoints requires exactly one of Points, line, plane or "
             "box.");

    PointSet result;
    if (hasPoints)
    {
        std::stringstream stream(parameters.RequiredString("Points"));
        std::vector<double> values;
        double value = 0.0;
        while (stream >> value)
        {
            values.push_back(value);
        }
        ASSERTL0(stream.eof() && !values.empty() && values.size() % 3 == 0,
                 "HistoryPoints Points must contain complete x y z triples.");
        ValidateFinite(values, "Points");
        result.m_points.reserve(values.size() / 3);
        for (size_t i = 0; i < values.size(); i += 3)
        {
            result.m_points.push_back(
                {values[i], values[i + 1], values[i + 2]});
        }
        return result;
    }

    const std::string key = hasLine ? "line" : hasPlane ? "plane" : "box";
    std::vector<double> values;
    ASSERTL0(ParseUtils::GenerateVector(parameters.RequiredString(key), values),
             "Failed to interpret HistoryPoints " + key + " string.");
    ValidateFinite(values, key);

    if (hasLine)
    {
        ASSERTL0(values.size() == 5 || values.size() == 7,
                 "HistoryPoints line requires N and two 2D or 3D "
                 "coordinates.");
        const size_t count = PointCount(values[0], "line point count");
        const size_t dim   = (values.size() - 1) / 2;
        result.m_points.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            const double fraction = static_cast<double>(i) / (count - 1);
            std::array<double, 3> point{};
            for (size_t d = 0; d < dim; ++d)
            {
                point[d] = values[1 + d] +
                           fraction * (values[1 + dim + d] - values[1 + d]);
            }
            result.m_points.push_back(point);
        }
    }
    else if (hasPlane)
    {
        ASSERTL0(values.size() == 10 || values.size() == 14,
                 "HistoryPoints plane requires N1,N2 and four 2D or 3D "
                 "coordinates.");
        const size_t count0 = PointCount(values[0], "plane point count N1");
        const size_t count1 = PointCount(values[1], "plane point count N2");
        CheckProduct(count0, count1);
        const size_t dim = (values.size() - 2) / 4;
        result.m_points.reserve(count0 * count1);
        for (size_t j = 0; j < count1; ++j)
        {
            const double t = static_cast<double>(j) / (count1 - 1);
            for (size_t i = 0; i < count0; ++i)
            {
                const double s = static_cast<double>(i) / (count0 - 1);
                std::array<double, 3> point{};
                for (size_t d = 0; d < dim; ++d)
                {
                    const double side0 =
                        values[2 + d] +
                        s * (values[2 + dim + d] - values[2 + d]);
                    const double side1 =
                        values[2 + 3 * dim + d] +
                        s * (values[2 + 2 * dim + d] - values[2 + 3 * dim + d]);
                    point[d] = (1.0 - t) * side0 + t * side1;
                }
                result.m_points.push_back(point);
            }
        }
    }
    else
    {
        ASSERTL0(values.size() == 9,
                 "HistoryPoints box requires N1,N2,N3,xmin,xmax,ymin,ymax,"
                 "zmin,zmax.");
        const std::array<size_t, 3> counts{
            PointCount(values[0], "box point count N1"),
            PointCount(values[1], "box point count N2"),
            PointCount(values[2], "box point count N3")};
        CheckProduct(counts[0], counts[1]);
        CheckProduct(counts[0] * counts[1], counts[2]);
        result.m_points.reserve(counts[0] * counts[1] * counts[2]);
        for (size_t k = 0; k < counts[2]; ++k)
        {
            for (size_t j = 0; j < counts[1]; ++j)
            {
                for (size_t i = 0; i < counts[0]; ++i)
                {
                    const std::array<size_t, 3> indices{i, j, k};
                    std::array<double, 3> point{};
                    for (size_t d = 0; d < 3; ++d)
                    {
                        const double fraction =
                            static_cast<double>(indices[d]) / (counts[d] - 1);
                        point[d] =
                            values[3 + 2 * d] +
                            fraction * (values[4 + 2 * d] - values[3 + 2 * d]);
                    }
                    result.m_points.push_back(point);
                }
            }
        }
    }

    ASSERTL0(!result.m_points.empty(),
             "HistoryPoints requires at least one point.");
    return result;
}

const std::vector<std::array<double, 3>> &PointSet::GetPoints() const
{
    return m_points;
}

} // namespace Nektar::SolverCore
