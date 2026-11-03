///////////////////////////////////////////////////////////////////////////////
//
// File: MetricWhitespaceTable.cpp
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
// Description: Tolerant comparison of whitespace-delimited numerical tables.
//
///////////////////////////////////////////////////////////////////////////////

#include <MetricWhitespaceTable.h>

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace Nektar
{
std::string MetricWhitespaceTable::type =
    GetMetricFactory().RegisterCreatorFunction("TABLE",
                                               MetricWhitespaceTable::create);

namespace
{
// Characters separating the values of a table row.
const StringUtils::IsAnyOf whitespace(" \t\r\n");
} // namespace

MetricWhitespaceTable::MetricWhitespaceTable(TiXmlElement *metric,
                                             bool generate)
    : Metric(metric, generate)
{
    for (auto *file = metric->FirstChildElement("file"); file;
         file       = file->NextSiblingElement("file"))
    {
        Table table;
        ASSERTL0(file->Attribute("filename"),
                 "Missing filename for TABLE metric file tag.");
        ASSERTL0(file->Attribute("tolerance"),
                 "Missing tolerance for TABLE metric file tag.");
        table.filename  = file->Attribute("filename");
        table.tolerance = std::stod(file->Attribute("tolerance"));
        ASSERTL0(std::isfinite(table.tolerance) && table.tolerance >= 0.0,
                 "TABLE metric tolerance must be finite and non-negative.");

        if (!generate)
        {
            auto *columns = file->FirstChildElement("columns");
            ASSERTL0(columns && !EmptyString(columns->GetText()),
                     "Missing columns for TABLE metric file " + table.filename);
            const std::string columnText =
                StringUtils::TrimCopy(columns->GetText());
            ASSERTL0(!columnText.empty(),
                     "Empty columns for TABLE metric file " + table.filename);
            StringUtils::Split(table.columns, columnText, whitespace,
                               StringUtils::eTokenCompressOn);

            for (auto *row = file->FirstChildElement("row"); row;
                 row       = row->NextSiblingElement("row"))
            {
                const std::string rowText =
                    EmptyString(row->GetText())
                        ? std::string()
                        : StringUtils::TrimCopy(row->GetText());
                ASSERTL0(!rowText.empty(),
                         "Empty row for TABLE metric file " + table.filename);
                std::vector<std::string> values;
                StringUtils::Split(values, rowText, whitespace,
                                   StringUtils::eTokenCompressOn);
                ASSERTL0(
                    values.size() == table.columns.size(),
                    "TABLE metric row has the wrong number of columns in " +
                        table.filename);
                std::vector<double> parsed;
                parsed.reserve(values.size());
                for (const auto &value : values)
                {
                    size_t used   = 0;
                    double number = std::stod(value, &used);
                    ASSERTL0(used == value.size() && std::isfinite(number),
                             "TABLE metric rows must contain finite numbers.");
                    parsed.push_back(number);
                }
                table.rows.push_back(std::move(parsed));
            }
            ASSERTL0(!table.rows.empty(),
                     "Missing rows for TABLE metric file " + table.filename);
        }
        m_tables.push_back(std::move(table));
    }
    ASSERTL0(!m_tables.empty(), "Missing file tag for TABLE metric.");
}

MetricWhitespaceTable::Table MetricWhitespaceTable::ReadFile(
    const Table &expected)
{
    std::ifstream input(expected.filename);
    ASSERTL0(input.is_open(), "Unable to open table file " + expected.filename);

    Table actual;
    actual.filename  = expected.filename;
    actual.tolerance = expected.tolerance;
    std::string line;
    while (std::getline(input, line))
    {
        StringUtils::Trim(line);
        if (line.empty() || line.front() == '#')
        {
            continue;
        }
        std::vector<std::string> values;
        StringUtils::Split(values, line, whitespace,
                           StringUtils::eTokenCompressOn);
        if (actual.columns.empty())
        {
            actual.columns = std::move(values);
            continue;
        }

        ASSERTL0(values.size() == actual.columns.size(),
                 "Wrong number of columns in table file " + expected.filename);
        std::vector<double> row;
        row.reserve(values.size());
        for (const auto &value : values)
        {
            size_t used   = 0;
            double number = std::stod(value, &used);
            ASSERTL0(used == value.size() && std::isfinite(number),
                     "Invalid numerical value in table file " +
                         expected.filename);
            row.push_back(number);
        }
        actual.rows.push_back(std::move(row));
    }
    ASSERTL0(!actual.columns.empty(),
             "No column-name row in table file " + expected.filename);
    return actual;
}

bool MetricWhitespaceTable::v_Test([[maybe_unused]] std::istream &pStdout,
                                   [[maybe_unused]] std::istream &pStderr)
{
    bool success = true;
    for (const auto &expected : m_tables)
    {
        const auto actual = ReadFile(expected);
        if (actual.columns != expected.columns)
        {
            std::cerr << "TABLE metric column mismatch in " << expected.filename
                      << std::endl;
            success = false;
            continue;
        }
        if (actual.rows.size() != expected.rows.size())
        {
            std::cerr << "TABLE metric row-count mismatch in "
                      << expected.filename << ": expected "
                      << expected.rows.size() << ", found "
                      << actual.rows.size() << std::endl;
            success = false;
            continue;
        }
        for (size_t row = 0; row < expected.rows.size(); ++row)
        {
            for (size_t column = 0; column < expected.columns.size(); ++column)
            {
                const double reference = expected.rows[row][column];
                const double error =
                    std::abs(actual.rows[row][column] - reference);
                const double limit =
                    expected.tolerance * (1.0 + std::abs(reference));
                if (error > limit)
                {
                    std::cerr << "TABLE metric mismatch in "
                              << expected.filename << " at row " << row
                              << ", column " << expected.columns[column]
                              << ": expected " << std::setprecision(17)
                              << reference << ", found "
                              << actual.rows[row][column] << std::endl;
                    success = false;
                }
            }
        }
    }
    return success;
}

void MetricWhitespaceTable::v_Generate([[maybe_unused]] std::istream &pStdout,
                                       [[maybe_unused]] std::istream &pStderr)
{
    auto *file = m_metric->FirstChildElement("file");
    for (const auto &tableSpec : m_tables)
    {
        ASSERTL0(file, "TABLE metric XML/file count mismatch.");
        const auto table = ReadFile(tableSpec);
        file->Clear();

        auto *columns = new TiXmlElement("columns");
        columns->LinkEndChild(
            new TiXmlText(StringUtils::Join(table.columns, " ")));
        file->LinkEndChild(columns);

        for (const auto &values : table.rows)
        {
            auto *row = new TiXmlElement("row");
            std::ostringstream rowText;
            rowText << std::setprecision(17);
            for (size_t i = 0; i < values.size(); ++i)
            {
                rowText << (i ? " " : "") << values[i];
            }
            row->LinkEndChild(new TiXmlText(rowText.str()));
            file->LinkEndChild(row);
        }
        file = file->NextSiblingElement("file");
    }
}
} // namespace Nektar
