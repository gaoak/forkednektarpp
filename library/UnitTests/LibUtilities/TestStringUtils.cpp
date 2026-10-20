///////////////////////////////////////////////////////////////////////////////
//
// File: TestStringUtils.cpp
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
// Description: Unit tests for the string utilities.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/StringUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <string>
#include <vector>

namespace Nektar::StringUtils::UnitTest
{

using Tokens = std::vector<std::string>;

BOOST_AUTO_TEST_CASE(TestSplit)
{
    Tokens tok;

    Split(tok, "a,b;c", IsAnyOf(",;"));
    BOOST_CHECK(tok == Tokens({"a", "b", "c"}));

    Split(tok, "", IsAnyOf(","));
    BOOST_CHECK(tok == Tokens({""}));

    Split(tok, "a,,b", IsAnyOf(","));
    BOOST_CHECK(tok == Tokens({"a", "", "b"}));

    Split(tok, ",a,", IsAnyOf(","));
    BOOST_CHECK(tok == Tokens({"", "a", ""}));

    Split(tok, "a, ,b", IsAnyOf(", "), eTokenCompressOn);
    BOOST_CHECK(tok == Tokens({"a", "b"}));

    Split(tok, ",,a,,", IsAnyOf(","), eTokenCompressOn);
    BOOST_CHECK(tok == Tokens({"", "a", ""}));
}

BOOST_AUTO_TEST_CASE(TestJoin)
{
    BOOST_CHECK_EQUAL(Join(Tokens({"x", "y", "z"}), ","), "x,y,z");
    BOOST_CHECK_EQUAL(Join(Tokens({"x"}), ","), "x");
    BOOST_CHECK_EQUAL(Join(Tokens(), ","), "");
}

BOOST_AUTO_TEST_CASE(TestCaseInsensitive)
{
    BOOST_CHECK(IEquals("Wall", "WALL"));
    BOOST_CHECK(!IEquals("Wall", "Walls"));
    BOOST_CHECK(IEquals(std::string(""), ""));

    BOOST_CHECK(IStartsWith("WomersleyX", "womersley"));
    BOOST_CHECK(!IStartsWith("Wom", "womersley"));
    BOOST_CHECK(StartsWith("XML_Key", "XML_"));
    BOOST_CHECK(!StartsWith("xml_Key", "XML_"));
    BOOST_CHECK(EndsWith("mesh.xml.gz", ".gz"));
    BOOST_CHECK(EndsWith("mesh.xml", "mesh.xml"));
    BOOST_CHECK(!EndsWith("gz", ".gz"));
    BOOST_CHECK(!EndsWith("mesh.XML", ".xml"));

    BOOST_CHECK(IContains("TimeDependent", "dEpEnD"));
    BOOST_CHECK(!IContains("TimeDependent", "steady"));
    BOOST_CHECK(IContains("abc", ""));
}

BOOST_AUTO_TEST_CASE(TestCaseConversion)
{
    BOOST_CHECK_EQUAL(ToUpperCopy("aBc1_"), "ABC1_");
    BOOST_CHECK_EQUAL(ToLowerCopy("AbC1_"), "abc1_");

    std::string s = "MiXeD";
    ToUpper(s);
    BOOST_CHECK_EQUAL(s, "MIXED");
    ToLower(s);
    BOOST_CHECK_EQUAL(s, "mixed");
}

BOOST_AUTO_TEST_CASE(TestTrim)
{
    std::string s = " \t a b \n";
    Trim(s);
    BOOST_CHECK_EQUAL(s, "a b");

    BOOST_CHECK_EQUAL(TrimCopy("   "), "");
    BOOST_CHECK_EQUAL(TrimCopy("x"), "x");

    std::string q = "\"name\"";
    TrimIf(q, IsAnyOf("\""));
    BOOST_CHECK_EQUAL(q, "name");
}

BOOST_AUTO_TEST_CASE(TestReplaceErase)
{
    std::string s = "a.b.c";
    ReplaceAll(s, ".", "..");
    BOOST_CHECK_EQUAL(s, "a..b..c");

    s = "aaa";
    ReplaceAll(s, "a", "");
    BOOST_CHECK_EQUAL(s, "");

    s = "# x # y";
    EraseFirst(s, "#");
    BOOST_CHECK_EQUAL(s, " x # y");

    s = "x";
    EraseFirst(s, "#");
    BOOST_CHECK_EQUAL(s, "x");
}

} // namespace Nektar::StringUtils::UnitTest
