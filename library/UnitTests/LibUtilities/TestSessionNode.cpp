///////////////////////////////////////////////////////////////////////////////
//
// File: TestSessionNode.cpp
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
// Description: unit tests for SessionNode and its XML adapter.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/SessionNode.h>
#include <LibUtilities/BasicUtils/SessionNodeXml.h>

#include <tinyxml.h>

#include <boost/test/unit_test.hpp>

namespace Nektar::SessionNodeUnitTests
{

using LibUtilities::SessionNode;

/// Builds the tree that the XML in ParseXml() below describes, but by hand,
/// so that the two construction routes can be compared.
static SessionNode BuildForcingSection()
{
    SessionNode coeff("COEFF", "SpongeCoefficient");
    SessionNode regions("BOUNDARYREGIONS", "0");

    SessionNode absorption("FORCE");
    absorption.SetAttribute("TYPE", "Absorption");
    absorption.AddChild(coeff);
    absorption.AddChild(regions);

    SessionNode noise("FORCE");
    noise.SetAttribute("TYPE", "Noise");
    noise.AddChild(SessionNode("WHITENOISE", "0.01"));
    noise.AddChild(SessionNode("UPDATEFREQ", "10"));

    SessionNode forcing("FORCING");
    forcing.AddChild(absorption);
    forcing.AddChild(noise);

    return forcing;
}

/// Parses @p xml and converts the root element into a SessionNode.
static SessionNode ParseXml(const std::string &xml,
                            const std::string &name = "")
{
    TiXmlDocument doc;
    doc.Parse(xml.c_str());
    BOOST_REQUIRE_MESSAGE(!doc.Error(), doc.ErrorDesc());
    return LibUtilities::SessionNodeFromXml(doc.RootElement(), name);
}

BOOST_AUTO_TEST_CASE(TestSessionNodeIdentity)
{
    SessionNode node("COEFF", "SpongeCoefficient");

    BOOST_CHECK_EQUAL(node.GetName(), "COEFF");
    BOOST_CHECK_EQUAL(node.GetText(), "SpongeCoefficient");
    BOOST_CHECK_EQUAL(node.GetPath(), "COEFF");
    BOOST_CHECK(node.DefinesText());

    SessionNode empty("EMPTY");
    BOOST_CHECK(!empty.DefinesText());
    BOOST_CHECK_EQUAL(empty.GetText(), "");
}

BOOST_AUTO_TEST_CASE(TestSessionNodeAttributes)
{
    SessionNode node("FORCE");
    node.SetAttribute("TYPE", "Absorption");
    node.SetAttribute("NVARS", "3");

    BOOST_CHECK(node.DefinesAttribute("TYPE"));
    BOOST_CHECK(!node.DefinesAttribute("MISSING"));
    BOOST_CHECK_EQUAL(node.GetAttribute("TYPE"), "Absorption");
    BOOST_CHECK_EQUAL(node.GetAttribute("MISSING", "fallback"), "fallback");
    BOOST_CHECK_EQUAL(node.GetAttributeAs<int>("NVARS"), 3);
    BOOST_CHECK_EQUAL(node.GetAttributeAs<int>("MISSING", 7), 7);

    // Attributes are kept in the order they were supplied, and setting an
    // existing attribute replaces it in place rather than appending.
    node.SetAttribute("TYPE", "Noise");
    BOOST_CHECK_EQUAL(node.GetAttributes().size(), 2u);
    BOOST_CHECK_EQUAL(node.GetAttributes()[0].first, "TYPE");
    BOOST_CHECK_EQUAL(node.GetAttributes()[0].second, "Noise");

    BOOST_CHECK_THROW(node.GetAttribute("MISSING"), ErrorUtil::NekError);
}

BOOST_AUTO_TEST_CASE(TestSessionNodeDescribe)
{
    SessionNode forcing = BuildForcingSection();

    BOOST_CHECK_EQUAL(forcing.GetChild("FORCE").Describe(),
                      "FORCING/FORCE [TYPE = \"Absorption\"]");

    // A node with no attributes describes as just its path.
    BOOST_CHECK_EQUAL(forcing.GetChild("FORCE").GetChild("COEFF").Describe(),
                      "FORCING/FORCE/COEFF");
}

BOOST_AUTO_TEST_CASE(TestSessionNodeChildren)
{
    SessionNode forcing = BuildForcingSection();

    BOOST_CHECK(forcing.DefinesChild("FORCE"));
    BOOST_CHECK(!forcing.DefinesChild("MAPPING"));
    BOOST_CHECK_EQUAL(forcing.GetChildren().size(), 2u);

    // Repeated names are how repeated entries are represented, and document
    // order is preserved.
    std::vector<const SessionNode *> forces = forcing.GetChildren("FORCE");
    BOOST_REQUIRE_EQUAL(forces.size(), 2u);
    BOOST_CHECK_EQUAL(forces[0]->GetAttribute("TYPE"), "Absorption");
    BOOST_CHECK_EQUAL(forces[1]->GetAttribute("TYPE"), "Noise");

    // GetChild() and GetChildPtr() return the first match.
    BOOST_CHECK_EQUAL(forcing.GetChild("FORCE").GetAttribute("TYPE"),
                      "Absorption");
    BOOST_CHECK(forcing.GetChildPtr("MAPPING") == nullptr);
    BOOST_CHECK_THROW(forcing.GetChild("MAPPING"), ErrorUtil::NekError);
}

BOOST_AUTO_TEST_CASE(TestSessionNodeValues)
{
    SessionNode force = *BuildForcingSection().GetChildren("FORCE")[1];

    BOOST_CHECK_EQUAL(force.GetValue("WHITENOISE"), "0.01");
    BOOST_CHECK_CLOSE(force.GetValueAs<NekDouble>("WHITENOISE"), 0.01, 1e-12);
    BOOST_CHECK_EQUAL(force.GetValueAs<int>("UPDATEFREQ"), 10);

    // The defaulted forms are used when the entry is absent.
    BOOST_CHECK_EQUAL(force.GetValue("NSTEPS", "none"), "none");
    BOOST_CHECK_EQUAL(force.GetValueAs<int>("NSTEPS", 0), 0);

    BOOST_CHECK_THROW(force.GetValue("NSTEPS"), ErrorUtil::NekError);
    BOOST_CHECK_THROW(force.GetValueAs<int>("WHITENOISE"), ErrorUtil::NekError);
}

BOOST_AUTO_TEST_CASE(TestSessionNodePaths)
{
    SessionNode forcing = BuildForcingSection();

    BOOST_CHECK_EQUAL(forcing.GetPath(), "FORCING");
    BOOST_CHECK_EQUAL(forcing.GetChild("FORCE").GetPath(), "FORCING/FORCE");
    BOOST_CHECK_EQUAL(forcing.GetChild("FORCE").GetChild("COEFF").GetPath(),
                      "FORCING/FORCE/COEFF");

    // Paths of an existing subtree are rewritten when it is grafted into a
    // parent, so they always describe the tree the node now belongs to.
    SessionNode root("NEKTAR");
    root.AddChild(forcing);
    BOOST_CHECK_EQUAL(
        root.GetChild("FORCING").GetChild("FORCE").GetChild("COEFF").GetPath(),
        "NEKTAR/FORCING/FORCE/COEFF");
}

BOOST_AUTO_TEST_CASE(TestSessionNodeConversions)
{
    SessionNode node("SETTINGS");
    node.AddChild(SessionNode("PADDED", "  42  "));
    node.AddChild(SessionNode("TRAILING", "3abc"));
    node.AddChild(SessionNode("NEGATIVE", "-1"));
    node.AddChild(SessionNode("HUGE", "999999999999999999999"));
    node.AddChild(SessionNode("EMPTY", ""));
    node.AddChild(SessionNode("REAL", "1.5e-3"));

    // Surrounding whitespace is ignored, so that <NSTEPS> 42 </NSTEPS> reads
    // as an integer for readers which do not strip it themselves.
    BOOST_CHECK_EQUAL(node.GetValueAs<int>("PADDED"), 42);
    BOOST_CHECK_CLOSE(node.GetValueAs<NekDouble>("REAL"), 1.5e-3, 1e-12);
    BOOST_CHECK_EQUAL(node.GetValueAs<int>("NEGATIVE"), -1);

    // Unlike std::stoi, trailing junk is rejected rather than ignored.
    BOOST_CHECK_THROW(node.GetValueAs<int>("TRAILING"), ErrorUtil::NekError);
    BOOST_CHECK_THROW(node.GetValueAs<int>("EMPTY"), ErrorUtil::NekError);

    // Out-of-range values are rejected rather than wrapping around, and a
    // negative value is never silently read as a large unsigned one.
    BOOST_CHECK_THROW(node.GetValueAs<int>("HUGE"), ErrorUtil::NekError);
    BOOST_CHECK_THROW(node.GetValueAs<size_t>("NEGATIVE"), ErrorUtil::NekError);

    // A string "conversion" always succeeds and is verbatim, so that values
    // which are later parsed by ParseUtils or the interpreter are untouched.
    BOOST_CHECK_EQUAL(node.GetValueAs<std::string>("PADDED"), "  42  ");
}

BOOST_AUTO_TEST_CASE(TestSessionNodeBooleans)
{
    SessionNode node("SETTINGS");
    node.AddChild(SessionNode("A", "true"));
    node.AddChild(SessionNode("B", "Yes"));
    node.AddChild(SessionNode("C", "ON"));
    node.AddChild(SessionNode("D", "1"));
    node.AddChild(SessionNode("E", "false"));
    node.AddChild(SessionNode("F", "No"));
    node.AddChild(SessionNode("G", "0"));
    node.AddChild(SessionNode("H", "maybe"));

    for (const std::string name : {"A", "B", "C", "D"})
    {
        BOOST_CHECK(node.GetValueAs<bool>(name));
    }
    for (const std::string name : {"E", "F", "G"})
    {
        BOOST_CHECK(!node.GetValueAs<bool>(name));
    }

    BOOST_CHECK_THROW(node.GetValueAs<bool>("H"), ErrorUtil::NekError);
    BOOST_CHECK(!node.GetValueAs<bool>("MISSING", false));
}

BOOST_AUTO_TEST_CASE(TestSessionNodeTimeLevel)
{
    // A section with no TIMELEVEL children applies at every time level.
    SessionNode plain = BuildForcingSection();
    BOOST_CHECK(plain.GetTimeLevel(0) == &plain);
    BOOST_CHECK(plain.GetTimeLevel(3) == &plain);

    SessionNode levelled("FORCE");
    SessionNode zero("TIMELEVEL");
    zero.SetAttribute("VALUE", "0");
    zero.AddChild(SessionNode("WHITENOISE", "0.01"));
    SessionNode one("TIMELEVEL");
    one.SetAttribute("VALUE", "1");
    one.AddChild(SessionNode("WHITENOISE", "0.02"));
    levelled.AddChild(zero);
    levelled.AddChild(one);

    const SessionNode *selected = levelled.GetTimeLevel(1);
    BOOST_REQUIRE(selected != nullptr);
    BOOST_CHECK_EQUAL(selected->GetValue("WHITENOISE"), "0.02");

    // A section which defines time levels but not this one simply does not
    // apply here, which the caller is expected to handle.
    BOOST_CHECK(levelled.GetTimeLevel(2, false) == nullptr);
    BOOST_CHECK_THROW(levelled.GetTimeLevel(2, true), ErrorUtil::NekError);
}

BOOST_AUTO_TEST_CASE(TestSessionNodeFromXml)
{
    const std::string xml = "<FORCING>"
                            "    <FORCE TYPE=\"Absorption\">"
                            "        <COEFF> SpongeCoefficient </COEFF>"
                            "        <BOUNDARYREGIONS> 0 </BOUNDARYREGIONS>"
                            "    </FORCE>"
                            "    <FORCE TYPE=\"Noise\">"
                            "        <WHITENOISE> 0.01 </WHITENOISE>"
                            "        <UPDATEFREQ> 10 </UPDATEFREQ>"
                            "    </FORCE>"
                            "</FORCING>";

    SessionNode parsed = ParseXml(xml);
    SessionNode built  = BuildForcingSection();

    // TinyXML condenses whitespace, so the text of an element arrives here
    // already stripped: the parsed tree should match the hand-built one
    // exactly.
    BOOST_CHECK_EQUAL(parsed.GetName(), built.GetName());
    BOOST_REQUIRE_EQUAL(parsed.GetChildren("FORCE").size(),
                        built.GetChildren("FORCE").size());

    for (size_t i = 0; i < parsed.GetChildren("FORCE").size(); ++i)
    {
        const SessionNode *p = parsed.GetChildren("FORCE")[i];
        const SessionNode *b = built.GetChildren("FORCE")[i];

        BOOST_CHECK_EQUAL(p->GetAttribute("TYPE"), b->GetAttribute("TYPE"));
        BOOST_REQUIRE_EQUAL(p->GetChildren().size(), b->GetChildren().size());

        for (size_t j = 0; j < p->GetChildren().size(); ++j)
        {
            BOOST_CHECK_EQUAL(p->GetChildren()[j].GetName(),
                              b->GetChildren()[j].GetName());
            BOOST_CHECK_EQUAL(p->GetChildren()[j].GetText(),
                              b->GetChildren()[j].GetText());
            BOOST_CHECK_EQUAL(p->GetChildren()[j].GetPath(),
                              b->GetChildren()[j].GetPath());
        }
    }
}

BOOST_AUTO_TEST_CASE(TestSessionNodeFromXmlEdgeCases)
{
    // An element whose children are all elements carries no text, and an
    // element with neither text nor children is still a valid, empty node.
    SessionNode parsed = ParseXml("<FORCE><EMPTY/></FORCE>");
    BOOST_CHECK(!parsed.DefinesText());
    BOOST_REQUIRE(parsed.DefinesChild("EMPTY"));
    BOOST_CHECK(!parsed.GetChild("EMPTY").DefinesText());
    BOOST_CHECK_EQUAL(parsed.GetChild("EMPTY").GetChildren().size(), 0u);

    // Comments carry no configuration and are dropped.
    parsed = ParseXml("<FORCE><!-- a comment --><A>1</A></FORCE>");
    BOOST_CHECK_EQUAL(parsed.GetChildren().size(), 1u);
    BOOST_CHECK_EQUAL(parsed.GetValue("A"), "1");

    // The root can be renamed, so that a section can be given its logical
    // name rather than whatever the document happened to call it.
    parsed = ParseXml("<Forcing><A>1</A></Forcing>", "FORCING");
    BOOST_CHECK_EQUAL(parsed.GetName(), "FORCING");
    BOOST_CHECK_EQUAL(parsed.GetChild("A").GetPath(), "FORCING/A");

    // A null element yields an empty node rather than a crash, so callers
    // may convert an optional section unconditionally.
    SessionNode none = LibUtilities::SessionNodeFromXml(nullptr, "FORCING");
    BOOST_CHECK_EQUAL(none.GetName(), "FORCING");
    BOOST_CHECK_EQUAL(none.GetChildren().size(), 0u);
}

} // namespace Nektar::SessionNodeUnitTests
