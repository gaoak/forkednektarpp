///////////////////////////////////////////////////////////////////////////////
//
// File: TestOptimization.cpp
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

#define BOOST_TEST_MODULE TestOptimization

#include <UnitTests/MultiRegions/TestOptimization.hpp>

#include <StdRegions/StdHexExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>

#include <boost/test/tools/output_test_stream.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

BOOST_AUTO_TEST_SUITE(TestSuiteOptimization)

#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
// GetOpImpl() gives SumFacTOP only to an operator registered under it, which
// none is in a build without the Device execution space.
static std::string Registered(const std::string &opName,
                              const std::string &impl)
{
    return impl != "SumFacTOP" ||
                   GetBlockOperatorFactory<double>().ModuleExists(
                       "Block" + opName + "Device" + impl)
               ? impl
               : "SumFac";
}

BOOST_FIXTURE_TEST_CASE(SerialBackend, TestOptimization)
{
    Configure("Serial");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "StdMat");
    }
}

BOOST_FIXTURE_TEST_CASE(AVXBackend, TestOptimization)
{
    Configure("AVX");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFac");
    }
}

BOOST_FIXTURE_TEST_CASE(DeviceBackend, TestOptimization)
{
    Configure("Device");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFacTOP");
    }
}

BOOST_FIXTURE_TEST_CASE(SerialBackendOverride, TestOptimization)
{
    Configure("Serial", "SumFac");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFac");
    }
}

BOOST_FIXTURE_TEST_CASE(AVXBackendOverride, TestOptimization)
{
    Configure("AVX", "StdMat");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "StdMat");
    }
}

BOOST_FIXTURE_TEST_CASE(DeviceBackendOverride, TestOptimization)
{
    Configure("Device", "SumFac");
    auto execStr = Operator<double>::GetOpExecSpace(session);
    auto implStr =
        ElmtOp<FieldState::Coeff, FieldState::Phys, double>::GetOpImpl(
            "BwdTrans", execStr, session);

    // Check results
    boost::test_tools::output_test_stream output;
    {
        BOOST_TEST(implStr == "SumFac");
    }
}

// An operator the session's backend sections do not name takes the order
// dependent default: on Device, for the segments of this mesh, SumFac below
// order six and SumFacTOP from it.
BOOST_FIXTURE_TEST_CASE(DeviceDefaultByOrder, TestOptimization)
{
    Configure("Device");
    using Op = ElmtOp<FieldState::Phys, FieldState::Phys, double>;
    for (const int nmodes : {1, 2, 3, 4, 5, 6, 7, 8})
    {
        const auto exp             = ExpansionList(nmodes)->GetExp(0);
        const std::string expected = nmodes - 1 >= 6 ? "SumFacTOP" : "SumFac";
        BOOST_TEST(Op::GetOpImpl("PhysDeriv", "Device", session, exp) ==
                   Registered("PhysDeriv", expected));
        BOOST_TEST(Operator<double>::GetDefaultOpImpl(exp, "Device") ==
                   expected);
    }
}

// The order at which the Device default changes to SumFacTOP is set by the
// row of the element's shape in the default table: currently order three for
// a hex and a tet, order six for a quad and a triangle, on every architecture.
BOOST_AUTO_TEST_CASE(DeviceDefaultByShape)
{
    using namespace LibUtilities;
    for (int nmodes = 1; nmodes <= 8; ++nmodes)
    {
        const BasisKey a(eOrtho_A, nmodes,
                         PointsKey(nmodes + 1, eGaussGaussLegendre));
        const BasisKey b(eOrtho_B, nmodes,
                         PointsKey(nmodes + 1, eGaussRadauMAlpha1Beta0));
        const BasisKey c(eOrtho_C, nmodes,
                         PointsKey(nmodes + 1, eGaussRadauMAlpha2Beta0));
        const std::vector<std::pair<StdRegions::StdExpansionSharedPtr, int>>
            expAndMinOrder = {
                {MemoryManager<StdRegions::StdQuadExp>::AllocateSharedPtr(a, a),
                 6},
                {MemoryManager<StdRegions::StdTriExp>::AllocateSharedPtr(a, b),
                 6},
                {MemoryManager<StdRegions::StdHexExp>::AllocateSharedPtr(a, a,
                                                                         a),
                 3},
                {MemoryManager<StdRegions::StdTetExp>::AllocateSharedPtr(a, b,
                                                                         c),
                 3}};
        for (const auto &[exp, minOrder] : expAndMinOrder)
        {
            const std::string expected =
                nmodes - 1 >= minOrder ? "SumFacTOP" : "SumFac";
            const auto rule = FindDefaultOpImplRule(
                "Device", "", exp->DetShapeType(), nmodes - 1);
            BOOST_TEST((rule ? std::string(rule->opImpl) : "SumFac") ==
                       expected);

            // On this build's device the default is the row of its
            // architecture.
            const auto deviceRule =
                FindDefaultOpImplRule("Device", GetDeviceArchitecture(),
                                      exp->DetShapeType(), nmodes - 1);
            BOOST_TEST(
                Operator<double>::GetDefaultOpImpl(exp, "Device") ==
                (deviceRule ? std::string(deviceRule->opImpl) : "SumFac"));
        }
    }
}

// The order dependent default falls back to SumFac for an operator with no
// SumFacTOP implementation registered, and applies on Device only.
BOOST_FIXTURE_TEST_CASE(DefaultFallbacks, TestOptimization)
{
    Configure("Device");
    using Op       = ElmtOp<FieldState::Phys, FieldState::Phys, double>;
    const auto exp = ExpansionList(8)->GetExp(0);
    BOOST_TEST(Op::GetOpImpl("NoSuchOperator", "Device", session, exp) ==
               "SumFac");
    BOOST_TEST(Op::GetOpImpl("PhysDeriv", "Device", session) == "SumFac");
    BOOST_TEST(Operator<double>::GetDefaultOpImpl(exp, "AVX") == "SumFac");
    BOOST_TEST(Operator<double>::GetDefaultOpImpl(exp, "Serial") == "SumFac");
    // A backend section still takes precedence over the default.
    BOOST_TEST(Op::GetOpImpl("BwdTrans", "Device", session,
                             ExpansionList(2)->GetExp(0)) == "SumFacTOP");
}

// On a mesh of mixed order each block takes the default of its own order,
// and the default interleave widths follow it block by block: on Device the
// vector width for a SumFac block, 1 for a SumFacTOP one.
BOOST_FIXTURE_TEST_CASE(DeviceDefaultPerBlock, TestOptimization)
{
    Configure("Device");
    using Op = ElmtOp<FieldState::Phys, FieldState::Phys, double>;
    // The default applies only to a session without backend sections. Reading
    // the mesh initialises the session again, which parses them again, so
    // they are cleared once the expansion list exists.
    const auto expansionList = ExpansionList(3, 8);
    session->GetBackendMap("Device").clear();
    BOOST_REQUIRE_EQUAL(GetCollections(expansionList).size(), 2u);
    const std::vector<std::string> expectedImpl    = {"SumFac", "SumFacTOP"};
    const std::vector<unsigned int> expectedWidths = {
        NektarSpaces::GetVectorWidth<double>("Device"), 1u};
    for (unsigned int blk = 0; blk < 2; ++blk)
    {
        const auto exp = GetCollection(expansionList, blk).GetExpVector()[0];
        BOOST_TEST(Op::GetOpImpl("PhysDeriv", "Device", session, exp) ==
                   Registered("PhysDeriv", expectedImpl[blk]));
    }

    const auto widths =
        Operator<double>::GetDefaultInterleaveWidths(expansionList);
    BOOST_TEST(widths == expectedWidths, boost::test_tools::per_element());
    // Without widths, GetBlockAttributes() takes the defaults.
    const auto blockAttr =
        GetBlockAttributes<double, FieldState::Phys>(expansionList);
    for (unsigned int blk = 0; blk < 2; ++blk)
    {
        BOOST_TEST(blockAttr[blk].GetInterleaveWidth() == expectedWidths[blk]);
    }
}

// A backend section, or an implementation on the command line, gives every
// block the width of that one choice.
BOOST_FIXTURE_TEST_CASE(DefaultWidthsUniform, TestOptimization)
{
    const std::vector<unsigned int> ones = {1u, 1u};
    Configure("Device");
    BOOST_TEST(Operator<double>::GetDefaultInterleaveWidths(
                   ExpansionList(3, 5)) == ones,
               boost::test_tools::per_element());

    Configure("Device", "SumFacTOP");
    BOOST_TEST(Operator<double>::GetDefaultInterleaveWidths(
                   ExpansionList(3, 5)) == ones,
               boost::test_tools::per_element());

    const unsigned int avxWidth = NektarSpaces::GetVectorWidth<double>("AVX");
    Configure("AVX");
    BOOST_TEST(Operator<double>::GetDefaultInterleaveWidths(ExpansionList(
                   3, 5)) == std::vector<unsigned int>({avxWidth, avxWidth}),
               boost::test_tools::per_element());

    // Widths given to GetBlockAttributes() replace the defaults.
    const auto blockAttr =
        GetBlockAttributes<double, FieldState::Phys>(ExpansionList(3, 5), ones);
    for (const auto &attr : blockAttr)
    {
        BOOST_TEST(attr.GetInterleaveWidth() == 1u);
    }
}

// Every row of the default table names a known execution space and
// implementation and a well-formed architecture, and no two rows of one
// execution space, architecture and shape start at the same order.
BOOST_AUTO_TEST_CASE(DefaultOpImplTableWellFormed)
{
    const std::set<std::string_view> execSpaces = {"Serial", "AVX", "Device"};
    const std::set<std::string_view> impls = {"StdMat", "SumFac", "SumFacTOP",
                                              "Generic"};
    std::set<std::tuple<std::string_view, std::string_view,
                        LibUtilities::ShapeType, int>>
        keys;
    for (const auto &rule : defaultOpImplTable)
    {
        BOOST_TEST_INFO(DescribeDefaultOpImplRule(rule));
        BOOST_TEST(execSpaces.count(rule.execSpace) == 1u);
        BOOST_TEST(impls.count(rule.opImpl) == 1u);
        BOOST_TEST(!rule.arch.empty());
        BOOST_TEST(rule.arch.find('*') >= rule.arch.size() - 1);
        BOOST_TEST(rule.fromOrder >= 0);
        BOOST_TEST(
            keys.emplace(rule.execSpace, rule.arch, rule.shape, rule.fromOrder)
                .second);
    }
}

// A block takes the rows of the architecture that matches its device most
// closely (an exact id, then the longest prefix, then "*"), and among them
// the highest order at or below its own.
// A table with rows for every architecture, a family and one device, to
// check the architecture matching of FindDefaultOpImplRule().
const DefaultOpImplRule archTestTable[] = {
    {"Device", "*", LibUtilities::eHexahedron, 3, "SumFacTOP"},
    {"Device", "*", LibUtilities::eHexahedron, 6, "StdMat"},
    {"Device", "gfx9*", LibUtilities::eHexahedron, 4, "SumFacTOP"},
    {"Device", "gfx90a", LibUtilities::eHexahedron, 2, "SumFacTOP"},
    {"Device", "*", LibUtilities::eTetrahedron, 2, "SumFacTOP"}};

// The Device row of archTestTable a block takes, as "implementation@arch",
// or "none".
std::string FindInArchTestTable(const std::string_view arch,
                                const LibUtilities::ShapeType shape,
                                const int order)
{
    const auto rule = FindDefaultOpImplRule("Device", arch, shape, order,
                                            std::begin(archTestTable),
                                            std::end(archTestTable));
    return rule ? std::string(rule->opImpl) + "@" + std::string(rule->arch)
                : "none";
}

BOOST_AUTO_TEST_CASE(DefaultOpImplTableArch)
{
    using LibUtilities::eHexahedron;
    using LibUtilities::eTetrahedron;
    BOOST_TEST(DefaultOpImplArchMatch("*", "gfx90a") == 0);
    BOOST_TEST(DefaultOpImplArchMatch("*", "") == 0);
    BOOST_TEST(DefaultOpImplArchMatch("gfx9*", "gfx90a") == 4);
    BOOST_TEST(DefaultOpImplArchMatch("gfx9*", "sm_90") == -1);
    BOOST_TEST(DefaultOpImplArchMatch("gfx9*", "") == -1);
    BOOST_TEST(DefaultOpImplArchMatch("gfx90a", "gfx90a") ==
               std::numeric_limits<int>::max());
    BOOST_TEST(DefaultOpImplArchMatch("gfx90a", "gfx90") == -1);

    // "*" rows, with the highest order at or below the block's.
    BOOST_TEST(FindInArchTestTable("sm_90", eHexahedron, 2) == "none");
    BOOST_TEST(FindInArchTestTable("sm_90", eHexahedron, 3) == "SumFacTOP@*");
    BOOST_TEST(FindInArchTestTable("sm_90", eHexahedron, 7) == "StdMat@*");
    BOOST_TEST(FindInArchTestTable("", eHexahedron, 3) == "SumFacTOP@*");
    // A prefix replaces the "*" rows for its shape as a whole.
    BOOST_TEST(FindInArchTestTable("gfx942", eHexahedron, 3) == "none");
    BOOST_TEST(FindInArchTestTable("gfx942", eHexahedron, 7) ==
               "SumFacTOP@gfx9*");
    // An exact id replaces the prefix.
    BOOST_TEST(FindInArchTestTable("gfx90a", eHexahedron, 2) ==
               "SumFacTOP@gfx90a");
    // ... for its own shape only.
    BOOST_TEST(FindInArchTestTable("gfx90a", eTetrahedron, 2) == "SumFacTOP@*");
    // Another execution space has no rows here.
    BOOST_TEST(!FindDefaultOpImplRule("AVX", "", eHexahedron, 7,
                                      std::begin(archTestTable),
                                      std::end(archTestTable)));
}

// Print the Device defaults as a grid of shape by order, for every
// architecture the table names and for this build's device, so that a
// reviewer can see them without reading the rules (run with
// --log_level=message).
BOOST_AUTO_TEST_CASE(DefaultOpImplTableGrid)
{
    using namespace LibUtilities;
    std::set<std::string_view> archs = {"*"};
    for (const auto &rule : defaultOpImplTable)
    {
        archs.insert(rule.arch);
    }
    const std::string device = GetDeviceArchitecture();
    if (!device.empty())
    {
        archs.insert(device);
    }

    std::stringstream grid;
    for (const auto arch : archs)
    {
        // Prefixes are shown by their stem, which they match.
        const std::string_view stem =
            arch.back() == '*' ? arch.substr(0, arch.size() - 1) : arch;
        grid << "Device default, architecture " << arch
             << " (F SumFac, T SumFacTOP)\n"
             << "               P0 P1 P2 P3 P4 P5 P6 P7 P8\n";
        for (const auto shape : {eSegment, eQuadrilateral, eTriangle,
                                 eHexahedron, eTetrahedron, ePrism, ePyramid})
        {
            grid << std::left << std::setw(15) << ShapeTypeMap[shape];
            for (int order = 0; order <= 8; ++order)
            {
                const auto rule =
                    FindDefaultOpImplRule("Device", stem, shape, order);
                const std::string impl =
                    rule ? std::string(rule->opImpl) : "SumFac";
                grid << (impl == "SumFac"      ? "F"
                         : impl == "SumFacTOP" ? "T"
                                               : impl.substr(0, 1))
                     << (order < 8 ? "  " : "\n");
            }
        }
    }
    BOOST_TEST_MESSAGE(grid.str());
    BOOST_TEST(!grid.str().empty());
}
#endif

BOOST_AUTO_TEST_SUITE_END()
