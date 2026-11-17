///////////////////////////////////////////////////////////////////////////////
//
// File: DefaultOpImplTable.hpp
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

/**
 * @file DefaultOpImplTable.hpp
 * @brief The table of default operator implementations, by execution space,
 * device architecture, element shape and expansion order.
 *
 * @details
 * Operator::GetDefaultOpImpl() reads the default implementation of a block
 * from this table when neither the command line ("--opImpl") nor a backend
 * section of the session names one. The table is kept as data, with the
 * evidence for each row beside it, so that the history of this file is the
 * history of the defaults: change a row only together with the timings that
 * justify it, and say in the row's comment and in the commit message where
 * they are.
 */

#pragma once

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>

#include <LibUtilities/BasicUtils/ShapeType.hpp>

namespace Nektar::MultiRegions
{

/**
 * @brief One row of the default implementation table.
 *
 * A row gives the implementation that blocks of one shape take, in one
 * execution space and on matching devices, from one expansion order upward.
 */
struct DefaultOpImplRule
{
    /// Execution space: "Serial", "AVX" or "Device".
    std::string_view execSpace;
    /// Device architecture the row applies to, as GetDeviceArchitecture()
    /// names it: "*" for every architecture, a prefix ending in '*' for a
    /// family (e.g. "gfx9*") or an exact id (e.g. "sm_90", "gfx90a").
    std::string_view arch;
    /// Element shape, as the block's expansion's DetShapeType() gives it.
    LibUtilities::ShapeType shape;
    /// Lowest expansion order (number of modes less one) the row applies to.
    int fromOrder;
    /// Implementation the row selects.
    std::string_view opImpl;
};

/**
 * @brief The default implementation table.
 *
 * A block takes the row of its execution space and shape whose @c arch
 * matches the device most closely (an exact id before a prefix, a longer
 * prefix before a shorter one, and any of them before "*"), and among the
 * rows of that @c arch the one with the highest @c fromOrder at or below the
 * block's order. A block that no row matches takes SumFac, so the table
 * lists only the departures from it: Serial and AVX need no rows.
 *
 * The rows of one @c arch replace those of a less specific one for that
 * shape as a whole, so a row added for one architecture must give every
 * switch that architecture makes.
 */
inline constexpr DefaultOpImplRule defaultOpImplTable[] = {
    // clang-format off
    // exec    arch shape                          from impl
    // Untimed; takes the 1D/2D order (!2931).
    {"Device", "*", LibUtilities::eSegment,        6,   "SumFacTOP"},
    // sm_90 (GH200): ProfilerElmtOps, BwdTrans, IProductWRTBase,
    // IProductWRTDerivBase and PhysDeriv on 4k and 65k elements, SumFac
    // faster through P5 (!2931). P6 is also where SumFac leaves its
    // mode-specialised kernels at the default NEKTAR_SWITCH_MAX. gfx90a
    // (MI210) untimed.
    {"Device", "*", LibUtilities::eQuadrilateral,  6,   "SumFacTOP"},
    {"Device", "*", LibUtilities::eTriangle,       6,   "SumFacTOP"},
    // sm_90 (GH200) and gfx90a (MI210): TGV P0-P4; sm_90 ProfilerElmtOps
    // crossing at P3 (!2931).
    {"Device", "*", LibUtilities::eHexahedron,     3,   "SumFacTOP"},
    // sm_90 (GH200) and gfx90a (MI210): TGV P0-P4 (!2931). The sm_90
    // ProfilerElmtOps crossing at P2 was timed on an all-GL point rule, not
    // the default GL x GL x GJ(1,0); rerun pending.
    {"Device", "*", LibUtilities::eTetrahedron,    3,   "SumFacTOP"},
    // Untimed; take the 3D order (!2931).
    {"Device", "*", LibUtilities::ePrism,          3,   "SumFacTOP"},
    {"Device", "*", LibUtilities::ePyramid,        3,   "SumFacTOP"},
    // clang-format on
};

/**
 * @brief Return how closely the architecture pattern @p pattern of a
 * DefaultOpImplRule matches the device architecture @p arch.
 *
 * @return -1 if it does not match; 0 for "*"; the length of the prefix for
 * a pattern "prefix*"; and, above every prefix, the largest int for an exact
 * id.
 */
constexpr int DefaultOpImplArchMatch(const std::string_view pattern,
                                     const std::string_view arch)
{
    int match = -1;
    if (!pattern.empty() && pattern.back() == '*')
    {
        const std::string_view prefix = pattern.substr(0, pattern.size() - 1);
        if (arch.substr(0, prefix.size()) == prefix)
        {
            match = static_cast<int>(prefix.size());
        }
    }
    else if (pattern == arch)
    {
        match = (std::numeric_limits<int>::max)();
    }
    return match;
}

/**
 * @brief Return whether the row @p rule of a default implementation table
 * is for execution space @p execSpace and element shape @p shape, whatever
 * its architecture and order.
 */
constexpr bool DefaultOpImplRuleApplies(const DefaultOpImplRule &rule,
                                        const std::string_view execSpace,
                                        const LibUtilities::ShapeType shape)
{
    return rule.execSpace == execSpace && rule.shape == shape;
}

/**
 * @brief Return the row of a default implementation table that a block
 * takes, or null if none does (the block then takes SumFac).
 *
 * @param execSpace  Execution space ("Serial", "AVX" or "Device").
 * @param arch       Architecture of the device, as GetDeviceArchitecture()
 *                   returns it; empty matches only "*".
 * @param shape      Shape of the block's elements.
 * @param order      Expansion order of the block's elements.
 * @param first      First row of the table.
 * @param last       One past the last row of the table.
 */
inline const DefaultOpImplRule *FindDefaultOpImplRule(
    const std::string_view execSpace, const std::string_view arch,
    const LibUtilities::ShapeType shape, const int order,
    const DefaultOpImplRule *first = std::begin(defaultOpImplTable),
    const DefaultOpImplRule *last  = std::end(defaultOpImplTable))
{
    // The rows of the most specific matching architecture.
    int closest = -1;
    for (auto rule = first; rule != last; ++rule)
    {
        if (DefaultOpImplRuleApplies(*rule, execSpace, shape))
        {
            closest =
                (std::max)(closest, DefaultOpImplArchMatch(rule->arch, arch));
        }
    }

    // Among them, the highest order switch at or below the block's order;
    // none if no row of the execution space and shape matches the
    // architecture.
    const DefaultOpImplRule *found = nullptr;
    if (closest >= 0)
    {
        for (auto rule = first; rule != last; ++rule)
        {
            if (DefaultOpImplRuleApplies(*rule, execSpace, shape) &&
                DefaultOpImplArchMatch(rule->arch, arch) == closest &&
                rule->fromOrder <= order &&
                (found == nullptr || rule->fromOrder > found->fromOrder))
            {
                found = rule;
            }
        }
    }
    return found;
}

/**
 * @brief Describe a row of the default implementation table, e.g.
 * "Device, *, Hexahedron, from order 3", for messages.
 */
inline std::string DescribeDefaultOpImplRule(const DefaultOpImplRule &rule)
{
    return std::string(rule.execSpace) + ", " + std::string(rule.arch) + ", " +
           LibUtilities::ShapeTypeMap[rule.shape] + ", from order " +
           std::to_string(rule.fromOrder);
}

} // namespace Nektar::MultiRegions
