///////////////////////////////////////////////////////////////////////////////
//
//  File: CompositeNames.h
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
#ifndef NEKTAR_SPATIALDOMAINS_COMPOSITENAMES_H
#define NEKTAR_SPATIALDOMAINS_COMPOSITENAMES_H

#include <SpatialDomains/SpatialDomainsDeclspec.h>

#include <map>
#include <string>
#include <vector>

namespace Nektar::SpatialDomains
{

/**
 * @brief Expand the body of a composite list into composite IDs, resolving
 * any entry that is a name rather than a number.
 *
 * @p str is what stands between the brackets of a `C[...]`: a comma-separated
 * list whose entries are a number, a `first-last` range of numbers, or the
 * name of a composite. So `0-3,7`, `WALL` and `0-3,WALL,INLET` are all
 * accepted, and a name may contain spaces since only commas separate entries.
 *
 * A numeric entry is always read as an ID, so a composite that happens to be
 * named "3" never shadows the composite whose ID is 3; this keeps every list
 * that could be written before names existed meaning exactly what it did.
 * Names are matched exactly, including case, because a label comes from the
 * mesh rather than from Nektar++ and folding it would be guessing. Where
 * several composites carry one name -- which is how a mesh says two sets of
 * faces are one patch -- the name yields all of them.
 *
 * Entries that are neither a number nor a known name are appended to
 * @p unresolved rather than failing here, since whether an unknown name is an
 * error depends on the caller: on one partition of a mesh a name may
 * legitimately refer to nothing local.
 *
 * @param str         Body of the composite list.
 * @param labels      Composite ID to name, as MeshGraph holds it.
 * @param ids         Resolved composite IDs, appended in the order listed.
 * @param unresolved  Entries that resolved to nothing, appended.
 *
 * @return True if every entry resolved and there was at least one; false
 *         otherwise, including for a list that is empty or all separators.
 */
/**
 * @brief Make a name read out of a mesh file usable as a composite label.
 *
 * Mesh formats are more permissive about names than the places a label has to
 * survive: it becomes an XML attribute, and it is meant to be referable from a
 * composite list, where a comma separates one entry from the next and brackets
 * delimit the list. A name containing any of those could never be written down
 * or referred to, so they are dropped along with surrounding whitespace.
 * Spaces within a name are kept, since a list can carry those.
 */
SPATIAL_DOMAINS_EXPORT std::string SanitiseCompositeName(
    const std::string &name);

SPATIAL_DOMAINS_EXPORT bool ParseCompositeList(
    const std::string &str, const std::map<int, std::string> &labels,
    std::vector<unsigned int> &ids,
    std::vector<std::string> *unresolved = nullptr);

} // namespace Nektar::SpatialDomains

#endif
