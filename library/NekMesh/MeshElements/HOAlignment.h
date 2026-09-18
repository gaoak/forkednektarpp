////////////////////////////////////////////////////////////////////////////////
//
//  File: HOAlignment.h
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
//  Description: High-order surfaces keyed on NekMesh nodes. The alignment
//               templates themselves live in SpatialDomains, since the
//               geometries need them too.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKMESH_MESHELEMENTS_ALIGNMENT
#define NEKMESH_MESHELEMENTS_ALIGNMENT

#include <SpatialDomains/HOAlignment.h>

namespace Nektar::NekMesh
{

using SpatialDomains::HOQuadrilateral;
using SpatialDomains::HOTriangle;

typedef HOTriangle<SpatialDomains::PointGeom *> HOSurf;
typedef std::shared_ptr<HOSurf> HOSurfSharedPtr;

/**
 * Hash class for high-order surfaces.
 */
struct HOSurfHash
{
    /**
     * Calculate hash of a given high-order surface p by taking
     * successive hashes of the vertex IDs.
     */
    std::size_t operator()(HOSurfSharedPtr const &p) const
    {
        std::vector<int> ids = p->vertId;
        std::sort(ids.begin(), ids.end());
        return hash_range(ids.begin(), ids.end());
    }
};

/**
 * @brief Equality for high-order surfaces: the same three vertices, in any
 * order.
 *
 * Stated explicitly as the set's comparator rather than left to
 * std::equal_to. That would resolve to a free operator== only by argument
 * dependent lookup, and whether it is found at all depends on the namespace
 * of HOSurf's template argument -- so changing that argument silently
 * demotes the set to comparing shared_ptr identity, whereupon no lookup by
 * value ever matches.
 */
struct HOSurfEqual
{
    bool operator()(HOSurfSharedPtr const &p1, HOSurfSharedPtr const &p2) const
    {
        if (p1->vertId.size() != p2->vertId.size())
        {
            return false;
        }

        std::vector<int> ids1 = p1->vertId, ids2 = p2->vertId;
        std::sort(ids1.begin(), ids1.end());
        std::sort(ids2.begin(), ids2.end());

        return ids1 == ids2;
    }
};

typedef std::unordered_set<HOSurfSharedPtr, HOSurfHash, HOSurfEqual> HOSurfSet;

} // namespace Nektar::NekMesh

#endif
