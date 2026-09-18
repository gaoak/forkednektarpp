////////////////////////////////////////////////////////////////////////////////
//
//  File: MacroElements.h
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
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//  Description: Elements a refinement replaced, kept for later mesh update.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKMESH_MODULE_MACROELEMENTS
#define NEKMESH_MODULE_MACROELEMENTS

#include <SpatialDomains/MeshGraph.h>

namespace Nektar::NekMesh
{

/**
 * @brief The coarse elements a refinement module has replaced.
 *
 * Boundary layer refinement takes an element out of the graph and puts
 * several in its place. It hands the original here rather than destroying it,
 * so that a later module can update the mesh dynamically -- re-refining from
 * the coarse element rather than from what the previous refinement left
 * behind. This is the sole owner of those entities: ProcessBL extracts them
 * from the MeshGraph, which no longer refers to them at all.
 *
 * Nothing in this branch reads them yet. They are here rather than on Mesh
 * because one module writing them and one module eventually reading them is
 * not a reason for every module to see them.
 *
 * Note that the default PayloadFollowsEntities applies: although the graph
 * does not own these entities, they refer to vertices and edges that it does,
 * so an orphan sweep can leave them unusable. Discarding them then is the
 * honest outcome -- a coarse element whose vertices have gone cannot be
 * refined again.
 */
struct MacroElements
{
    SpatialDomains::EntityHolder entities;
};

} // namespace Nektar::NekMesh

#endif
