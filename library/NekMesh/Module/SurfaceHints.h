////////////////////////////////////////////////////////////////////////////////
//
//  File: SurfaceHints.h
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
//  Description: Surface reconstruction hints passed between modules.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKMESH_MODULE_SURFACEHINTS
#define NEKMESH_MODULE_SURFACEHINTS

#include <array>
#include <unordered_map>
#include <vector>

#include <SpatialDomains/Geometry.h>
#include <SpatialDomains/PointGeom.h>

namespace Nektar::NekMesh
{

/**
 * @brief True surface normals at mesh vertices, where a file supplies them.
 *
 * Spherigon smoothing reconstructs a curved surface from a faceted one, and
 * does so far better when told the normal of the true surface at each vertex
 * than when left to average the normals of the surrounding facets. Some input
 * formats carry exactly that: a .ply may have nx/ny/nz vertex properties, and
 * a Nektar .rea file's spherigon curved sides give a normal per face vertex.
 *
 * Neither the normals nor the surfaces below are mesh data -- nothing writes
 * them to an output file and only the spherigon module reads them -- so they
 * travel through the mesh's ModuleContext rather than as Mesh members.
 *
 * Keyed on the vertex itself rather than its id: ids are reassigned by
 * Module::ReorderPrisms() and friends, and prismatic boundary layer meshes
 * are precisely where spherigons get used.
 */
struct VertexNormals
{
    std::unordered_map<SpatialDomains::PointGeom *, std::array<NekDouble, 3>>
        normals;
};

/**
 * @brief Faces or edges an input module has asked to be spherigon smoothed.
 *
 * A .rea file marks individual element sides as spherigon sides, which is
 * finer grained than a composite tag and is not recoverable from the mesh
 * once read. Where an input module has no opinion this is simply absent and
 * the spherigon module's @c surf option selects the surfaces instead.
 */
struct SpherigonSurfs
{
    std::vector<SpatialDomains::Geometry *> surfs;
};

} // namespace Nektar::NekMesh

#endif
