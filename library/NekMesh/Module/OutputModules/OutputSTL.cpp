////////////////////////////////////////////////////////////////////////////////
//
//  File: OutputSTL.cpp
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
//  Description: STL surface writer.
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>

#include "OutputSTL.h"

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey OutputSTL::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "stl"), OutputSTL::create, "Writes STL file.");

OutputSTL::OutputSTL(MeshSharedPtr m) : OutputModule(m)
{
}

OutputSTL::~OutputSTL()
{
}

void OutputSTL::Process()
{
    std::string filename = m_config["outfile"].as<std::string>();
    m_log(VERBOSE) << "Writing STL file '" << filename << "'." << endl;

    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    if (meshDim != 2 && meshDim != 3)
    {
        m_log(FATAL) << "Only 2D or 3D meshes are supported." << endl;
    }

    // Binary, despite the contents being ASCII: STL is consumed by other
    // tools and compared byte-for-byte by the regression test, so the line
    // endings must not depend on the platform we write it from.
    bool stream_open = OpenStream(true);
    if (!stream_open)
    {
        return;
    }

    m_mshFile << std::scientific << setprecision(8);

    // One facet per element: the outward normal from its first three
    // vertices, then the vertices themselves.
    auto writeFacet = [&](SpatialDomains::Geometry *geom) {
        const int nVerts = geom->GetNumVerts();
        ASSERTL0(nVerts >= 3, "An STL facet needs at least three vertices");

        auto v = [&](int i, int d) { return (*geom->GetVertex(i))[d]; };

        std::array<NekDouble, 3> n;
        for (int d = 0; d < 3; ++d)
        {
            const int a = (d + 1) % 3, b = (d + 2) % 3;
            n[d] = (v(1, a) - v(0, a)) * (v(2, b) - v(0, b)) -
                   (v(1, b) - v(0, b)) * (v(2, a) - v(0, a));
        }

        NekDouble mt = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int d = 0; d < 3; ++d)
        {
            n[d] /= mt;
        }

        m_mshFile << "facet normal " << n[0] << " " << n[1] << " " << n[2]
                  << endl;
        m_mshFile << "outer loop" << endl;
        for (int j = 0; j < nVerts; j++)
        {
            m_mshFile << "vertex " << v(j, 0) << " " << v(j, 1) << " "
                      << v(j, 2) << endl;
        }
        m_mshFile << "endloop" << endl << "endfacet" << endl;
    };

    if (meshDim == 2)
    {
        m_mshFile << "solid comp:" << 0 << endl;

        for ([[maybe_unused]] auto &[geom, tag] : m_mesh->m_elementTags[2])
        {
            writeFacet(geom);
        }

        m_mshFile << "endsolid" << endl;
        return;
    }

    // In 3D the surface is whatever the two-dimensional composites hold.
    for (auto &it : m_mesh->m_meshGraph->GetComposites())
    {
        if (it.second->m_geomVec.empty() ||
            it.second->m_geomVec[0]->GetShapeDim() != 2)
        {
            continue;
        }

        m_mshFile << "solid comp:" << it.first << endl;

        for (auto &geom : it.second->m_geomVec)
        {
            writeFacet(geom);
        }

        m_mshFile << "endsolid" << endl;
    }
}
} // namespace Nektar::NekMesh
