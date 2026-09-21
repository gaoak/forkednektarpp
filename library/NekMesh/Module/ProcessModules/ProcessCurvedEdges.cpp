////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessCurvedEdges.cpp
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
//  Description: Abstract base class for creating curved edges on boundaries.
//
////////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/NodalTriExp.h>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/SegExp.h>
#include <LocalRegions/TriExp.h>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <NekMesh/MeshElements/Element.h>

#include "ProcessCurvedEdges.h"

using namespace std;

namespace Nektar::NekMesh
{
/**
 * @brief Default constructor.
 */
ProcessCurvedEdges::ProcessCurvedEdges(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["surf"] =
        ConfigOption(false, "-1", "Tag identifying surface to process.");
    m_config["N"] = ConfigOption(false, "7", "Number of points along edge.");
}

/**
 * @brief Destructor.
 */
ProcessCurvedEdges::~ProcessCurvedEdges()
{
}

void ProcessCurvedEdges::Process()
{
    int surfTag         = m_config["surf"].as<int>();
    int prismedge[2][3] = {{0, 5, 4}, {2, 6, 7}};
    int dim             = m_mesh->m_meshGraph->GetMeshDimension();

    ASSERTL0(dim == 2 || dim == 3, "Dimension not supported");

    // The boundary element that a face or edge of an element belongs to is
    // found by looking the geometry itself up among the tagged entities one
    // dimension down, which is what Element::GetBoundaryLink() used to
    // record.
    auto &bndTags = m_mesh->m_elementTags[dim - 1];

    for (auto &[el, elTag] : m_mesh->m_elementTags[dim])
    {
        (void)elTag; // unused

        int nSurf = dim == 3 ? el->GetNumFaces() : el->GetNumEdges();

        for (int j = 0; j < nSurf; ++j)
        {
            SpatialDomains::Geometry *bnd =
                dim == 3
                    ? static_cast<SpatialDomains::Geometry *>(el->GetFace(j))
                    : static_cast<SpatialDomains::Geometry *>(el->GetEdge(j));

            auto blIt = bndTags.find(bnd);
            if (blIt == bndTags.end() || blIt->second != surfTag)
            {
                continue;
            }

            if (dim == 2)
            {
                GenerateEdgeNodes(
                    static_cast<SpatialDomains::SegGeom *>(el->GetEdge(j)));
            }
            else
            {
                ASSERTL0(j == 1 || j == 3,
                         "Curved edge needs to be on prism triangular face");

                // Check all edge interior points.
                for (int k = 0; k < 3; ++k)
                {
                    GenerateEdgeNodes(static_cast<SpatialDomains::SegGeom *>(
                        el->GetEdge(prismedge[(j - 1) / 2][k])));
                }
            }
        }
    }
}
} // namespace Nektar::NekMesh
