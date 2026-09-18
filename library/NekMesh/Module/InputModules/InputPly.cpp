////////////////////////////////////////////////////////////////////////////////
//
//  File: InputPly.cpp
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
//  Description: PLY converter.
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/Module/SurfaceHints.h>

#include "InputPly.h"

using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey InputPly::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "ply"), InputPly::create,
    "Reads ply triangulation format.");

InputPly::InputPly(MeshSharedPtr m) : InputModule(m)
{
}

InputPly::~InputPly()
{
}

/**
 *
 * @param   pFilename           Filename of Gmsh file to read.
 */
void InputPly::Process()
{
    // Open the file stream.
    OpenStream();

    ReadPly(m_mshFile);

    m_mshFile.reset();

    // Finishing processes
    ProcessElements();
    ProcessComposites();

    PrintSummary();
}

void InputPly::ReadPly(io::filtering_istream &mshFile, NekDouble scale)
{
    std::string line;
    int nVertices                  = 0;
    int nEntities                  = 0;
    int nProperties                = 0;
    LibUtilities::ShapeType elType = LibUtilities::eTriangle;
    std::map<std::string, int> propMap;

    m_log(VERBOSE) << "Reading .ply file '"
                   << m_config["infile"].as<std::string>() << "'" << std::endl;

    // Keep track of spaceDim as we read the grid. Will reset nodes later with
    // correct coordim.
    int spaceDim = 0;

    // Always assume a triangulation.
    m_mesh->m_meshGraph->SetMeshDimension(2);

    while (!mshFile.eof())
    {
        std::getline(mshFile, line);
        std::stringstream s(line);
        std::string word;
        s >> word;
        if (word == "format")
        {
            s >> word;
            if (word != "ascii")
            {
                m_log(FATAL) << "Currently only ASCII-formatted .ply files are "
                             << "supported." << std::endl;
            }
        }
        else if (word == "element")
        {
            s >> word;
            if (word == "vertex")
            {
                s >> nVertices;
            }
            else if (word == "face")
            {
                s >> nEntities;
            }
            continue;
        }
        else if (word == "property")
        {
            s >> word >> word;
            propMap[word] = nProperties++;
        }
        else if (word == "end_header")
        {
            // Read nodes
            std::vector<double> data(nProperties);
            for (int i = 0; i < nVertices; ++i)
            {
                std::getline(mshFile, line);
                std::stringstream st(line);

                for (int j = 0; j < nProperties; ++j)
                {
                    st >> data[j];
                }

                double x = data[propMap["x"]];
                double y = data[propMap["y"]];
                double z = data[propMap["z"]];

                if ((x * x) > 0.000001 && spaceDim < 1)
                {
                    spaceDim = 1;
                    UpdateCoordim(m_mesh->m_meshGraph, i, spaceDim);
                }
                if ((y * y) > 0.000001 && spaceDim < 2)
                {
                    spaceDim = 2;
                    UpdateCoordim(m_mesh->m_meshGraph, i, spaceDim);
                }
                if ((z * z) > 0.000001 && spaceDim < 3)
                {
                    spaceDim = 3;
                    UpdateCoordim(m_mesh->m_meshGraph, i, spaceDim);
                }

                x *= scale;
                y *= scale;
                z *= scale;

                m_mesh->m_meshGraph->CreatePointGeom(spaceDim, i, x, y, z);

                // A .ply may carry the true surface normal at each vertex in
                // nx/ny/nz. Nothing writes those to an output file, so they
                // are handed to whichever later module wants them -- the
                // spherigon module, which smooths far better with them than
                // with normals averaged from the facets.
                if (propMap.count("nx") > 0)
                {
                    m_mesh->GetContext()
                        .Get<VertexNormals>()
                        .normals[m_mesh->m_meshGraph->GetPointGeom(i)] = {
                        data[propMap["nx"]], data[propMap["ny"]],
                        data[propMap["nz"]]};
                }
            }

            // Read elements
            for (int i = 0; i < nEntities; ++i)
            {
                std::getline(mshFile, line);
                std::stringstream st(line);
                int id = 0;

                // Read element node list
                st >> id;
                std::vector<SpatialDomains::PointGeom *> nodeList;
                for (int k = 0; k < 3; ++k)
                {
                    int node = 0;
                    st >> node;
                    nodeList.push_back(m_mesh->m_meshGraph->GetPointGeom(node));
                }

                // Create element
                ElmtConfig conf(elType, 1, false, false);
                SpatialDomains::Geometry *element =
                    GetElementFactory().CreateInstance(
                        elType, nodeList, m_mesh->m_meshGraph,
                        m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr,
                        nullptr, nullptr, nullptr);

                auto shapeDim = LibUtilities::ShapeTypeDimMap[elType];
                m_mesh->m_elementTags[shapeDim][element] = 0;
            }
        }
    }

    // Set final space dimension on the meshgraph.
    m_mesh->m_meshGraph->SetSpaceDimension(spaceDim);
}
} // namespace Nektar::NekMesh
