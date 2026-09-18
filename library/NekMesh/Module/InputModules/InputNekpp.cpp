////////////////////////////////////////////////////////////////////////////////
//
//  File: InputNekpp.cpp
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
//  Description: GMSH converter.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/CppCommandLine.hpp>
#include <iostream>
#include <string>

#include "InputNekpp.h"
#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace std;

namespace Nektar::NekMesh
{

using namespace Nektar::NekMesh;

ModuleKey InputNekpp::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "xml"), InputNekpp::create,
    "Reads Nektar++ xml file.");
/**
 * @brief Set up InputNekpp object.
 */
InputNekpp::InputNekpp(MeshSharedPtr m) : InputModule(m)
{
    m_config["prismreorder"] = ConfigOption(
        true, "0",
        "Reorder prisms to align vertices and faces along lines of prisms");
}

InputNekpp::~InputNekpp()
{
}

/**
 *
 */
void InputNekpp::Process()
{
    vector<string> filename;
    filename.push_back(m_config["infile"].as<string>());

    m_log(VERBOSE) << "Reading Nektar++ XML file '" << filename[0] << "'"
                   << endl;

    LibUtilities::CommSharedPtr pComm =
        m_mesh->m_comm ? m_mesh->m_comm : LibUtilities::CommSharedPtr();

    LibUtilities::CppCommandLine cmd({"NekMesh"});
    LibUtilities::SessionReaderSharedPtr pSession =
        LibUtilities::SessionReader::CreateInstance(cmd.GetArgc(),
                                                    cmd.GetArgv(), filename);
    m_mesh->m_meshGraph = SpatialDomains::MeshGraphIO::Read(pSession);

    for (const auto &pair : m_mesh->m_meshGraph->GetComposites())
    {
        int compDim = pair.second->m_geomVec[0]->GetShapeDim();
        for (SpatialDomains::Geometry *geomPtr : pair.second->m_geomVec)
        {
            m_mesh->m_elementTags[compDim][geomPtr] = pair.first;
        }
        if (compDim == 3)
        {
            for (SpatialDomains::Geometry *geomPtr : pair.second->m_geomVec)
            {
                m_mesh->m_meshGraph->PopulateFaceToElMap(
                    static_cast<SpatialDomains::Geometry3D *>(geomPtr),
                    geomPtr->GetNumFaces());
            }
        }
    }

    for (auto [id, geom] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        m_mesh->m_edgeSet[std::pair(geom->GetVid(0), geom->GetVid(1))] = geom;
    }
    for (auto [id, geom] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        std::array<int, 4> vids = {geom->GetVid(0), geom->GetVid(1),
                                   geom->GetVid(2), -1};
        m_mesh->m_faceSet[vids] = geom;
    }
    for (auto [id, geom] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        std::array<int, 4> vids = {geom->GetVid(0), geom->GetVid(1),
                                   geom->GetVid(2), geom->GetVid(3)};
        m_mesh->m_faceSet[vids] = geom;
    }

    auto comm = pSession->GetComm();

    if (comm->GetType().find("MPI") != std::string::npos)
    {
        m_mesh->m_comm = comm;
    }

    if (m_config["prismreorder"].beenSet)
    {
        PerMap perFaces;
        ReorderPrisms(perFaces);
    }
}
} // namespace Nektar::NekMesh