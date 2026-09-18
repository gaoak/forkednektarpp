////////////////////////////////////////////////////////////////////////////////
//
//  File: OutputVtk.cpp
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
//  Description: VTK file format output.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/VtkUtil.hpp>
#include <NekMesh/MeshElements/Element.h>

#include "OutputVtk.h"
#include <vtkCellType.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey OutputVtk::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "vtk"), OutputVtk::create, "Writes a VTK file.");

OutputVtk::OutputVtk(MeshSharedPtr m) : OutputModule(m)
{
    m_config["uncompress"] = ConfigOption(true, "0", "Uncompress xml sections");
    m_config["legacy"]     = ConfigOption(true, "0", "Output in legacy format");
}

OutputVtk::~OutputVtk()
{
}

int OutputVtk::GetVtkCellType(LibUtilities::ShapeType pType)
{
    switch (pType)
    {
        case LibUtilities::eSegment:
            return VTK_LINE;
        case LibUtilities::eTriangle:
            return VTK_TRIANGLE;
        case LibUtilities::eQuadrilateral:
            return VTK_QUAD;
        case LibUtilities::eTetrahedron:
            return VTK_TETRA;
        case LibUtilities::ePyramid:
            return VTK_PYRAMID;
        case LibUtilities::ePrism:
            return VTK_WEDGE;
        case LibUtilities::eHexahedron:
            return VTK_HEXAHEDRON;
        default:
            m_log(FATAL) << "Element type " << LibUtilities::ShapeTypeMap[pType]
                         << " is not supported by the VTK writer." << endl;
            return 0;
    }
}

void OutputVtk::Process()
{
    std::string filename = m_config["outfile"].as<string>();
    m_log(VERBOSE) << "Writing VTK file '" << filename << "'." << endl;

    // Check whether file exists.
    if (!CheckOverwrite(filename))
    {
        return;
    }

    vtkSmartPointer<vtkUnstructuredGrid> vtkMesh =
        vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> vtkMeshPoints =
        vtkSmartPointer<vtkPoints>::New();

    // VTK indexes points by position in its own array, so the mesh's global
    // ids -- which need be neither contiguous nor start at zero -- have to be
    // mapped rather than used directly. The geometry map is ordered by id, so
    // the resulting point order is deterministic.
    std::unordered_map<int, vtkIdType> ptIdMap;

    for (auto &[id, vert] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        ptIdMap[id] =
            vtkMeshPoints->InsertNextPoint((*vert)[0], (*vert)[1], (*vert)[2]);
    }

    // Write the elements out in a deterministic order: the tag map is keyed on
    // pointers, whose iteration order varies between runs.
    std::vector<SpatialDomains::Geometry *> elmt;
    for (auto &entry :
         m_mesh->m_elementTags[m_mesh->m_meshGraph->GetMeshDimension()])
    {
        elmt.push_back(entry.first);
    }
    std::sort(elmt.begin(), elmt.end(),
              [](SpatialDomains::Geometry *a, SpatialDomains::Geometry *b) {
                  return std::make_pair(static_cast<int>(a->GetShapeType()),
                                        a->GetGlobalID()) <
                         std::make_pair(static_cast<int>(b->GetShapeType()),
                                        b->GetGlobalID());
              });

    vtkIdType p[8];
    for (auto &el : elmt)
    {
        int vertexCount = el->GetNumVerts();
        for (int j = 0; j < vertexCount; ++j)
        {
            p[j] = ptIdMap[el->GetVertex(j)->GetGlobalID()];
        }
        // Adjust vertex order to the vtk convention
        if (el->GetShapeType() == LibUtilities::ePrism)
        {
            std::swap(p[2], p[4]);
        }
        vtkMesh->InsertNextCell(GetVtkCellType(el->GetShapeType()), vertexCount,
                                &p[0]);
    }

    vtkMesh->SetPoints(vtkMeshPoints);

    // Write out the new mesh in XML or legacy format
    if (m_config["legacy"].beenSet)
    {
        vtkSmartPointer<vtkUnstructuredGridWriter> vtkMeshWriter =
            vtkSmartPointer<vtkUnstructuredGridWriter>::New();
        vtkMeshWriter->SetFileName(filename.c_str());

#if VTK_MAJOR_VERSION <= 5
        vtkMeshWriter->SetInput(vtkMesh);
#else
        vtkMeshWriter->SetInputData(vtkMesh);
#endif
        vtkMeshWriter->Update();
    }
    else // XML format
    {

        vtkSmartPointer<vtkXMLUnstructuredGridWriter> vtkMeshWriter =
            vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        vtkMeshWriter->SetFileName(filename.c_str());

#if VTK_MAJOR_VERSION <= 5
        vtkMeshWriter->SetInput(vtkMesh);
#else
        vtkMeshWriter->SetInputData(vtkMesh);
#endif
        if (m_config["uncompress"].beenSet)
        {
            vtkMeshWriter->SetDataModeToAscii();
        }
        vtkMeshWriter->Update();
    }
}
} // namespace Nektar::NekMesh
