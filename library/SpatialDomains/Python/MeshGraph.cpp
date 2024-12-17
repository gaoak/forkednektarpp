////////////////////////////////////////////////////////////////////////////////
//
//  File: MeshGraph.cpp
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
//  Description: Python wrapper for MeshGraph.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Python/NekPyConfig.hpp>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <SpatialDomains/Movement/Movement.h>
#include <SpatialDomains/Python/SpatialDomains.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;

/*
 * @brief Simple wrapper to build Composite objects from lists of
 * Geometry objects.
 */
CompositeSharedPtr Composite_Init(py::list geometries)
{
    CompositeSharedPtr composite = std::make_shared<Composite>();
    composite->m_geomVec.clear();
    for (int i = 0; i < py::len(geometries); i++)
    {
        composite->m_geomVec.emplace_back(
            py::cast<GeometrySharedPtr>(geometries[i]));
    }
    return composite;
}

/**
 * @brief MeshGraph exports.
 */
void export_MeshGraph(py::module &m)
{
    py::bind_map<LibUtilities::FieldMetaDataMap>(m, "FieldMetaDataMap");
    py::bind_vector<std::vector<GeometrySharedPtr>>(m, "GeometryList");

    py::class_<Composite, std::shared_ptr<Composite>>(m, "Composite")
        .def(py::init<>())
        .def(py::init<>(&Composite_Init))
        .def_readwrite("geometries", &Composite::m_geomVec);

    py::bind_map<PointGeomMap>(m, "PointGeomMap");
    py::bind_map<SegGeomMap>(m, "SegGeomMap");
    py::bind_map<QuadGeomMap>(m, "QuadGeomMap");
    py::bind_map<TriGeomMap>(m, "TriGeomMap");
    py::bind_map<TetGeomMap>(m, "TetGeomMap");
    py::bind_map<PrismGeomMap>(m, "PrismGeomMap");
    py::bind_map<PyrGeomMap>(m, "PyrGeomMap");
    py::bind_map<HexGeomMap>(m, "HexGeomMap");
    py::bind_map<CurveMap>(m, "CurveMap");
    py::bind_map<CompositeMap>(m, "CompositeMap");
    py::bind_map<std::map<int, CompositeMap>>(m, "DomainMap");

    py::class_<MeshGraph, std::shared_ptr<MeshGraph>>(m, "MeshGraph")
        .def(py::init<>())

        .def("Empty", &MeshGraph::Empty)

        .def("GetMeshDimension", &MeshGraph::GetMeshDimension)
        .def("GetSpaceDimension", &MeshGraph::GetSpaceDimension)

        .def("SetMeshDimension", &MeshGraph::SetMeshDimension)
        .def("SetSpaceDimension", &MeshGraph::SetSpaceDimension)

        .def("GetAllPointGeoms", &MeshGraph::GetAllPointGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllSegGeoms", &MeshGraph::GetAllSegGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllQuadGeoms", &MeshGraph::GetAllQuadGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllTriGeoms", &MeshGraph::GetAllTriGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllTetGeoms", &MeshGraph::GetAllTetGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllPrismGeoms", &MeshGraph::GetAllPrismGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllPyrGeoms", &MeshGraph::GetAllPyrGeoms,
             py::return_value_policy::reference_internal)
        .def("GetAllHexGeoms", &MeshGraph::GetAllHexGeoms,
             py::return_value_policy::reference_internal)
        .def("GetCurvedEdges", &MeshGraph::GetCurvedEdges,
             py::return_value_policy::reference_internal)
        .def("GetCurvedFaces", &MeshGraph::GetCurvedFaces,
             py::return_value_policy::reference_internal)
        .def("GetComposites", &MeshGraph::GetComposites,
             py::return_value_policy::reference_internal)
        .def<std::map<int, CompositeMap> &(MeshGraph::*)()>(
            "GetDomain", &MeshGraph::GetDomain,
            py::return_value_policy::reference_internal)

        .def("GetMovement", &MeshGraph::GetMovement,
             py::return_value_policy::reference_internal)

        .def("GetNumElements", &MeshGraph::GetNumElements)

        .def("SetExpansionInfosToEvenlySpacedPoints",
             &MeshGraph::SetExpansionInfoToEvenlySpacedPoints)
        .def("SetExpansionInfosToPolyOrder",
             &MeshGraph::SetExpansionInfoToNumModes)
        .def("SetExpansionInfosToPointOrder",
             &MeshGraph::SetExpansionInfoToPointOrder);
}
