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
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

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
            py::extract<GeometrySharedPtr>(geometries[i]));
    }
    return composite;
}

/**
 * @brief MeshGraph exports.
 */
void export_MeshGraph()
{
    py::class_<LibUtilities::FieldMetaDataMap>("FieldMetaDataMap")
        .def(py::map_indexing_suite<LibUtilities::FieldMetaDataMap, true>());

    py::class_<std::vector<GeometrySharedPtr>>("GeometryList")
        .def(py::vector_indexing_suite<std::vector<GeometrySharedPtr>, true>());
    py::class_<Composite, std::shared_ptr<Composite>>("Composite", py::init<>())
        .def("__init__", py::make_constructor(&Composite_Init))
        .def_readwrite("geometries", &Composite::m_geomVec);

    py::class_<PointGeomMap>("PointGeomMap")
        .def(py::map_indexing_suite<PointGeomMap, true>());
    py::class_<SegGeomMap>("SegGeomMap")
        .def(py::map_indexing_suite<SegGeomMap, true>());
    py::class_<QuadGeomMap>("QuadGeomMap")
        .def(py::map_indexing_suite<QuadGeomMap, true>());
    py::class_<TriGeomMap>("TriGeomMap")
        .def(py::map_indexing_suite<TriGeomMap, true>());
    py::class_<TetGeomMap>("TetGeomMap")
        .def(py::map_indexing_suite<TetGeomMap, true>());
    py::class_<PrismGeomMap>("PrismGeomMap")
        .def(py::map_indexing_suite<PrismGeomMap, true>());
    py::class_<PyrGeomMap>("PyrGeomMap")
        .def(py::map_indexing_suite<PyrGeomMap, true>());
    py::class_<HexGeomMap>("HexGeomMap")
        .def(py::map_indexing_suite<HexGeomMap, true>());
    py::class_<CurveMap>("CurveMap")
        .def(py::map_indexing_suite<CurveMap, true>());
    py::class_<CompositeMap>("CompositeMap")
        .def(py::map_indexing_suite<CompositeMap, true>());
    py::class_<std::map<int, CompositeMap>>("DomainMap")
        .def(py::map_indexing_suite<std::map<int, CompositeMap>, true>());

    py::class_<MeshGraph, std::shared_ptr<MeshGraph>, boost::noncopyable>(
        "MeshGraph", py::init<>())

        .def("Empty", &MeshGraph::Empty)

        .def("GetMeshDimension", &MeshGraph::GetMeshDimension)
        .def("GetSpaceDimension", &MeshGraph::GetSpaceDimension)

        .def("SetMeshDimension", &MeshGraph::SetMeshDimension)
        .def("SetSpaceDimension", &MeshGraph::SetSpaceDimension)

        .def("GetAllPointGeoms", &MeshGraph::GetAllPointGeoms,
             py::return_internal_reference<>())
        .def("GetAllSegGeoms", &MeshGraph::GetAllSegGeoms,
             py::return_internal_reference<>())
        .def("GetAllQuadGeoms", &MeshGraph::GetAllQuadGeoms,
             py::return_internal_reference<>())
        .def("GetAllTriGeoms", &MeshGraph::GetAllTriGeoms,
             py::return_internal_reference<>())
        .def("GetAllTetGeoms", &MeshGraph::GetAllTetGeoms,
             py::return_internal_reference<>())
        .def("GetAllPrismGeoms", &MeshGraph::GetAllPrismGeoms,
             py::return_internal_reference<>())
        .def("GetAllPyrGeoms", &MeshGraph::GetAllPyrGeoms,
             py::return_internal_reference<>())
        .def("GetAllHexGeoms", &MeshGraph::GetAllHexGeoms,
             py::return_internal_reference<>())
        .def("GetCurvedEdges", &MeshGraph::GetCurvedEdges,
             py::return_internal_reference<>())
        .def("GetCurvedFaces", &MeshGraph::GetCurvedFaces,
             py::return_internal_reference<>())
        .def("GetComposites", &MeshGraph::GetComposites,
             py::return_internal_reference<>())
        .def<std::map<int, CompositeMap> &(MeshGraph::*)()>(
            "GetDomain", &MeshGraph::GetDomain,
            py::return_internal_reference<>())

        .def("GetMovement", &MeshGraph::GetMovement,
             py::return_value_policy<py::return_by_value>())

        .def("GetNumElements", &MeshGraph::GetNumElements)

        .def("SetExpansionInfosToEvenlySpacedPoints",
             &MeshGraph::SetExpansionInfoToEvenlySpacedPoints)
        .def("SetExpansionInfosToPolyOrder",
             &MeshGraph::SetExpansionInfoToNumModes)
        .def("SetExpansionInfosToPointOrder",
             &MeshGraph::SetExpansionInfoToPointOrder);
}
