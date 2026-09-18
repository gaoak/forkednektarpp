///////////////////////////////////////////////////////////////////////////////
//
//  File: Mesh.cpp
//
//  For more information, please see: http://www.nektar.info
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
//  Description: Python wrapper for Mesh.
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/core/ignore_unused.hpp>

#include <LibUtilities/Python/NekPyConfig.hpp>
#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/MeshElements/Mesh.h>
#include <NekMesh/Python/NekMesh.h>

using namespace Nektar;
using namespace Nektar::NekMesh;

/**
 * @brief The mesh's expansion dimension, which lives on the MeshGraph.
 */
int Mesh_GetExpDim(MeshSharedPtr mesh)
{
    return mesh->m_meshGraph->GetMeshDimension();
}

void Mesh_SetExpDim(MeshSharedPtr mesh, int dim)
{
    mesh->m_meshGraph->SetMeshDimension(dim);
}

int Mesh_GetSpaceDim(MeshSharedPtr mesh)
{
    return mesh->m_meshGraph->GetSpaceDimension();
}

void Mesh_SetSpaceDim(MeshSharedPtr mesh, int dim)
{
    mesh->m_meshGraph->SetSpaceDimension(dim);
}

/**
 * @brief Create a vertex, owned by the MeshGraph.
 *
 * A mesh vertex is a SpatialDomains::PointGeom held by the MeshGraph, so it
 * cannot be built free-standing the way NekMesh's own Node could: the mesh has
 * to make it. Pass -1 as the id to have one assigned.
 */
SpatialDomains::PointGeom *Mesh_CreateVertex(MeshSharedPtr mesh, int id,
                                             NekDouble x, NekDouble y,
                                             NekDouble z)
{
    auto &graph = mesh->m_meshGraph;

    if (id < 0)
    {
        id = NextPointId(graph);
    }

    return graph->CreatePointGeom(graph->GetSpaceDimension(), id, x, y, z);
}

/**
 * @brief Create an element from a list of vertices and tag it.
 *
 * This replaces the old Element.Create followed by an append to
 * Mesh.element[dim]: the element factory writes the element, and the edges and
 * faces it needs, straight into the graph, sharing them with any neighbour
 * that already created them, and the tag is what puts it in a composite.
 */
SpatialDomains::Geometry *Mesh_CreateElement(
    MeshSharedPtr mesh, ElmtConfig conf,
    std::vector<SpatialDomains::PointGeom *> nodeList, int tag)
{
    SpatialDomains::Geometry *el = GetElementFactory().CreateInstance(
        conf.m_e, nodeList, mesh->m_meshGraph, mesh->m_edgeSet, mesh->m_faceSet,
        conf, nullptr, nullptr, nullptr, nullptr);

    mesh->m_elementTags[LibUtilities::ShapeTypeDimMap[conf.m_e]][el] = tag;

    return el;
}

/**
 * @brief The mesh's vertices, in ascending order of id.
 */
std::vector<SpatialDomains::PointGeom *> Mesh_GetVertices(MeshSharedPtr mesh)
{
    std::vector<SpatialDomains::PointGeom *> ret;

    for (auto &[id, vert] :
         mesh->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        boost::ignore_unused(id);
        ret.push_back(vert);
    }

    return ret;
}

/**
 * @brief The tagged entities of a given dimension, in ascending order of id.
 *
 * The mesh's elements and its boundary elements are the same thing at
 * different dimensions, so this covers both: dimension expDim gives the
 * elements, expDim - 1 their boundaries.
 */
std::vector<SpatialDomains::Geometry *> Mesh_GetElements(MeshSharedPtr mesh,
                                                         int dim)
{
    if (dim < 0 || dim > 3)
    {
        throw py::index_error();
    }

    std::vector<SpatialDomains::Geometry *> ret;

    for (auto &[el, tag] : mesh->m_elementTags[dim])
    {
        boost::ignore_unused(tag);
        ret.push_back(el);
    }

    std::sort(ret.begin(), ret.end(),
              [](SpatialDomains::Geometry *a, SpatialDomains::Geometry *b) {
                  return std::make_pair(static_cast<int>(a->GetShapeType()),
                                        a->GetGlobalID()) <
                         std::make_pair(static_cast<int>(b->GetShapeType()),
                                        b->GetGlobalID());
              });

    return ret;
}

/**
 * @brief The composite tag of an entity.
 */
int Mesh_GetTag(MeshSharedPtr mesh, SpatialDomains::Geometry *el)
{
    auto &tags = mesh->m_elementTags[el->GetShapeDim()];

    auto it = tags.find(el);
    if (it == tags.end())
    {
        throw py::key_error("entity is not tagged in this mesh");
    }

    return it->second;
}

/**
 * @brief Tag an entity, putting it in the corresponding composite.
 */
void Mesh_SetTag(MeshSharedPtr mesh, SpatialDomains::Geometry *el, int tag)
{
    mesh->m_elementTags[el->GetShapeDim()][el] = tag;
}

/**
 * @brief Untag an entity, taking it out of its composite.
 *
 * The geometry itself stays in the graph, where its neighbours may still be
 * using its edges and faces; Module.RemoveOrphanedEntities() is what clears
 * out whatever is left unreferenced.
 */
void Mesh_RemoveElement(MeshSharedPtr mesh, SpatialDomains::Geometry *el)
{
    if (mesh->m_elementTags[el->GetShapeDim()].erase(el) == 0)
    {
        throw py::key_error("entity is not tagged in this mesh");
    }
}

void export_Mesh(py::module &m)
{
    py::bind_map<GeomTagMap>(m, "GeomTagMap");

    py::class_<ElmtConfig>(m, "ElmtConfig")
        .def(py::init<LibUtilities::ShapeType, unsigned int, bool, bool, bool,
                      LibUtilities::PointsType, LibUtilities::PointsType>(),
             py::arg("shapeType"), py::arg("order"), py::arg("faceNodes"),
             py::arg("volumeNodes"), py::arg("reorient") = true,
             py::arg("edgeNodeType") = LibUtilities::ePolyEvenlySpaced,
             py::arg("faceNodeType") = LibUtilities::ePolyEvenlySpaced);

    py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
        .def(py::init<>())

        .def_property("expDim", &Mesh_GetExpDim, &Mesh_SetExpDim)
        .def_property("spaceDim", &Mesh_GetSpaceDim, &Mesh_SetSpaceDim)

        .def_property_readonly(
            "graph", [](MeshSharedPtr mesh) { return mesh->m_meshGraph; },
            "The MeshGraph holding the geometry.")

        .def("CreateVertex", &Mesh_CreateVertex, py::arg("id"), py::arg("x"),
             py::arg("y"), py::arg("z") = 0.0,
             py::return_value_policy::reference_internal,
             "Create a vertex owned by this mesh; id -1 assigns one.")
        .def("CreateElement", &Mesh_CreateElement, py::arg("config"),
             py::arg("nodes"), py::arg("tag") = 0,
             py::return_value_policy::reference_internal,
             "Create an element from its vertices and give it a composite "
             "tag.")

        .def("GetVertices", &Mesh_GetVertices,
             py::return_value_policy::reference_internal)
        .def("GetElements", &Mesh_GetElements, py::arg("dim"),
             py::return_value_policy::reference_internal,
             "The tagged entities of a given dimension.")

        .def("GetTag", &Mesh_GetTag, py::arg("entity"))
        .def("SetTag", &Mesh_SetTag, py::arg("entity"), py::arg("tag"))
        .def("RemoveElement", &Mesh_RemoveElement, py::arg("entity"),
             "Untag an entity, taking it out of its composite.")

        .def("GetNumElements", &Mesh::GetNumElements)
        .def("GetNumBndryElements", &Mesh::GetNumBndryElements)
        .def("GetNumTaggedEntities", &Mesh::GetNumTaggedEntities)

        .def("GetAllElementTags",
             py::overload_cast<int>(&Mesh::GetAllElementTags), py::arg("dim"),
             py::return_value_policy::reference_internal,
             "The entity to tag map for a given dimension.");
}
