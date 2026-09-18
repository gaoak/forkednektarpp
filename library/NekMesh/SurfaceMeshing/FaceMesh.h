////////////////////////////////////////////////////////////////////////////////
//
//  File: FaceMesh.h
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
//  Description:
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NekMesh_SURFACEMESHING_FACEMESH
#define NekMesh_SURFACEMESHING_FACEMESH

#include <NekMesh/MeshElements/Mesh.h>
#include <NekMesh/SurfaceMeshing/CurveMesh.h>
#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <SpatialDomains/CADSystem/CADSurf.h>

namespace Nektar::NekMesh
{

/**
 * @brief class for surface meshes on individual surfaces (paramter plane
 * meshes)
 */
class FaceMesh
{
public:
    friend class MemoryManager<FaceMesh>;

    /**
     * @brief Default constructor
     */
    FaceMesh(const int id, MeshSharedPtr m,
             const std::map<int, CurveMeshSharedPtr> &cmeshes, const int comp,
             Logger l)
        : m_mesh(m), m_curvemeshes(cmeshes), m_id(id), m_compId(comp), m_log(l)
    {
        m_cadsurf   = m_mesh->m_meshGraph->GetCAD()->GetSurf(m_id);
        m_edgeloops = m_cadsurf->GetEdges();
        m_log.SetPrefix("FaceMesh");
    };

    /**
     * @brief mesh exectuation command
     */
    void Mesh();

    /**
     * @brief validate the curve meshes
     */
    bool ValidateCurves();

    /**
     * @brief validate the curve meshes considering the loops
     */
    void ValidateLoops();

private:
    /**
     * @brief Get the boundries of the surface and extracts the nodes from
     * the curve meshes in the correct order
     */
    void OrientateCurves();

    /**
     * @brief Calculate the paramter plane streching factor
     */
    void Stretching();

    /**
     * @brief performs node smoothing on face
     */
    void Smoothing();

    /**
     * @brief swapping
     */
    void DiagonalSwap();

    /**
     * @brief build a local version of mesh elements
     */
    void BuildLocalMesh();

    /**
     * @brief function which calls the optimisation routines
     */
    void OptimiseLocalMesh();

    /**
     * @brief Validate the surface mesh base on the octree and real
     * dimensions of the edges
     */
    bool Validate();

    /**
     * @brief adds a new stiener point to the triangulation for meshing
     */
    void AddNewPoint(std::array<NekDouble, 2> uv);

    /**
     * @brief adds a quad layer around any interior loops
     */
    void MakeBL();

    /// mesh pointer
    MeshSharedPtr m_mesh;
    /// CAD surface
    SpatialDomains::CADSurfSharedPtr m_cadsurf;
    /// Map of the curve meshes which bound the surfaces
    std::map<int, CurveMeshSharedPtr> m_curvemeshes;
    /// data structure containing the edges, their order and oreientation for
    /// the surface
    std::vector<SpatialDomains::EdgeLoopSharedPtr> m_edgeloops;
    /// id of the surface mesh
    int m_id;
    /// list of boundary nodes in their order loops
    std::vector<std::vector<SpatialDomains::PointGeom *>> orderedLoops;
    /// list of stiener points in the triangulation
    std::vector<SpatialDomains::PointGeom *> m_stienerpoints;
    /// pplane stretching
    NekDouble m_str;
    /// triangle connectiviities
    std::vector<std::vector<SpatialDomains::PointGeom *>> m_connec;
    /// local set of nodes
    /// Nodes of this face, ordered by ID rather than by address: the elastic
    /// relaxation in Smoothing() moves each node onto the average of its
    /// neighbours' current positions, so the order they are visited in decides
    /// where they end up. Safe to order on the ID because these are created
    /// with NextPointId and nothing renumbers them while the face is meshed.
    std::set<SpatialDomains::PointGeom *, GeometryPtrIdLess> m_localNodes;
    /// local set of edges
    EdgeMap m_localEdges;
    /// local list of elements
    std::vector<SpatialDomains::Geometry *> m_localElements;
    /// set of nodes which are in the boundary (easier to identify conflicts
    /// with)
    std::unordered_set<SpatialDomains::PointGeom *> m_inBoundary;
    /// identity to put into element tags
    int m_compId;
    /// Logger
    Logger m_log;
};

typedef std::shared_ptr<FaceMesh> FaceMeshSharedPtr;
} // namespace Nektar::NekMesh

#endif
