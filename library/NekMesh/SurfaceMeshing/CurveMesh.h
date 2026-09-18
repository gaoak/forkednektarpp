////////////////////////////////////////////////////////////////////////////////
//
//  File: CurveMesh.h
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
//  Description: object for individual curve meshes.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_SURFACEMESHING_CURVEMESH_H
#define NEKTAR_MESHUTILS_SURFACEMESHING_CURVEMESH_H

#include <NekMesh/MeshElements/Mesh.h>
#include <SpatialDomains/CADSystem/CADCurve.h>
#include <SpatialDomains/CADSystem/CADVert.h>

#include <LibUtilities/Interpreter/Interpreter.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SpatialDomains/CADSystem/CADAssociation.h>

namespace Nektar::NekMesh
{

class CurveMesh;
typedef std::shared_ptr<CurveMesh> CurveMeshSharedPtr;

/**
 * @brief The mesh vertex sitting on each CAD vertex, by CAD vertex id.
 *
 * Curves meeting at a CAD vertex have to share the mesh vertex there or their
 * curve meshes come out disconnected, and the MeshGraph's CAD association
 * runs the other way -- from mesh entity to CAD object -- so it cannot answer
 * the question. This is shared by all the CurveMesh instances the surface and
 * 2D generators create, which is why it cannot simply be a member of one.
 */
struct CADVertPoints
{
    std::unordered_map<int, SpatialDomains::PointGeom *> points;
};

/**
 * @brief class for meshing individual curves (1d meshing)
 */
class CurveMesh
{
public:
    friend class MemoryManager<CurveMesh>;

    /**
     * @brief default constructor
     */
    CurveMesh(int id, MeshSharedPtr m, Logger log, std::string expr = "0.0")
        : m_id(id), m_mesh(m), m_log(log)
    {
        m_blID     = m_bl.DefineFunction("x y z", expr);
        m_cadcurve = m_mesh->m_meshGraph->GetCAD()->GetCurve(m_id);
        m_log.SetPrefix("CurveMesh");
    }

    CurveMesh(int id, MeshSharedPtr m,
              std::vector<SpatialDomains::PointGeom *> ns, Logger l)
        : m_id(id), m_mesh(m), m_meshpoints(ns), m_log(l)
    {
        m_cadcurve = m_mesh->m_meshGraph->GetCAD()->GetCurve(m_id);
        m_log.SetPrefix("CurveMesh");
    }

    /**
     * @brief execute meshing
     */
    void Mesh(bool forceThree = false);

    /**
     * @brief Delete old mesh and mesh with forceThree on
     */
    void ReMesh();

    /**
     * @brief get id of first node
     */
    SpatialDomains::PointGeom *GetFirstPoint()
    {
        return m_meshpoints[0];
    }

    /**
     * @brief get id of last node
     */
    SpatialDomains::PointGeom *GetLastPoint()
    {
        return m_meshpoints.back();
    }

    /**
     * @brief get list of mesh nodes
     */
    std::vector<SpatialDomains::PointGeom *> GetMeshPoints()
    {
        return m_meshpoints;
    }

    std::vector<SpatialDomains::SegGeom *> GetMeshEdges()
    {
        return m_meshedges;
    }

    /**
     * @brief get the number of points in the curve
     */
    int GetNumPoints()
    {
        return m_meshpoints.size();
    }

    /**
     * @brief get the length of the curve
     */
    NekDouble GetLength()
    {
        return m_curvelength;
    }

    void PeriodicOverwrite(CurveMeshSharedPtr from);

    int GetId()
    {
        return m_id;
    }

    void SetOffset(unsigned i, NekDouble offset)
    {
        m_endoffset[i] = offset;
    }

private:
    /**
     * @brief get the mesh vertex sitting on a CAD vertex, creating it on first
     * use so that curves meeting there share it
     */
    SpatialDomains::PointGeom *GetOrCreateCADVertPoint(
        SpatialDomains::CADVertSharedPtr cadVert);

    /**
     * @brief get node spacing sampling function
     */
    void GetSampleFunction();

    /**
     * @brief get node spacing phi function
     */
    void GetPhiFunction();

    /**
     * @brief evaluate paramter ds at curve location s
     */
    NekDouble EvaluateDS(NekDouble s);

    /**
     * @brief evaluate paramter ps at curve location s
     */
    NekDouble EvaluatePS(NekDouble s);

    /// CAD curve
    SpatialDomains::CADCurveSharedPtr m_cadcurve;
    /// length of the curve in real space
    NekDouble m_curvelength;
    /// number of sampling points used in algorithm
    int m_numSamplePoints;
    /// coords of the ends of the parametric curve
    std::array<NekDouble, 2> m_bounds;
    /// array of function ds evaluations
    std::vector<std::vector<NekDouble>> m_dst;
    /// array of function ps evaluations
    std::vector<std::vector<NekDouble>> m_ps;
    /// spacing function evaluation
    NekDouble Ae;
    /// ds
    NekDouble ds;
    /// number of edges to be made in the curve as defined by the spacing
    /// funtion
    int Ne;
    /// paramteric coordiates of the mesh nodes
    std::vector<NekDouble> meshsvalue;
    /// list of mesh edges in the curvemesh
    std::vector<SpatialDomains::SegGeom *> m_meshedges;
    /// id of the curvemesh
    int m_id;
    ///
    MeshSharedPtr m_mesh;
    /// ids of the mesh nodes
    std::vector<SpatialDomains::PointGeom *> m_meshpoints;
    LibUtilities::Interpreter m_bl;
    int m_blID;
    /// offset of second point at each end
    std::map<unsigned, NekDouble> m_endoffset;
    /// Logger
    Logger m_log;
};

} // namespace Nektar::NekMesh

#endif
