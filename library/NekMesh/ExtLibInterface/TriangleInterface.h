////////////////////////////////////////////////////////////////////////////////
//
//  File: TriangleInterface.h
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
//  Description: class for interfacing with triangle
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_EXTLIBINTERFACE_TRIANGLEINTERFACE_H
#define NEKTAR_MESHUTILS_EXTLIBINTERFACE_TRIANGLEINTERFACE_H

#include <memory>

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SpatialDomains/PointGeom.h>

namespace Nektar::NekMesh
{

/// Parametric (u,v) location of a mesh vertex on the CAD surface.
typedef std::unordered_map<SpatialDomains::PointGeom *,
                           std::array<NekDouble, 2>>
    NodeUVMap;

/**
 * @brief class for interfacing with external library triangle
 */
class TriangleInterface
{
public:
    friend class MemoryManager<TriangleInterface>;

    /**
     * @brief default constructor
     */
    TriangleInterface();

    /**
     * @brief destructor
     *
     * Out of line so that the incomplete DelaunayTriangle below can be
     * destroyed where its definition is visible.
     */
    ~TriangleInterface();

    /**
     * @brief assign meshing paramters
     *
     * @p nodeUV gives the (u,v) on the surface being meshed for every node
     * passed in.
     */
    void Assign(
        std::vector<std::vector<SpatialDomains::PointGeom *>> &boundingloops,
        std::vector<std::array<NekDouble, 2>> &centers, int i,
        const NodeUVMap &nodeUV, NekDouble str = 1.0)
    {
        m_boundingloops = boundingloops;
        m_centers       = centers;
        m_str           = str;
        sid             = i;
        m_nodeUV        = nodeUV;
    }

    void AssignStiener(std::vector<SpatialDomains::PointGeom *> stiner,
                       const NodeUVMap &nodeUV)
    {
        m_stienerpoints = stiner;
        m_nodeUV.insert(nodeUV.begin(), nodeUV.end());
    }

    /**
     * @brief Execute meshing
     */
    void Mesh(bool Quality = false);

    /**
     * @brief Extract mesh
     */
    void Extract(std::vector<std::vector<SpatialDomains::PointGeom *>> &Connec);

private:
    /**
     * @brief Clear memory
     */
    void SetUp();

    /// Triangle's input and output data structures. Defined in the source
    /// file: triangle.h declares lower-case function-like macros (dest, org,
    /// apex, ...) that clash with ordinary member names, so it must not be
    /// dragged into anything that includes this header.
    struct DelaunayTriangle;

    /// List of bounding nodes to the surface
    std::vector<std::vector<SpatialDomains::PointGeom *>> m_boundingloops;
    /// List of additional nodes
    std::vector<SpatialDomains::PointGeom *> m_stienerpoints;
    /// (u,v) on this surface of every node handed to Assign/AssignStiener
    NodeUVMap m_nodeUV;
    /// Coordinates of the centers of the loops
    std::vector<std::array<NekDouble, 2>> m_centers;
    /// Map from NekMesh id to triangle id
    std::map<int, SpatialDomains::PointGeom *> nodemap;
    /// ID of the surface
    int sid;
    /// Stretching factor of parameter plane
    NekDouble m_str;
    /// Triangle data strucutres
    std::unique_ptr<DelaunayTriangle> dt;
};

typedef std::shared_ptr<TriangleInterface> TriangleInterfaceSharedPtr;
} // namespace Nektar::NekMesh

#endif
