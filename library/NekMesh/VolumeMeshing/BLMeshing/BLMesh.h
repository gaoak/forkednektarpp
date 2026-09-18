////////////////////////////////////////////////////////////////////////////////
//
//  File: BLMesh.h
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
//  Description: class for boundary layer meshing
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_BLMESHING_BLMESH_H
#define NEKTAR_MESHUTILS_BLMESHING_BLMESH_H

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <NekMesh/MeshElements/Mesh.h>

namespace Nektar::NekMesh
{

class BLMesh
{
public:
    friend class MemoryManager<BLMesh>;

    /**
     *@brief default constructor
     */
    BLMesh(MeshSharedPtr m, std::vector<unsigned int> bls, NekDouble b, int l,
           NekDouble p, int id, Logger log)
        : m_mesh(m), m_blsurfs(bls), m_bl(b), m_prog(p), m_layer(l), m_id(id),
          m_log(log)
    {
        m_log.SetPrefix("BLMesh");
    }

    /**
     * @brief Execute boundary layer meshing
     */
    void Mesh();

    std::vector<unsigned int> GetSymSurfs()
    {
        return m_symSurfs;
    }
    std::vector<unsigned int> GetBLSurfs()
    {
        return m_blsurfs;
    }

    std::map<SpatialDomains::PointGeom *, SpatialDomains::PointGeom *>
    GetSymNodes();

    std::vector<SpatialDomains::Geometry *> GetPseudoSurface()
    {
        return m_psuedoSurface;
    }

    struct blInfo
    {
        SpatialDomains::PointGeom *pNode;
        SpatialDomains::PointGeom *oNode;
        int bl;
        std::array<NekDouble, 3> N;
        int symsurf;
        bool onSym;
        std::vector<SpatialDomains::Geometry *> els;
        std::set<int> surfs;

        bool stopped;

        void AlignNode(NekDouble t)
        {
            NekDouble x, y, z;
            oNode->GetCoords(x, y, z);
            pNode->UpdatePosition(x + t * N[0], y + t * N[1], z + t * N[2]);
        }
    };
    typedef std::shared_ptr<blInfo> blInfoSharedPtr;

private:
    void Setup();
    void GrowLayers();
    void Shrink();
    void BuildElements();
    bool TestIntersectionEl(SpatialDomains::Geometry *e1,
                            SpatialDomains::Geometry *e2);
    bool IsPrismValid(SpatialDomains::Geometry *el);
    NekDouble Proximity(SpatialDomains::PointGeom *n,
                        SpatialDomains::Geometry *el);

    NekDouble Visability(std::vector<SpatialDomains::Geometry *> tris,
                         std::array<NekDouble, 3> N);
    std::array<NekDouble, 3> GetNormal(
        std::vector<SpatialDomains::Geometry *> tris);

    /// mesh object containing surface mesh
    MeshSharedPtr m_mesh;
    /// List of surfaces onto which boundary layers are placed
    std::vector<unsigned int> m_blsurfs;
    /// thickness of the boundary layer
    NekDouble m_bl;
    NekDouble m_prog;
    int m_layer;
    int m_id;
    std::vector<NekDouble> m_layerT;
    /// list of surfaces to be remeshed due to the boundary layer
    std::vector<unsigned int> m_symSurfs;
    /// data structure used to store and develop bl information. Ordered by
    /// node ID, not by address: growing the layers visits these in order and
    /// stops nodes against the ones already grown, so the order decides the
    /// mesh. Safe to key on the ID because nothing renumbers vertices while
    /// the boundary layer is being built; VolumeMesh only calls
    /// ProcessVertices once the meshing is finished.
    std::map<SpatialDomains::PointGeom *, blInfoSharedPtr, GeometryPtrIdLess>
        m_blData;
    std::map<SpatialDomains::PointGeom *, std::vector<blInfoSharedPtr>,
             GeometryPtrIdLess>
        m_nToNInfo; // node to neighbouring information
    std::map<SpatialDomains::Geometry *, SpatialDomains::Geometry *> m_priToTri;
    std::vector<SpatialDomains::Geometry *> m_psuedoSurface;
    NekMatrix<NekDouble> m_deriv[3];
    /// Logger
    Logger m_log;
};

typedef std::shared_ptr<BLMesh> BLMeshSharedPtr;
} // namespace Nektar::NekMesh

#endif
