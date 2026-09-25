////////////////////////////////////////////////////////////////////////////////
//
//  File: TetGeom.cpp
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
//  Description: Tetrahedral geometry information.
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/TetGeom.h>

#include <LibUtilities/Foundations/Interp.h>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/HOAlignment.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TriGeom.h>
#include <SpatialDomains/XmapFactory.hpp>
#include <StdRegions/StdNodalTetExp.h>
#include <StdRegions/StdTetExp.h>

namespace Nektar::SpatialDomains
{
const unsigned int TetGeom::VertexEdgeConnectivity[4][3] = {
    {0, 2, 3}, {0, 1, 4}, {1, 2, 5}, {3, 4, 5}};
const unsigned int TetGeom::VertexFaceConnectivity[4][3] = {
    {0, 1, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}};
const unsigned int TetGeom::EdgeFaceConnectivity[6][2] = {
    {0, 1}, {0, 2}, {0, 3}, {1, 3}, {1, 2}, {2, 3}};
const unsigned int TetGeom::EdgeNormalToFaceVert[4][3] = {
    {3, 4, 5}, {1, 2, 5}, {0, 2, 3}, {0, 1, 4}};

XmapFactory<StdRegions::StdTetExp, 3> &GetStdTetFactory()
{
    static XmapFactory<StdRegions::StdTetExp, 3> factory;
    return factory;
}

TetGeom::TetGeom()
{
    m_shapeType = LibUtilities::eTetrahedron;
}

TetGeom::TetGeom(int id, std::array<TriGeom *, kNfaces> faces, Curve *curve)
    : Geometry3D(faces[0]->GetEdge(0)->GetVertex(0)->GetCoordim(), curve)
{
    m_shapeType = LibUtilities::eTetrahedron;
    m_globalID  = id;
    m_faces     = faces;

    SetUpLocalEdges();
    SetUpLocalVertices();
    SetUpEdgeOrientation();
    SetUpFaceOrientation();
}

TetGeom::TetGeom(int id, std::array<TriGeom *, 4> faces,
                 std::array<SegGeom *, 6> edges,
                 std::array<PointGeom *, 4> verts, bool skipSetUp, Curve *curve)
    : Geometry3D(faces[0]->GetEdge(0)->GetVertex(0)->GetCoordim(), curve)
{
    m_shapeType = LibUtilities::eTetrahedron;
    m_globalID  = id;

    /// Copy the face & edge & vert pointers
    m_faces = faces;
    m_edges = edges;
    m_verts = verts;

    if (!skipSetUp)
    {
        SetUpEdgeOrientation();
        SetUpFaceOrientation();
    }
}

int TetGeom::v_GetDir(const int faceidx, const int facedir) const
{
    if (faceidx == 0)
    {
        return facedir;
    }
    else if (faceidx == 1)
    {
        return 2 * facedir;
    }
    else
    {
        return 1 + facedir;
    }
}

int TetGeom::v_GetVertexEdgeMap(const int i, const int j) const
{
    return VertexEdgeConnectivity[i][j];
}

int TetGeom::v_GetVertexFaceMap(const int i, const int j) const
{
    return VertexFaceConnectivity[i][j];
}

int TetGeom::v_GetEdgeFaceMap(const int i, const int j) const
{
    return EdgeFaceConnectivity[i][j];
}

int TetGeom::v_GetEdgeNormalToFaceVert(const int i, const int j) const
{
    return EdgeNormalToFaceVert[i][j];
}

void TetGeom::SetUpLocalEdges()
{

    // find edge 0
    int i, j;
    unsigned int check;

    // First set up the 3 bottom edges

    if (m_faces[0]->GetEid(0) != m_faces[1]->GetEid(0))
    {
        std::ostringstream errstrm;
        errstrm << "Local edge 0 (eid=" << m_faces[0]->GetEid(0);
        errstrm << ") on face " << m_faces[0]->GetGlobalID();
        errstrm << " must be the same as local edge 0 (eid="
                << m_faces[1]->GetEid(0);
        errstrm << ") on face " << m_faces[1]->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }

    int faceConnected;
    for (faceConnected = 1; faceConnected < 4; faceConnected++)
    {
        check = 0;
        for (i = 0; i < 3; i++)
        {
            if ((m_faces[0])->GetEid(i) == (m_faces[faceConnected])->GetEid(0))
            {
                m_edges[faceConnected - 1] =
                    static_cast<SegGeom *>((m_faces[0])->GetEdge(i));
                check++;
            }
        }

        if (check < 1)
        {
            std::ostringstream errstrm;
            errstrm << "Face 0 does not share an edge with first edge of "
                       "adjacent face. Faces ";
            errstrm << (m_faces[0])->GetGlobalID() << ", "
                    << (m_faces[faceConnected])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
        else if (check > 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces share more than one edge. Faces ";
            errstrm << (m_faces[0])->GetGlobalID() << ", "
                    << (m_faces[faceConnected])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
    }

    // Then, set up the 3 vertical edges
    check = 0;
    for (i = 0; i < 3; i++) // Set up the vertical edge :face(1) and face(3)
    {
        for (j = 0; j < 3; j++)
        {
            if ((m_faces[1])->GetEid(i) == (m_faces[3])->GetEid(j))
            {
                m_edges[3] = static_cast<SegGeom *>((m_faces[1])->GetEdge(i));
                check++;
            }
        }
    }
    if (check < 1)
    {
        std::ostringstream errstrm;
        errstrm << "Connected faces do not share an edge. Faces ";
        errstrm << (m_faces[1])->GetGlobalID() << ", "
                << (m_faces[3])->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
    else if (check > 1)
    {
        std::ostringstream errstrm;
        errstrm << "Connected faces share more than one edge. Faces ";
        errstrm << (m_faces[1])->GetGlobalID() << ", "
                << (m_faces[3])->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
    // Set up vertical edges: face(1) through face(3)
    for (faceConnected = 1; faceConnected < 3; faceConnected++)
    {
        check = 0;
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 3; j++)
            {
                if ((m_faces[faceConnected])->GetEid(i) ==
                    (m_faces[faceConnected + 1])->GetEid(j))
                {
                    m_edges[faceConnected + 3] = static_cast<SegGeom *>(
                        (m_faces[faceConnected])->GetEdge(i));
                    check++;
                }
            }
        }

        if (check < 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces do not share an edge. Faces ";
            errstrm << (m_faces[faceConnected])->GetGlobalID() << ", "
                    << (m_faces[faceConnected + 1])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
        else if (check > 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces share more than one edge. Faces ";
            errstrm << (m_faces[faceConnected])->GetGlobalID() << ", "
                    << (m_faces[faceConnected + 1])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
    }
}

void TetGeom::SetUpLocalVertices()
{
    // Set up the first 2 vertices (i.e. vertex 0,1)
    if ((m_edges[0]->GetVid(0) == m_edges[1]->GetVid(0)) ||
        (m_edges[0]->GetVid(0) == m_edges[1]->GetVid(1)))
    {
        m_verts[0] = m_edges[0]->GetVertex(1);
        m_verts[1] = m_edges[0]->GetVertex(0);
    }
    else if ((m_edges[0]->GetVid(1) == m_edges[1]->GetVid(0)) ||
             (m_edges[0]->GetVid(1) == m_edges[1]->GetVid(1)))
    {
        m_verts[0] = m_edges[0]->GetVertex(0);
        m_verts[1] = m_edges[0]->GetVertex(1);
    }
    else
    {
        std::ostringstream errstrm;
        errstrm << "Connected edges do not share a vertex. Edges ";
        errstrm << m_edges[0]->GetGlobalID() << ", "
                << m_edges[1]->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }

    // set up the other bottom vertices (i.e. vertex 2)
    for (int i = 1; i < 2; i++)
    {
        if (m_edges[i]->GetVid(0) == m_verts[i]->GetGlobalID())
        {
            m_verts[2] = m_edges[i]->GetVertex(1);
        }
        else if (m_edges[i]->GetVid(1) == m_verts[i]->GetGlobalID())
        {
            m_verts[2] = m_edges[i]->GetVertex(0);
        }
        else
        {
            std::ostringstream errstrm;
            errstrm << "Connected edges do not share a vertex. Edges ";
            errstrm << m_edges[i]->GetGlobalID() << ", "
                    << m_edges[i - 1]->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
    }

    // set up top vertex
    if (m_edges[3]->GetVid(0) == m_verts[0]->GetGlobalID())
    {
        m_verts[3] = m_edges[3]->GetVertex(1);
    }
    else
    {
        m_verts[3] = m_edges[3]->GetVertex(0);
    }

    // Check the other edges match up.
    int check = 0;
    for (int i = 4; i < 6; ++i)
    {
        if ((m_edges[i]->GetVid(0) == m_verts[i - 3]->GetGlobalID() &&
             m_edges[i]->GetVid(1) == m_verts[3]->GetGlobalID()) ||
            (m_edges[i]->GetVid(1) == m_verts[i - 3]->GetGlobalID() &&
             m_edges[i]->GetVid(0) == m_verts[3]->GetGlobalID()))
        {
            check++;
        }
    }
    if (check != 2)
    {
        std::ostringstream errstrm;
        errstrm << "Connected edges do not share a vertex. Edges ";
        errstrm << m_edges[3]->GetGlobalID() << ", "
                << m_edges[2]->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
}

void TetGeom::SetUpEdgeOrientation()
{

    // This 2D array holds the local id's of all the vertices
    // for every edge. For every edge, they are ordered to what we
    // define as being Forwards
    const unsigned int edgeVerts[kNedges][2] = {{0, 1}, {1, 2}, {0, 2},
                                                {0, 3}, {1, 3}, {2, 3}};

    int i;
    for (i = 0; i < kNedges; i++)
    {
        if (m_edges[i]->GetVid(0) == m_verts[edgeVerts[i][0]]->GetGlobalID())
        {
            m_eorient[i] = StdRegions::eForwards;
        }
        else if (m_edges[i]->GetVid(0) ==
                 m_verts[edgeVerts[i][1]]->GetGlobalID())
        {
            m_eorient[i] = StdRegions::eBackwards;
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "Could not find matching vertex for the edge");
        }
    }
}

void TetGeom::SetUpFaceOrientation()
{

    int f, i;

    // These arrays represent the vector of the A and B
    // coordinate of the local elemental coordinate system
    // where A corresponds with the coordinate direction xi_i
    // with the lowest index i (for that particular face)
    // Coordinate 'B' then corresponds to the other local
    // coordinate (i.e. with the highest index)
    Array<OneD, NekDouble> elementAaxis(m_coordim);
    Array<OneD, NekDouble> elementBaxis(m_coordim);

    // These arrays correspond to the local coordinate
    // system of the face itself (i.e. the Geometry2D)
    // faceAaxis correspond to the xi_0 axis
    // faceBaxis correspond to the xi_1 axis
    Array<OneD, NekDouble> faceAaxis(m_coordim);
    Array<OneD, NekDouble> faceBaxis(m_coordim);

    // This is the base vertex of the face (i.e. the Geometry2D)
    // This corresponds to thevertex with local ID 0 of the
    // Geometry2D
    unsigned int baseVertex;

    // The lenght of the vectors above
    NekDouble elementAaxis_length;
    NekDouble elementBaxis_length;
    NekDouble faceAaxis_length;
    NekDouble faceBaxis_length;

    // This 2D array holds the local id's of all the vertices
    // for every face. For every face, they are ordered in such
    // a way that the implementation below allows a unified approach
    // for all faces.
    const unsigned int faceVerts[kNfaces][TriGeom::kNverts] = {
        {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};

    NekDouble dotproduct1 = 0.0;
    NekDouble dotproduct2 = 0.0;

    unsigned int orientation;

    // Loop over all the faces to set up the orientation
    for (f = 0; f < kNqfaces + kNtfaces; f++)
    {
        // initialisation
        elementAaxis_length = 0.0;
        elementBaxis_length = 0.0;
        faceAaxis_length    = 0.0;
        faceBaxis_length    = 0.0;

        dotproduct1 = 0.0;
        dotproduct2 = 0.0;

        baseVertex = m_faces[f]->GetVid(0);

        // We are going to construct the vectors representing the A
        // and B axis of every face. These vectors will be constructed
        // as a vector-representation of the edges of the
        // face. However, for both coordinate directions, we can
        // represent the vectors by two different edges. That's why we
        // need to make sure that we pick the edge to which the
        // baseVertex of the Geometry2D-representation of the face
        // belongs...

        // Compute the length of edges on a base-face
        if (baseVertex == m_verts[faceVerts[f][0]]->GetGlobalID())
        {
            for (i = 0; i < m_coordim; i++)
            {
                elementAaxis[i] = (*m_verts[faceVerts[f][1]])[i] -
                                  (*m_verts[faceVerts[f][0]])[i];
                elementBaxis[i] = (*m_verts[faceVerts[f][2]])[i] -
                                  (*m_verts[faceVerts[f][0]])[i];
            }
        }
        else if (baseVertex == m_verts[faceVerts[f][1]]->GetGlobalID())
        {
            for (i = 0; i < m_coordim; i++)
            {
                elementAaxis[i] = (*m_verts[faceVerts[f][1]])[i] -
                                  (*m_verts[faceVerts[f][0]])[i];
                elementBaxis[i] = (*m_verts[faceVerts[f][2]])[i] -
                                  (*m_verts[faceVerts[f][1]])[i];
            }
        }
        else if (baseVertex == m_verts[faceVerts[f][2]]->GetGlobalID())
        {
            for (i = 0; i < m_coordim; i++)
            {
                elementAaxis[i] = (*m_verts[faceVerts[f][1]])[i] -
                                  (*m_verts[faceVerts[f][2]])[i];
                elementBaxis[i] = (*m_verts[faceVerts[f][2]])[i] -
                                  (*m_verts[faceVerts[f][0]])[i];
            }
        }
        else
        {
            NEKERROR(ErrorUtil::efatal,
                     "Could not find matching vertex for the face");
        }

        // Now, construct the edge-vectors of the local coordinates of
        // the Geometry2D-representation of the face
        for (i = 0; i < m_coordim; i++)
        {
            faceAaxis[i] =
                (*m_faces[f]->GetVertex(1))[i] - (*m_faces[f]->GetVertex(0))[i];
            faceBaxis[i] =
                (*m_faces[f]->GetVertex(2))[i] - (*m_faces[f]->GetVertex(0))[i];

            elementAaxis_length += pow(elementAaxis[i], 2);
            elementBaxis_length += pow(elementBaxis[i], 2);
            faceAaxis_length += pow(faceAaxis[i], 2);
            faceBaxis_length += pow(faceBaxis[i], 2);
        }

        elementAaxis_length = sqrt(elementAaxis_length);
        elementBaxis_length = sqrt(elementBaxis_length);
        faceAaxis_length    = sqrt(faceAaxis_length);
        faceBaxis_length    = sqrt(faceBaxis_length);

        // Calculate the inner product of both the A-axis
        // (i.e. Elemental A axis and face A axis)
        for (i = 0; i < m_coordim; i++)
        {
            dotproduct1 += elementAaxis[i] * faceAaxis[i];
        }

        NekDouble norm =
            fabs(dotproduct1) / elementAaxis_length / faceAaxis_length;
        orientation = 0;

        // if the innerproduct is equal to the (absolute value of the ) products
        // of the lengths of both vectors, then, the coordinate systems will NOT
        // be transposed
        if (fabs(norm - 1.0) < NekConstants::kNekZeroTol)
        {
            // if the inner product is negative, both A-axis point
            // in reverse direction
            if (dotproduct1 < 0.0)
            {
                orientation += 2;
            }

            // calculate the inner product of both B-axis
            for (i = 0; i < m_coordim; i++)
            {
                dotproduct2 += elementBaxis[i] * faceBaxis[i];
            }

            norm = fabs(dotproduct2) / elementBaxis_length / faceBaxis_length;

            // check that both these axis are indeed parallel
            if (fabs(norm - 1.0) >= NekConstants::kNekZeroTol)
            {
                NEKERROR(ErrorUtil::ewarning,
                         "These vectors should be parallel");
            }

            // if the inner product is negative, both B-axis point
            // in reverse direction
            if (dotproduct2 < 0.0)
            {
                orientation++;
            }
        }
        // The coordinate systems are transposed
        else
        {
            orientation = 4;

            // Calculate the inner product between the elemental A-axis
            // and the B-axis of the face (which are now the corresponding axis)
            dotproduct1 = 0.0;
            for (i = 0; i < m_coordim; i++)
            {
                dotproduct1 += elementAaxis[i] * faceBaxis[i];
            }

            norm = fabs(dotproduct1) / elementAaxis_length / faceBaxis_length;

            // check that both these axis are indeed parallel
            if (fabs(norm - 1.0) >= NekConstants::kNekZeroTol)
            {
                NEKERROR(ErrorUtil::ewarning,
                         "These vectors should be parallel");
            }

            // if the result is negative, both axis point in reverse
            // directions
            if (dotproduct1 < 0.0)
            {
                orientation += 2;
            }

            // Do the same for the other two corresponding axis
            dotproduct2 = 0.0;
            for (i = 0; i < m_coordim; i++)
            {
                dotproduct2 += elementBaxis[i] * faceAaxis[i];
            }

            norm = fabs(dotproduct2) / elementBaxis_length / faceAaxis_length;

            // check that both these axis are indeed parallel
            if (fabs(norm - 1.0) >= NekConstants::kNekZeroTol)
            {
                NEKERROR(ErrorUtil::ewarning,
                         "These vectors should be parallel");
            }

            if (dotproduct2 < 0.0)
            {
                orientation++;
            }
        }

        orientation = orientation + 5;

        ASSERTL0(orientation < StdRegions::eDir1FwdDir2_Dir2FwdDir1,
                 "Orientation of triangular face (id = " +
                     std::to_string(m_faces[f]->GetGlobalID()) +
                     ") is inconsistent with face " + std::to_string(f) +
                     " of tet element (id = " + std::to_string(m_globalID) +
                     ") since Dir2 is aligned with Dir1. Mesh setup "
                     "needs investigation");

        // Fill the m_forient array
        m_forient[f] = (StdRegions::Orientation)orientation;
    }
}

void TetGeom::v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    Geometry::v_Reset(curvedEdges, curvedFaces);

    for (int i = 0; i < 4; ++i)
    {
        m_faces[i]->Reset(curvedEdges, curvedFaces);
    }

    SetUpXmap();
    SetUpCoeffs(m_xmap->GetNcoeffs());
}

void TetGeom::v_ResetLite()
{
    SetUpEdgeOrientation();
    SetUpFaceOrientation();
}

void TetGeom::v_Setup()
{
    if (!m_setupState)
    {
        for (int i = 0; i < 4; ++i)
        {
            m_faces[i]->Setup();
        }
        SetUpXmap();
        SetUpCoeffs(m_xmap->GetNcoeffs());

        m_setupState = true;
    }
}

/**
 * Generate the geometry factors for this element.
 */
GeomType TetGeom::v_CalcGeomType(IsoParam &iso)
{
    if (!m_setupState)
    {
        TetGeom::v_Setup();
    }
    v_FillGeom();

    GeomType Gtype = eRegular;

    // check to see if expansions are linear
    if (m_xmap->GetBasisNumModes(0) != 2 || m_xmap->GetBasisNumModes(1) != 2 ||
        m_xmap->GetBasisNumModes(2) != 2)
    {
        Gtype = eDeformed;
    }

    if (Gtype == eRegular)
    {
        iso.m_nCoeff = 4;
        for (int i = 0; i < 3; ++i)
        {
            NekDouble A       = (*m_verts[0])(i);
            NekDouble B       = (*m_verts[1])(i);
            NekDouble C       = (*m_verts[2])(i);
            NekDouble D       = (*m_verts[3])(i);
            iso.m_coeff[i][0] = 0.5 * (-A + B + C + D);

            iso.m_coeff[i][1] = 0.5 * (-A + B); // xi1
            iso.m_coeff[i][2] = 0.5 * (-A + C); // xi2
            iso.m_coeff[i][3] = 0.5 * (-A + D); // xi3
        }
    }

    return Gtype;
}

GeomFactorsUniquePtr TetGeom::v_GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    GeomType Gtype = CalcGeomType();

    return ObjPoolManager<GeomFactors>::AllocateUniquePtr(
        Gtype, m_coordim, m_xmap, m_coeffs, keyTgt);
}

/**
 * @brief Set up the #m_xmap object by determining the order of each
 * direction from derived faces.
 */
void TetGeom::SetUpXmap()
{
    std::vector<int> tmp;
    tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(0));
    int order0 = *std::max_element(tmp.begin(), tmp.end());

    tmp.clear();
    tmp.push_back(order0);
    tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(1));
    tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(2));
    int order1 = *std::max_element(tmp.begin(), tmp.end());

    tmp.clear();
    tmp.push_back(order0);
    tmp.push_back(order1);
    tmp.push_back(m_faces[1]->GetXmap()->GetTraceNcoeffs(1));
    tmp.push_back(m_faces[1]->GetXmap()->GetTraceNcoeffs(2));
    tmp.push_back(m_faces[3]->GetXmap()->GetTraceNcoeffs(1));
    int order2 = *std::max_element(tmp.begin(), tmp.end());

    std::array<LibUtilities::BasisKey, 3> basis = {
        LibUtilities::BasisKey(
            LibUtilities::eModified_A, order0,
            LibUtilities::PointsKey(order0 + 1,
                                    LibUtilities::eGaussLobattoLegendre)),
        LibUtilities::BasisKey(
            LibUtilities::eModified_B, order1,
            LibUtilities::PointsKey(order1,
                                    LibUtilities::eGaussRadauMAlpha1Beta0)),
        LibUtilities::BasisKey(
            LibUtilities::eModified_C, order2,
            LibUtilities::PointsKey(order2,
                                    LibUtilities::eGaussRadauMAlpha2Beta0))};

    m_xmap = GetStdTetFactory().CreateInstance(basis);
}

/**
 * @brief Put all quadrature information into face/edge structure and
 * backward transform.
 *
 * Note verts, edges, and faces are listed according to anticlockwise
 * convention but points in _coeffs have to be in array format from left
 * to right.
 */
void TetGeom::v_FillGeom()
{
    if (m_state == ePtsFilled)
    {
        return;
    }

    int i, j, k;

    if (m_curve)
    {
        // Interior nodes of the element itself. A tetrahedron's nodes are a
        // nodal distribution rather than a tensor grid, so this follows the
        // triangle: interpolate through a nodal expansion of matching order,
        // then transform. The faces below then overwrite the boundary
        // coefficients, which are shared and so authoritative, leaving only
        // the interior taken from here.
        const int N = m_curve->m_points.size();

        // N = n(n+1)(n+2)/6; recover n.
        int nEdgePts = 1;
        while (nEdgePts * (nEdgePts + 1) * (nEdgePts + 2) / 6 < N)
        {
            ++nEdgePts;
        }
        ASSERTL0(nEdgePts * (nEdgePts + 1) * (nEdgePts + 2) / 6 == N,
                 "NUMPOINTS should be a tetrahedral number in tetrahedron " +
                     std::to_string(m_globalID));

        const LibUtilities::PointsKey P0(nEdgePts,
                                         LibUtilities::eGaussLobattoLegendre);
        const LibUtilities::PointsKey P1(nEdgePts,
                                         LibUtilities::eGaussRadauMAlpha1Beta0);
        const LibUtilities::PointsKey P2(nEdgePts,
                                         LibUtilities::eGaussRadauMAlpha2Beta0);
        const LibUtilities::BasisKey T0(LibUtilities::eOrtho_A, nEdgePts, P0);
        const LibUtilities::BasisKey T1(LibUtilities::eOrtho_B, nEdgePts, P1);
        const LibUtilities::BasisKey T2(LibUtilities::eOrtho_C, nEdgePts, P2);

        const int nq =
            P0.GetNumPoints() * P1.GetNumPoints() * P2.GetNumPoints();
        Array<OneD, NekDouble> nodal(N);
        Array<OneD, NekDouble> tmp(nq);
        Array<OneD, NekDouble> phys(m_xmap->GetTotPoints());

        for (i = 0; i < m_coordim; ++i)
        {
            StdRegions::StdNodalTetExpSharedPtr t =
                MemoryManager<StdRegions::StdNodalTetExp>::AllocateSharedPtr(
                    T0, T1, T2, m_curve->m_ptype);

            for (j = 0; j < N; ++j)
            {
                nodal[j] = (m_curve->m_points[j]->GetPtr())[i];
            }

            t->BwdTrans(nodal, tmp);

            LibUtilities::Interp3D(P0, P1, P2, tmp,
                                   m_xmap->GetBasis(0)->GetPointsKey(),
                                   m_xmap->GetBasis(1)->GetPointsKey(),
                                   m_xmap->GetBasis(2)->GetPointsKey(), phys);

            m_xmap->FwdTrans(phys, m_coeffs[i]);
        }
    }

    for (i = 0; i < kNfaces; i++)
    {
        m_faces[i]->FillGeom();

        int nFaceCoeffs = m_faces[i]->GetXmap()->GetNcoeffs();

        Array<OneD, unsigned int> mapArray(nFaceCoeffs);
        Array<OneD, int> signArray(nFaceCoeffs);

        if (m_forient[i] < 9)
        {
            m_xmap->GetTraceToElementMap(
                i, mapArray, signArray, m_forient[i],
                m_faces[i]->GetXmap()->GetTraceNcoeffs(0),
                m_faces[i]->GetXmap()->GetTraceNcoeffs(1));
        }
        else
        {
            m_xmap->GetTraceToElementMap(
                i, mapArray, signArray, m_forient[i],
                m_faces[i]->GetXmap()->GetTraceNcoeffs(1),
                m_faces[i]->GetXmap()->GetTraceNcoeffs(0));
        }

        for (j = 0; j < m_coordim; j++)
        {
            const Array<OneD, const NekDouble> &coeffs =
                m_faces[i]->GetCoeffs(j);

            for (k = 0; k < nFaceCoeffs; k++)
            {
                NekDouble v              = signArray[k] * coeffs[k];
                m_coeffs[j][mapArray[k]] = v;
            }
        }
    }

    m_state = ePtsFilled;
}

std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> TetGeom::v_MakeOrder(
    int order, const LibUtilities::PointsType pType)
{
    int nPoints = order + 1;

    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsKey pKey(nPoints, pType);
    ASSERTL1(pKey.GetPointsDim() == 3, "Points distribution must be 3D");
    LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

    // A nodal tetrahedron is laid out as four vertices, then the interior of
    // each of the six edges, then the interior of each of the four faces, then
    // the interior of the element.
    const int nTetPts    = nPoints * (nPoints + 1) * (nPoints + 2) / 6;
    const int nEdgeNodes = nPoints - 2;
    const int nFaceNodes = (nPoints - 2) * (nPoints - 3) / 2;

    std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> cd;

    cd.first = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
        m_globalID, pType);

    Curve *c = cd.first.get();
    c->m_points.resize(nTetPts);

    m_curve = c;

    for (int i = 0; i < kNverts; ++i)
    {
        c->m_points[i] = m_verts[i];
    }

    // Edge interiors, taken from the edge curves so that they stay shared with
    // the neighbours that own them. m_eorient says directly whether the
    // segment runs along the element's local edge direction; unlike the
    // triangle and quadrilateral, nothing was flipped during construction.
    for (int e = 0; e < kNedges; ++e)
    {
        Curve *edgeCurve = m_edges[e]->GetCurve();
        ASSERTL1(edgeCurve != nullptr,
                 "Edge curve not set; call MakeOrder on edges before volumes");

        // The nodal distribution emits edge 2 running from vertex 2 to
        // vertex 0, whereas edgeVerts[2] is {0, 2}, so that one block is
        // traversed against the element's local edge direction. Verified
        // against eNodalTetEvenlySpaced rather than assumed.
        const bool alongLocal = (m_eorient[e] == StdRegions::eForwards);
        const bool forward    = (e == 2) ? !alongLocal : alongLocal;

        const int offset = kNverts + e * nEdgeNodes;
        for (int j = 0; j < nEdgeNodes; ++j)
        {
            c->m_points[offset + j] =
                forward ? edgeCurve->m_points[j + 1]
                        : edgeCurve->m_points[nPoints - 2 - j];
        }
    }

    // Face interiors. A face is numbered by whichever element built it, so its
    // interior nodes have to be rotated and reflected into this element's view
    // of that face before they can be copied across.
    const unsigned int faceVerts[kNfaces][TriGeom::kNverts] = {
        {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};

    if (nFaceNodes > 0)
    {
        for (int f = 0; f < kNfaces; ++f)
        {
            Curve *faceCurve = m_faces[f]->GetCurve();
            ASSERTL1(
                faceCurve != nullptr,
                "Face curve not set; call MakeOrder on faces before volumes");

            // The face's own interior nodes are the tail of its curve.
            const int faceStart =
                TriGeom::kNverts + TriGeom::kNedges * nEdgeNodes;
            std::vector<PointGeom *> faceNodes(
                faceCurve->m_points.begin() + faceStart,
                faceCurve->m_points.begin() + faceStart + nFaceNodes);

            std::vector<int> faceOwnIds(TriGeom::kNverts);
            std::vector<int> elmtIds(TriGeom::kNverts);
            for (int v = 0; v < TriGeom::kNverts; ++v)
            {
                faceOwnIds[v] = m_faces[f]->GetVertex(v)->GetGlobalID();
                elmtIds[v]    = m_verts[faceVerts[f][v]]->GetGlobalID();
            }

            HOTriangle<PointGeom *> hoTri(faceOwnIds, faceNodes);
            hoTri.Align(elmtIds);

            const int offset = kNverts + kNedges * nEdgeNodes + f * nFaceNodes;
            for (int j = 0; j < nFaceNodes; ++j)
            {
                c->m_points[offset + j] = hoTri.surfVerts[j];
            }
        }
    }

    // Interior nodes are new, and are evaluated on this element's own mapping.
    const int volStart = kNverts + kNedges * nEdgeNodes + kNfaces * nFaceNodes;
    if (volStart < nTetPts)
    {
        Array<OneD, Array<OneD, NekDouble>> phys(m_coordim);
        for (int i = 0; i < m_coordim; ++i)
        {
            phys[i] = Array<OneD, NekDouble>(m_xmap->GetTotPoints());
            m_xmap->BwdTrans(GetCoeffs(i), phys[i]);
        }

        for (int i = volStart; i < nTetPts; ++i)
        {
            Array<OneD, NekDouble> xp(3);
            xp[0] = px[i];
            xp[1] = py[i];
            xp[2] = pz[i];

            Array<OneD, NekDouble> x(3, 0.0);
            for (int j = 0; j < m_coordim; ++j)
            {
                x[j] = m_xmap->PhysEvaluate(xp, phys[j]);
            }

            cd.second.push_back(
                ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                    m_coordim, 0, x[0], x[1], x[2]));
            c->m_points[i] = cd.second.back().get();
        }
    }

    return cd;
}

} // namespace Nektar::SpatialDomains
