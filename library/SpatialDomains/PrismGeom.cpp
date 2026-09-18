////////////////////////////////////////////////////////////////////////////////
//
//  File: PrismGeom.cpp
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
//  Description: Prismatic geometry definition.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/Interp.h>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/HOAlignment.h>
#include <SpatialDomains/PrismGeom.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TriGeom.h>
#include <SpatialDomains/XmapFactory.hpp>
#include <StdRegions/StdNodalPrismExp.h>
#include <StdRegions/StdPrismExp.h>

namespace Nektar::SpatialDomains
{

const unsigned int PrismGeom::VertexEdgeConnectivity[6][3] = {
    {0, 3, 4}, {0, 1, 5}, {1, 2, 6}, {2, 3, 7}, {4, 5, 8}, {6, 7, 8}};
const unsigned int PrismGeom::VertexFaceConnectivity[6][3] = {
    {0, 1, 4}, {0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {1, 2, 4}, {2, 3, 4}};
const unsigned int PrismGeom::EdgeFaceConnectivity[9][2] = {
    {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 4}, {1, 2}, {2, 3}, {3, 4}, {2, 4}};
const unsigned int PrismGeom::EdgeNormalToFaceVert[5][4] = {
    {4, 5, 6, 7}, {1, 3, 8, -1}, {0, 2, 4, 7}, {1, 3, 8, -1}, {0, 2, 5, 6}};

XmapFactory<StdRegions::StdPrismExp, 3> &GetStdPrismFactory()
{
    static XmapFactory<StdRegions::StdPrismExp, 3> factory;
    return factory;
}

PrismGeom::PrismGeom()
{
    m_shapeType = LibUtilities::ePrism;
}

PrismGeom::PrismGeom(int id, std::array<Geometry2D *, kNfaces> faces,
                     Curve *curve)
    : Geometry3D(faces[0]->GetEdge(0)->GetVertex(0)->GetCoordim(), curve)
{
    m_shapeType = LibUtilities::ePrism;
    m_globalID  = id;
    m_faces     = faces;

    /// Set up local objects.
    SetUpLocalEdges();
    SetUpLocalVertices();
    SetUpEdgeOrientation();
    SetUpFaceOrientation();
}

PrismGeom::PrismGeom(int id, std::array<Geometry2D *, 5> faces,
                     std::array<SegGeom *, 9> edges,
                     std::array<PointGeom *, 6> verts, bool skipSetUp,
                     Curve *curve)
    : Geometry3D(faces[0]->GetEdge(0)->GetVertex(0)->GetCoordim(), curve)
{
    m_shapeType = LibUtilities::ePrism;
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

int PrismGeom::v_GetDir(const int faceidx, const int facedir) const
{
    if (faceidx == 0)
    {
        return facedir;
    }
    else if (faceidx == 1 || faceidx == 3)
    {
        return 2 * facedir;
    }
    else
    {
        return 1 + facedir;
    }
}

GeomType PrismGeom::v_CalcGeomType()
{
    if (!m_setupState)
    {
        v_Setup();
    }
    v_FillGeom();

    GeomType Gtype = eRegular;

    // check to see if expansions are linear
    if (m_xmap->GetBasisNumModes(0) != 2 || m_xmap->GetBasisNumModes(1) != 2 ||
        m_xmap->GetBasisNumModes(2) != 2)
    {
        Gtype = eDeformed;
    }

    // check to see if all quadrilateral faces are parallelograms
    if (Gtype == eRegular)
    {
        m_isoParameter = Array<OneD, Array<OneD, NekDouble>>(3);
        for (int i = 0; i < 3; ++i)
        {
            m_isoParameter[i]    = Array<OneD, NekDouble>(6, 0.);
            NekDouble A          = (*m_verts[0])(i);
            NekDouble B          = (*m_verts[1])(i);
            NekDouble C          = (*m_verts[2])(i);
            NekDouble D          = (*m_verts[3])(i);
            NekDouble E          = (*m_verts[4])(i);
            NekDouble F          = (*m_verts[5])(i);
            m_isoParameter[i][0] = 0.25 * (B + C + E + F);

            m_isoParameter[i][1] = 0.25 * (-A + B + C - D); // xi1
            m_isoParameter[i][2] = 0.25 * (-B + C - E + F); // xi2
            m_isoParameter[i][3] = 0.25 * (-A - D + E + F); // xi3

            m_isoParameter[i][4] = 0.25 * (A - B + C - D); // xi1*xi2
            m_isoParameter[i][5] = 0.25 * (A - D - E + F); // xi2*xi3
            NekDouble tmp        = fabs(m_isoParameter[i][1]) +
                            fabs(m_isoParameter[i][2]) +
                            fabs(m_isoParameter[i][3]);
            tmp *= NekConstants::kNekZeroTol;
            for (int d = 4; d < 6; ++d)
            {
                if (fabs(m_isoParameter[i][d]) > tmp)
                {
                    Gtype = eDeformed;
                }
            }
        }
    }

    if (Gtype == eRegular)
    {
        v_CalculateInverseIsoParam();
    }
    return Gtype;
}

GeomFactorsUniquePtr PrismGeom::v_GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    GeomType Gtype = CalcGeomType();

    return ObjPoolManager<GeomFactors>::AllocateUniquePtr(
        Gtype, m_coordim, m_xmap, m_coeffs, keyTgt);
}

int PrismGeom::v_GetVertexEdgeMap(const int i, const int j) const
{
    return VertexEdgeConnectivity[i][j];
}

int PrismGeom::v_GetVertexFaceMap(const int i, const int j) const
{
    return VertexFaceConnectivity[i][j];
}

int PrismGeom::v_GetEdgeFaceMap(const int i, const int j) const
{
    return EdgeFaceConnectivity[i][j];
}

int PrismGeom::v_GetEdgeNormalToFaceVert(const int i, const int j) const
{
    return EdgeNormalToFaceVert[i][j];
}

void PrismGeom::SetUpLocalEdges()
{
    // find edge 0
    int i, j;
    unsigned int check;

    // First set up the 4 bottom edges
    int f; //  Connected face index
    for (f = 1; f < 5; f++)
    {
        int nEdges = m_faces[f]->GetNumEdges();
        check      = 0;
        for (i = 0; i < 4; i++)
        {
            for (j = 0; j < nEdges; j++)
            {
                if (m_faces[0]->GetEid(i) == m_faces[f]->GetEid(j))
                {
                    m_edges[f - 1] =
                        static_cast<SegGeom *>((m_faces[0])->GetEdge(i));
                    check++;
                }
            }
        }

        if (check < 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces do not share an edge. Faces ";
            errstrm << (m_faces[0])->GetGlobalID() << ", "
                    << (m_faces[f])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
        else if (check > 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces share more than one edge. Faces ";
            errstrm << (m_faces[0])->GetGlobalID() << ", "
                    << (m_faces[f])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
    }

    // Then, set up the 4 vertical edges
    check = 0;
    for (i = 0; i < 3; i++) // Set up the vertical edge :face(1) and face(4)
    {
        for (j = 0; j < 4; j++)
        {
            if ((m_faces[1])->GetEid(i) == (m_faces[4])->GetEid(j))
            {
                m_edges[4] = static_cast<SegGeom *>((m_faces[1])->GetEdge(i));
                check++;
            }
        }
    }
    if (check < 1)
    {
        std::ostringstream errstrm;
        errstrm << "Connected faces do not share an edge. Faces ";
        errstrm << (m_faces[1])->GetGlobalID() << ", "
                << (m_faces[4])->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
    else if (check > 1)
    {
        std::ostringstream errstrm;
        errstrm << "Connected faces share more than one edge. Faces ";
        errstrm << (m_faces[1])->GetGlobalID() << ", "
                << (m_faces[4])->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
    // Set up vertical edges: face(1) through face(4)
    for (f = 1; f < 4; f++)
    {
        check = 0;
        for (i = 0; i < m_faces[f]->GetNumEdges(); i++)
        {
            for (j = 0; j < m_faces[f + 1]->GetNumEdges(); j++)
            {
                if ((m_faces[f])->GetEid(i) == (m_faces[f + 1])->GetEid(j))
                {
                    m_edges[f + 4] =
                        static_cast<SegGeom *>((m_faces[f])->GetEdge(i));
                    check++;
                }
            }
        }

        if (check < 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces do not share an edge. Faces ";
            errstrm << (m_faces[f])->GetGlobalID() << ", "
                    << (m_faces[f + 1])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
        else if (check > 1)
        {
            std::ostringstream errstrm;
            errstrm << "Connected faces share more than one edge. Faces ";
            errstrm << (m_faces[f])->GetGlobalID() << ", "
                    << (m_faces[f + 1])->GetGlobalID();
            NEKERROR(ErrorUtil::efatal, errstrm.str());
        }
    }

    // Finally, set up the 1 top edge
    check = 0;
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            if ((m_faces[2])->GetEid(i) == (m_faces[4])->GetEid(j))
            {
                m_edges[8] = static_cast<SegGeom *>((m_faces[2])->GetEdge(i));
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
}

void PrismGeom::SetUpLocalVertices()
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

    // set up the other bottom vertices (i.e. vertex 2,3)
    for (int i = 1; i < 3; i++)
    {
        if (m_edges[i]->GetVid(0) == m_verts[i]->GetGlobalID())
        {
            m_verts[i + 1] = m_edges[i]->GetVertex(1);
        }
        else if (m_edges[i]->GetVid(1) == m_verts[i]->GetGlobalID())
        {
            m_verts[i + 1] = m_edges[i]->GetVertex(0);
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

    // set up top vertices
    // First, set up vertices 4,5
    if ((m_edges[8]->GetVid(0) == m_edges[4]->GetVid(0)) ||
        (m_edges[8]->GetVid(0) == m_edges[4]->GetVid(1)))
    {
        m_verts[4] = m_edges[8]->GetVertex(0);
        m_verts[5] = m_edges[8]->GetVertex(1);
    }
    else if ((m_edges[8]->GetVid(1) == m_edges[4]->GetVid(0)) ||
             (m_edges[8]->GetVid(1) == m_edges[4]->GetVid(1)))
    {
        m_verts[4] = m_edges[8]->GetVertex(1);
        m_verts[5] = m_edges[8]->GetVertex(0);
    }
    else
    {
        std::ostringstream errstrm;
        errstrm << "Connected edges do not share a vertex. Edges ";
        errstrm << m_edges[8]->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
}

void PrismGeom::SetUpEdgeOrientation()
{

    // This 2D array holds the local id's of all the vertices
    // for every edge. For every edge, they are ordered to what we
    // define as being Forwards
    const unsigned int edgeVerts[kNedges][2] = {
        {0, 1}, {1, 2}, {3, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 5}, {3, 5}, {4, 5}};

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

void PrismGeom::SetUpFaceOrientation()
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

    // The length of the vectors above
    NekDouble elementAaxis_length;
    NekDouble elementBaxis_length;
    NekDouble faceAaxis_length;
    NekDouble faceBaxis_length;

    // This 2D array holds the local id's of all the vertices
    // for every face. For every face, they are ordered in such
    // a way that the implementation below allows a unified approach
    // for all faces.
    const unsigned int faceVerts[kNfaces][QuadGeom::kNverts] = {
        {0, 1, 2, 3},
        {0, 1, 4, 0}, // This is triangle requires only three vertices
        {1, 2, 5, 4},
        {3, 2, 5, 0}, // This is triangle requires only three vertices
        {0, 3, 5, 4},
    };

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
        if (f == 1 || f == 3)
        { // Face is a Triangle
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
        }
        else
        { // Face is a Quad
            if (baseVertex == m_verts[faceVerts[f][0]]->GetGlobalID())
            {
                for (i = 0; i < m_coordim; i++)
                {
                    elementAaxis[i] = (*m_verts[faceVerts[f][1]])[i] -
                                      (*m_verts[faceVerts[f][0]])[i];
                    elementBaxis[i] = (*m_verts[faceVerts[f][3]])[i] -
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
                    elementAaxis[i] = (*m_verts[faceVerts[f][2]])[i] -
                                      (*m_verts[faceVerts[f][3]])[i];
                    elementBaxis[i] = (*m_verts[faceVerts[f][2]])[i] -
                                      (*m_verts[faceVerts[f][1]])[i];
                }
            }
            else if (baseVertex == m_verts[faceVerts[f][3]]->GetGlobalID())
            {
                for (i = 0; i < m_coordim; i++)
                {
                    elementAaxis[i] = (*m_verts[faceVerts[f][2]])[i] -
                                      (*m_verts[faceVerts[f][3]])[i];
                    elementBaxis[i] = (*m_verts[faceVerts[f][3]])[i] -
                                      (*m_verts[faceVerts[f][0]])[i];
                }
            }
            else
            {
                NEKERROR(ErrorUtil::efatal,
                         "Could not find matching vertex for the face");
            }
        }
        // Now, construct the edge-vectors of the local coordinates of
        // the Geometry2D-representation of the face
        for (i = 0; i < m_coordim; i++)
        {
            int v = m_faces[f]->GetNumVerts() - 1;
            faceAaxis[i] =
                (*m_faces[f]->GetVertex(1))[i] - (*m_faces[f]->GetVertex(0))[i];
            faceBaxis[i] =
                (*m_faces[f]->GetVertex(v))[i] - (*m_faces[f]->GetVertex(0))[i];

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

        orientation = 0;

        // if the innerproduct is equal to the (absolute value of the ) products
        // of the lengths of both vectors, then, the coordinate systems will NOT
        // be transposed
        if (fabs(elementAaxis_length * faceAaxis_length - fabs(dotproduct1)) <
            NekConstants::kNekZeroTol)
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
            if (fabs(
                    fabs(dotproduct2 / elementBaxis_length / faceBaxis_length) -
                    1.0) >= NekConstants::kNekZeroTol)
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

            // check that both these axis are indeed parallel
            if (fabs(fabs(dotproduct1) / elementAaxis_length /
                         faceBaxis_length -
                     1.0) >= NekConstants::kNekZeroTol)
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

            if (fabs(fabs(dotproduct2) / elementBaxis_length /
                         faceAaxis_length -
                     1.0) >= NekConstants::kNekZeroTol)
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

        if ((f == 1) || (f == 3)) // check triange orientation
        {
            ASSERTL0(
                orientation < StdRegions::eDir1FwdDir2_Dir2FwdDir1,
                "Orientation of triangular face (id = " +
                    std::to_string(m_faces[f]->GetGlobalID()) +
                    ") is inconsistent with face " + std::to_string(f) +
                    " of prism element (id = " + std::to_string(m_globalID) +
                    ") since Dir2 is aligned with Dir1. Mesh setup "
                    "needs investigation");
        }

        // Fill the m_forient array
        m_forient[f] = (StdRegions::Orientation)orientation;
    }
}

void PrismGeom::v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    Geometry::v_Reset(curvedEdges, curvedFaces);

    for (int i = 0; i < 5; ++i)
    {
        m_faces[i]->Reset(curvedEdges, curvedFaces);
    }

    SetUpXmap();
    SetUpCoeffs(m_xmap->GetNcoeffs());
}

void PrismGeom::v_ResetLite()
{
    SetUpEdgeOrientation();
    SetUpFaceOrientation();
}

void PrismGeom::v_Setup()
{
    if (!m_setupState)
    {
        for (int i = 0; i < 5; ++i)
        {
            m_faces[i]->Setup();
        }
        SetUpXmap();
        SetUpCoeffs(m_xmap->GetNcoeffs());

        // check to see if expansions are linear
        m_straightEdge = 1;
        if (m_xmap->GetBasisNumModes(0) != 2 ||
            m_xmap->GetBasisNumModes(1) != 2 ||
            m_xmap->GetBasisNumModes(2) != 2)
        {
            m_straightEdge = 0;
        }

        m_setupState = true;
    }
}

/**
 * @brief Set up the #m_xmap object by determining the order of each
 * direction from derived faces.
 */
void PrismGeom::SetUpXmap()
{
    std::vector<int> tmp;
    int order0, order1;

    if (m_forient[0] < 9)
    {
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(0));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(2));
        order0 = *std::max_element(tmp.begin(), tmp.end());
    }
    else
    {
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(1));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(3));
        order0 = *std::max_element(tmp.begin(), tmp.end());
    }

    if (m_forient[0] < 9)
    {
        tmp.clear();
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(1));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(3));
        tmp.push_back(m_faces[2]->GetXmap()->GetTraceNcoeffs(2));
        order1 = *std::max_element(tmp.begin(), tmp.end());
    }
    else
    {
        tmp.clear();
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(0));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(2));
        tmp.push_back(m_faces[2]->GetXmap()->GetTraceNcoeffs(2));
        order1 = *std::max_element(tmp.begin(), tmp.end());
    }

    tmp.clear();
    tmp.push_back(order0);
    tmp.push_back(order1);
    tmp.push_back(m_faces[1]->GetXmap()->GetTraceNcoeffs(1));
    tmp.push_back(m_faces[1]->GetXmap()->GetTraceNcoeffs(2));
    tmp.push_back(m_faces[3]->GetXmap()->GetTraceNcoeffs(1));
    tmp.push_back(m_faces[3]->GetXmap()->GetTraceNcoeffs(2));
    int order2 = *std::max_element(tmp.begin(), tmp.end());

    std::array<LibUtilities::BasisKey, 3> basis = {
        LibUtilities::BasisKey(
            LibUtilities::eModified_A, order0,
            LibUtilities::PointsKey(order0 + 1,
                                    LibUtilities::eGaussLobattoLegendre)),
        LibUtilities::BasisKey(
            LibUtilities::eModified_A, order1,
            LibUtilities::PointsKey(order1 + 1,
                                    LibUtilities::eGaussLobattoLegendre)),
        LibUtilities::BasisKey(
            LibUtilities::eModified_B, order2,
            LibUtilities::PointsKey(order2,
                                    LibUtilities::eGaussRadauMAlpha1Beta0))};

    m_xmap = GetStdPrismFactory().CreateInstance(basis);
}

/**
 * @brief Put all quadrature information into face/edge structure and
 * backward transform.
 *
 * Note verts, edges, and faces are listed according to anticlockwise
 * convention but points in _coeffs have to be in array format from left
 * to right.
 */
void PrismGeom::v_FillGeom()
{
    if (m_state == ePtsFilled)
    {
        return;
    }

    if (m_curve)
    {
        // Interior nodes of the element itself, taken through a nodal prism
        // expansion of matching order in the same way the tetrahedron does.
        // The face loop below then overwrites the boundary coefficients, which
        // are shared and so authoritative.
        const int N = m_curve->m_points.size();

        // N = n(n+1)/2 * n; recover n.
        int nEdgePts = 1;
        while (nEdgePts * (nEdgePts + 1) / 2 * nEdgePts < N)
        {
            ++nEdgePts;
        }
        ASSERTL0(nEdgePts * (nEdgePts + 1) / 2 * nEdgePts == N,
                 "NUMPOINTS should be a prism number in prism " +
                     std::to_string(m_globalID));

        const LibUtilities::PointsKey P0(nEdgePts,
                                         LibUtilities::eGaussLobattoLegendre);
        const LibUtilities::PointsKey P1(nEdgePts,
                                         LibUtilities::eGaussLobattoLegendre);
        const LibUtilities::PointsKey P2(nEdgePts,
                                         LibUtilities::eGaussRadauMAlpha1Beta0);
        const LibUtilities::BasisKey T0(LibUtilities::eOrtho_A, nEdgePts, P0);
        const LibUtilities::BasisKey T1(LibUtilities::eOrtho_A, nEdgePts, P1);
        const LibUtilities::BasisKey T2(LibUtilities::eOrtho_B, nEdgePts, P2);

        const int nq =
            P0.GetNumPoints() * P1.GetNumPoints() * P2.GetNumPoints();
        Array<OneD, NekDouble> nodal(N);
        Array<OneD, NekDouble> tmp(nq);
        Array<OneD, NekDouble> phys(m_xmap->GetTotPoints());

        for (int i = 0; i < m_coordim; ++i)
        {
            StdRegions::StdNodalPrismExpSharedPtr t =
                MemoryManager<StdRegions::StdNodalPrismExp>::AllocateSharedPtr(
                    T0, T1, T2, m_curve->m_ptype);

            for (int j = 0; j < N; ++j)
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

    int i, j, k;

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

std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> PrismGeom::
    v_MakeOrder(int order, const LibUtilities::PointsType pType)
{
    int nPoints = order + 1;

    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsKey pKey(nPoints, pType);
    ASSERTL1(pKey.GetPointsDim() == 3, "Points distribution must be 3D");
    LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

    // A nodal prism is laid out as six vertices, then the interior of each of
    // the nine edges, then the interior of each of the five faces in
    // faceVerts order -- quadrilateral, triangle, quadrilateral, triangle,
    // quadrilateral -- and finally the interior of the element.
    const int nPrismPts  = nPoints * (nPoints + 1) / 2 * nPoints;
    const int nEdgeNodes = nPoints - 2;
    const int nTriNodes  = (nPoints - 2) * (nPoints - 3) / 2;
    const int nQuadNodes = (nPoints - 2) * (nPoints - 2);

    std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> cd;

    cd.first = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
        m_globalID, pType);

    Curve *c = cd.first.get();
    c->m_points.resize(nPrismPts);

    m_curve = c;

    // The nodal distribution numbers its vertices in raster order, which
    // transposes v2 and v3 with respect to the standard element: nodal slot 2
    // is at (-1, 1, -1), which is where the standard element puts vertex 3.
    const int vertPerm[kNverts] = {0, 1, 3, 2, 4, 5};
    for (int i = 0; i < kNverts; ++i)
    {
        c->m_points[i] = m_verts[vertPerm[i]];
    }

    // Edge interiors. Measured against eNodalPrismEvenlySpaced: the blocks
    // come in edgeVerts order and each runs along its edge, so unlike the
    // tetrahedron there is no block to reverse.
    for (int e = 0; e < kNedges; ++e)
    {
        Curve *edgeCurve = m_edges[e]->GetCurve();
        ASSERTL1(edgeCurve != nullptr,
                 "Edge curve not set; call MakeOrder on edges before volumes");

        const int offset = kNverts + e * nEdgeNodes;
        for (int j = 0; j < nEdgeNodes; ++j)
        {
            c->m_points[offset + j] =
                (m_eorient[e] == StdRegions::eForwards)
                    ? edgeCurve->m_points[j + 1]
                    : edgeCurve->m_points[nPoints - 2 - j];
        }
    }

    // Face interiors. A face is numbered by whichever element built it, so its
    // nodes have to be brought into this element's view first.
    const unsigned int faceVerts[kNfaces][QuadGeom::kNverts] = {
        {0, 1, 2, 3}, {0, 1, 4, 0}, {1, 2, 5, 4}, {3, 2, 5, 0}, {0, 3, 5, 4}};
    const bool isTri[kNfaces] = {false, true, false, true, false};

    int offset = kNverts + kNedges * nEdgeNodes;
    for (int f = 0; f < kNfaces; ++f)
    {
        Curve *faceCurve = m_faces[f]->GetCurve();
        ASSERTL1(faceCurve != nullptr,
                 "Face curve not set; call MakeOrder on faces before volumes");

        if (isTri[f])
        {
            if (nTriNodes > 0)
            {
                // A triangle's curve is nodal, so its interior is the tail.
                const int faceStart =
                    TriGeom::kNverts + TriGeom::kNedges * nEdgeNodes;
                std::vector<PointGeom *> faceNodes(
                    faceCurve->m_points.begin() + faceStart,
                    faceCurve->m_points.begin() + faceStart + nTriNodes);

                std::vector<int> faceOwnIds(TriGeom::kNverts);
                std::vector<int> elmtIds(TriGeom::kNverts);
                for (int v = 0; v < TriGeom::kNverts; ++v)
                {
                    faceOwnIds[v] = m_faces[f]->GetVertex(v)->GetGlobalID();
                    elmtIds[v]    = m_verts[faceVerts[f][v]]->GetGlobalID();
                }

                HOTriangle<PointGeom *> hoTri(faceOwnIds, faceNodes);
                hoTri.Align(elmtIds);

                for (int j = 0; j < nTriNodes; ++j)
                {
                    c->m_points[offset + j] = hoTri.surfVerts[j];
                }
            }
            offset += nTriNodes;
        }
        else
        {
            if (nQuadNodes > 0)
            {
                // A quadrilateral's curve is a tensor grid, so reindex it into
                // this element's ordering of the face and then take the
                // interior, first face direction fastest.
                Array<OneD, int> idmap;
                m_xmap->ReOrientTracePhysMap(m_forient[f], idmap, nPoints,
                                             nPoints);

                std::vector<PointGeom *> elemFace(nPoints * nPoints, nullptr);
                for (int t = 0; t < nPoints * nPoints; ++t)
                {
                    elemFace[idmap[t]] = faceCurve->m_points[t];
                }

                int cnt = 0;
                for (int b = 1; b < nPoints - 1; ++b)
                {
                    for (int a = 1; a < nPoints - 1; ++a)
                    {
                        c->m_points[offset + cnt++] = elemFace[a + nPoints * b];
                    }
                }
            }
            offset += nQuadNodes;
        }
    }

    // Interior nodes are new, and are evaluated on this element's own mapping.
    if (offset < nPrismPts)
    {
        Array<OneD, Array<OneD, NekDouble>> phys(m_coordim);
        for (int i = 0; i < m_coordim; ++i)
        {
            phys[i] = Array<OneD, NekDouble>(m_xmap->GetTotPoints());
            m_xmap->BwdTrans(GetCoeffs(i), phys[i]);
        }

        for (int i = offset; i < nPrismPts; ++i)
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
