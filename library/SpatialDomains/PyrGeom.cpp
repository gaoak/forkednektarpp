////////////////////////////////////////////////////////////////////////////////
//
//  File: PyrGeom.cpp
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
//  Description: Pyramidic geometry information.
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/PyrGeom.h>

#include <LibUtilities/Foundations/Interp.h>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/HOAlignment.h>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TriGeom.h>
#include <SpatialDomains/XmapFactory.hpp>
#include <StdRegions/StdNodalPyrExp.h>
#include <StdRegions/StdPyrExp.h>

namespace Nektar::SpatialDomains
{
const unsigned int PyrGeom::EdgeNormalToFaceVert[5][4] = {
    {4, 5, 6, 7}, {1, 3, 6, 7}, {0, 2, 4, 7}, {1, 3, 4, 5}, {0, 2, 5, 6}};

XmapFactory<StdRegions::StdPyrExp, 3> &GetStdPyrFactory()
{
    static XmapFactory<StdRegions::StdPyrExp, 3> factory;
    return factory;
}

PyrGeom::PyrGeom()
{
    m_shapeType = LibUtilities::ePyramid;
}

PyrGeom::PyrGeom(int id, std::array<Geometry2D *, 5> faces, Curve *curve)
    : Geometry3D(faces[0]->GetEdge(0)->GetVertex(0)->GetCoordim(), curve)
{
    m_shapeType = LibUtilities::ePyramid;
    m_globalID  = id;

    /// Copy the face pointers
    for (int i = 0; i < kNfaces; i++)
    {
        m_faces[i] = faces[i];
    }

    SetUpLocalEdges();
    SetUpLocalVertices();
    SetUpEdgeOrientation();
    SetUpFaceOrientation();
}

GeomType PyrGeom::v_CalcGeomType(IsoParam &iso)
{
    if (!m_setupState)
    {
        v_Setup();
    }

    GeomType Gtype = eRegular;

    // check to see if expansions are linear
    if (m_xmap->GetBasisNumModes(0) != 2 || m_xmap->GetBasisNumModes(1) != 2 ||
        m_xmap->GetBasisNumModes(2) != 2)
    {
        Gtype = eDeformed;
    }

    // Check to see if the quadrilateral base is a parallelogram.
    //
    // Unlike the other elements, a pyramid's mapping is not a polynomial in
    // the local coordinates. Both of the base directions collapse against the
    // same denominator -- eta_i = 2(1 + xi_i)/(1 - xi_2) - 1 for i = 0,1 --
    // so the bilinear base term carries a factor 1/(1 - xi_2). The polynomial
    // below is therefore only the true mapping when the coefficient of that
    // term, A - B + C - D, vanishes, which is exactly when the base is a
    // parallelogram and the whole mapping collapses to an affine one.
    //
    // @see StdRegions::StdPyrExp::v_LocCoordToLocCollapsed
    if (Gtype == eRegular)
    {
        iso.m_nCoeff = 5;
        for (int i = 0; i < 3; ++i)
        {
            NekDouble A       = (*m_verts[0])(i);
            NekDouble B       = (*m_verts[1])(i);
            NekDouble C       = (*m_verts[2])(i);
            NekDouble D       = (*m_verts[3])(i);
            NekDouble E       = (*m_verts[4])(i);
            iso.m_coeff[i][0] = 0.25 * (-A + B + C + D + E + E);

            iso.m_coeff[i][1] = 0.25 * (-A + B + C - D); // xi1
            iso.m_coeff[i][2] = 0.25 * (-A - B + C + D); // xi2
            iso.m_coeff[i][3] = 0.5 * (-A + E);          // xi3

            iso.m_coeff[i][4] = 0.25 * (A - B + C - D); // xi1*xi2
            NekDouble tmp = fabs(iso.m_coeff[i][1]) + fabs(iso.m_coeff[i][2]) +
                            fabs(iso.m_coeff[i][3]);
            if (fabs(iso.m_coeff[i][4]) > tmp * NekConstants::kNekZeroTol)
            {
                Gtype = eDeformed;
            }
        }

        if (Gtype != eRegular)
        {
            // The base is not a parallelogram, so the coefficients above do
            // not describe this mapping away from the vertices. Report that
            // there is no polynomial form, which sends the inverse mapping to
            // the general Newton iteration on the chi mapping itself.
            iso.m_nCoeff = 0;
        }
    }

    return Gtype;
}

GeomFactorsUniquePtr PyrGeom::v_GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    FillGeom();
    GeomType Gtype = CalcGeomType();

    return ObjPoolManager<GeomFactors>::AllocateUniquePtr(
        Gtype, m_coordim, m_xmap, m_coeffs, keyTgt);
}

int PyrGeom::v_GetDir(const int faceidx, const int facedir) const
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

int PyrGeom::v_GetEdgeNormalToFaceVert(const int i, const int j) const
{
    return EdgeNormalToFaceVert[i][j];
}

void PyrGeom::SetUpLocalEdges()
{
    // find edge 0
    int i, j;
    unsigned int check;

    // First set up the 4 bottom edges
    int f;
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
        for (j = 0; j < 3; j++)
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
}

void PyrGeom::SetUpLocalVertices()
{
    // Set up the first 2 vertices (i.e. vertex 0,1)
    if (m_edges[0]->GetVid(0) == m_edges[1]->GetVid(0) ||
        m_edges[0]->GetVid(0) == m_edges[1]->GetVid(1))
    {
        m_verts[0] = m_edges[0]->GetVertex(1);
        m_verts[1] = m_edges[0]->GetVertex(0);
    }
    else if (m_edges[0]->GetVid(1) == m_edges[1]->GetVid(0) ||
             m_edges[0]->GetVid(1) == m_edges[1]->GetVid(1))
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

    // set up top vertex
    if (m_edges[4]->GetVid(0) == m_verts[0]->GetGlobalID())
    {
        m_verts[4] = m_edges[4]->GetVertex(1);
    }
    else
    {
        m_verts[4] = m_edges[4]->GetVertex(0);
    }

    int check = 0;
    for (int i = 5; i < 8; ++i)
    {
        if ((m_edges[i]->GetVid(0) == m_verts[i - 4]->GetGlobalID() &&
             m_edges[i]->GetVid(1) == m_verts[4]->GetGlobalID()) ||
            (m_edges[i]->GetVid(1) == m_verts[i - 4]->GetGlobalID() &&
             m_edges[i]->GetVid(0) == m_verts[4]->GetGlobalID()))
        {
            check++;
        }
    }
    if (check != 3)
    {
        std::ostringstream errstrm;
        errstrm << "Connected edges do not share a vertex. Edges ";
        errstrm << m_edges[3]->GetGlobalID() << ", "
                << m_edges[2]->GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }
}

void PyrGeom::SetUpEdgeOrientation()
{
    // This 2D array holds the local id's of all the vertices for every
    // edge. For every edge, they are ordered to what we define as being
    // Forwards.
    const unsigned int edgeVerts[kNedges][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3},
                                                {0, 4}, {1, 4}, {2, 4}, {3, 4}};

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

void PyrGeom::SetUpFaceOrientation()
{
    // This 2D array holds the local id's of all the vertices for every face,
    // in the cyclic order in which they bound that face. Faces 1 to 4 are
    // triangles and use only the first three entries.
    const unsigned int faceVerts[kNfaces][QuadGeom::kNverts] = {
        {0, 1, 2, 3}, {0, 1, 4, 0}, {1, 2, 4, 0}, {3, 2, 4, 0}, {0, 3, 4, 0}};

    // Quadrilateral faces: the face geometry runs around the same four points
    // as the element's face does, so its vertex list is that cycle started at
    // some vertex and traversed in one of the two directions. Those eight
    // cases are exactly the eight quadrilateral orientations.
    const StdRegions::Orientation quadFwd[QuadGeom::kNverts] = {
        StdRegions::eDir1FwdDir1_Dir2FwdDir2, // starts at face vertex 0
        StdRegions::eDir1BwdDir2_Dir2FwdDir1, // starts at face vertex 1
        StdRegions::eDir1BwdDir1_Dir2BwdDir2, // starts at face vertex 2
        StdRegions::eDir1FwdDir2_Dir2BwdDir1  // starts at face vertex 3
    };
    const StdRegions::Orientation quadBwd[QuadGeom::kNverts] = {
        StdRegions::eDir1FwdDir2_Dir2FwdDir1, // starts at face vertex 0
        StdRegions::eDir1BwdDir1_Dir2FwdDir2, // starts at face vertex 1
        StdRegions::eDir1BwdDir2_Dir2BwdDir1, // starts at face vertex 2
        StdRegions::eDir1FwdDir1_Dir2BwdDir2  // starts at face vertex 3
    };

    for (int f = 0; f < kNfaces; ++f)
    {
        const int nVerts  = m_faces[f]->GetNumVerts();
        const int baseVid = m_faces[f]->GetVid(0);

        // Which of the element's face vertices the face geometry starts at.
        int base = -1;
        for (int v = 0; v < nVerts; ++v)
        {
            if (baseVid == m_verts[faceVerts[f][v]]->GetGlobalID())
            {
                base = v;
                break;
            }
        }

        if (base < 0)
        {
            NEKERROR(ErrorUtil::efatal,
                     "Face " + std::to_string(f) + " of pyramid " +
                         std::to_string(m_globalID) +
                         " does not start at any of the element vertices that "
                         "bound it.");
        }
        else if (nVerts == TriGeom::kNverts)
        {
            // Triangular face. SetUpLocalVertices has already put the vertices
            // in a canonical order by the time this runs, which leaves three
            // of the six permutations reachable; the other three all give an
            // orientation in which Dir2 is aligned with Dir1, which the vector
            // implementation this replaced rejected outright.
            const int nextVid = m_faces[f]->GetVid(1);

            if (base == 0 && nextVid == m_verts[faceVerts[f][1]]->GetGlobalID())
            {
                // Face vertices are element face vertices (0, 1, 2).
                m_forient[f] = StdRegions::eDir1FwdDir1_Dir2FwdDir2;
            }
            else if (base == 1 &&
                     nextVid == m_verts[faceVerts[f][0]]->GetGlobalID())
            {
                // Face vertices are element face vertices (1, 0, 2).
                m_forient[f] = StdRegions::eDir1BwdDir1_Dir2FwdDir2;
            }
            else if (base == 2 &&
                     nextVid == m_verts[faceVerts[f][1]]->GetGlobalID())
            {
                // Face vertices are element face vertices (2, 1, 0).
                m_forient[f] = StdRegions::eDir1FwdDir1_Dir2BwdDir2;
            }
            else
            {
                NEKERROR(ErrorUtil::efatal,
                         "Orientation of triangular face (id = " +
                             std::to_string(m_faces[f]->GetGlobalID()) +
                             ") is inconsistent with face " +
                             std::to_string(f) + " of pyramid element (id = " +
                             std::to_string(m_globalID) +
                             ") since Dir2 is aligned with Dir1. Mesh setup "
                             "needs investigation");
            }
        }
        else
        {
            const int nextVid = m_faces[f]->GetVid(1);

            if (nextVid == m_verts[faceVerts[f][(base + 1) % QuadGeom::kNverts]]
                               ->GetGlobalID())
            {
                m_forient[f] = quadFwd[base];
            }
            else if (nextVid ==
                     m_verts[faceVerts[f][(base + 3) % QuadGeom::kNverts]]
                         ->GetGlobalID())
            {
                m_forient[f] = quadBwd[base];
            }
            else
            {
                NEKERROR(ErrorUtil::efatal,
                         "The second vertex of quadrilateral face " +
                             std::to_string(f) + " of pyramid " +
                             std::to_string(m_globalID) +
                             " is not adjacent to its first one, so the face "
                             "does not traverse the element's face cycle.");
            }
        }
    }
}

void PyrGeom::v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    Geometry::v_Reset(curvedEdges, curvedFaces);

    for (int i = 0; i < 5; ++i)
    {
        m_faces[i]->Reset(curvedEdges, curvedFaces);
    }

    SetUpXmap();
}

void PyrGeom::v_ResetLite()
{
    SetUpEdgeOrientation();
    SetUpFaceOrientation();
}

void PyrGeom::v_Setup()
{
    if (!m_setupState)
    {
        for (int i = 0; i < 5; ++i)
        {
            m_faces[i]->Setup();
        }
        SetUpXmap();

        m_setupState = true;
    }
}

/**
 * @brief Set up the #m_xmap object by determining the order of each
 * direction from derived faces.
 */
void PyrGeom::SetUpXmap()
{
    std::vector<int> tmp;
    int order0, order1;

    if (m_forient[0] < 9)
    {
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(0));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(2));
        order0 = *max_element(tmp.begin(), tmp.end());
    }
    else
    {
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(1));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(3));
        order0 = *max_element(tmp.begin(), tmp.end());
    }

    if (m_forient[0] < 9)
    {
        tmp.clear();
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(1));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(3));
        tmp.push_back(m_faces[2]->GetXmap()->GetTraceNcoeffs(2));
        order1 = *max_element(tmp.begin(), tmp.end());
    }
    else
    {
        tmp.clear();
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(0));
        tmp.push_back(m_faces[0]->GetXmap()->GetTraceNcoeffs(2));
        tmp.push_back(m_faces[2]->GetXmap()->GetTraceNcoeffs(2));
        order1 = *max_element(tmp.begin(), tmp.end());
    }

    tmp.clear();
    tmp.push_back(order0);
    tmp.push_back(order1);
    tmp.push_back(m_faces[1]->GetXmap()->GetTraceNcoeffs(1));
    tmp.push_back(m_faces[1]->GetXmap()->GetTraceNcoeffs(2));
    tmp.push_back(m_faces[3]->GetXmap()->GetTraceNcoeffs(1));
    tmp.push_back(m_faces[3]->GetXmap()->GetTraceNcoeffs(2));
    int order2 = *max_element(tmp.begin(), tmp.end());

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
            LibUtilities::eModifiedPyr_C, order2,
            LibUtilities::PointsKey(order2,
                                    LibUtilities::eGaussRadauMAlpha2Beta0))};

    m_xmap = GetStdPyrFactory().CreateInstance(basis).get();
}

/**
 * @brief Put all quadrature information into face/edge structure and
 * backward transform.
 *
 * Note verts, edges, and faces are listed according to anticlockwise
 * convention but points in _coeffs have to be in array format from left
 * to right.
 */
void PyrGeom::v_FillGeom()
{

    if (m_curve)
    {
        // Interior nodes of the element itself, taken through a nodal pyramid
        // expansion of matching order in the same way the tetrahedron and
        // prism do. The face loop below then overwrites the boundary
        // coefficients, which are shared and so authoritative.
        const int N = m_curve->m_points.size();

        // N is the square pyramidal number n(n+1)(2n+1)/6; recover n.
        int nEdgePts = 1;
        while (nEdgePts * (nEdgePts + 1) * (2 * nEdgePts + 1) / 6 < N)
        {
            ++nEdgePts;
        }
        ASSERTL0(nEdgePts * (nEdgePts + 1) * (2 * nEdgePts + 1) / 6 == N,
                 "NUMPOINTS should be a square pyramidal number in pyramid " +
                     std::to_string(m_globalID));

        const LibUtilities::PointsKey P0(nEdgePts,
                                         LibUtilities::eGaussLobattoLegendre);
        const LibUtilities::PointsKey P1(nEdgePts,
                                         LibUtilities::eGaussLobattoLegendre);
        const LibUtilities::PointsKey P2(nEdgePts,
                                         LibUtilities::eGaussRadauMAlpha2Beta0);
        const LibUtilities::BasisKey T0(LibUtilities::eOrtho_A, nEdgePts, P0);
        const LibUtilities::BasisKey T1(LibUtilities::eOrtho_A, nEdgePts, P1);
        const LibUtilities::BasisKey T2(LibUtilities::eOrthoPyr_C, nEdgePts,
                                        P2);

        const int nq =
            P0.GetNumPoints() * P1.GetNumPoints() * P2.GetNumPoints();
        Array<OneD, NekDouble> nodal(N);
        Array<OneD, NekDouble> tmp(nq);
        Array<OneD, NekDouble> phys(m_xmap->GetTotPoints());

        for (int i = 0; i < m_coordim; ++i)
        {
            StdRegions::StdNodalPyrExpSharedPtr t =
                MemoryManager<StdRegions::StdNodalPyrExp>::AllocateSharedPtr(
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
}

std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> PyrGeom::v_MakeOrder(
    int order, const LibUtilities::PointsType pType)
{
    int nPoints = order + 1;

    Array<OneD, NekDouble> px, py, pz;
    LibUtilities::PointsKey pKey(nPoints, pType);
    ASSERTL1(pKey.GetPointsDim() == 3, "Points distribution must be 3D");
    LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

    // A nodal pyramid is laid out as five vertices, then the interior of each
    // of the eight edges, then the interior of each of the five faces in
    // faceVerts order -- the quadrilateral base followed by the four
    // triangles -- and finally the interior of the element.
    //
    // The lattice stacks an (nPoints - k) by (nPoints - k) layer at each
    // height k, so the total is the square pyramidal number. Both extents
    // shrink together, unlike the prism, because the pyramid tapers to a
    // point rather than to an edge. Stripping the boundary leaves the same
    // lattice three sizes down.
    const int nPyrPts    = nPoints * (nPoints + 1) * (2 * nPoints + 1) / 6;
    const int nEdgeNodes = nPoints - 2;
    const int nTriNodes  = (nPoints - 2) * (nPoints - 3) / 2;
    const int nQuadNodes = (nPoints - 2) * (nPoints - 2);

    std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> cd;

    cd.first = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
        m_globalID, pType);

    Curve *c = cd.first.get();
    c->m_points.resize(nPyrPts);

    m_curve = c;

    // As for the prism, the nodal distribution numbers the base vertices in
    // raster order, which transposes v2 and v3 with respect to the standard
    // element: nodal slot 2 is at (-1, 1, -1), which is where the standard
    // element puts vertex 3. The apex is unaffected.
    const int vertPerm[kNverts] = {0, 1, 3, 2, 4};
    for (int i = 0; i < kNverts; ++i)
    {
        c->m_points[i] = m_verts[vertPerm[i]];
    }

    // Edge interiors. Measured against eNodalPyrEvenlySpaced: the blocks come
    // in edgeVerts order and each runs along its edge, so as with the prism
    // there is no block to reverse. Note this relies on edges 2 and 3 being
    // stored as 3->2 and 0->3, which is the direction the distribution emits.
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
        {0, 1, 2, 3}, {0, 1, 4, 0}, {1, 2, 4, 0}, {3, 2, 4, 0}, {0, 3, 4, 0}};
    const bool isTri[kNfaces] = {false, true, true, true, true};

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
                // interior, first face direction fastest -- which is the
                // raster order the distribution emits over the base.
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
    if (offset < nPyrPts)
    {
        Array<OneD, Array<OneD, NekDouble>> phys(m_coordim);
        for (int i = 0; i < m_coordim; ++i)
        {
            phys[i] = Array<OneD, NekDouble>(m_xmap->GetTotPoints());
            m_xmap->BwdTrans(GetCoeffs(i), phys[i]);
        }

        for (int i = offset; i < nPyrPts; ++i)
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
