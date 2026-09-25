////////////////////////////////////////////////////////////////////////////////
//
//  File: SegGeom.cpp
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
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/XmapFactory.hpp>

#include <LibUtilities/Foundations/ManagerAccess.h> // for PointsManager, etc
#include <StdRegions/StdRegions.hpp>
#include <StdRegions/StdSegExp.h>

namespace Nektar::SpatialDomains
{

XmapFactory<StdRegions::StdSegExp, 1> &GetStdSegFactory()
{
    static XmapFactory<StdRegions::StdSegExp, 1> factory;
    return factory;
}
SegGeom::SegGeom()
{
    m_shapeType = LibUtilities::eSegment;
}

SegGeom::SegGeom(int id, int coordim, std::array<PointGeom *, kNverts> vertex,
                 Curve *curve)
    : Geometry1D(coordim)
{
    m_shapeType = LibUtilities::eSegment;
    m_globalID  = id;
    m_state     = eNotFilled;
    m_curve     = curve;
    m_verts     = vertex;
}

SegGeom::SegGeom(const SegGeom &in) : Geometry1D(in)
{
    // From Geometry class
    m_shapeType = in.m_shapeType;

    // info from EdgeComponent class
    m_globalID = in.m_globalID;
    m_xmap     = in.m_xmap;
    SetUpCoeffs(m_xmap->GetNcoeffs());

    // info from SegGeom class
    m_coordim  = in.m_coordim;
    m_verts[0] = in.m_verts[0];
    m_verts[1] = in.m_verts[1];

    m_state = in.m_state;
}

void SegGeom::SetUpXmap()
{
    if (m_curve)
    {
        int npts = m_curve->m_points.size();
        LibUtilities::PointsKey pkey(npts + 1,
                                     LibUtilities::eGaussLobattoLegendre);
        const LibUtilities::BasisKey B(LibUtilities::eModified_A, npts, pkey);
        m_xmap = MemoryManager<StdRegions::StdSegExp>::AllocateSharedPtr(B);
    }
    else
    {
        const LibUtilities::BasisKey B(
            LibUtilities::eModified_A, 2,
            LibUtilities::PointsKey(2, LibUtilities::eGaussLobattoLegendre));
        m_xmap = MemoryManager<StdRegions::StdSegExp>::AllocateSharedPtr(B);
    }
}

/**
 * \brief Generate a one dimensional space segment geometry where the vert[0]
 * has the same x value and vert[1] is set to vert[0] plus the length of the
 * original segment
 **/
SegGeomUniquePtr SegGeom::GenerateOneSpaceDimGeom(EntityHolder1D &holder)
{
    SegGeomUniquePtr returnval = ObjPoolManager<SegGeom>::AllocateUniquePtr();

    // info about numbering
    returnval->m_globalID = m_globalID;

    // geometric information.
    returnval->m_coordim     = 1;
    NekDouble x0             = (*m_verts[0])[0];
    PointGeomUniquePtr vert0 = ObjPoolManager<PointGeom>::AllocateUniquePtr(
        1, m_verts[0]->GetGlobalID(), x0, 0.0, 0.0);
    vert0->SetGlobalID(vert0->GetGlobalID());
    returnval->m_verts[0] = vert0.get();
    holder.m_pointVec.push_back(std::move(vert0));

    // Get information to calculate length.
    const Array<OneD, const LibUtilities::BasisSharedPtr> base =
        m_xmap->GetBase();
    LibUtilities::PointsKeyVector v;
    v.push_back(base[0]->GetPointsKey());

    const Array<OneD, const NekDouble> jac = v_GenGeomFactors(v)->GetJac();

    NekDouble len = 0.0;
    if (jac.size() == 1)
    {
        len = jac[0] * 2.0;
    }
    else
    {
        Array<OneD, const NekDouble> w0 = base[0]->GetW();
        len                             = 0.0;

        for (int i = 0; i < jac.size(); ++i)
        {
            len += jac[i] * w0[i];
        }
    }
    // Set up second vertex.
    PointGeomUniquePtr vert1 = ObjPoolManager<PointGeom>::AllocateUniquePtr(
        1, m_verts[1]->GetGlobalID(), x0 + len, 0.0, 0.0);
    vert1->SetGlobalID(vert1->GetGlobalID());

    returnval->m_verts[1] = vert1.get();
    holder.m_pointVec.push_back(std::move(vert1));

    // at present just use previous m_xmap[0];
    returnval->m_xmap = m_xmap;
    returnval->SetUpCoeffs(m_xmap->GetNcoeffs());
    returnval->m_state = eNotFilled;

    return returnval;
}

LibUtilities::ShapeType SegGeom::v_GetShapeType() const
{
    return LibUtilities::eSegment;
}

NekDouble SegGeom::v_GetCoord(const int i,
                              const Array<OneD, const NekDouble> &Lcoord)
{
    if (m_state != ePtsFilled)
    {
        NEKERROR(ErrorUtil::ewarning, "Geometry is not in physical space");
    }

    Array<OneD, NekDouble> tmp(m_xmap->GetTotPoints());
    m_xmap->BwdTrans(m_coeffs[i], tmp);

    return m_xmap->PhysEvaluate(Lcoord, tmp);
}

/**
 * @brief Get the orientation of @p edge1.
 *
 * If @p edge1 is connected to @p edge2 in the same direction as the points
 * comprising @p edge1 then it is forward, otherwise it is backward.
 *
 * For example, assume @p edge1 is comprised of points 1 and 2, and @p edge2 is
 * comprised of points 2 and 3, then @p edge1 is forward.
 *
 * If @p edge1 is comprised of points 2 and 1 and @p edge2 is comprised of
 * points 3 and 2, then @p edge1 is backward.
 *
 * Since both edges are passed, it does not need any information from the
 * EdgeComponent instance.
 */
StdRegions::Orientation SegGeom::GetEdgeOrientation(const SegGeom &edge1,
                                                    const SegGeom &edge2)
{
    StdRegions::Orientation returnval = StdRegions::eForwards;

    if ((*edge1.GetVertex(0) == *edge2.GetVertex(0)) ||
        (*edge1.GetVertex(0) == *edge2.GetVertex(1)))
    {
        // Backward direction.  Vertex 0 is connected to edge 2.
        returnval = StdRegions::eBackwards;
    }
    else if ((*edge1.GetVertex(1) != *edge2.GetVertex(0)) &&
             (*edge1.GetVertex(1) != *edge2.GetVertex(1)))
    {
        // Not forward either, then we have a problem.
        std::ostringstream errstrm;
        errstrm << "Connected edges do not share a vertex. Edges ";
        errstrm << edge1.GetGlobalID() << ", " << edge2.GetGlobalID();
        NEKERROR(ErrorUtil::efatal, errstrm.str());
    }

    return returnval;
}

GeomType SegGeom::v_CalcGeomType([[maybe_unused]] IsoParam &iso)
{
    if (!m_setupState)
    {
        SegGeom::v_Setup();
    }
    SegGeom::v_FillGeom();

    SpatialDomains::GeomType gType = eRegular;

    if (m_xmap->GetBasisNumModes(0) != 2)
    {
        gType = eDeformed;
    }

    return gType;
}

GeomFactorsUniquePtr SegGeom::v_GenGeomFactors(
    LibUtilities::PointsKeyVector &keyTgt)
{
    GeomType Gtype = CalcGeomType();
    return ObjPoolManager<GeomFactors>::AllocateUniquePtr(
        Gtype, m_coordim, m_xmap, m_coeffs, keyTgt);
}

void SegGeom::v_FillGeom()
{
    if (m_state != ePtsFilled)
    {
        int i;

        if (m_coordim > 0 && m_curve)
        {
            int npts = m_curve->m_points.size();
            LibUtilities::PointsKey pkey(npts + 1,
                                         LibUtilities::eGaussLobattoLegendre);
            Array<OneD, NekDouble> tmp(npts);

            if (m_verts[0]->dist(*(m_curve->m_points[0])) >
                NekConstants::kVertexTheSameDouble)
            {
                std::string err =
                    "Vertex 0 is separated from first point by more than ";
                std::stringstream strstrm;
                strstrm << NekConstants::kVertexTheSameDouble << " in edge "
                        << m_globalID;
                err += strstrm.str();
                NEKERROR(ErrorUtil::ewarning, err.c_str());
            }

            if (m_verts[1]->dist(*(m_curve->m_points[npts - 1])) >
                NekConstants::kVertexTheSameDouble)
            {
                std::string err =
                    "Vertex 1 is separated from last point by more than ";
                std::stringstream strstrm;
                strstrm << NekConstants::kVertexTheSameDouble << " in edge "
                        << m_globalID;
                err += strstrm.str();
                NEKERROR(ErrorUtil::ewarning, err.c_str());
            }

            LibUtilities::PointsKey fkey(npts, m_curve->m_ptype);
            DNekMatSharedPtr I0 =
                LibUtilities::PointsManager()[fkey]->GetI(pkey);
            NekVector<NekDouble> out(npts + 1);

            for (int i = 0; i < m_coordim; ++i)
            {
                // Load up coordinate values into tmp
                for (int j = 0; j < npts; ++j)
                {
                    tmp[j] = (m_curve->m_points[j]->GetPtr())[i];
                }

                // Interpolate to GLL points
                NekVector<NekDouble> in(npts, tmp, eWrapper);
                out = (*I0) * in;

                m_xmap->FwdTrans(out.GetPtr(), m_coeffs[i]);
            }
        }

        for (i = 0; i < m_coordim; ++i)
        {
            m_coeffs[i][0] = (*m_verts[0])[i];
            m_coeffs[i][1] = (*m_verts[1])[i];
        }

        m_state = ePtsFilled;
    }
}

std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> SegGeom::v_MakeOrder(
    int order, const LibUtilities::PointsType pType)
{
    int nPoints = order + 1;

    Array<OneD, NekDouble> edgePoints;
    LibUtilities::PointsKey edgeKey(nPoints, pType);
    LibUtilities::PointsManager()[edgeKey]->GetPoints(edgePoints);

    Array<OneD, Array<OneD, NekDouble>> phys(m_coordim);

    for (int i = 0; i < m_coordim; ++i)
    {
        phys[i] = Array<OneD, NekDouble>(m_xmap->GetTotPoints());
        m_xmap->BwdTrans(GetCoeffs(i), phys[i]);
    }

    std::pair<CurveUniquePtr, std::vector<PointGeomUniquePtr>> cd;

    cd.first = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
        m_globalID, pType);

    Curve *c = cd.first.get();
    m_curve  = c;

    c->m_points.resize(nPoints);

    // Assuming that pType includes vertices in the curve points
    c->m_points[0] = m_verts[0];
    for (int i = 1; i < nPoints - 1; ++i)
    {
        Array<OneD, NekDouble> x(3, 0.0);
        for (int j = 0; j < m_coordim; ++j)
        {
            x[j] = m_xmap->PhysEvaluate(edgePoints + i, phys[j]);
        }

        cd.second.push_back(
            ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                m_coordim, 0, x[0], x[1], x[2]));
        c->m_points[i] = cd.second.back().get();
    }
    c->m_points[nPoints - 1] = m_verts[1];

    return cd;
}

void SegGeom::v_Reset(CurveMap &curvedEdges, CurveMap &curvedFaces)
{
    Geometry::v_Reset(curvedEdges, curvedFaces);
    CurveMap::iterator it = curvedEdges.find(m_globalID);

    if (it != curvedEdges.end())
    {
        m_curve = it->second.get();
    }
    else
    {
        m_curve = nullptr;
    }

    SetUpXmap();
    SetUpCoeffs(m_xmap->GetNcoeffs());
}

void SegGeom::v_Setup()
{
    if (!m_setupState)
    {
        SetUpXmap();
        SetUpCoeffs(m_xmap->GetNcoeffs());
        m_setupState = true;
    }
}

PointGeom *SegGeom::v_GetVertex(const int i) const
{
    PointGeom *returnval = nullptr;

    if (i >= 0 && i < kNverts)
    {
        returnval = m_verts[i];
    }

    return returnval;
}

int SegGeom::v_GetNumVerts() const
{
    return kNverts;
}

} // namespace Nektar::SpatialDomains
