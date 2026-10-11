////////////////////////////////////////////////////////////////////////////////
//
//  File: CADSurfOCE.cpp
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
//  Description: cad object surface methods.
//
////////////////////////////////////////////////////////////////////////////////

#include <BRep_Tool.hxx>
#include <Geom2d_Curve.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <SpatialDomains/CADSystem/OCE/CADCurveOCE.h>
#include <SpatialDomains/CADSystem/OCE/CADSurfOCE.h>
#include <SpatialDomains/CADSystem/OCE/CADUnits.hpp>
#include <SpatialDomains/CADSystem/OCE/TransfiniteSurface.h>
#include <TopoDS.hxx>

using namespace std;

namespace Nektar::SpatialDomains
{

std::string CADSurfOCE::key = GetCADSurfFactory().RegisterCreatorFunction(
    "oce", CADSurfOCE::create, "CADSurfOCE");

void CADSurfOCE::Initialise(int i, TopoDS_Shape in)
{
    m_s = BRep_Tool::Surface(TopoDS::Face(in));

    // Test to see if this surface is a transfinite surface, since some OCC
    // functions will not work on our custom type.
    Handle(Geom_TransfiniteSurface) tf =
        Handle(Geom_TransfiniteSurface)::DownCast(m_s);
    m_isTransfiniteSurf = !tf.IsNull();

    if (in.Orientation() == TopAbs_REVERSED)
    {
        m_orientation = CADOrientation::eBackwards;
    }

    m_id = i;

    BRepTools::UVBounds(TopoDS::Face(in), m_bounds[0], m_bounds[1], m_bounds[2],
                        m_bounds[3]);
    m_sas = new ShapeAnalysis_Surface(m_s);
    m_sas->SetDomain(m_bounds[0], m_bounds[1], m_bounds[2], m_bounds[3]);

    m_shape = in;

    m_2Dclass = new BRepTopAdaptor_FClass2d(TopoDS::Face(m_shape), 1e-4);
}

std::array<NekDouble, 4> CADSurfOCE::GetBounds()
{
    return m_bounds;
}

void CADSurfOCE::GetBounds(NekDouble &umin, NekDouble &umax, NekDouble &vmin,
                           NekDouble &vmax)
{
    umin = m_bounds[0];
    umax = m_bounds[1];
    vmin = m_bounds[2];
    vmax = m_bounds[3];
}

bool CADSurfOCE::IsPlanar()
{
    if (m_sas->Adaptor3d()->GetType() == GeomAbs_Plane)
    {
        return true;
    }

    return false;
}

std::array<NekDouble, 6> CADSurfOCE::BoundingBox()
{
    BRepMesh_IncrementalMesh brmsh(m_shape, 0.005);

    Bnd_Box B;
    BRepBndLib::Add(m_shape, B);
    NekDouble e = sqrt(B.SquareExtent()) * 0.01;
    e           = min(e, 5e-3);
    B.Enlarge(e);

    std::array<NekDouble, 6> ret;
    B.Get(ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]);
    for (auto &r : ret)
    {
        r = CADUnits::FromCAD(r);
    }

    return ret;
}

std::array<NekDouble, 6> CADSurfOCE::BoundingBox(NekDouble scale)
{
    BRepMesh_IncrementalMesh brmsh(m_shape, 0.005 * scale);

    Bnd_Box B;
    BRepBndLib::Add(m_shape, B);
    NekDouble e = sqrt(B.SquareExtent()) * 0.01 * scale;
    e           = min(e, 5e-3 * scale);
    B.Enlarge(e);

    std::array<NekDouble, 6> ret;
    B.Get(ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]);
    for (auto &r : ret)
    {
        r = CADUnits::FromCAD(r);
    }

    return ret;
}

std::array<NekDouble, 2> CADSurfOCE::locuv(std::array<NekDouble, 3> p,
                                           NekDouble &dist)
{
    gp_Pnt loc = CADUnits::PntToCAD(p);
    std::array<NekDouble, 2> uv;

    if (!m_isTransfiniteSurf)
    {
        gp_Pnt2d p2 = m_sas->ValueOfUV(loc, Precision::Confusion());

        TopAbs_State s = m_2Dclass->Perform(p2);

        if (s == TopAbs_OUT)
        {
            BRepBuilderAPI_MakeVertex v(loc);
            BRepExtrema_DistShapeShape dss(
                BRepTools::OuterWire(TopoDS::Face(m_shape)), v.Shape());
            dss.Perform();
            gp_Pnt np = dss.PointOnShape1(1);
            p2        = m_sas->ValueOfUV(np, Precision::Confusion());
        }

        uv[0] = p2.X();
        uv[1] = p2.Y();

        gp_Pnt p3 = m_sas->Value(p2);

        dist = CADUnits::FromCAD(p3.Distance(loc));
    }
    else
    {
        GeomAPI_ProjectPointOnSurf proj(loc, m_s, Precision::Confusion());
        proj.Perform(loc);
        ASSERTL1(proj.NbPoints() > 0, "Unable to find a projection!");
        proj.LowerDistanceParameters(uv[0], uv[1]);
        dist = CADUnits::FromCAD(proj.LowerDistance());
    }

    return uv;
}

std::array<NekDouble, 2> CADSurfOCE::locuv(std::array<NekDouble, 3> p,
                                           NekDouble &dist, NekDouble Umin,
                                           NekDouble Usup, NekDouble Vmin,
                                           NekDouble Vsup)
{
    gp_Pnt loc = CADUnits::PntToCAD(p);
    std::array<NekDouble, 2> uv;

    if (!m_isTransfiniteSurf)
    {
        gp_Pnt2d p2 = m_sas->ValueOfUV(loc, Precision::Confusion());

        TopAbs_State s = m_2Dclass->Perform(p2);

        if (s == TopAbs_OUT)
        {
            BRepBuilderAPI_MakeVertex v(loc);
            BRepExtrema_DistShapeShape dss(
                BRepTools::OuterWire(TopoDS::Face(m_shape)), v.Shape());
            dss.Perform();
            gp_Pnt np = dss.PointOnShape1(1);
            p2        = m_sas->ValueOfUV(np, Precision::Confusion());
        }

        uv[0] = p2.X();
        uv[1] = p2.Y();

        gp_Pnt p3 = m_sas->Value(p2);

        dist = CADUnits::FromCAD(p3.Distance(loc));
    }
    else
    {
        GeomAPI_ProjectPointOnSurf proj(loc, m_s, Umin, Usup, Vmin, Vsup,
                                        Precision::Confusion(),
                                        Extrema_ExtAlgo_Grad);
        proj.Perform(loc);
        ASSERTL1(proj.NbPoints() > 0, "Unable to find a projection!");
        proj.LowerDistanceParameters(uv[0], uv[1]);
        dist = CADUnits::FromCAD(proj.LowerDistance());
    }

    return uv;
}

NekDouble CADSurfOCE::Curvature(std::array<NekDouble, 2> uv)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    GeomLProp_SLProps d(m_s, 2, Precision::Confusion());
    d.SetParameters(uv[0], uv[1]);

    if (!d.IsCurvatureDefined())
    {
        return -1.0;
    }

    NekDouble d1 = abs(d.MaxCurvature());
    NekDouble d2 = abs(d.MinCurvature());

    return CADUnits::CurvatureFromCAD(d1 > d2 ? d1 : d2);
}

NekDouble CADSurfOCE::CurvatureAndPosition(std::array<NekDouble, 2> uv,
                                           std::array<NekDouble, 3> &loc)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    // One evaluation for both: SLProps computes the surface's position along
    // with the derivatives the curvature comes from, so asking it for the
    // point costs nothing beyond the call already made.
    GeomLProp_SLProps d(m_s, 2, Precision::Confusion());
    d.SetParameters(uv[0], uv[1]);

    loc = CADUnits::PntFromCAD(d.Value());

    if (!d.IsCurvatureDefined())
    {
        return -1.0;
    }

    NekDouble d1 = abs(d.MaxCurvature());
    NekDouble d2 = abs(d.MinCurvature());

    return CADUnits::CurvatureFromCAD(d1 > d2 ? d1 : d2);
}

std::array<NekDouble, 2> CADSurfOCE::locuvFrom(std::array<NekDouble, 3> p,
                                               std::array<NekDouble, 2> guess,
                                               NekDouble &dist)
{
    // Only the plain projection takes a starting point; a transfinite
    // surface goes the long way round as before.
    if (m_isTransfiniteSurf)
    {
        return locuv(p, dist);
    }

    gp_Pnt loc  = CADUnits::PntToCAD(p);
    gp_Pnt2d p2 = m_sas->NextValueOfUV(gp_Pnt2d(guess[0], guess[1]), loc,
                                       Precision::Confusion());

    TopAbs_State state = m_2Dclass->Perform(p2);

    if (state == TopAbs_OUT)
    {
        // Outside the face's own boundary, which means the guess has led the
        // search somewhere it should not be. Fall back to the cold search,
        // which handles this case by pulling the point onto the outer wire.
        return locuv(p, dist);
    }

    gp_Pnt p3 = m_sas->Value(p2);
    dist      = CADUnits::FromCAD(p3.Distance(loc));

    return {p2.X(), p2.Y()};
}

std::array<NekDouble, 2> CADSurfOCE::locuvOnCurve(
    const CADCurveSharedPtr &curve, NekDouble t)
{
    // A bounding curve is stored twice by OpenCASCADE: once in space, and
    // once as a curve in this surface's parameter plane. Reading the second
    // gives the (u,v) exactly and immediately, where projecting the first
    // back onto the surface is an iterative minimisation costing some
    // hundreds of microseconds. On brakedisc that projection was 22 of the
    // 34 seconds the whole mesh took.
    auto oceCurve = std::dynamic_pointer_cast<CADCurveOCE>(curve);

    if (oceCurve)
    {
        const TopoDS_Edge &edge = oceCurve->GetOccEdge();
        TopoDS_Face face        = TopoDS::Face(m_shape);

        // A seam edge has two parameter-plane curves on this face, one for
        // each side of the seam, and which of them comes back depends on the
        // orientation the edge is asked for rather than on anything known
        // here. Project in that case rather than pick the wrong one.
        if (!BRep_Tool::IsClosed(edge, face))
        {
            Standard_Real first, last;
            Handle(Geom2d_Curve) pcurve =
                BRep_Tool::CurveOnSurface(edge, face, first, last);

            if (!pcurve.IsNull())
            {
                gp_Pnt2d p = pcurve->Value(t);
                return {p.X(), p.Y()};
            }
        }
    }

    return CADSurf::locuv(curve->P(t));
}

std::array<NekDouble, 3> CADSurfOCE::P(std::array<NekDouble, 2> uv)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    return CADUnits::PntFromCAD(m_s->Value(uv[0], uv[1]));
}

void CADSurfOCE::P(std::array<NekDouble, 2> uv, NekDouble &x, NekDouble &y,
                   NekDouble &z)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    auto loc = CADUnits::PntFromCAD(m_s->Value(uv[0], uv[1]));
    x        = loc[0];
    y        = loc[1];
    z        = loc[2];
}

std::array<NekDouble, 3> CADSurfOCE::N(std::array<NekDouble, 2> uv)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    GeomLProp_SLProps d(m_s, 2, Precision::Confusion());
    d.SetParameters(uv[0], uv[1]);

    std::array<NekDouble, 3> normal = {0.0, 0.0, 0.0};

    if (!d.IsNormalDefined())
    {
        return normal;
    }

    gp_Dir n = d.Normal();

    if (m_orientation == CADOrientation::eBackwards)
    {
        n.Reverse();
    }

    normal[0] = n.X();
    normal[1] = n.Y();
    normal[2] = n.Z();

    return normal;
}

std::array<NekDouble, 9> CADSurfOCE::D1(std::array<NekDouble, 2> uv)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    std::array<NekDouble, 9> r;
    gp_Pnt Loc;
    gp_Vec D1U, D1V;
    m_s->D1(uv[0], uv[1], Loc, D1U, D1V);

    auto l  = CADUnits::PntFromCAD(Loc);
    auto du = CADUnits::VecFromCAD(D1U);
    auto dv = CADUnits::VecFromCAD(D1V);

    r = {l[0],  l[1],  l[2],  // x, y, z
         du[0], du[1], du[2], // dx/du, dy/du, dz/du
         dv[0], dv[1], dv[2]};

    return r;
}

std::array<NekDouble, 18> CADSurfOCE::D2(std::array<NekDouble, 2> uv)
{
#if defined(NEKTAR_DEBUG)
    Test(uv);
#endif

    std::array<NekDouble, 18> r;
    gp_Pnt Loc;
    gp_Vec D1U, D1V, D2U, D2V, D2UV;
    m_s->D2(uv[0], uv[1], Loc, D1U, D1V, D2U, D2V, D2UV);

    auto l   = CADUnits::PntFromCAD(Loc);
    auto du  = CADUnits::VecFromCAD(D1U);
    auto dv  = CADUnits::VecFromCAD(D1V);
    auto duu = CADUnits::VecFromCAD(D2U);
    auto dvv = CADUnits::VecFromCAD(D2V);
    auto duv = CADUnits::VecFromCAD(D2UV);

    r = {l[0],   l[1],   l[2],   // x, y, z
         du[0],  du[1],  du[2],  // dx/du, dy/du, dz/du
         dv[0],  dv[1],  dv[2],  // dx/dv, dy/dv, dz/dv
         duu[0], duu[1], duu[2], // d2x/du2, d2y/du2, d2z/du2
         dvv[0], dvv[1], dvv[2], // d2x/dv2, d2y/dv2, d2z/dv2
         duv[0], duv[1], duv[2]};

    return r;
}

void CADSurfOCE::Test(std::array<NekDouble, 2> uv)
{
    stringstream error;

    error << "Point not within parameter plane: " << endl;

    bool passed = true;

    if (uv[0] < m_bounds[0])
    {
        if (fabs(uv[0] - m_bounds[0]) > 1E-6)
        {
            error << "U(" << uv[0] << ") is less than Umin(" << m_bounds[0]
                  << ")";
            passed = false;
        }
    }
    else if (uv[0] > m_bounds[1])
    {
        if (fabs(uv[0] - m_bounds[1]) > 1E-6)
        {
            error << "U(" << uv[0] << ") is greater than Umax(" << m_bounds[1]
                  << ")";
            passed = false;
        }
    }
    else if (uv[1] < m_bounds[2])
    {
        if (fabs(uv[1] - m_bounds[2]) > 1E-6)
        {
            error << "V(" << uv[1] << ") is less than Vmin(" << m_bounds[2]
                  << ")";
            passed = false;
        }
    }
    else if (uv[1] > m_bounds[3])
    {
        if (fabs(uv[1] - m_bounds[3]) > 1E-6)
        {
            error << "V(" << uv[1] << ") is greater than Vmax(" << m_bounds[3]
                  << ")";
            passed = false;
        }
    }

    error << " On Surface: " << GetId();
    WARNINGL1(passed, "Warning: " + error.str());
    (void)passed; // suppress warning
}

} // namespace Nektar::SpatialDomains
