////////////////////////////////////////////////////////////////////////////////
//
//  File: CADCurveOCE.cpp
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
//  Description: CAD object curve methods.
//
////////////////////////////////////////////////////////////////////////////////

#include <SpatialDomains/CADSystem/OCE/CADCurveOCE.h>
#include <SpatialDomains/CADSystem/OCE/CADUnits.hpp>
#include <array>

using namespace std;

namespace Nektar::SpatialDomains
{

std::string CADCurveOCE::key = GetCADCurveFactory().RegisterCreatorFunction(
    "oce", CADCurveOCE::create, "CADCurveOCE");

void CADCurveOCE::Initialise(int i, TopoDS_Shape in)
{
    m_occEdge = TopoDS::Edge(in);

    GProp_GProps System;
    BRepGProp::LinearProperties(m_occEdge, System);
    m_length = CADUnits::FromCAD(System.Mass());

    m_c  = BRep_Tool::Curve(TopoDS::Edge(in), m_b[0], m_b[1]);
    m_id = i;
}

NekDouble CADCurveOCE::tAtArcLength(NekDouble s)
{
    GeomAdaptor_Curve c(m_c);
    GCPnts_AbscissaPoint ap(c, CADUnits::ToCAD(s), m_b[0]);
    return ap.Parameter();
}

NekDouble CADCurveOCE::Length(NekDouble ti, NekDouble tf)
{
    Handle(Geom_Curve) NewCurve = new Geom_TrimmedCurve(m_c, ti, tf);
    TopoDS_Edge NewEdge         = BRepBuilderAPI_MakeEdge(NewCurve);
    GProp_GProps System;
    BRepGProp::LinearProperties(NewEdge, System);
    return CADUnits::FromCAD(System.Mass());
}

NekDouble CADCurveOCE::GetMinDistance(std::array<NekDouble, 3> &xyz)
{
    gp_Pnt loc = CADUnits::PntToCAD(xyz);
    GeomAPI_ProjectPointOnCurve proj(loc, m_c, m_b[0], m_b[1]);
    if (proj.NbPoints())
    {
        return CADUnits::FromCAD(proj.LowerDistance());
    }
    else
    {
        return CADUnits::FromCAD(std::min(loc.Distance(m_c->Value(m_b[0])),
                                          loc.Distance(m_c->Value(m_b[1]))));
    }
}

NekDouble CADCurveOCE::loct(std::array<NekDouble, 3> xyz, NekDouble &t)
{
    t          = 0.0;
    gp_Pnt loc = CADUnits::PntToCAD(xyz);
    ShapeAnalysis_Curve sac;
    gp_Pnt p;
    sac.Project(m_c, loc, Precision::Confusion(), p, t);
    return CADUnits::FromCAD(p.Distance(loc));
}

// This function updates the parametric coordinate t of a cartesian point xyz
// within the bounds of the CADCurve cf and cl ; Fixes the bug with respect to
// Periodic CADCurves like Circles
NekDouble CADCurveOCE::loct(std::array<NekDouble, 3> xyz, NekDouble &t,
                            NekDouble cf, NekDouble cl)
{
    t = 0.0;

    gp_Pnt loc = CADUnits::PntToCAD(xyz);

    ShapeAnalysis_Curve sac;
    gp_Pnt p;
    sac.Project(m_c, loc, Precision::Confusion(), p, t, cf, cl);

    return CADUnits::FromCAD(p.Distance(loc));
}

std::array<NekDouble, 3> CADCurveOCE::P(NekDouble t)
{
    return CADUnits::PntFromCAD(m_c->Value(t));
}

void CADCurveOCE::P(NekDouble t, NekDouble &x, NekDouble &y, NekDouble &z)
{
    auto loc = CADUnits::PntFromCAD(m_c->Value(t));
    x        = loc[0];
    y        = loc[1];
    z        = loc[2];
}

std::array<NekDouble, 9> CADCurveOCE::D2(NekDouble t)
{
    gp_Pnt loc;
    gp_Vec d1, d2;
    m_c->D2(t, loc, d1, d2);

    auto l = CADUnits::PntFromCAD(loc);
    auto a = CADUnits::VecFromCAD(d1);
    auto b = CADUnits::VecFromCAD(d2);

    return {l[0], l[1], l[2], a[0], a[1], a[2], b[0], b[1], b[2]};
}

std::array<NekDouble, 3> CADCurveOCE::N(NekDouble t)
{
    GeomLProp_CLProps d(m_c, 2, Precision::Confusion());
    d.SetParameter(t + 1e-8);

    gp_Vec d2 = d.D2();
    if (d2.Magnitude() < 1e-8)
    {
        // no normal, stright line
        return {0.0, 0.0, 0.0};
    }

    gp_Dir n;
    d.Normal(n);

    return {n.X(), n.Y(), n.Z()};
}

NekDouble CADCurveOCE::Curvature(NekDouble t)
{
    GeomLProp_CLProps d(m_c, 2, Precision::Confusion());
    d.SetParameter(t);

    return CADUnits::CurvatureFromCAD(d.Curvature());
}

std::array<NekDouble, 2> CADCurveOCE::GetBounds()
{
    return m_b;
}

void CADCurveOCE::GetBounds(NekDouble &tmin, NekDouble &tmax)
{
    tmin = m_b[0];
    tmax = m_b[1];
}

std::array<NekDouble, 6> CADCurveOCE::GetMinMax()
{
    gp_Pnt start =
        BRep_Tool::Pnt(TopExp::FirstVertex(m_occEdge, Standard_True));
    gp_Pnt end = BRep_Tool::Pnt(TopExp::LastVertex(m_occEdge, Standard_True));

    auto s0 = CADUnits::PntFromCAD(start);
    auto s1 = CADUnits::PntFromCAD(end);

    return {s0[0], s0[1], s0[2], s1[0], s1[1], s1[2]};
}

std::array<NekDouble, 6> CADCurveOCE::BoundingBox(NekDouble scale)
{
    BRepMesh_IncrementalMesh brmsh(m_occEdge, 0.005 * scale);

    Bnd_Box B;
    BRepBndLib::Add(m_occEdge, B);
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

} // namespace Nektar::SpatialDomains
