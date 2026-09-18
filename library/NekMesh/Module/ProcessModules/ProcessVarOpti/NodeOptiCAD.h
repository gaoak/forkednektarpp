////////////////////////////////////////////////////////////////////////////////
//
//  File: NodeOptiCAD.h
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

#ifndef UTILITIES_NEKMESH_NODEOPTICAD
#define UTILITIES_NEKMESH_NODEOPTICAD

#include "NodeOpti.h"

#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <SpatialDomains/CADSystem/CADCurve.h>

namespace Nektar::NekMesh
{
using SpatialDomains::CADCurve;
using SpatialDomains::CADCurveSharedPtr;
using SpatialDomains::CADSurf;
using SpatialDomains::CADSurfSharedPtr;

class NodeOpti1D3D : public NodeOpti // 1D optimsation in 3D space
{
public:
    NodeOpti1D3D(SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
                 ResidualSharedPtr r,
                 std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d,
                 optiType o, SpatialDomains::MeshGraph *g, CADCurveSharedPtr c)
        : NodeOpti(n, e, r, d, o, 3, g), curve(c)
    {
    }

    ~NodeOpti1D3D() override {};

    void Optimise() override;

    static int m_type;
    static NodeOptiSharedPtr create(
        SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
        ResidualSharedPtr r,
        std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d, optiType o,
        SpatialDomains::MeshGraph *g)
    {
        return NodeOptiSharedPtr(new NodeOpti1D3D(
            n, e, r, d, o, g, g->GetCADAssociation()->GetCurve(n)));
    }

private:
    void ProcessGradient();
    CADCurveSharedPtr curve;
};

class NodeOpti2D3D : public NodeOpti // 1D optimsation in 3D space
{
public:
    NodeOpti2D3D(SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
                 ResidualSharedPtr r,
                 std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d,
                 optiType o, SpatialDomains::MeshGraph *g, CADSurfSharedPtr s)
        : NodeOpti(n, e, r, d, o, 3, g), surf(s)
    {
    }

    ~NodeOpti2D3D() override {};

    void Optimise() override;

    static int m_type;
    static NodeOptiSharedPtr create(
        SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
        ResidualSharedPtr r,
        std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d, optiType o,
        SpatialDomains::MeshGraph *g)
    {
        return NodeOptiSharedPtr(new NodeOpti2D3D(
            n, e, r, d, o, g, g->GetCADAssociation()->GetSurf(n)));
    }

private:
    void ProcessGradient();
    CADSurfSharedPtr surf;
};

class NodeOpti1D2D : public NodeOpti // 1D optimsation in 2D space
{
public:
    NodeOpti1D2D(SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
                 ResidualSharedPtr r,
                 std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d,
                 optiType o, SpatialDomains::MeshGraph *g, CADCurveSharedPtr c)
        : NodeOpti(n, e, r, d, o, 2, g), curve(c)
    {
        m_bd = curve->GetBounds();
    }

    ~NodeOpti1D2D() override {};

    void Optimise() override;

    static int m_type;
    static NodeOptiSharedPtr create(
        SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
        ResidualSharedPtr r,
        std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d, optiType o,
        SpatialDomains::MeshGraph *g)
    {
        return NodeOptiSharedPtr(new NodeOpti1D2D(
            n, e, r, d, o, g, g->GetCADAssociation()->GetCurve(n)));
    }

private:
    void ProcessGradient();
    CADCurveSharedPtr curve;
    std::array<NekDouble, 2> m_bd;
};
} // namespace Nektar::NekMesh

#endif
