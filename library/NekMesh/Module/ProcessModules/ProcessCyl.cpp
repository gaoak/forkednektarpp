////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessCyl.cpp
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
//  Description: create cylinder curved edges
//
////////////////////////////////////////////////////////////////////////////////

#include <LocalRegions/NodalTriExp.h>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/SegExp.h>
#include <LocalRegions/TriExp.h>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

#include <NekMesh/MeshElements/Element.h>

#include "ProcessCyl.h"

#include <SpatialDomains/Curve.hpp>

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey ProcessCyl::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "cyl"), ProcessCyl::create);

/**
 * @brief Default constructor.
 */
ProcessCyl::ProcessCyl(MeshSharedPtr m) : ProcessCurvedEdges(m)
{
    m_config["r"]  = ConfigOption(false, "0.0", "Radius of cylinder.");
    m_config["xc"] = ConfigOption(false, "0.0", "Radius of cylinder.");
    m_config["yc"] = ConfigOption(false, "0.0", "Radius of cylinder.");
}

/**
 * @brief Destructor.
 */
ProcessCyl::~ProcessCyl()
{
}

void ProcessCyl::v_GenerateEdgeNodes(SpatialDomains::SegGeom *edge)
{
    SpatialDomains::PointGeom *n1 = edge->GetVertex(0);
    SpatialDomains::PointGeom *n2 = edge->GetVertex(1);

    int nq    = m_config["N"].as<int>();
    double r  = m_config["r"].as<double>();
    double xc = m_config["xc"].as<double>();
    double yc = m_config["yc"].as<double>();
    double t1 = atan2((*n1)[1] - yc, (*n1)[0] - xc);
    double t2 = atan2((*n2)[1] - yc, (*n2)[0] - xc);
    double dt;
    double dz;

    if (t1 < -M_PI / 2.0 && t2 > 0.0)
    {
        t1 += 2 * M_PI;
    }
    if (t2 < -M_PI / 2.0 && t1 > 0.0)
    {
        t2 += 2 * M_PI;
    }

    dt = (t2 - t1) / (nq - 1);
    dz = ((*n2)[2] - (*n1)[2]) / (nq - 1);

    // A curve runs end to end, so the two vertices bracket the interior
    // points that are generated here.
    auto curve = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
        edge->GetGlobalID(), LibUtilities::ePolyEvenlySpaced);

    curve->m_points.push_back(n1);
    for (int i = 1; i < nq - 1; ++i)
    {
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            n1->GetCoordim(), 0, xc + r * cos(t1 + i * dt),
            yc + r * sin(t1 + i * dt), (*n1)[2] + i * dz);
        curve->m_points.push_back(pt.get());
        m_mesh->m_meshGraph->GetAllCurveNodes().push_back(std::move(pt));
    }
    curve->m_points.push_back(n2);

    edge->SetCurve(curve.get());
    m_mesh->m_meshGraph->AddCurvedEdge(std::move(curve));
}
} // namespace Nektar::NekMesh
