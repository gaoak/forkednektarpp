////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessInsertSurface.cpp
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
//  Description: Calculate Jacobians of elements.
//
////////////////////////////////////////////////////////////////////////////////

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "ProcessInsertSurface.h"
#include <NekMesh/MeshElements/Element.h>
#include <SpatialDomains/Curve.hpp>
#include <boost/core/ignore_unused.hpp>

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey ProcessInsertSurface::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "insertsurface"),
        ProcessInsertSurface::create,
        "Insert high-order surface mesh into current working mesh.");

ProcessInsertSurface::ProcessInsertSurface(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["mesh"] = ConfigOption(false, "", "Mesh to be inserted.");
    m_config["nonconforming"] =
        ConfigOption(false, "", "Relax tests for nonconforming boundries");
}

ProcessInsertSurface::~ProcessInsertSurface()
{
}

void ProcessInsertSurface::Process()
{
    typedef bg::model::point<NekDouble, 3, bg::cs::cartesian> Point;
    typedef pair<Point, unsigned int> PointI;

    string file     = m_config["mesh"].as<string>();
    bool nonconform = m_config["nonconforming"].beenSet;

    m_log(VERBOSE) << "Inserting surface from file '" << file << "'." << endl;

    MeshSharedPtr inMsh = std::shared_ptr<Mesh>(new Mesh());
    ModuleSharedPtr mod = GetModuleFactory().CreateInstance(
        ModuleKey(eInputModule, "xml"), inMsh);
    mod->RegisterConfig("infile", file);
    mod->Process();

    // build ann tree of surface verticies from inMsh
    // match surface vertices in ccm mesh to inMsh and copy information

    // tolerance of matching vertices
    NekDouble tol = 1e-5;

    auto surfaceVerts = [](MeshSharedPtr &m) {
        std::unordered_set<SpatialDomains::PointGeom *> verts;
        for (auto &[geom, tag] : m->m_elementTags[2])
        {
            boost::ignore_unused(tag);
            for (int j = 0; j < geom->GetNumVerts(); ++j)
            {
                verts.insert(geom->GetVertex(j));
            }
        }
        return verts;
    };

    auto inSurfVerts = surfaceVerts(inMsh);
    vector<SpatialDomains::PointGeom *> inMshnodeList(inSurfVerts.begin(),
                                                      inSurfVerts.end());

    vector<PointI> dataPts;
    for (int i = 0; i < inMshnodeList.size(); i++)
    {
        dataPts.push_back(
            make_pair(Point((*inMshnodeList[i])[0], (*inMshnodeList[i])[1],
                            (*inMshnodeList[i])[2]),
                      i));
    }

    // Build tree
    bgi::rtree<PointI, bgi::rstar<16>> rtree;
    rtree.insert(dataPts.begin(), dataPts.end());

    if (!nonconform && surfaceVerts(m_mesh).size() != inMshnodeList.size())
    {
        m_log(FATAL) << "Surface mesh node count mismatch, inserting surface "
                     << "will not work" << endl;
    }

    // The boundary elements are the surface geometries themselves, so what
    // used to be reached through Element::GetFaceLink() is the tagged entity.
    std::unordered_set<SpatialDomains::SegGeom *> surfEdges;
    for (auto &[face, tag] : m_mesh->m_elementTags[2])
    {
        boost::ignore_unused(tag);
        for (int j = 0; j < face->GetNumEdges(); ++j)
        {
            surfEdges.insert(
                static_cast<SpatialDomains::SegGeom *>(face->GetEdge(j)));
        }
    }

    // Locate a working-mesh vertex among the inserted mesh's surface
    // vertices, or return nullptr if nothing lies within tolerance.
    auto locate =
        [&](SpatialDomains::PointGeom *v) -> SpatialDomains::PointGeom * {
        Point queryPt((*v)[0], (*v)[1], (*v)[2]);
        vector<PointI> result;
        rtree.query(bgi::nearest(queryPt, 1), std::back_inserter(result));

        if (bg::distance(result[0].first, queryPt) > tol)
        {
            if (!nonconform)
            {
                m_log(FATAL)
                    << "Cannot locate point accurately enough." << endl;
            }
            return nullptr;
        }

        return inMshnodeList[result[0].second];
    };

    for (auto &edge : surfEdges)
    {
        SpatialDomains::PointGeom *inN1 = locate(edge->GetVertex(0));
        SpatialDomains::PointGeom *inN2 = locate(edge->GetVertex(1));

        if (inN1 == nullptr || inN2 == nullptr)
        {
            continue;
        }

        auto f = inMsh->m_edgeSet.find(
            std::make_pair(inN1->GetGlobalID(), inN2->GetGlobalID()));

        if (f == inMsh->m_edgeSet.end())
        {
            m_log(FATAL) << "Could not find edge in input" << endl;
        }

        SpatialDomains::Curve *inCurve = f->second->GetCurve();
        if (inCurve == nullptr || inCurve->m_points.size() < 3)
        {
            continue;
        }

        // The interior of the inserted curve, in the direction of this edge.
        // The inserted mesh is a local object whose graph owns its curvature
        // nodes, so the coordinates have to be copied into nodes owned by
        // this mesh rather than the pointers shared.
        const bool reversed =
            f->second->GetVertex(0)->dist(*edge->GetVertex(0)) > tol;

        auto curve = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
            edge->GetGlobalID(), inCurve->m_ptype);

        const int nInterior = inCurve->m_points.size() - 2;
        curve->m_points.push_back(edge->GetVertex(0));
        for (int k = 0; k < nInterior; ++k)
        {
            SpatialDomains::PointGeom *src =
                inCurve->m_points[reversed ? nInterior - k : k + 1];

            auto pt =
                ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                    edge->GetVertex(0)->GetCoordim(), 0, (*src)[0], (*src)[1],
                    (*src)[2]);
            curve->m_points.push_back(pt.get());
            m_mesh->m_meshGraph->GetAllCurveNodes().push_back(std::move(pt));
        }
        curve->m_points.push_back(edge->GetVertex(1));

        edge->SetCurve(curve.get());
        m_mesh->m_meshGraph->AddCurvedEdge(std::move(curve));
    }
}
} // namespace Nektar::NekMesh
