////////////////////////////////////////////////////////////////////////////////
//
//  File: BLMesh.cpp
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
//  Description: BL meshing methods
//
////////////////////////////////////////////////////////////////////////////////

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <NekMesh/VolumeMeshing/BLMeshing/BLMesh.h>
#include <SpatialDomains/CADSystem/CADSurf.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/NodalUtil.h>

#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <algorithm>

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<double, 3, bg::cs::cartesian> point;
typedef bg::model::box<point> box;
typedef std::pair<box, unsigned int> boxI;

using namespace std;
namespace Nektar::NekMesh
{

inline vector<SpatialDomains::PointGeom *> GetVerts(
    SpatialDomains::Geometry *el)
{
    vector<SpatialDomains::PointGeom *> ns(el->GetNumVerts());
    for (int i = 0; i < el->GetNumVerts(); i++)
    {
        ns[i] = el->GetVertex(i);
    }
    return ns;
}

/**
 * @brief Unit normal of a boundary triangle, pointing into the domain.
 *
 * This is the convention the old Element::Normal(true) provided. The raw
 * cross product of the first two edges points out of the surface only when the
 * surface is traversed forwards, so it is reversed when the CAD surface is
 * itself reversed, and then reversed again to point into the domain, which is
 * the direction a boundary layer grows in.
 *
 * The CAD surface therefore has to come in with the triangle: a node shared by
 * two surfaces of opposite orientation would otherwise average two normals
 * pointing opposite ways.
 */
inline std::array<NekDouble, 3> Normal(
    SpatialDomains::Geometry *el, const SpatialDomains::CADSurfSharedPtr &surf)
{
    vector<SpatialDomains::PointGeom *> ns = GetVerts(el);

    std::array<NekDouble, 3> e0 = {ns[1]->x() - ns[0]->x(),
                                   ns[1]->y() - ns[0]->y(),
                                   ns[1]->z() - ns[0]->z()};
    std::array<NekDouble, 3> e1 = {ns[2]->x() - ns[0]->x(),
                                   ns[2]->y() - ns[0]->y(),
                                   ns[2]->z() - ns[0]->z()};

    std::array<NekDouble, 3> N = {e0[1] * e1[2] - e0[2] * e1[1],
                                  e0[2] * e1[0] - e0[0] * e1[2],
                                  e0[0] * e1[1] - e0[1] * e1[0]};

    // Reversed surface and inward normal are two sign flips, so they cancel;
    // a forwards surface keeps the single flip that makes the normal inward.
    NekDouble mag =
        sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]) *
        (surf->Orientation() == SpatialDomains::CADOrientation::eBackwards
             ? 1.0
             : -1.0);
    N[0] /= mag;
    N[1] /= mag;
    N[2] /= mag;

    return N;
}

inline box GetBox(SpatialDomains::Geometry *el, NekDouble ov)
{
    NekDouble xmin = numeric_limits<double>::max(),
              xmax = -1.0 * numeric_limits<double>::max(),
              ymin = numeric_limits<double>::max(),
              ymax = -1.0 * numeric_limits<double>::max(),
              zmin = numeric_limits<double>::max(),
              zmax = -1.0 * numeric_limits<double>::max();

    vector<SpatialDomains::PointGeom *> ns = GetVerts(el);
    for (int i = 0; i < ns.size(); i++)
    {
        xmin = min(xmin, ns[i]->x());
        xmax = max(xmax, ns[i]->x());
        ymin = min(ymin, ns[i]->y());
        ymax = max(ymax, ns[i]->y());
        zmin = min(zmin, ns[i]->z());
        zmax = max(zmax, ns[i]->z());
    }

    return box(point(xmin - ov, ymin - ov, zmin - ov),
               point(xmax + ov, ymax + ov, zmax + ov));
}

inline box GetBox(vector<SpatialDomains::Geometry *> els, NekDouble ov)
{
    NekDouble xmin = numeric_limits<double>::max(),
              xmax = -1.0 * numeric_limits<double>::max(),
              ymin = numeric_limits<double>::max(),
              ymax = -1.0 * numeric_limits<double>::max(),
              zmin = numeric_limits<double>::max(),
              zmax = -1.0 * numeric_limits<double>::max();

    for (int j = 0; j < els.size(); j++)
    {
        vector<SpatialDomains::PointGeom *> ns = GetVerts(els[j]);
        for (int i = 0; i < ns.size(); i++)
        {
            xmin = min(xmin, ns[i]->x());
            xmax = max(xmax, ns[i]->x());
            ymin = min(ymin, ns[i]->y());
            ymax = max(ymax, ns[i]->y());
            zmin = min(zmin, ns[i]->z());
            zmax = max(zmax, ns[i]->z());
        }
    }

    return box(point(xmin - ov, ymin - ov, zmin - ov),
               point(xmax + ov, ymax + ov, zmax + ov));
}

inline box GetBox(SpatialDomains::PointGeom *n, NekDouble ov)
{
    return box(point(n->x() - ov, n->y() - ov, n->z() - ov),
               point(n->x() + ov, n->y() + ov, n->z() + ov));
}

void BLMesh::Mesh()
{
    Setup();

    BuildElements();

    GrowLayers();

    Shrink();

    for (auto bit = m_blData.begin(); bit != m_blData.end(); bit++)
    {
        vector<blInfoSharedPtr> infos = m_nToNInfo[bit->first];
        for (int i = 0; i < infos.size(); i++)
        {
            if (bit->second->bl > infos[i]->bl + 1)
            {
                m_log(TRACE) << "non smooth error " << bit->second->bl << " "
                             << infos[i]->bl << endl;
            }
        }
    }

    for (auto &[el, tag] : m_mesh->m_elementTags[3])
    {
        if (!IsPrismValid(el))
        {
            m_log(TRACE) << "validity error " << el->GetGlobalID() << endl;
        }
    }

    // The pseudo surface is tagged with the CAD surface its prisms grew from
    // so that GrowLayers can group it by surface, but it is not on that
    // surface: it is the far face of the layer, and once the layers have
    // stopped growing it is an interior face of the mesh. Anything later that
    // trusts the association -- MakeOrder projects a CAD-associated face's
    // nodes on to the CAD -- would pull the face back down on to the wall and
    // fold the prism flat, so drop it here.
    for (auto *T : m_psuedoSurface)
    {
        m_mesh->m_meshGraph->GetCADAssociation()->Remove(T);
    }
}

map<SpatialDomains::PointGeom *, SpatialDomains::PointGeom *> BLMesh::
    GetSymNodes()
{
    map<SpatialDomains::PointGeom *, SpatialDomains::PointGeom *> ret;

    auto &m_graph = m_mesh->m_meshGraph;

    for (auto bit = m_blData.begin(); bit != m_blData.end(); bit++)
    {
        if (!bit->second->onSym)
        {
            continue;
        }
        SpatialDomains::CADSurfSharedPtr s =
            m_graph->GetCAD()->GetSurf(bit->second->symsurf);

        std::array<NekDouble, 3> loc;
        bit->second->pNode->GetCoords(loc[0], loc[1], loc[2]);

        auto uv = s->locuv(loc);
        m_graph->GetCADAssociation()->Add(bit->second->pNode,
                                          {s, {uv[0], uv[1]}});
        ret[bit->first] = bit->second->pNode;
    }
    return ret;
}

inline bool Infont(SpatialDomains::PointGeom *n, SpatialDomains::Geometry *el,
                   const SpatialDomains::CADSurfSharedPtr &surf)
{
    vector<SpatialDomains::PointGeom *> ns1 = GetVerts(el);
    auto N1                                 = Normal(el, surf);

    std::array<NekDouble, 3> V;
    V[0] = n->x() - ns1[0]->x();
    V[1] = n->y() - ns1[0]->y();
    V[2] = n->z() - ns1[0]->z();

    NekDouble Vmag = sqrt(V[0] * V[0] + V[1] * V[1] + V[2] * V[2]);

    NekDouble ang = (N1[0] * V[0] + N1[1] * V[1] + N1[2] * V[2]) / Vmag;

    return ang > 0.17;
}

void BLMesh::GrowLayers()
{
    auto &m_graph = m_mesh->m_meshGraph;
    decltype(m_blData)::iterator bit;

    // setup up a tree which is formed of boxes of each surface plus some
    // extra room (ideal bl thick)

    // in each iteration a tree is made for all the triangles in each surface
    // when considering to stop a bounary layer growing, it first
    // looks at the top tree to find surfaces which are canditates
    // it then searches the subtrees for each triangle canditate
    // it then does a distance calcation.
    // if a boundary layer should be close to that from another surface, it
    // should stop

    map<int, vector<SpatialDomains::Geometry *>> psElements;
    for (auto &[el, tag] : m_mesh->m_elementTags[2])
    {
        int surfId = m_graph->GetCADAssociation()->GetSurf(el)->GetId();

        vector<unsigned int>::iterator f =
            find(m_blsurfs.begin(), m_blsurfs.end(), surfId);

        vector<unsigned int>::iterator s =
            find(m_symSurfs.begin(), m_symSurfs.end(), surfId);

        if (f == m_blsurfs.end() && s == m_symSurfs.end())
        {
            psElements[surfId].push_back(el);
        }
    }
    for (int i = 0; i < m_psuedoSurface.size(); i++)
    {
        psElements
            [m_graph->GetCADAssociation()->GetSurf(m_psuedoSurface[i])->GetId()]
                .push_back(m_psuedoSurface[i]);
    }

    bgi::rtree<boxI, bgi::quadratic<16>> TopTree;
    map<int, bgi::rtree<boxI, bgi::quadratic<16>>> SubTrees;

    // ofstream file;
    // file.open("pts.3D");
    // file << "x y z value" << endl;

    for (int l = 1; l < m_layer; l++)
    {
        NekDouble delta = (m_layerT[l] - m_layerT[l - 1]);
        TopTree.clear();
        SubTrees.clear();
        map<int, vector<SpatialDomains::Geometry *>>::iterator it;
        for (it = psElements.begin(); it != psElements.end(); it++)
        {
            TopTree.insert(make_pair(GetBox(it->second, m_bl), it->first));
            vector<boxI> toInsert;
            for (int i = 0; i < it->second.size(); i++)
            {
                toInsert.push_back(make_pair(GetBox(it->second[i], m_bl), i));
            }
            SubTrees[it->first].insert(toInsert.begin(), toInsert.end());
        }

        for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
        {
            if (bit->second->stopped)
            {
                continue;
            }

            vector<boxI> results;
            TopTree.query(bgi::intersects(point(bit->second->pNode->x(),
                                                bit->second->pNode->y(),
                                                bit->second->pNode->z())),
                          back_inserter(results));
            set<int> surfs;
            for (int i = 0; i < results.size(); i++)
            {
                set<int>::iterator f =
                    bit->second->surfs.find(results[i].second);
                if (f == bit->second->surfs.end())
                {
                    // hit
                    surfs.insert(results[i].second);
                }
            }

            set<int>::iterator iit;
            bool hit = false;
            for (iit = surfs.begin(); iit != surfs.end(); iit++)
            {
                results.clear();
                SubTrees[*iit].query(
                    bgi::intersects(GetBox(bit->second->pNode, m_bl)),
                    back_inserter(results));
                for (int i = 0; i < results.size(); i++)
                {
                    SpatialDomains::Geometry *psEl =
                        psElements[*iit][results[i].second];
                    if (Infont(bit->second->pNode, psEl,
                               m_graph->GetCADAssociation()->GetSurf(psEl)))
                    {
                        NekDouble prox =
                            Proximity(bit->second->pNode,
                                      psElements[*iit][results[i].second]);
                        if (prox < delta * 2.5)
                        {
                            hit                  = true;
                            bit->second->stopped = true;
                            break;
                        }
                    }
                }
                if (hit)
                {
                    break;
                }
            }
        }

        // if after proximity scanning all is okay, advance the layer
        for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
        {
            if (bit->second->stopped)
            {
                continue;
            }

            // test the smoothness
            bool shouldStop            = false;
            vector<blInfoSharedPtr> ne = m_nToNInfo[bit->first];
            for (int i = 0; i < ne.size(); i++)
            {
                if (ne[i]->bl < bit->second->bl)
                {
                    shouldStop = true;
                    break;
                }
            }
            if (shouldStop)
            {
                bit->second->stopped = true;
                continue;
            }

            bit->second->AlignNode(m_layerT[l]);
            bit->second->bl = l;
        }
    }
    // file.close();
}

inline bool sign(NekDouble a, NekDouble b)
{
    return (a * b > 0.0);
}

inline NekDouble Dot(std::array<NekDouble, 3> a, std::array<NekDouble, 3> b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

NekDouble BLMesh::Proximity(SpatialDomains::PointGeom *n,
                            SpatialDomains::Geometry *el)
{
    vector<SpatialDomains::PointGeom *> ns = GetVerts(el);

    std::array<NekDouble, 3> B;
    ns[0]->GetCoords(B[0], B[1], B[2]);

    std::array<NekDouble, 3> E0 = {ns[1]->x() - ns[0]->x(),
                                   ns[1]->y() - ns[0]->y(),
                                   ns[1]->z() - ns[0]->z()};
    std::array<NekDouble, 3> E1 = {ns[2]->x() - ns[0]->x(),
                                   ns[2]->y() - ns[0]->y(),
                                   ns[2]->z() - ns[0]->z()};

    std::array<NekDouble, 3> P;
    n->GetCoords(P[0], P[1], P[2]);

    NekDouble a = Dot(E0, E0);
    NekDouble b = Dot(E0, E1);
    NekDouble c = Dot(E1, E1);

    std::array<NekDouble, 3> BP = {B[0] - P[0], B[1] - P[1], B[2] - P[2]};

    NekDouble d = Dot(E0, BP);
    NekDouble e = Dot(E1, BP);

    NekDouble det = a * c - b * b;
    NekDouble s = b * e - c * d, t = b * d - a * e;

    if (s + t <= det)
    {
        if (s < 0)
        {
            if (t < 0)
            {
                t = 0;
                s = 0;
            }
            else
            {
                s = 0;
                t = (e >= 0 ? 0 : (-e >= c ? 1 : -e / c));
            }
        }
        else if (t < 0)
        {
            t = 0;
            s = (d >= 0 ? 0 : (-d >= a ? 1 : -d / a));
        }
        else
        {
            NekDouble invDet = 1.0 / det;
            s *= invDet;
            t *= invDet;
        }
    }
    else
    {
        if (s < 0)
        {
            s = 0;
            t = 1;
        }
        else if (t < 0)
        {
            s = 1;
            t = 0;
        }
        else
        {
            NekDouble numer = c + e - b - d;
            if (numer <= 0)
            {
                s = 0;
            }
            else
            {
                NekDouble denom = a - 2 * b + c;
                s               = (numer >= denom ? 1 : numer / denom);
            }
            t = 1 - s;
        }
    }

    // closest point on the triangle; the old code built a throwaway Node here
    // purely to measure the distance to it
    std::array<NekDouble, 3> cp = {B[0] + s * E0[0] + t * E1[0],
                                   B[1] + s * E0[1] + t * E1[1],
                                   B[2] + s * E0[2] + t * E1[2]};

    return sqrt((P[0] - cp[0]) * (P[0] - cp[0]) +
                (P[1] - cp[1]) * (P[1] - cp[1]) +
                (P[2] - cp[2]) * (P[2] - cp[2]));
}

bool BLMesh::TestIntersectionEl(SpatialDomains::Geometry *e1,
                                SpatialDomains::Geometry *e2)
{
    vector<SpatialDomains::PointGeom *> ns1 = GetVerts(e1);
    vector<SpatialDomains::PointGeom *> ns2 = GetVerts(e2);
    if (ns1[0] == ns2[0] || ns1[0] == ns2[1] || ns1[0] == ns2[2] ||
        ns1[1] == ns2[0] || ns1[1] == ns2[1] || ns1[1] == ns2[2] ||
        ns1[2] == ns2[0] || ns1[2] == ns2[1] || ns1[2] == ns2[2])
    {
        return false;
    }

    std::array<NekDouble, 3> N1, N2;
    NekDouble d1, d2;

    N1[0] = (ns1[1]->y() - ns1[0]->y()) * (ns1[2]->z() - ns1[0]->z()) -
            (ns1[2]->y() - ns1[0]->y()) * (ns1[1]->z() - ns1[0]->z());
    N1[1] = -1.0 * ((ns1[1]->x() - ns1[0]->x()) * (ns1[2]->z() - ns1[0]->z()) -
                    (ns1[2]->x() - ns1[0]->x()) * (ns1[1]->z() - ns1[0]->z()));
    N1[2] = (ns1[1]->x() - ns1[0]->x()) * (ns1[2]->y() - ns1[0]->y()) -
            (ns1[2]->x() - ns1[0]->x()) * (ns1[1]->y() - ns1[0]->y());

    N2[0] = (ns2[1]->y() - ns2[0]->y()) * (ns2[2]->z() - ns2[0]->z()) -
            (ns2[2]->y() - ns2[0]->y()) * (ns2[1]->z() - ns2[0]->z());
    N2[1] = -1.0 * ((ns2[1]->x() - ns2[0]->x()) * (ns2[2]->z() - ns2[0]->z()) -
                    (ns2[2]->x() - ns2[0]->x()) * (ns2[1]->z() - ns2[0]->z()));
    N2[2] = (ns2[1]->x() - ns2[0]->x()) * (ns2[2]->y() - ns2[0]->y()) -
            (ns2[2]->x() - ns2[0]->x()) * (ns2[1]->y() - ns2[0]->y());

    d1 = -1.0 *
         (N1[0] * ns1[0]->x() + N1[1] * ns1[0]->y() + N1[2] * ns1[0]->z());
    d2 = -1.0 *
         (N2[0] * ns2[0]->x() + N2[1] * ns2[0]->y() + N2[2] * ns2[0]->z());

    std::array<NekDouble, 3> dv1, dv2;

    dv1[0] =
        N2[0] * ns1[0]->x() + N2[1] * ns1[0]->y() + N2[2] * ns1[0]->z() + d2;
    dv1[1] =
        N2[0] * ns1[1]->x() + N2[1] * ns1[1]->y() + N2[2] * ns1[1]->z() + d2;
    dv1[2] =
        N2[0] * ns1[2]->x() + N2[1] * ns1[2]->y() + N2[2] * ns1[2]->z() + d2;

    dv2[0] =
        N1[0] * ns2[0]->x() + N1[1] * ns2[0]->y() + N1[2] * ns2[0]->z() + d1;
    dv2[1] =
        N1[0] * ns2[1]->x() + N1[1] * ns2[1]->y() + N1[2] * ns2[1]->z() + d1;
    dv2[2] =
        N1[0] * ns2[2]->x() + N1[1] * ns2[2]->y() + N1[2] * ns2[2]->z() + d1;

    if (sign(dv1[0], dv1[1]) && sign(dv1[1], dv1[2]))
    {
        return false;
    }
    if (sign(dv2[0], dv2[1]) && sign(dv2[1], dv2[2]))
    {
        return false;
    }

    std::array<NekDouble, 3> D;
    D[0] = N1[1] * N2[2] - N1[2] * N2[1];
    D[1] = -1.0 * (N1[0] * N2[2] - N1[2] * N2[0]);
    D[2] = N1[0] * N2[1] - N1[0] * N2[1];

    int base1 = 0, base2 = 0;
    if (!sign(dv2[0], dv2[1]) && sign(dv2[1], dv2[2]))
    {
        base2 = 0;
    }
    else if (!sign(dv2[1], dv2[2]) && sign(dv2[2], dv2[0]))
    {
        base2 = 1;
    }
    else if (!sign(dv2[2], dv2[0]) && sign(dv2[0], dv2[1]))
    {
        base2 = 2;
    }
    else
    {
        m_log(TRACE) << "base not set" << endl;
    }

    if (!sign(dv1[0], dv1[1]) && sign(dv1[1], dv1[2]))
    {
        base1 = 0;
    }
    else if (!sign(dv1[1], dv1[2]) && sign(dv1[2], dv1[0]))
    {
        base1 = 1;
    }
    else if (!sign(dv1[2], dv1[0]) && sign(dv1[0], dv1[1]))
    {
        base1 = 2;
    }
    else
    {
        m_log(TRACE) << "base not set" << endl;
    }

    std::array<NekDouble, 3> p1, p2;

    p1[0] = D[0] * ns1[0]->x() + D[1] * ns1[0]->y() + D[2] * ns1[0]->z();
    p1[1] = D[0] * ns1[1]->x() + D[1] * ns1[1]->y() + D[2] * ns1[1]->z();
    p1[2] = D[0] * ns1[2]->x() + D[1] * ns1[2]->y() + D[2] * ns1[2]->z();

    p2[0] = D[0] * ns2[0]->x() + D[1] * ns2[0]->y() + D[2] * ns2[0]->z();
    p2[1] = D[0] * ns2[1]->x() + D[1] * ns2[1]->y() + D[2] * ns2[1]->z();
    p2[2] = D[0] * ns2[2]->x() + D[1] * ns2[2]->y() + D[2] * ns2[2]->z();

    NekDouble t11, t12, t21, t22;
    int o1 = 0, o2 = 0;
    if (base1 == 0)
    {
        o1 = 1;
        o2 = 2;
    }
    else if (base1 == 1)
    {
        o1 = 2;
        o2 = 0;
    }
    else if (base1 == 2)
    {
        o1 = 0;
        o2 = 1;
    }

    t11 = p1[o1] + (p1[base1] - p1[o1]) * dv1[o1] / (dv1[o1] - dv1[base1]);
    t12 = p1[o2] + (p1[base1] - p1[o2]) * dv1[o2] / (dv1[o2] - dv1[base1]);

    if (base2 == 0)
    {
        o1 = 1;
        o2 = 2;
    }
    else if (base2 == 1)
    {
        o1 = 2;
        o2 = 0;
    }
    else if (base2 == 2)
    {
        o1 = 0;
        o2 = 1;
    }

    t21 = p2[o1] + (p2[base2] - p2[o1]) * dv2[o1] / (dv2[o1] - dv2[base2]);
    t22 = p2[o2] + (p2[base2] - p2[o2]) * dv2[o2] / (dv2[o2] - dv2[base2]);

    if (t11 > t12)
    {
        swap(t11, t12);
    }
    if (t21 > t22)
    {
        swap(t21, t22);
    }

    if (t21 < t11)
    {
        swap(t11, t21);
        swap(t12, t22);
    }

    if (!sign(t21 - t11, t22 - t11) || !sign(t21 - t12, t22 - t12))
    {
        return true;
    }

    return false;
}

void BLMesh::Shrink()
{
    decltype(m_blData)::iterator bit;
    bool smsh = true;

    while (smsh)
    {
        smsh = false;

        vector<SpatialDomains::Geometry *> inv;
        for (auto &[el, tag] : m_mesh->m_elementTags[3])
        {
            if (!IsPrismValid(el))
            {
                inv.push_back(el);
            }
        }

        smsh = (inv.size() > 0);

        for (int i = 0; i < inv.size(); i++)
        {
            SpatialDomains::Geometry *t = m_priToTri[inv[i]];
            vector<blInfoSharedPtr> bls;
            vector<SpatialDomains::PointGeom *> ns = GetVerts(t);
            for (int j = 0; j < ns.size(); j++)
            {
                bls.push_back(m_blData[ns[j]]);
            }
            bool repeat = true;
            while (repeat)
            {
                repeat = false;
                int mx = 0;
                for (int j = 0; j < 3; j++)
                {
                    mx = max(mx, bls[j]->bl);
                }
                ASSERTL0(mx > 0, "shrinking to nothing");
                for (int j = 0; j < 3; j++)
                {
                    if (bls[j]->bl < mx)
                    {
                        continue;
                    }
                    bls[j]->bl--;
                    bls[j]->AlignNode(m_layerT[bls[j]->bl]);
                }
                if (!IsPrismValid(inv[i]))
                {
                    repeat = true;
                }
            }
        }

        bool repeat = true;
        while (repeat)
        {
            repeat = false;
            for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
            {
                vector<blInfoSharedPtr> infos = m_nToNInfo[bit->first];
                for (int i = 0; i < infos.size(); i++)
                {
                    if (bit->second->bl > infos[i]->bl + 1)
                    {
                        bit->second->bl--;
                        bit->second->AlignNode(m_layerT[bit->second->bl]);
                        repeat = true;
                    }
                }
            }
        }
    }
}

bool BLMesh::IsPrismValid(SpatialDomains::Geometry *el)
{
    NekDouble mn = numeric_limits<double>::max();
    NekDouble mx = -1.0 * numeric_limits<double>::max();
    vector<SpatialDomains::PointGeom *> ns = GetVerts(el);
    NekVector<NekDouble> X(6), Y(6), Z(6);
    for (int j = 0; j < ns.size(); j++)
    {
        X(j) = ns[j]->x();
        Y(j) = ns[j]->y();
        Z(j) = ns[j]->z();
    }
    NekVector<NekDouble> x1(6), y1(6), z1(6), x2(6), y2(6), z2(6), x3(6), y3(6),
        z3(6);

    x1 = m_deriv[0] * X;
    y1 = m_deriv[0] * Y;
    z1 = m_deriv[0] * Z;
    x2 = m_deriv[1] * X;
    y2 = m_deriv[1] * Y;
    z2 = m_deriv[1] * Z;
    x3 = m_deriv[2] * X;
    y3 = m_deriv[2] * Y;
    z3 = m_deriv[2] * Z;

    for (int j = 0; j < 6; j++)
    {
        DNekMat dxdz(3, 3, 1.0, eFULL);
        dxdz(0, 0) = x1(j);
        dxdz(0, 1) = x2(j);
        dxdz(0, 2) = x3(j);
        dxdz(1, 0) = y1(j);
        dxdz(1, 1) = y2(j);
        dxdz(1, 2) = y3(j);
        dxdz(2, 0) = z1(j);
        dxdz(2, 1) = z2(j);
        dxdz(2, 2) = z3(j);

        NekDouble jacDet =
            dxdz(0, 0) * (dxdz(1, 1) * dxdz(2, 2) - dxdz(2, 1) * dxdz(1, 2)) -
            dxdz(0, 1) * (dxdz(1, 0) * dxdz(2, 2) - dxdz(2, 0) * dxdz(1, 2)) +
            dxdz(0, 2) * (dxdz(1, 0) * dxdz(2, 1) - dxdz(2, 0) * dxdz(1, 1));
        mn = min(mn, jacDet);
        mx = max(mx, jacDet);
    }

    return mn > 0;
}

void BLMesh::BuildElements()
{
    auto &m_graph = m_mesh->m_meshGraph;

    // make prisms
    map<SpatialDomains::CADOrientation::Orientation, vector<int>> baseTri;
    map<SpatialDomains::CADOrientation::Orientation, vector<int>> topTri;

    vector<int> tmp;
    // back-base
    tmp.push_back(0);
    tmp.push_back(4);
    tmp.push_back(1);
    baseTri[SpatialDomains::CADOrientation::eBackwards] = tmp;
    tmp.clear();
    // for-base
    tmp.push_back(0);
    tmp.push_back(1);
    tmp.push_back(4);
    baseTri[SpatialDomains::CADOrientation::eForwards] = tmp;
    // back-top
    tmp.clear();
    tmp.push_back(3);
    tmp.push_back(5);
    tmp.push_back(2);
    topTri[SpatialDomains::CADOrientation::eBackwards] = tmp;
    // for-top
    tmp.clear();
    tmp.push_back(3);
    tmp.push_back(2);
    tmp.push_back(5);
    topTri[SpatialDomains::CADOrientation::eForwards] = tmp;

    ElmtConfig pconf(LibUtilities::ePrism, 1, false, false);
    ElmtConfig tconf(LibUtilities::eTriangle, 1, false, false);

    // surface triags
    vector<SpatialDomains::Geometry *> blTris;
    for (auto &[el, tag] : m_mesh->m_elementTags[2])
    {
        vector<unsigned int>::iterator f =
            find(m_blsurfs.begin(), m_blsurfs.end(),
                 m_graph->GetCADAssociation()->GetSurf(el)->GetId());

        if (f == m_blsurfs.end())
        {
            // for quad do we should extend for hexes in the future
            continue;
        }
        blTris.push_back(el);
    }

    // The prism starts from the surf triag, hence the face has to be reset to
    // comply with the prism.
    std::unordered_set<int> naiveTriIDs;
    for (auto *el : blTris)
    {
        naiveTriIDs.insert(el->GetGlobalID());
    }

    for (auto *el : blTris)
    {
        SpatialDomains::CADSurfSharedPtr cadSurf =
            m_graph->GetCADAssociation()->GetSurf(el);

        vector<SpatialDomains::PointGeom *> tn(3); // nodes for pseduo surface
        vector<SpatialDomains::PointGeom *> pn(6); // all prism nodes
        vector<SpatialDomains::PointGeom *> n = GetVerts(el);

        SpatialDomains::CADOrientation::Orientation o = cadSurf->Orientation();

        for (int j = 0; j < 3; j++)
        {
            pn[baseTri[o][j]] = n[j];
            pn[topTri[o][j]]  = m_blData[n[j]]->pNode;
            tn[j]             = m_blData[n[j]]->pNode;
        }

        SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
            LibUtilities::ePrism, pn, m_graph, m_mesh->m_edgeSet,
            m_mesh->m_faceSet, pconf, nullptr, nullptr, &naiveTriIDs, nullptr);

        m_mesh->m_elementTags[3][E] = m_id;

        SpatialDomains::Geometry *T = GetElementFactory().CreateInstance(
            LibUtilities::eTriangle, tn, m_graph, m_mesh->m_edgeSet,
            m_mesh->m_faceSet, tconf, nullptr, nullptr, nullptr, nullptr);

        m_psuedoSurface.push_back(T);

        m_graph->GetCADAssociation()->Set(T, {cadSurf});

        m_priToTri[E] = el;
    }
}

NekDouble BLMesh::Visability(vector<SpatialDomains::Geometry *> tris,
                             std::array<NekDouble, 3> N)
{
    NekDouble mn = numeric_limits<double>::max();

    for (int i = 0; i < tris.size(); i++)
    {
        auto tmp =
            Normal(tris[i],
                   m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(tris[i]));
        NekDouble dt = tmp[0] * N[0] + tmp[1] * N[1] + tmp[2] * N[2];
        mn           = min(mn, dt);
    }
    return mn;
}

std::array<NekDouble, 3> BLMesh::GetNormal(
    vector<SpatialDomains::Geometry *> tris)
{
    // compile list of normals
    vector<std::array<NekDouble, 3>> N;
    for (int i = 0; i < tris.size(); i++)
    {
        N.push_back(
            Normal(tris[i],
                   m_mesh->m_meshGraph->GetCADAssociation()->GetSurf(tris[i])));
    }

    vector<NekDouble> w(N.size());
    std::array<NekDouble, 3> Np = {0.0, 0.0, 0.0};

    for (int i = 0; i < N.size(); i++)
    {
        w[i] = 1.0 / N.size();
    }

    for (int i = 0; i < N.size(); i++)
    {
        Np[0] += w[i] * N[i][0];
        Np[1] += w[i] * N[i][1];
        Np[2] += w[i] * N[i][2];
    }

    NekDouble mag = sqrt(Np[0] * Np[0] + Np[1] * Np[1] + Np[2] * Np[2]);
    Np[0] /= mag;
    Np[1] /= mag;
    Np[2] /= mag;

    std::array<NekDouble, 3> Ninital = Np;

    NekDouble dot = 0.0;
    int ct        = 0;
    vector<NekDouble> a(N.size());
    while (fabs(dot - 1) > 1e-6)
    {
        ct++;
        std::array<NekDouble, 3> Nplast = Np;

        NekDouble aSum = 0.0;
        for (int i = 0; i < N.size(); i++)
        {
            NekDouble dot2 =
                Np[0] * N[i][0] + Np[1] * N[i][1] + Np[2] * N[i][2];
            if (fabs(dot2 - 1) < 1e-9)
            {
                a[i] = dot2 / fabs(dot2) * 1e-9;
            }
            else
            {
                a[i] = acos(dot2);
            }

            aSum += a[i];
        }

        NekDouble wSum = 0.0;
        for (int i = 0; i < N.size(); i++)
        {
            w[i] = w[i] * a[i] / aSum;
            wSum += w[i];
        }

        for (int i = 0; i < N.size(); i++)
        {
            w[i] = w[i] / wSum;
        }

        std::array<NekDouble, 3> NpN = {0.0, 0.0, 0.0};
        for (int i = 0; i < N.size(); i++)
        {
            NpN[0] += w[i] * N[i][0];
            NpN[1] += w[i] * N[i][1];
            NpN[2] += w[i] * N[i][2];
        }
        mag = sqrt(NpN[0] * NpN[0] + NpN[1] * NpN[1] + NpN[2] * NpN[2]);
        NpN[0] /= mag;
        NpN[1] /= mag;
        NpN[2] /= mag;

        Np[0] = 0.8 * NpN[0] + (1.0 - 0.8) * Np[0];
        Np[1] = 0.8 * NpN[1] + (1.0 - 0.8) * Np[1];
        Np[2] = 0.8 * NpN[2] + (1.0 - 0.8) * Np[2];
        mag   = sqrt(Np[0] * Np[0] + Np[1] * Np[1] + Np[2] * Np[2]);
        Np[0] /= mag;
        Np[1] /= mag;
        Np[2] /= mag;

        dot = Np[0] * Nplast[0] + Np[1] * Nplast[1] + Np[2] * Nplast[2];

        if (ct > 100000)
        {
            m_log(TRACE) << "run out of iterations" << endl;
            Np = Ninital;
            break;
        }
    }

    return Np;
}

void BLMesh::Setup()
{
    NekDouble a = m_prog == 1.0
                      ? 1.0 / m_layer
                      : 2.0 * (1.0 - m_prog) / (1.0 - pow(m_prog, m_layer + 1));
    m_layerT.resize(m_layer);
    m_layerT[0] = a * m_prog * m_bl;
    for (int i = 1; i < m_layer; i++)
    {
        m_layerT[i] = m_layerT[i - 1] + a * pow(m_prog, i) * m_bl;
    }

    m_log(VERBOSE) << "    - First layer height " << m_layerT[0] << endl;

    auto &m_graph = m_mesh->m_meshGraph;

    // this sets up all the boundary layer normals data holder
    set<int> symSurfs;
    int failed = 0;

    // ofstream file1;
    // file1.open("pts.3D");
    // file1 << "X Y Z value" << endl;
    for (auto [vid, vert] : m_graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        auto ss = m_graph->GetCADAssociation()->GetLinks(
            vert, SpatialDomains::CADType::eSurf);
        vector<unsigned int> surfs;
        for (int i = 0; i < ss.size(); i++)
        {
            surfs.push_back(ss[i].Id());
        }
        sort(surfs.begin(), surfs.end());
        vector<unsigned int> inter, diff;

        set_intersection(m_blsurfs.begin(), m_blsurfs.end(), surfs.begin(),
                         surfs.end(), back_inserter(inter));
        set_symmetric_difference(inter.begin(), inter.end(), surfs.begin(),
                                 surfs.end(), back_inserter(diff));

        // is somewhere on a bl surface
        if (inter.size() > 0)
        {
            // initialise a new bl boudnary node
            blInfoSharedPtr bln = std::shared_ptr<blInfo>(new blInfo);
            bln->oNode          = vert;
            bln->stopped        = false;

            // file1 << (*it)->x() << " " << (*it)->y() << " " << (*it)->z() <<
            // " " << ss.size() << endl;

            if (diff.size() > 0)
            {
                // if the diff size is greater than 1 there is a curve that
                // needs remeshing
                ASSERTL0(diff.size() <= 1, "not setup for curve bl refinement");
                symSurfs.insert(diff[0]);
                bln->symsurf = diff[0];
                bln->onSym   = true;
            }
            else
            {
                bln->onSym = false;
            }

            m_blData[vert] = bln;
        }
    }
    // file1.close();

    // need a map from vertex idx to surface elements
    // but do not care about triangles which are not in the bl
    for (auto &[el, tag] : m_mesh->m_elementTags[2])
    {
        int surfId = m_graph->GetCADAssociation()->GetSurf(el)->GetId();

        vector<unsigned int>::iterator f =
            find(m_blsurfs.begin(), m_blsurfs.end(), surfId);

        if (f == m_blsurfs.end())
        {
            // if this triangle is not in bl surfs continue
            continue;
        }

        vector<SpatialDomains::PointGeom *> ns = GetVerts(el);
        for (int j = 0; j < ns.size(); j++)
        {
            m_blData[ns[j]]->els.push_back(el);
            m_blData[ns[j]]->surfs.insert(surfId);
        }
    }

    decltype(m_blData)::iterator bit;
    for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
    {
        // calculate mesh normal
        bit->second->N = GetNormal(bit->second->els);

        if (Visability(bit->second->els, bit->second->N) < 0.0)
        {
            m_log(WARNING) << "failed " << bit->first->x() << " "
                           << bit->first->y() << " " << bit->first->z() << " "
                           << Visability(bit->second->els, bit->second->N)
                           << endl;
            failed++;
        }

        std::array<NekDouble, 3> loc;
        bit->first->GetCoords(loc[0], loc[1], loc[2]);
        for (int k = 0; k < 3; k++)
        {
            loc[k] += bit->second->N[k] * m_layerT[0];
        }

        int newId = NextPointId(m_graph);
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            3, newId, loc[0], loc[1], loc[2]);

        bit->second->pNode = pt.get();
        m_graph->AddGeom<SpatialDomains::PointGeom>(newId, std::move(pt));

        bit->second->bl = 0;
    }

    m_symSurfs = vector<unsigned int>(symSurfs.begin(), symSurfs.end());

    // now need to enforce that all symmetry plane nodes have their normal
    // forced onto the symmetry surface
    for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
    {
        if (!bit->second->onSym)
        {
            continue;
        }

        std::array<NekDouble, 3> loc;
        bit->second->pNode->GetCoords(loc[0], loc[1], loc[2]);

        auto uv = m_graph->GetCAD()->GetSurf(bit->second->symsurf)->locuv(loc);
        auto nl = m_graph->GetCAD()->GetSurf(bit->second->symsurf)->P(uv);

        std::array<NekDouble, 3> N = {nl[0] - bit->first->x(),
                                      nl[1] - bit->first->y(),
                                      nl[2] - bit->first->z()};

        NekDouble mag = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
        N[0] /= mag;
        N[1] /= mag;
        N[2] /= mag;

        bit->second->N = N;
        bit->second->AlignNode(m_layerT[0]);
    }

    // now smooth all the normals by distance weighted average
    // keep normals on curves constant
    for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
    {
        set<int> added;
        added.insert(bit->first->GetGlobalID());
        for (int i = 0; i < bit->second->els.size(); i++)
        {
            vector<SpatialDomains::PointGeom *> ns =
                GetVerts(bit->second->els[i]);
            for (int j = 0; j < ns.size(); j++)
            {
                set<int>::iterator t = added.find(ns[j]->GetGlobalID());
                if (t == added.end())
                {
                    m_nToNInfo[bit->first].push_back(m_blData[ns[j]]);
                }
            }
        }
    }

    for (int l = 0; l < 10; l++)
    {
        for (bit = m_blData.begin(); bit != m_blData.end(); bit++)
        {
            if (m_graph->GetCADAssociation()->Count(
                    bit->first, SpatialDomains::CADType::eSurf) > 1)
            {
                continue;
            }

            std::array<NekDouble, 3> sumV = {0.0, 0.0, 0.0};
            vector<blInfoSharedPtr> data  = m_nToNInfo[bit->first];
            NekDouble Dtotal              = 0.0;
            for (int i = 0; i < data.size(); i++)
            {
                NekDouble d = bit->first->dist(*data[i]->oNode);
                Dtotal += d;
                sumV[0] += data[i]->N[0] / d;
                sumV[1] += data[i]->N[1] / d;
                sumV[2] += data[i]->N[2] / d;
            }
            sumV[0] *= Dtotal;
            sumV[1] *= Dtotal;
            sumV[2] *= Dtotal;
            NekDouble mag =
                sqrt(sumV[0] * sumV[0] + sumV[1] * sumV[1] + sumV[2] * sumV[2]);
            sumV[0] /= mag;
            sumV[1] /= mag;
            sumV[2] /= mag;

            std::array<NekDouble, 3> N = {
                (1.0 - 0.8) * bit->second->N[0] + 0.8 * sumV[0],
                (1.0 - 0.8) * bit->second->N[1] + 0.8 * sumV[1],
                (1.0 - 0.8) * bit->second->N[2] + 0.8 * sumV[2]};

            mag = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
            N[0] /= mag;
            N[1] /= mag;
            N[2] /= mag;

            bit->second->N = N;
            bit->second->AlignNode(m_layerT[0]);
        }
    }

    /*ofstream file;
    file.open("bl.lines");
    for(bit = m_blData.begin(); bit != m_blData.end(); bit++)
    {
        NekDouble l = 0.05;
        file << bit->first->x() << ", " << bit->first->y() << ", " <<
    bit->first->z() << endl;
        file << bit->first->x() + bit->second->N[0]*l << ", "
             << bit->first->y() + bit->second->N[1]*l << ", "
             << bit->first->z() + bit->second->N[2]*l << endl;
        file << endl;
    }
    file.close();*/

    ASSERTL0(failed == 0, "some normals failed to generate");

    LibUtilities::PointsKey pkey1(2, LibUtilities::eNodalPrismElec);

    Array<OneD, NekDouble> u1, v1, w1;
    LibUtilities::PointsManager()[pkey1]->GetPoints(u1, v1, w1);

    LibUtilities::NodalUtilPrism nodalPrism(1, u1, v1, w1);

    NekMatrix<NekDouble> Vandermonde  = *nodalPrism.GetVandermonde();
    NekMatrix<NekDouble> VandermondeI = Vandermonde;
    VandermondeI.Invert();

    m_deriv[0] = *nodalPrism.GetVandermondeForDeriv(0) * VandermondeI;
    m_deriv[1] = *nodalPrism.GetVandermondeForDeriv(1) * VandermondeI;
    m_deriv[2] = *nodalPrism.GetVandermondeForDeriv(2) * VandermondeI;
}
} // namespace Nektar::NekMesh
