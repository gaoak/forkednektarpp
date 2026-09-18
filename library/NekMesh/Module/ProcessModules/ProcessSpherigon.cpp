///////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessSpherigon.cpp
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
//  Description: Apply Spherigon surface smoothing technique to a 3D mesh.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <unordered_set>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Foundations/ManagerAccess.h>

#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/QuadGeom.h>
#include <SpatialDomains/SegGeom.h>
#include <SpatialDomains/TriGeom.h>

#include "ProcessSpherigon.h"

namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

using namespace std;

#define TOL_BLEND 1.0e-8

namespace Nektar::NekMesh
{
ModuleKey ProcessSpherigon::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "spherigon"), ProcessSpherigon::create);

namespace
{

/**
 * @brief Minimal 3-vector, so that the spherigon algebra below reads like the
 * paper it comes from.
 */
struct Vec3
{
    NekDouble x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(NekDouble px, NekDouble py, NekDouble pz) : x(px), y(py), z(pz)
    {
    }
    explicit Vec3(const SpatialDomains::PointGeom *p)
        : x((*p)[0]), y((*p)[1]), z((*p)[2])
    {
    }
    explicit Vec3(const std::array<NekDouble, 3> &a) : x(a[0]), y(a[1]), z(a[2])
    {
    }

    Vec3 operator+(const Vec3 &r) const
    {
        return {x + r.x, y + r.y, z + r.z};
    }
    Vec3 operator-(const Vec3 &r) const
    {
        return {x - r.x, y - r.y, z - r.z};
    }
    Vec3 operator*(NekDouble s) const
    {
        return {x * s, y * s, z * s};
    }
    Vec3 &operator+=(const Vec3 &r)
    {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    NekDouble dot(const Vec3 &r) const
    {
        return x * r.x + y * r.y + z * r.z;
    }
    NekDouble abs2() const
    {
        return x * x + y * y + z * z;
    }
    Vec3 unit() const
    {
        NekDouble m = sqrt(abs2());
        return m > 0.0 ? *this * (1.0 / m) : *this;
    }
    std::array<NekDouble, 3> arr() const
    {
        return {x, y, z};
    }
};

Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

/**
 * @brief Normal to an entity, from its own vertex ordering.
 *
 * The sign is whatever that ordering happens to give and is not comparable
 * between neighbours; see ConsistentWinding().
 */
Vec3 FacetNormal(SpatialDomains::Geometry *e, int spaceDim)
{
    if (spaceDim == 3)
    {
        // Two tangent vectors and their unit cross product.
        return Cross(Vec3(e->GetVertex(1)) - Vec3(e->GetVertex(0)),
                     Vec3(e->GetVertex(2)) - Vec3(e->GetVertex(0)))
            .unit();
    }

    // Gradient vector, inverted.
    Vec3 dx = (Vec3(e->GetVertex(1)) - Vec3(e->GetVertex(0))).unit();
    return {-dx.y, dx.x, 0.0};
}

/**
 * @brief Entities whose facet normal must be negated to agree with their
 * neighbours.
 *
 * Averaging facet normals at a vertex is only meaningful if the facets agree
 * on which side of the surface is which, and the geometry does not remember
 * the winding its input file gave: a closed triangulation read from a .ply
 * comes back with the vertices of some faces in one rotational sense and some
 * in the other. Left uncorrected the contributions cancel and the averaged
 * normals are worse than useless.
 *
 * Two entities sharing an edge agree exactly when they traverse that edge in
 * opposite directions, so a breadth-first sweep of the surface fixes every
 * sign relative to the entity it started from. Which way round a connected
 * patch ends up does not matter: negating every normal, N and vN alike,
 * leaves the spherigon's output unchanged.
 */
std::unordered_set<SpatialDomains::Geometry *> ConsistentWinding(
    const std::vector<SpatialDomains::Geometry *> &el)
{
    // Entities either side of each edge.
    std::unordered_map<SpatialDomains::Geometry *, std::vector<size_t>>
        edgeToEnt;
    for (size_t i = 0; i < el.size(); ++i)
    {
        for (int k = 0; k < el[i]->GetNumEdges(); ++k)
        {
            edgeToEnt[el[i]->GetEdge(k)].push_back(i);
        }
    }

    // Direction in which entity i runs along its local edge k.
    auto traversal = [&](size_t i, int k) {
        const int nV = el[i]->GetNumVerts();
        return std::make_pair(el[i]->GetVertex(k),
                              el[i]->GetVertex((k + 1) % nV));
    };

    std::vector<int> sign(el.size(), 0);
    std::vector<size_t> stack;

    for (size_t seed = 0; seed < el.size(); ++seed)
    {
        if (sign[seed] != 0)
        {
            continue;
        }

        sign[seed] = 1;
        stack.push_back(seed);

        while (!stack.empty())
        {
            size_t a = stack.back();
            stack.pop_back();

            for (int k = 0; k < el[a]->GetNumEdges(); ++k)
            {
                auto *seg = el[a]->GetEdge(k);
                auto dirA = traversal(a, k);

                for (size_t b : edgeToEnt[seg])
                {
                    if (b == a || sign[b] != 0)
                    {
                        continue;
                    }

                    for (int k2 = 0; k2 < el[b]->GetNumEdges(); ++k2)
                    {
                        if (el[b]->GetEdge(k2) != seg)
                        {
                            continue;
                        }

                        auto dirB        = traversal(b, k2);
                        const bool agree = dirB.first == dirA.second &&
                                           dirB.second == dirA.first;

                        sign[b] = agree ? sign[a] : -sign[a];
                        stack.push_back(b);
                        break;
                    }
                }
            }
        }
    }

    std::unordered_set<SpatialDomains::Geometry *> flipped;
    for (size_t i = 0; i < el.size(); ++i)
    {
        if (sign[i] < 0)
        {
            flipped.insert(el[i]);
        }
    }

    return flipped;
}

/**
 * @brief Magnitude of the cross product \f$ \vec{a}\times\vec{b} \f$.
 */
NekDouble CrossProdMag(const Vec3 &a, const Vec3 &b)
{
    return sqrt(Cross(a, b).abs2());
}

/**
 * @brief Calculate the \f$ C^1 \f$ blending function for the spherigon.
 *
 * See equation (10) of the paper.
 *
 * @param r      Generalised barycentric coordinates of the point P.
 * @param Q      Vector of vertices denoting this triangle/quad.
 * @param P      Point in the triangle to apply blending to.
 * @param blend  The resulting blending components for each vertex.
 */
void SuperBlend(vector<NekDouble> &r, vector<Vec3> &Q, Vec3 &P,
                vector<NekDouble> &blend)
{
    int nV = r.size();
    vector<NekDouble> tmp(nV);
    NekDouble totBlend = 0.0;

    for (int i = 0; i < nV; ++i)
    {
        blend[i] = 0.0;
        tmp[i]   = (Q[i] - P).abs2();
    }

    for (int i = 0; i < nV; ++i)
    {
        int ip = (i + 1) % nV, im = (i - 1 + nV) % nV;

        if (r[i] > TOL_BLEND && r[i] < (1 - TOL_BLEND))
        {
            blend[i] = r[i] * r[i] *
                       (r[im] * r[im] * tmp[im] / (tmp[im] + tmp[i]) +
                        r[ip] * r[ip] * tmp[ip] / (tmp[ip] + tmp[i]));
            totBlend += blend[i];
        }
    }

    for (int i = 0; i < nV; ++i)
    {
        blend[i] /= totBlend;
        if (r[i] >= (1 - TOL_BLEND))
        {
            blend[i] = 1.0;
        }
        if (r[i] <= TOL_BLEND)
        {
            blend[i] = 0.0;
        }
    }
}

/**
 * @brief Number of points in the curve of an entity of @p shape at order
 * @p nq - 1.
 *
 * Matches the layout that SegGeom, TriGeom and QuadGeom::MakeOrder produce,
 * which is what the graph and the output modules expect to find.
 */
int NumCurvePoints(LibUtilities::ShapeType shape, int nq)
{
    switch (shape)
    {
        case LibUtilities::eSegment:
            return nq;
        case LibUtilities::eTriangle:
            return nq * (nq + 1) / 2;
        default:
            return nq * nq;
    }
}

/**
 * @brief Curve point indices along local edge @p e of a 2D entity, running
 * from the entity's local vertex @p e to local vertex @p e+1.
 *
 * A triangle's curve is three vertices, then each edge's interior nodes in
 * turn, then the face interior. A quadrilateral's is the tensor grid, point
 * (i, j) at index i*nq+j with @f$\xi_1 = \xi[j]@f$, @f$\xi_2 = \xi[i]@f$.
 */
vector<int> LocalEdgeCurveIndices(LibUtilities::ShapeType shape, int e, int nq)
{
    vector<int> idx(nq);

    if (shape == LibUtilities::eTriangle)
    {
        const int nEdgeNodes = nq - 2;
        idx[0]               = e;
        idx[nq - 1]          = (e + 1) % 3;
        for (int j = 0; j < nEdgeNodes; ++j)
        {
            idx[j + 1] = 3 + e * nEdgeNodes + j;
        }
    }
    else
    {
        for (int k = 0; k < nq; ++k)
        {
            switch (e)
            {
                case 0:
                    idx[k] = k;
                    break;
                case 1:
                    idx[k] = k * nq + (nq - 1);
                    break;
                case 2:
                    idx[k] = (nq - 1) * nq + (nq - 1 - k);
                    break;
                default:
                    idx[k] = (nq - 1 - k) * nq;
                    break;
            }
        }
    }

    return idx;
}

/**
 * @brief Reference coordinates of an entity's curve points, in curve order.
 */
vector<std::array<NekDouble, 2>> CurvePointCoords(LibUtilities::ShapeType shape,
                                                  int nq)
{
    vector<std::array<NekDouble, 2>> xi;

    if (shape == LibUtilities::eTriangle)
    {
        Array<OneD, NekDouble> px, py;
        LibUtilities::PointsManager()[LibUtilities::PointsKey(
                                          nq, LibUtilities::eNodalTriElec)]
            ->GetPoints(px, py);

        const int nPts = nq * (nq + 1) / 2;
        xi.resize(nPts);
        for (int i = 0; i < nPts; ++i)
        {
            xi[i] = {px[i], py[i]};
        }
    }
    else
    {
        Array<OneD, NekDouble> px;
        LibUtilities::PointsManager()
            [LibUtilities::PointsKey(nq, LibUtilities::eGaussLobattoLegendre)]
                ->GetPoints(px);

        if (shape == LibUtilities::eSegment)
        {
            xi.resize(nq);
            for (int i = 0; i < nq; ++i)
            {
                xi[i] = {px[i], 0.0};
            }
        }
        else
        {
            xi.resize(nq * nq);
            for (int i = 0; i < nq; ++i)
            {
                for (int j = 0; j < nq; ++j)
                {
                    xi[i * nq + j] = {px[j], px[i]};
                }
            }
        }
    }

    return xi;
}

/**
 * @brief Map a reference coordinate through the straight-sided entity.
 *
 * The spherigon reconstructs a curved surface from a faceted one, so it is
 * defined entirely by the vertices: any curvature the entity already carries
 * is ignored, exactly as it was when this module built a linear element of
 * its own to sample. That makes the mapping the element's linear shape
 * functions, which is both cheaper and more accurate than interpolating a
 * quadrature representation of them.
 */
Vec3 StraightSidedMap(LibUtilities::ShapeType shape, const vector<Vec3> &v,
                      const std::array<NekDouble, 2> &xi)
{
    switch (shape)
    {
        case LibUtilities::eSegment:
            return v[0] * (0.5 * (1.0 - xi[0])) + v[1] * (0.5 * (1.0 + xi[0]));
        case LibUtilities::eTriangle:
            return v[0] * (-0.5 * (xi[0] + xi[1])) +
                   v[1] * (0.5 * (1.0 + xi[0])) + v[2] * (0.5 * (1.0 + xi[1]));
        default:
            return v[0] * (0.25 * (1.0 - xi[0]) * (1.0 - xi[1])) +
                   v[1] * (0.25 * (1.0 + xi[0]) * (1.0 - xi[1])) +
                   v[2] * (0.25 * (1.0 + xi[0]) * (1.0 + xi[1])) +
                   v[3] * (0.25 * (1.0 - xi[0]) * (1.0 + xi[1]));
    }
}

} // namespace

/**
 * @class ProcessSpherigon
 *
 * This class implements the spherigon surface smoothing technique which
 * is documented in
 *
 *   "The SPHERIGON: A Simple Polygon Patch for Smoothing Quickly your
 *   Polygonal Meshes": P. Volino and N. Magnenat Thalmann, Computer
 *   Animation Proceedings (1998).
 *
 * This implementation works in both a 2D manifold setting (for
 * triangles and quadrilaterals embedded in 3-space) and in a full 3D
 * enviroment (prisms, tetrahedra and hexahedra).
 *
 * The surfaces to smooth are selected with the @c surf option. An input
 * module that knows better -- a Nektar .rea file marks individual element
 * sides as spherigon sides -- may instead leave a SpherigonSurfs hint in the
 * mesh's ModuleContext, which is used when @c surf is not given.
 *
 * The algorithm assumes normals are supplied which are perpendicular to the
 * true surface at each vertex. Those may come from an input module as a
 * VertexNormals hint, or from a separate ply file via @c usenormalfile;
 * failing both, they are approximated by averaging the normals of all the
 * edges or faces meeting at the vertex, which smooths less well.
 */

/**
 * @brief Default constructor.
 */
ProcessSpherigon::ProcessSpherigon(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["N"] =
        ConfigOption(false, "5", "Number of points to add to face edges.");
    m_config["surf"] =
        ConfigOption(false, "-1", "Tag identifying surface to process.");
    m_config["BothTriFacesOnPrism"] = ConfigOption(
        true, "-1", "Curve both triangular faces of prism on boundary.");
    m_config["usenormalfile"] = ConfigOption(
        false, "NoFile", "Use alternative file for Spherigon definition");
    m_config["scalefile"] = ConfigOption(
        false, "1.0", "Apply scaling factor to coordinates in file ");
    m_config["normalnoise"] =
        ConfigOption(false, "NotSpecified",
                     "Add randowm noise to normals of amplitude AMP "
                     "in specified region. input string is "
                     "Amp,xmin,xmax,ymin,ymax,zmin,zmax");
}

/**
 * @brief Destructor.
 */
ProcessSpherigon::~ProcessSpherigon()
{
}

void ProcessSpherigon::FindNormalFromPlyFile(
    MeshSharedPtr &plymesh,
    const vector<SpatialDomains::PointGeom *> &surfverts)
{
    typedef bg::model::point<NekDouble, 3, bg::cs::cartesian> Point;
    typedef pair<Point, unsigned int> PointI;

    const int n_neighbs = 1;

    // The ply mesh has its own context, so its normals are entirely separate
    // from the ones being assembled for the mesh under construction.
    VertexNormals &plyNormals = plymesh->GetContext().Get<VertexNormals>();

    vector<PointI> dataPts;
    vector<SpatialDomains::PointGeom *> treeIdToPlyVert;

    unsigned int j = 0;
    for (auto &[id, pt] :
         plymesh->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        boost::ignore_unused(id);
        dataPts.push_back(make_pair(Point((*pt)[0], (*pt)[1], (*pt)[2]), j++));
        treeIdToPlyVert.push_back(pt);
    }

    bgi::rtree<PointI, bgi::rstar<16>> rtree;
    rtree.insert(dataPts.begin(), dataPts.end());

    int cnt = 0, prog = 0;
    for (auto *v : surfverts)
    {
        m_log(VERBOSE).Progress(cnt++, surfverts.size(), "Nearest ply verts",
                                prog);

        Point queryPt((*v)[0], (*v)[1], (*v)[2]);
        vector<PointI> result;
        rtree.query(bgi::nearest(queryPt, n_neighbs),
                    std::back_inserter(result));

        auto *plyVert = treeIdToPlyVert[result[0].second];
        auto nIt      = plyNormals.normals.find(plyVert);

        ASSERTL1(nIt != plyNormals.normals.end(),
                 "Nearest ply vertex has no normal");

        m_normals.normals[v] = nIt->second;
    }

    m_log(VERBOSE).Newline();
}

/**
 * @brief Generate a set of approximate vertex normals to a surface
 * represented by line segments in 2D and a hybrid
 * triangular/quadrilateral mesh in 3D.
 *
 * This routine approximates the true vertex normals to a surface by
 * averaging the normals of all edges/faces which connect to the
 * vertex. It is better to use the exact surface normals where an input module
 * or a ply file can supply them, but where they are not available this gives
 * the spherigon implementation something to work with.
 *
 * @param el        Entities denoting the surface mesh.
 * @param spaceDim  Dimension of the space the surface is embedded in.
 * @param out       Normals, accumulated into any already present.
 */
void ProcessSpherigon::GenerateNormals(
    const vector<SpatialDomains::Geometry *> &el, int spaceDim,
    VertexNormals &out)
{
    // Facet normals only average sensibly once they agree on a side.
    auto flipped = ConsistentWinding(el);

    for (auto *e : el)
    {
        Vec3 n = FacetNormal(e, spaceDim);

        if (flipped.find(e) != flipped.end())
        {
            n = n * -1.0;
        }

        // Insert face normal into vertex normal list or add to existing
        // value.
        for (int j = 0; j < e->GetNumVerts(); ++j)
        {
            auto *v = e->GetVertex(j);
            auto it = out.normals.find(v);

            if (it == out.normals.end())
            {
                out.normals[v] = n.arr();
            }
            else
            {
                it->second = (Vec3(it->second) + n).arr();
            }
        }
    }

    // Normalize resulting vectors.
    for (auto &[v, nrm] : out.normals)
    {
        boost::ignore_unused(v);
        nrm = Vec3(nrm).unit().arr();
    }
}

/**
 * @brief Perform the spherigon smoothing technique on the mesh.
 */
void ProcessSpherigon::Process()
{
    m_log(VERBOSE) << "Smoothing mesh with spherigons." << endl;

    auto &graph        = m_mesh->m_meshGraph;
    const int spaceDim = graph->GetSpaceDimension();
    const int expDim   = graph->GetMeshDimension();
    ModuleContext &ctx = m_mesh->GetContext();

    if (spaceDim != 3 && spaceDim != 2)
    {
        m_log(FATAL) << "Spherigon implementation only valid in 2D/3D." << endl;
    }

    // First construct the list of entities to smooth. These are real
    // geometries in the graph -- the faces or edges themselves -- where the
    // module used to have to synthesise an Element to stand in for each one.
    vector<SpatialDomains::Geometry *> el;

    if (expDim == 2 && spaceDim == 3)
    {
        // Manifold case: the elements are the surface.
        for (auto &[geom, tag] : m_mesh->m_elementTags[2])
        {
            boost::ignore_unused(tag);
            el.push_back(geom);
        }
    }
    else if (expDim == spaceDim)
    {
        const bool prismTag = m_config["BothTriFacesOnPrism"].beenSet;

        // The boundary entity carrying a tag is the face or edge itself, so
        // the tagged entities one dimension down are exactly what
        // Element::GetBoundaryLink() used to have to look up.
        auto &bndTags = m_mesh->m_elementTags[expDim - 1];

        if (m_config["surf"].beenSet)
        {
            vector<unsigned int> surfTags;
            ParseUtils::GenerateSeqVector(m_config["surf"].as<string>(),
                                          surfTags);
            set<int> surfs(surfTags.begin(), surfTags.end());

            for (auto &[elmt, tag] : m_mesh->m_elementTags[expDim])
            {
                boost::ignore_unused(tag);

                const int nSurf =
                    expDim == 3 ? elmt->GetNumFaces() : elmt->GetNumEdges();

                for (int j = 0; j < nSurf; ++j)
                {
                    auto *bnd = expDim == 3
                                    ? static_cast<SpatialDomains::Geometry *>(
                                          elmt->GetFace(j))
                                    : static_cast<SpatialDomains::Geometry *>(
                                          elmt->GetEdge(j));

                    auto blIt = bndTags.find(bnd);
                    if (blIt == bndTags.end() ||
                        surfs.find(blIt->second) == surfs.end())
                    {
                        continue;
                    }

                    el.push_back(bnd);

                    // Curve other tri face on Prism. Note could be
                    // problem on pyramid when implemented.
                    if (nSurf == 5 && prismTag)
                    {
                        el.push_back(elmt->GetFace(j == 1 ? 3 : 1));
                    }
                }
            }
        }
        else if (auto *hint = ctx.TryGet<SpherigonSurfs>())
        {
            // An input module has nominated the sides itself.
            el = hint->surfs;
        }
    }
    else
    {
        m_log(FATAL) << "Spherigon expansions must be 2/3 dimensional." << endl;
    }

    // m_elementTags is unordered, so fix an order: an edge shared by two
    // smoothed faces takes its curvature from whichever face reaches it
    // first, and that choice must not depend on the hash layout -- pointer
    // keys would otherwise make the output differ from run to run. The same
    // pass drops the duplicates that BothTriFacesOnPrism can introduce.
    sort(el.begin(), el.end(),
         [](SpatialDomains::Geometry *a, SpatialDomains::Geometry *b) {
             return make_pair(static_cast<int>(a->GetShapeType()),
                              a->GetGlobalID()) <
                    make_pair(static_cast<int>(b->GetShapeType()),
                              b->GetGlobalID());
         });
    el.erase(unique(el.begin(), el.end()), el.end());

    if (el.size() == 0)
    {
        m_log(WARNING) << "Spherigon surfaces have not been defined "
                       << "-- ignoring smoothing." << endl;
        return;
    }

    for (auto *e : el)
    {
        LibUtilities::ShapeType s = e->GetShapeType();
        if (s != LibUtilities::eSegment && s != LibUtilities::eTriangle &&
            s != LibUtilities::eQuadrilateral)
        {
            m_log(FATAL) << "Spherigon expansions must be lines, triangles or "
                         << "quadrilaterals." << endl;
        }
    }

    // Vertices of the surface, in the order the entities present them.
    vector<SpatialDomains::PointGeom *> surfverts;
    {
        unordered_set<SpatialDomains::PointGeom *> seen;
        for (auto *e : el)
        {
            for (int j = 0; j < e->GetNumVerts(); ++j)
            {
                if (seen.insert(e->GetVertex(j)).second)
                {
                    surfverts.push_back(e->GetVertex(j));
                }
            }
        }
    }

    // Establish the vertex normals, in decreasing order of fidelity: a ply
    // file named on the command line, then a hint left by an input module,
    // then an approximation from the surface itself.
    std::string normalfile = m_config["usenormalfile"].as<string>();

    if (normalfile.compare("NoFile") != 0)
    {
        NekDouble scale = m_config["scalefile"].as<NekDouble>();

        m_log(VERBOSE) << " - Using normal file: '" << normalfile
                       << "' with scaling of " << scale << endl;

        ifstream inplyTmp;
        io::filtering_istream inply;

        inplyTmp.open(normalfile.c_str());
        if (!inplyTmp)
        {
            m_log(FATAL) << "Could not open input ply file: '" << normalfile
                         << "'." << endl;
        }

        inply.push(inplyTmp);

        MeshSharedPtr m           = std::make_shared<Mesh>();
        InputPlySharedPtr plyfile = std::make_shared<InputPly>(m);
        plyfile->ReadPly(inply, scale);
        plyfile->ProcessVertices();

        MeshSharedPtr plymesh = plyfile->GetMesh();

        // Any normals the ply file itself carried are already in its context;
        // fill the gaps by approximating from its own facets.
        vector<SpatialDomains::Geometry *> plyEl;
        for (auto &[geom, tag] : plymesh->m_elementTags[2])
        {
            boost::ignore_unused(tag);
            plyEl.push_back(geom);
        }

        m_log(VERBOSE) << " - Generating ply normals." << endl;
        GenerateNormals(plyEl, 3, plymesh->GetContext().Get<VertexNormals>());

        m_log(VERBOSE) << " - Processing surface normals." << endl;
        FindNormalFromPlyFile(plymesh, surfverts);
    }
    else if (auto *hint = ctx.TryGet<VertexNormals>();
             hint != nullptr && !hint->normals.empty())
    {
        // Exact normals from the input file: better smoothing than anything
        // that can be recovered from the faceted surface.
        m_normals = *hint;
    }
    else
    {
        GenerateNormals(el, spaceDim, m_normals);
    }

    // See if we should add noise to normals
    std::string normalnoise = m_config["normalnoise"].as<string>();
    if (normalnoise.compare("NotSpecified") != 0)
    {
        vector<NekDouble> values;
        if (!ParseUtils::GenerateVector(normalnoise, values))
        {
            m_log(FATAL) << "Failed to interpret normal noise string" << endl;
        }

        int nvalues   = values.size() / 2;
        NekDouble amp = values[0];

        m_log(VERBOSE) << " - Adding noise to normals of amplitude: " << amp
                       << " in range: ";
        for (int i = 0; i < nvalues; ++i)
        {
            m_log(VERBOSE) << values[2 * i + 1] << "," << values[2 * i + 2]
                           << " ";
        }
        m_log(VERBOSE) << endl;

        for (auto *v : surfverts)
        {
            bool AddNoise = false;

            for (int i = 0; i < nvalues; ++i)
            {
                // check to see if point is in range
                switch (spaceDim)
                {
                    case 1:
                        if (((*v)[0] > values[2 * i + 1]) &&
                            ((*v)[0] < values[2 * i + 2]))
                        {
                            AddNoise = true;
                        }
                        break;
                    case 2:
                        if (((*v)[0] > values[2 * i + 1]) &&
                            ((*v)[0] < values[2 * i + 2]) &&
                            ((*v)[1] > values[2 * i + 3]) &&
                            ((*v)[1] < values[2 * i + 4]))
                        {
                            AddNoise = true;
                        }
                        break;
                    case 3:
                        if (((*v)[0] > values[2 * i + 1]) &&
                            ((*v)[0] < values[2 * i + 2]) &&
                            ((*v)[1] > values[2 * i + 3]) &&
                            ((*v)[1] < values[2 * i + 4]) &&
                            ((*v)[2] > values[2 * i + 5]) &&
                            ((*v)[2] < values[2 * i + 6]))
                        {
                            AddNoise = true;
                        }
                        break;
                }
            }

            if (AddNoise)
            {
                auto nIt = m_normals.normals.find(v);
                if (nIt == m_normals.normals.end())
                {
                    continue;
                }

                // generate random unit vector;
                Vec3 rvec(rand(), rand(), rand());
                rvec = rvec.unit() * values[0];

                nIt->second = (Vec3(nIt->second) + rvec).unit().arr();
            }
        }
    }

    const int nq = m_config["N"].as<int>();

    if (nq <= 2)
    {
        m_log(FATAL) << "Number of points in resulting high-order element must "
                     << "be greater than 2." << endl;
    }

    // Reference coordinates of the curve points, cached per shape.
    map<LibUtilities::ShapeType, vector<std::array<NekDouble, 2>>> xiCache;
    for (auto s : {LibUtilities::eSegment, LibUtilities::eTriangle,
                   LibUtilities::eQuadrilateral})
    {
        xiCache[s] = CurvePointCoords(s, nq);
    }

    // Smoothed positions of every curve point of every entity, computed
    // before anything is written so that edges shared between entities can be
    // made to agree afterwards.
    vector<vector<Vec3>> smoothed(el.size());

    for (size_t i = 0; i < el.size(); ++i)
    {
        SpatialDomains::Geometry *e   = el[i];
        LibUtilities::ShapeType shape = e->GetShapeType();

        const int nV    = e->GetNumVerts();
        const int nquad = NumCurvePoints(shape, nq);
        const auto &xi  = xiCache[shape];

        // Find vertices and their normals.
        vector<Vec3> v(nV), vN(nV);
        for (int j = 0; j < nV; ++j)
        {
            auto *vert = e->GetVertex(j);
            v[j]       = Vec3(vert);

            auto nIt = m_normals.normals.find(vert);
            ASSERTL1(nIt != m_normals.normals.end(),
                     "Normal has not been defined");
            vN[j] = Vec3(nIt->second);
        }

        vector<Vec3> tmp(nV), K(nV), Q(nV), Qp(nV);
        vector<NekDouble> r(nV), blend(nV);
        vector<Vec3> &out = smoothed[i];
        out.resize(nquad);

        // Calculate segment length for 2D spherigon routine.
        NekDouble segLength = sqrt((v[0] - v[1]).abs2());

        // Perform Spherigon method to smooth manifold.
        for (int j = 0; j < nquad; ++j)
        {
            Vec3 P = StraightSidedMap(shape, v, xi[j]);
            Vec3 N;

            // Calculate generalised barycentric coordinates r[] and the
            // Phong normal N = vN . r for this point of the element.
            if (spaceDim == 2)
            {
                // In 2D the coordinates are given by a ratio of the
                // segment length to the distance from one of the
                // endpoints.
                r[0] = sqrt((P - v[0]).abs2()) / segLength;
                r[0] = max(min(1.0, r[0]), 0.0);
                r[1] = 1.0 - r[0];

                // Calculate Phong normal.
                N = vN[0] * r[0] + vN[1] * r[1];
            }
            else
            {
                for (int k = 0; k < nV; ++k)
                {
                    tmp[k] = P - v[k];
                }

                // Calculate generalized barycentric coordinate system
                // (see equation 6 of paper).
                NekDouble weight = 0.0;
                for (int k = 0; k < nV; ++k)
                {
                    r[k] = 1.0;
                    for (int l = 0; l < nV - 2; ++l)
                    {
                        r[k] *= CrossProdMag(tmp[(k + l + 1) % nV],
                                             tmp[(k + l + 2) % nV]);
                    }
                    weight += r[k];
                }

                // Calculate Phong normal (equation 1).
                for (int k = 0; k < nV; ++k)
                {
                    r[k] /= weight;
                    N += vN[k] * r[k];
                }
            }

            // Normalise Phong normal.
            N = N.unit();

            for (int k = 0; k < nV; ++k)
            {
                // Perform steps denoted in equations 2, 3, 8 for C1
                // smoothing.
                NekDouble tmp1;
                K[k]  = P + N * ((v[k] - P).dot(N));
                tmp1  = (v[k] - K[k]).dot(vN[k]) / (1.0 + N.dot(vN[k]));
                Q[k]  = K[k] + N * tmp1;
                Qp[k] = v[k] - N * ((v[k] - P).dot(N));
            }

            // Apply C1 blending function to the surface. TODO: Add
            // option to do (more efficient) C0 blending function.
            SuperBlend(r, Qp, P, blend);
            P = Vec3();

            // Apply blending (equation 4).
            for (int k = 0; k < nV; ++k)
            {
                P += Q[k] * blend[k];
            }

            if ((boost::math::isnan)(P.x) || (boost::math::isnan)(P.y) ||
                (boost::math::isnan)(P.z))
            {
                m_log(FATAL) << "Found a spherigon point with a NaN value. "
                             << "Check to see if ply file is correct if using "
                             << "input normal file" << endl;
            }

            // Zero z-coordinate in 2D.
            if (spaceDim == 2)
            {
                P.z = 0.0;
            }

            out[j] = P;
        }
    }

    // Write the results into the graph, edges before faces so that a face
    // curve can take its boundary nodes from the edge curves rather than
    // recomputing points that must agree to the last bit.
    const int coordim = spaceDim;
    auto newNode      = [&](const Vec3 &p) {
        auto pt = ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
            coordim, 0, p.x, p.y, p.z);
        auto *raw = pt.get();
        graph->GetAllCurveNodes().push_back(std::move(pt));
        return raw;
    };

    unordered_set<SpatialDomains::Geometry *> touchedEdges, touchedFaces;

    for (size_t i = 0; i < el.size(); ++i)
    {
        SpatialDomains::Geometry *e   = el[i];
        LibUtilities::ShapeType shape = e->GetShapeType();
        const vector<Vec3> &out       = smoothed[i];

        if (shape == LibUtilities::eSegment)
        {
            // The entity is the edge; its curve is the smoothed points
            // directly, with the real vertices at either end.
            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    e->GetGlobalID(), LibUtilities::eGaussLobattoLegendre);

            curve->m_points.resize(nq);
            curve->m_points[0]      = e->GetVertex(0);
            curve->m_points[nq - 1] = e->GetVertex(1);
            for (int j = 1; j < nq - 1; ++j)
            {
                curve->m_points[j] = newNode(out[j]);
            }

            static_cast<SpatialDomains::SegGeom *>(e)->SetCurve(curve.get());
            graph->AddCurvedEdge(std::move(curve));
            touchedEdges.insert(e);
            continue;
        }

        for (int edge = 0; edge < e->GetNumEdges(); ++edge)
        {
            auto *seg = e->GetEdge(edge);

            if (touchedEdges.find(seg) != touchedEdges.end())
            {
                continue;
            }

            vector<int> idx = LocalEdgeCurveIndices(shape, edge, nq);

            // The local edge runs from vertex `edge` to the next one round;
            // the SegGeom may be stored the other way about.
            const bool reverse = seg->GetVertex(0) != e->GetVertex(edge);

            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    seg->GetGlobalID(), LibUtilities::eGaussLobattoLegendre);

            curve->m_points.resize(nq);
            for (int k = 0; k < nq; ++k)
            {
                const int slot = reverse ? nq - 1 - k : k;
                curve->m_points[slot] =
                    (k == 0 || k == nq - 1)
                        ? e->GetVertex((edge + (k == 0 ? 0 : 1)) %
                                       e->GetNumVerts())
                        : newNode(out[idx[k]]);
            }

            static_cast<SpatialDomains::SegGeom *>(seg)->SetCurve(curve.get());
            graph->AddCurvedEdge(std::move(curve));
            touchedEdges.insert(seg);
        }
    }

    for (size_t i = 0; i < el.size(); ++i)
    {
        SpatialDomains::Geometry *e   = el[i];
        LibUtilities::ShapeType shape = e->GetShapeType();

        if (shape == LibUtilities::eSegment)
        {
            continue;
        }

        const vector<Vec3> &out = smoothed[i];
        const int nquad         = NumCurvePoints(shape, nq);

        auto curve = ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
            e->GetGlobalID(), shape == LibUtilities::eTriangle
                                  ? LibUtilities::eNodalTriElec
                                  : LibUtilities::eGaussLobattoLegendre);

        curve->m_points.assign(nquad, nullptr);

        // Vertices, then each edge's nodes read back out of the edge curve so
        // that neighbouring entities share the identical points.
        for (int j = 0; j < e->GetNumVerts(); ++j)
        {
            curve->m_points[shape == LibUtilities::eTriangle
                                ? j
                                : LocalEdgeCurveIndices(shape, j, nq)[0]] =
                e->GetVertex(j);
        }

        for (int edge = 0; edge < e->GetNumEdges(); ++edge)
        {
            auto *seg                     = e->GetEdge(edge);
            SpatialDomains::Curve *eCurve = seg->GetCurve();
            ASSERTL1(eCurve != nullptr, "Edge curve was not written");

            vector<int> idx    = LocalEdgeCurveIndices(shape, edge, nq);
            const bool reverse = seg->GetVertex(0) != e->GetVertex(edge);

            for (int k = 1; k < nq - 1; ++k)
            {
                curve->m_points[idx[k]] =
                    eCurve->m_points[reverse ? nq - 1 - k : k];
            }
        }

        // Whatever is left is interior to this entity alone.
        for (int j = 0; j < nquad; ++j)
        {
            if (curve->m_points[j] == nullptr)
            {
                curve->m_points[j] = newNode(out[j]);
            }
        }

        static_cast<SpatialDomains::Geometry2D *>(e)->SetCurve(curve.get());
        graph->AddCurvedFace(std::move(curve));
        touchedFaces.insert(e);
    }

    // The curves changed the number of points on these entities, so drop the
    // cached xmap and coefficients of everything that reads them: the
    // entities themselves, and any element with one on its boundary.
    auto &curvedEdges = graph->GetCurvedEdges();
    auto &curvedFaces = graph->GetCurvedFaces();

    for (auto *seg : touchedEdges)
    {
        seg->Reset(curvedEdges, curvedFaces);
    }
    for (auto *face : touchedFaces)
    {
        face->Reset(curvedEdges, curvedFaces);
    }

    for (auto &[elmt, tag] : m_mesh->m_elementTags[expDim])
    {
        boost::ignore_unused(tag);

        bool affected = false;

        for (int j = 0; j < elmt->GetNumEdges() && !affected; ++j)
        {
            affected = touchedEdges.count(elmt->GetEdge(j)) > 0;
        }
        for (int j = 0; j < elmt->GetNumFaces() && !affected; ++j)
        {
            affected = touchedFaces.count(elmt->GetFace(j)) > 0;
        }

        if (affected && touchedEdges.count(elmt) == 0 &&
            touchedFaces.count(elmt) == 0)
        {
            elmt->Reset(curvedEdges, curvedFaces);
        }
    }
}
} // namespace Nektar::NekMesh
