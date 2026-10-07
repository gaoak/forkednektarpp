////////////////////////////////////////////////////////////////////////////////
//
//  File: PreProcessing.cpp
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

#include "ProcessVarOpti.h"

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/NodalUtil.h>

#include <LibUtilities/BasicUtils/ParseUtils.h>

#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace Nektar::NekMesh
{

namespace
{

/**
 * @brief Every node belonging to @p g: the nodes of its curve if it carries
 * one, and otherwise just its vertices.
 */
std::vector<SpatialDomains::PointGeom *> AllNodes(SpatialDomains::Geometry *g)
{
    SpatialDomains::Curve *curve = g->GetCurve();

    if (curve != nullptr && !curve->m_points.empty())
    {
        return curve->m_points;
    }

    std::vector<SpatialDomains::PointGeom *> verts(g->GetNumVerts());
    for (int i = 0; i < g->GetNumVerts(); ++i)
    {
        verts[i] = g->GetVertex(i);
    }
    return verts;
}

/**
 * @brief The nodes strictly interior to @p g.
 *
 * Taken as every node of @p g less those belonging to its vertices, edges and
 * faces, rather than by indexing into the curve. That keeps this independent
 * of the nodal orderings, which differ between shapes.
 */
std::vector<SpatialDomains::PointGeom *> InteriorNodes(
    SpatialDomains::Geometry *g)
{
    std::unordered_set<SpatialDomains::PointGeom *> onBoundary;

    for (int i = 0; i < g->GetNumVerts(); ++i)
    {
        onBoundary.insert(g->GetVertex(i));
    }
    for (int i = 0; i < g->GetNumEdges(); ++i)
    {
        for (auto *n : AllNodes(g->GetEdge(i)))
        {
            onBoundary.insert(n);
        }
    }
    for (int i = 0; i < g->GetNumFaces(); ++i)
    {
        for (auto *n : AllNodes(g->GetFace(i)))
        {
            onBoundary.insert(n);
        }
    }

    std::vector<SpatialDomains::PointGeom *> interior;
    for (auto *n : AllNodes(g))
    {
        if (onBoundary.find(n) == onBoundary.end())
        {
            interior.push_back(n);
        }
    }
    return interior;
}

/**
 * @brief Map each facet (edge in 2D, face in 3D) of the mesh dimension's
 * elements to the elements either side of it. A facet reached by only one
 * element lies on the boundary of the mesh.
 */
std::unordered_map<SpatialDomains::Geometry *,
                   std::vector<SpatialDomains::Geometry *>>
BuildFacetElementMap(MeshSharedPtr mesh, int meshDim)
{
    std::unordered_map<SpatialDomains::Geometry *,
                       std::vector<SpatialDomains::Geometry *>>
        facetMap;

    for (auto &[geom, tag] : mesh->m_elementTags[meshDim])
    {
        const int nFacets =
            (meshDim == 3) ? geom->GetNumFaces() : geom->GetNumEdges();

        for (int f = 0; f < nFacets; ++f)
        {
            SpatialDomains::Geometry *facet =
                (meshDim == 3)
                    ? static_cast<SpatialDomains::Geometry *>(geom->GetFace(f))
                    : static_cast<SpatialDomains::Geometry *>(geom->GetEdge(f));
            facetMap[facet].push_back(geom);
        }
    }

    return facetMap;
}

} // namespace

map<LibUtilities::ShapeType, DerivUtilSharedPtr> BuildDerivUtil(int nummode,
                                                                int o)
{
    // build Vandermonde information
    map<LibUtilities::ShapeType, DerivUtilSharedPtr> ret;

    // Typedef for points types used in the variational optimser. First entry is
    // the evaluation points; second entry is the distributions used for the
    // full element (includes element boundary)
    typedef std::pair<LibUtilities::PointsType, LibUtilities::PointsType>
        PTypes;

    map<LibUtilities::ShapeType, PTypes> typeMap;

    if (nummode + o <= 11)
    {
        typeMap[LibUtilities::eTriangle] =
            PTypes(LibUtilities::eNodalTriSPI, LibUtilities::eNodalTriElec);
        typeMap[LibUtilities::eTetrahedron] =
            PTypes(LibUtilities::eNodalTetSPI, LibUtilities::eNodalTetElec);
        typeMap[LibUtilities::ePrism] =
            PTypes(LibUtilities::eNodalPrismSPI, LibUtilities::eNodalPrismElec);
    }

    // Quadrilaterals and hexahedra are built from Gauss-Lobatto points, which
    // are available at any order, so unlike the shapes above they are not
    // limited by the tabulated symmetric rules.
    typeMap[LibUtilities::eQuadrilateral] =
        PTypes(LibUtilities::eNodalQuadElec, LibUtilities::eNodalQuadElec);
    typeMap[LibUtilities::eHexahedron] =
        PTypes(LibUtilities::eNodalHexElec, LibUtilities::eNodalHexElec);

    for (auto &it : typeMap)
    {
        PTypes pType           = it.second;
        DerivUtilSharedPtr der = std::shared_ptr<DerivUtil>(new DerivUtil());

        LibUtilities::PointsKey pkey1(nummode, pType.second);
        LibUtilities::PointsKey pkey2(nummode + o, pType.first);

        const int pDim  = pkey1.GetPointsDim();
        const int order = nummode - 1;

        Array<OneD, Array<OneD, NekDouble>> u1(pDim), u2(pDim);

        switch (pDim)
        {
            case 2:
            {
                LibUtilities::PointsManager()[pkey1]->GetPoints(u1[0], u1[1]);
                LibUtilities::PointsManager()[pkey2]->GetPoints(u2[0], u2[1]);
                break;
            }
            case 3:
            {
                LibUtilities::PointsManager()[pkey1]->GetPoints(u1[0], u1[1],
                                                                u1[2]);
                LibUtilities::PointsManager()[pkey2]->GetPoints(u2[0], u2[1],
                                                                u2[2]);
                break;
            }
        }

        der->ptsStd = u1[0].size();
        der->pts    = u2[0].size();

        LibUtilities::NodalUtil *nodalUtil = nullptr;

        if (it.first == LibUtilities::eTriangle)
        {
            nodalUtil =
                new LibUtilities::NodalUtilTriangle(order, u1[0], u1[1]);
        }
        else if (it.first == LibUtilities::eQuadrilateral)
        {
            nodalUtil = new LibUtilities::NodalUtilQuad(order, u1[0], u1[1]);
        }
        else if (it.first == LibUtilities::eTetrahedron)
        {
            nodalUtil = new LibUtilities::NodalUtilTetrahedron(order, u1[0],
                                                               u1[1], u1[2]);
        }
        else if (it.first == LibUtilities::ePrism)
        {
            nodalUtil =
                new LibUtilities::NodalUtilPrism(order, u1[0], u1[1], u1[2]);
        }
        else if (it.first == LibUtilities::eHexahedron)
        {
            nodalUtil =
                new LibUtilities::NodalUtilHex(order, u1[0], u1[1], u1[2]);
        }
        else
        {
            ASSERTL0(false,
                     "Unknown element type for derivative utility setup");
        }

        NekMatrix<NekDouble> interp = *nodalUtil->GetInterpolationMatrix(u2);
        NekMatrix<NekDouble> Vandermonde  = *nodalUtil->GetVandermonde();
        NekMatrix<NekDouble> VandermondeI = Vandermonde;
        VandermondeI.Invert();

        for (int i = 0; i < pDim; ++i)
        {
            der->VdmDStd[i] =
                *nodalUtil->GetVandermondeForDeriv(i) * VandermondeI;
            der->VdmD[i] = interp * der->VdmDStd[i];
        }

        Array<OneD, NekDouble> qds =
            LibUtilities::PointsManager()[pkey2]->GetW();
        NekVector<NekDouble> quadWi(qds);
        der->quadW = quadWi;

        ret[it.first] = der;
        delete nodalUtil;
    }

    return ret;
}

struct NodeComparator
{
    const vector<int> &value_vector;

    NodeComparator(const vector<int> &val_vec) : value_vector(val_vec)
    {
    }

    bool operator()(int i1, int i2)
    {
        return value_vector[i1] > value_vector[i2];
    }
};

vector<vector<SpatialDomains::PointGeom *>> ProcessVarOpti::GetColouredNodes(
    vector<ElUtilSharedPtr> elLock)
{
    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    // create set of nodes to be ignored and hence not included in the
    // coloursets
    std::unordered_set<SpatialDomains::PointGeom *> ignoredNodes;
    for (int i = 0; i < elLock.size(); i++)
    {
        for (auto *n : GetCurvedNodes(elLock[i]->GetEl()))
        {
            ignoredNodes.insert(n);
        }
    }

    // Null when the mesh has no CAD attached. Fetched once: asking the graph
    // for it creates it.
    SpatialDomains::CADAssociationSharedPtr cad =
        m_mesh->m_meshGraph->HasCAD() ? m_mesh->m_meshGraph->GetCADAssociation()
                                      : nullptr;

    // Whether a node on the boundary is free to move along the CAD it belongs
    // to. Without CAD nothing says where a boundary node could go, so it
    // stays put; with CAD it is instead confined to its curve or surface and
    // may slide within it, which is the only way a surface mesh that is
    // itself invalid can ever be repaired. A node on a CAD vertex is where
    // two curves meet and has nowhere to go.
    auto slidesOnCAD = [&cad](SpatialDomains::PointGeom *n) {
        if (cad == nullptr || cad->GetVert(n) != nullptr)
        {
            return false;
        }

        return cad->Has(n, SpatialDomains::CADType::eCurve) ||
               cad->Has(n, SpatialDomains::CADType::eSurf);
    };

    // The nodes on the boundary of the mesh that are not free to move. A
    // facet reached by only one element lies on that boundary.
    std::unordered_set<SpatialDomains::PointGeom *> boundaryNodes;

    for (auto &[facet, elmts] : BuildFacetElementMap(m_mesh, meshDim))
    {
        if (elmts.size() == 2)
        {
            continue;
        }

        for (auto *n : AllNodes(facet))
        {
            if (!slidesOnCAD(n))
            {
                boundaryNodes.insert(n);
            }
        }
    }

    // Free nodes, split into the tiers the colouring below works through:
    // nodes on element vertices and edges, then nodes interior to a face,
    // then nodes interior to a volume.
    vector<SpatialDomains::PointGeom *> remainEdgeVertex;
    vector<SpatialDomains::PointGeom *> remainFace;
    vector<SpatialDomains::PointGeom *> remainVolume;
    m_res->nDoF   = 0;
    m_res->nOnCAD = 0;

    const int spaceDim = m_mesh->m_meshGraph->GetSpaceDimension();

    int orphans = 0;

    auto consider = [&](SpatialDomains::PointGeom *n,
                        vector<SpatialDomains::PointGeom *> &remain) {
        if (boundaryNodes.find(n) != boundaryNodes.end() ||
            ignoredNodes.find(n) != ignoredNodes.end())
        {
            return;
        }

        // The candidates are gathered from the mesh graph, which can hold
        // nodes that no element of the mesh dimension uses -- a surface mesh
        // the volume mesher did not take up, say, of which a boundary layer
        // case can leave thousands. There is no energy attached to such a
        // node and nothing to optimise, and reaching for the elements around
        // it is what used to end the run.
        if (m_nodeElMap.find(n) == m_nodeElMap.end())
        {
            ++orphans;
            return;
        }

        remain.push_back(n);

        // A node confined to a CAD curve has one degree of freedom and one
        // confined to a surface two; everything else moves in the full space.
        // The order of the two tests matches the one Process() uses to decide
        // which optimiser a node gets.
        const bool onCurve =
            cad != nullptr && cad->Has(n, SpatialDomains::CADType::eCurve);
        const bool onSurf = !onCurve && cad != nullptr &&
                            cad->Has(n, SpatialDomains::CADType::eSurf);

        m_res->nOnCAD += (onCurve || onSurf) ? 1 : 0;
        m_res->nDoF += onCurve ? 1 : onSurf ? 2 : spaceDim;
    };

    for (auto &[id, vert] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        consider(vert, remainEdgeVertex);
    }

    for (auto &[id, edge] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
    {
        for (auto *n : InteriorNodes(edge))
        {
            consider(n, remainEdgeVertex);
        }
    }

    for (auto &[id, face] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        for (auto *n : InteriorNodes(face))
        {
            consider(n, remainFace);
        }
    }
    for (auto &[id, face] :
         m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>())
    {
        for (auto *n : InteriorNodes(face))
        {
            consider(n, remainFace);
        }
    }

    if (meshDim == 3)
    {
        for (auto &[geom, tag] : m_mesh->m_elementTags[3])
        {
            for (auto *n : InteriorNodes(geom))
            {
                consider(n, remainVolume);
            }
        }
    }

    if (orphans)
    {
        m_log(VERBOSE) << "  - Nodes in no element: " << orphans
                       << " (left alone)" << endl;
    }

    // size of all free nodes to be included in the coloursets
    m_res->n =
        remainEdgeVertex.size() + remainFace.size() + remainVolume.size();

    // data structure for coloursets, that will ultimately contain all free
    // nodes
    vector<vector<SpatialDomains::PointGeom *>> ret;
    vector<vector<SpatialDomains::PointGeom *>> retPart;

    // edge and vertex nodes
    // create vector num_el of number of associated elements of each node
    vector<int> num_el(remainEdgeVertex.size());
    for (int i = 0; i < remainEdgeVertex.size(); i++)
    {
        // try to find node within all elements
        auto it = m_nodeElMap.find(remainEdgeVertex[i]);
        vector<ElUtilSharedPtr> &elUtils = it->second;
        num_el[i]                        = elUtils.size();
    }
    // finding the permutation according to num_el
    vector<int> permNode(remainEdgeVertex.size());
    for (int i = 0; i < remainEdgeVertex.size(); ++i)
    {
        permNode[i] = i;
    }
    std::sort(permNode.begin(), permNode.end(), NodeComparator(num_el));
    // applying the permutation to remainEdgeVertex
    vector<SpatialDomains::PointGeom *> remainEdgeVertexSort(
        remainEdgeVertex.size());
    for (int i = 0; i < remainEdgeVertex.size(); ++i)
    {
        int j                   = permNode[i];
        remainEdgeVertexSort[i] = remainEdgeVertex[j];
    }

    retPart = CreateColoursets(remainEdgeVertexSort);

    m_log(VERBOSE).Newline();
    m_log(VERBOSE) << "  - Number of Edge/Vertex Coloursets: " << retPart.size()
                   << endl;

    for (int i = 0; i < retPart.size(); i++)
    {
        ret.push_back(retPart[i]);
    }

    // face nodes
    retPart = CreateColoursets(remainFace);
    m_log(VERBOSE) << "  - Number of Face Coloursets: " << retPart.size()
                   << endl;

    for (int i = 0; i < retPart.size(); i++)
    {
        ret.push_back(retPart[i]);
    }

    // volume nodes
    retPart = CreateColoursets(remainVolume);
    m_log(VERBOSE) << "  - Number of Volume Coloursets: " << retPart.size()
                   << endl;

    for (int i = 0; i < retPart.size(); i++)
    {
        ret.push_back(retPart[i]);
    }

    return ret;
}

/**
 * @brief Split @p remain into sets of nodes that share no element.
 *
 * Greedy: sweep the nodes in order, take one whenever none of the elements
 * around it has already been claimed for this colour, and claim its elements.
 * Whatever is left over goes round again as the next colour. Each set can then
 * be optimised in parallel, since no two of its nodes touch the same element.
 *
 * The order the nodes are swept in, and so the colouring produced, is the one
 * the caller handed over; nothing here reorders them.
 */
vector<vector<SpatialDomains::PointGeom *>> ProcessVarOpti::CreateColoursets(
    vector<SpatialDomains::PointGeom *> remain)
{
    vector<vector<SpatialDomains::PointGeom *>> retPart;

    // The elements around each node, resolved once and flattened to their
    // ids. m_nodeElMap is keyed by node pointer, so a node that is not taken
    // for a while is looked up again on every pass, and reaching an id from
    // there is three dependent loads -- shared_ptr, ElUtil, Geometry. Both
    // are most of the work on a mesh of any size.
    struct Candidate
    {
        SpatialDomains::PointGeom *node;
        /// Where this node's element ids start in elIds, and how many.
        int offset;
        int count;
    };

    vector<Candidate> left;
    vector<int> elIds;
    left.reserve(remain.size());

    int maxId = -1;
    for (auto *n : remain)
    {
        auto it = m_nodeElMap.find(n);
        ASSERTL0(it != m_nodeElMap.end(), "could not find node");

        left.push_back({n, static_cast<int>(elIds.size()),
                        static_cast<int>(it->second.size())});

        for (auto &el : it->second)
        {
            elIds.push_back(el->GetId());
            maxId = std::max(maxId, elIds.back());
        }
    }

    // Which colour each element is claimed for. Stamping with the colour
    // number rather than marking and clearing means the marks left by the
    // previous colour need not be visited again, and the test is one array
    // read where it used to be a search of a std::set.
    vector<unsigned int> claimedBy(maxId + 1, 0);
    unsigned int colour = 0;

    while (!left.empty())
    {
        ++colour;

        vector<SpatialDomains::PointGeom *> layer;
        size_t keep = 0;

        for (size_t i = 0; i < left.size(); ++i)
        {
            const int *ids = elIds.data() + left[i].offset;
            const int nIds = left[i].count;

            bool islocked = false;
            for (int j = 0; j < nIds; ++j)
            {
                if (claimedBy[ids[j]] == colour)
                {
                    islocked = true;
                    break;
                }
            }

            if (islocked)
            {
                // Still waiting for a colour; compacted in place, which keeps
                // the sweep order for the passes to come.
                left[keep++] = left[i];
                continue;
            }

            layer.push_back(left[i].node);
            for (int j = 0; j < nIds; ++j)
            {
                claimedBy[ids[j]] = colour;
            }
        }

        left.resize(keep);
        retPart.push_back(std::move(layer));

        m_log(VERBOSE).Progress(m_res->n - left.size(), m_res->n,
                                "Node Coloring");
    }

    return retPart;
}

void ProcessVarOpti::GetElementMap(
    int o, map<LibUtilities::ShapeType, DerivUtilSharedPtr> derMap)
{
    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    // Build the bounding boxes around each CAD curve when r-adaption on CAD
    // curves is set.
    m_radaptCAD = (m_config["radaptcurves"].beenSet);
    if (m_radaptCAD)
    {
        NekDouble radaptrad = (m_config["radaptrad"].as<NekDouble>() > 0.0)
                                  ? m_config["radaptrad"].as<NekDouble>() * 1.5
                                  : 0.0;
        std::vector<unsigned int> curveIds;
        ParseUtils::GenerateSeqVector(
            m_config["radaptcurves"].as<std::string>().c_str(), curveIds);

        auto cad = m_mesh->m_meshGraph->GetCAD();

        for (auto Id : curveIds)
        {
            // GetMinMax() gives the curve's two end points; widen the box
            // they span by the radius of influence. Held in the same
            // {xmin, ymin, zmin, xmax, ymax, zmax} order that
            // Geometry::GetBoundingBox() uses.
            auto locs = cad->GetCurve(Id)->GetMinMax();

            std::array<NekDouble, 6> box;
            for (int d = 0; d < 3; ++d)
            {
                box[d]     = std::min(locs[d], locs[d + 3]) - radaptrad;
                box[d + 3] = std::max(locs[d], locs[d + 3]) + radaptrad;
            }

            m_adaptCurves.push_back(std::make_pair(cad->GetCurve(Id), box));
        }
    }

    // Take the elements in a fixed order: m_dataSet is indexed by position,
    // so the second pass below has to see the same order as this one.
    std::vector<SpatialDomains::Geometry *> elmts;
    elmts.reserve(m_mesh->m_elementTags[meshDim].size());
    for (auto &[geom, tag] : m_mesh->m_elementTags[meshDim])
    {
        elmts.push_back(geom);
    }

    for (int i = 0; i < elmts.size(); i++)
    {
        SpatialDomains::Geometry *el = elmts[i];
        ElUtilSharedPtr d            = std::shared_ptr<ElUtil>(
            new ElUtil(el, derMap[el->GetShapeType()], m_res, m_nummode, o));
        m_dataSet.push_back(d);

        if (m_radaptCAD) // Initial r-adaption on CAD curve step.
        {
            bool update = m_dataSet.back()->PreUpdateMapping(
                m_adaptCurves, m_config["radaptscale"].as<NekDouble>(),
                m_config["radaptrad"].as<NekDouble>(),
                m_mesh->m_meshGraph.get());
            if (update)
            {
                m_dataSet.back()->UpdateMapping();
            }
        }
    }

    if (m_config["scalingfile"].beenSet)
    {
        LibUtilities::Interpolator interp = GetScalingFieldFromFile(
            m_config["scalingfile"].as<string>().c_str());

        for (int i = 0; i < m_dataSet.size(); ++i)
        {
            m_dataSet[i]->SetScaling(interp);
        }
    }

    for (int i = 0; i < elmts.size(); i++)
    {
        SpatialDomains::Geometry *el                = elmts[i];
        std::vector<SpatialDomains::PointGeom *> ns = GetCurvedNodes(el);

        for (int j = 0; j < ns.size(); j++)
        {
            m_nodeElMap[ns[j]].push_back(m_dataSet[i]);
        }

        ASSERTL0(derMap[el->GetShapeType()]->ptsStd == ns.size(),
                 "mismatch node count");
    }
}

vector<ElUtilSharedPtr> ProcessVarOpti::GetLockedElements(NekDouble thres)
{
    vector<ElUtilSharedPtr> elBelowThres;
    for (int i = 0; i < m_dataSet.size(); ++i)
    {
        if (m_dataSet[i]->GetScaledJac() < thres)
        {
            elBelowThres.push_back(m_dataSet[i]);
        }
    }

    // m_dataSet is indexed by position, not by element id, so map each
    // element back to its slot.
    std::unordered_map<SpatialDomains::Geometry *, int> elIndex;
    for (int i = 0; i < m_dataSet.size(); ++i)
    {
        elIndex[m_dataSet[i]->GetEl()] = i;
    }

    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();
    auto facetMap     = BuildFacetElementMap(m_mesh, meshDim);

    std::unordered_set<int> inmesh;
    vector<ElUtilSharedPtr> totest;

    // Collect the elements neighbouring @p el, through the facets it shares
    // with them, keeping only those not already gathered.
    auto addNeighbours = [&](ElUtilSharedPtr el, vector<ElUtilSharedPtr> &out) {
        SpatialDomains::Geometry *g = el->GetEl();
        const int nFacets =
            (meshDim == 3) ? g->GetNumFaces() : g->GetNumEdges();

        for (int f = 0; f < nFacets; ++f)
        {
            SpatialDomains::Geometry *facet =
                (meshDim == 3)
                    ? static_cast<SpatialDomains::Geometry *>(g->GetFace(f))
                    : static_cast<SpatialDomains::Geometry *>(g->GetEdge(f));

            for (auto *neighbour : facetMap[facet])
            {
                if (neighbour == g)
                {
                    continue;
                }

                if (inmesh.insert(neighbour->GetGlobalID()).second)
                {
                    out.push_back(m_dataSet[elIndex[neighbour]]);
                }
            }
        }
    };

    for (int i = 0; i < elBelowThres.size(); i++)
    {
        inmesh.insert(elBelowThres[i]->GetId());
        addNeighbours(elBelowThres[i], totest);
    }

    for (int i = 0; i < 6; i++)
    {
        vector<ElUtilSharedPtr> tmp = totest;
        totest.clear();
        for (int j = 0; j < tmp.size(); j++)
        {
            addNeighbours(tmp[j], totest);
        }
    }

    // now need to invert the list
    vector<ElUtilSharedPtr> ret;
    for (int i = 0; i < m_dataSet.size(); ++i)
    {
        if (inmesh.find(m_dataSet[i]->GetId()) == inmesh.end())
        {
            ret.push_back(m_dataSet[i]);
        }
    }

    return ret;
}

void ProcessVarOpti::RemoveLinearCurvature()
{
    // This used to strip the high-order nodes back off elements, faces and
    // edges that had come out essentially straight, to keep the output
    // smaller. Doing that now means dropping the curves those entities hold
    // in MeshGraph, and since the only call site has long been commented out
    // there is nothing exercising the result. Left unimplemented rather than
    // ported blind.
    NEKERROR(ErrorUtil::efatal,
             "ProcessVarOpti::RemoveLinearCurvature has not been ported to "
             "the SpatialDomains geometry classes.");
}

LibUtilities::Interpolator ProcessVarOpti::GetScalingFieldFromFile(string file)
{
    vector<vector<NekDouble>> data;

    ifstream f;
    f.open(file);
    ASSERTL0(f.is_open(), "No such scaling file")

    string fline;

    while (!f.eof())
    {
        getline(f, fline);

        vector<string> tmp;
        boost::split(tmp, fline, boost::is_any_of(" "));

        int i = 0;
        while (i < tmp.size())
        {
            if (tmp[i].size() == 0)
            {
                tmp.erase(tmp.begin() + i);
            }
            else
            {
                ++i;
            }
        }

        if (tmp.size() < 4)
        {
            continue;
        }

        vector<NekDouble> tmpD;
        tmpD.push_back(std::stod(tmp[0])); // x
        tmpD.push_back(std::stod(tmp[1])); // y
        tmpD.push_back(std::stod(tmp[3])); // scaling

        data.push_back(tmpD);
    }

    int dim = m_mesh->m_meshGraph->GetMeshDimension();

    Array<OneD, Array<OneD, NekDouble>> inPts(dim + 1);
    for (int i = 0; i < dim + 1; ++i)
    {
        inPts[i] = Array<OneD, NekDouble>(data.size());

        for (int j = 0; j < data.size(); ++j)
        {
            inPts[i][j] = data[j][i];
        }
    }

    return GetField(inPts);
}

LibUtilities::Interpolator ProcessVarOpti::GetField(
    Array<OneD, Array<OneD, NekDouble>> inPts)
{
    int dim = m_mesh->m_meshGraph->GetMeshDimension();

    vector<string> fieldNames;
    fieldNames.push_back("");

    map<LibUtilities::PtsInfo, int> ptsInfo = LibUtilities::NullPtsInfoMap;

    PtsFieldSharedPtr inField =
        MemoryManager<LibUtilities::PtsField>::AllocateSharedPtr(
            dim, fieldNames, inPts, ptsInfo);

    Array<OneD, Array<OneD, NekDouble>> dummyPts(dim + 1);
    for (int i = 0; i < dim + 1; ++i)
    {
        dummyPts[i] = Array<OneD, NekDouble>(0);
    }

    PtsFieldSharedPtr dummyField =
        MemoryManager<LibUtilities::PtsField>::AllocateSharedPtr(
            dim, fieldNames, dummyPts, ptsInfo);

    LibUtilities::Interpolator ret;
    ret.Interpolate(inField, dummyField);

    return ret;
}
} // namespace Nektar::NekMesh
