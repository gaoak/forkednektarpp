////////////////////////////////////////////////////////////////////////////////
//
//  File: Module.cpp
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
//  Description: Abstract input/output modules.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <algorithm>
#include <boost/iostreams/filter/gzip.hpp>
#include <fstream>
#include <functional>

#include "Module.h"
#include <SpatialDomains/CADSystem/CADAssociation.h>

using namespace std;
namespace io = boost::iostreams;

namespace Nektar::NekMesh
{

/**
 * Returns an instance of the module factory, held as a singleton.
 */
ModuleFactory &GetModuleFactory()
{
    static ModuleFactory instance;
    return instance;
}

/**
 * Prints a given module key to a stream.
 */
std::ostream &operator<<(std::ostream &os, const ModuleKey &rhs)
{
    return os << ModuleTypeMap[rhs.first] << ": " << rhs.second;
}

InputModule::InputModule(MeshSharedPtr m) : Module(m)
{
    m_config["infile"] = ConfigOption(false, "", "Input filename.");
}

OutputModule::OutputModule(MeshSharedPtr m) : Module(m)
{
    m_config["outfile"] = ConfigOption(false, "", "Output filename.");
    m_config["forceoutput"] =
        ConfigOption(true, "", "Force-write output even if file exists.");
}

/**
 * @brief Open a file for input.
 */
void InputModule::OpenStream()
{
    string filename = m_config["infile"].as<string>();

    // Check to see if filename exists.
    if (!fs::exists(filename))
    {
        m_log(FATAL) << "Unable to read file: '" << filename << "'"
                     << std::endl;
    }

    if (filename.size() > 3 && filename.substr(filename.size() - 3, 3) == ".gz")
    {
        m_mshFileStream.open(filename.c_str(), ios_base::in | ios_base::binary);
        m_mshFile.push(io::gzip_decompressor());
        m_mshFile.push(m_mshFileStream);
    }
    else
    {
        m_mshFileStream.open(filename.c_str());
        m_mshFile.push(m_mshFileStream);
    }

    if (!m_mshFile.good())
    {
        m_log(FATAL) << "Error opening file: " << filename << endl;
        abort();
    }
}

/**
 * @brief Open a file for output.
 *
 * @param binary  Open the file in binary mode. Text mode translates '\n' to
 *                the platform's line ending, so a format that is compared
 *                byte-for-byte across platforms has to ask for binary even
 *                though its contents are ASCII.
 */
bool OutputModule::OpenStream(bool binary)
{
    string filename = m_config["outfile"].as<string>();
    bool overwrite  = CheckOverwrite(filename);

    if (overwrite)
    {
        if (filename.size() > 3 &&
            filename.substr(filename.size() - 3, 3) == ".gz")
        {
            m_mshFileStream.open(filename.c_str(),
                                 ios_base::out | ios_base::binary);
            m_mshFile.push(io::gzip_compressor());
            m_mshFile.push(m_mshFileStream);
        }
        else
        {
            m_mshFileStream.open(filename.c_str(),
                                 binary ? ios_base::out | ios_base::binary
                                        : ios_base::out);
            m_mshFile.push(m_mshFileStream);
        }

        if (!m_mshFile.good())
        {
            m_log(FATAL) << "Error opening file: '" << filename << "'" << endl;
        }
    }

    return overwrite;
}

/**
 * @brief Check to see whether we would overwrite this file and prompt the user,
 * unless forceoverwrite option is enabled.
 *
 * @param filename   Filename to check.
 */
bool OutputModule::CheckOverwrite(const std::string &filename)
{
    if (m_config["forceoutput"].beenSet)
    {
        return true;
    }

    if (!fs::exists(filename))
    {
        return true;
    }

    // Assume 'n' in case of non-interactive terminal
    if (!m_log.IsTty())
    {
        return false;
    }

    std::string answer;
    m_log(WARNING) << "Did you wish to overwrite " << filename << " (y/n)? "
                   << std::flush;
    std::getline(std::cin, answer);

    if (answer.compare("y") == 0)
    {
        return true;
    }
    else
    {
        m_log(WARNING) << "Not writing file '" << filename
                       << "' because it already exists" << endl;
    }

    return false;
}

/**
 * @brief Create a unique set of mesh vertices from elements stored in
 * Mesh::element.
 *
 * Each element is processed in turn and the vertices extracted and
 * inserted into #m_vertexSet, which at the end of the routine
 * contains all unique vertices in the mesh.
 */
namespace
{

/// Remove every geometry of type @p T that is not in @p live, returning how
/// many went.
template <typename T>
int RemoveUnreachable(
    SpatialDomains::MeshGraphSharedPtr &graph,
    const std::unordered_set<SpatialDomains::Geometry *> &live,
    std::vector<int> *removedIds = nullptr)
{
    std::vector<int> doomed;
    std::vector<SpatialDomains::Geometry *> doomedGeoms;

    for (auto &[id, geom] : graph->GetGeomMap<T>())
    {
        if (live.find(static_cast<SpatialDomains::Geometry *>(geom)) ==
            live.end())
        {
            doomed.push_back(id);
            doomedGeoms.push_back(geom);
        }
    }

    // Drop any CAD association first: the pool allocator reuses freed
    // addresses, so an entry left behind here would reappear attached to
    // whatever geometry is allocated next.
    if (graph->HasCAD())
    {
        for (auto *geom : doomedGeoms)
        {
            graph->GetCADAssociation()->Remove(geom);
        }
    }

    for (int id : doomed)
    {
        graph->ExtractGeom<T>(id, true);
    }

    if (removedIds != nullptr)
    {
        removedIds->insert(removedIds->end(), doomed.begin(), doomed.end());
    }

    return doomed.size();
}

} // namespace

/**
 * @brief Drop any geometry the mesh no longer refers to.
 *
 * A module that replaces elements leaves the geometry of what it replaced
 * behind in the MeshGraph, which is still written out. Call this once such a
 * module is finished: every element the mesh declares, at any dimension, is
 * taken as a root, everything reachable from one is kept, and the rest --
 * geometries, the curves attached to them, and the curvature nodes those
 * curves held -- is removed.
 */
void Module::RemoveOrphanedEntities()
{
    auto &graph = m_mesh->m_meshGraph;

    std::unordered_set<SpatialDomains::Geometry *> live;

    // Mark a geometry and everything beneath it. The recursion is only as
    // deep as the dimension.
    std::function<void(SpatialDomains::Geometry *)> mark =
        [&](SpatialDomains::Geometry *g) {
            if (g == nullptr || !live.insert(g).second)
            {
                return;
            }

            // A point has nothing beneath it, and asking is fatal rather
            // than empty.
            if (g->GetShapeDim() < 1)
            {
                return;
            }

            for (int i = 0; i < g->GetNumVerts(); ++i)
            {
                mark(g->GetVertex(i));
            }
            for (int i = 0; i < g->GetNumEdges(); ++i)
            {
                mark(g->GetEdge(i));
            }
            for (int i = 0; i < g->GetNumFaces(); ++i)
            {
                mark(g->GetFace(i));
            }
        };

    for (auto &tags : m_mesh->m_elementTags)
    {
        for (auto &[geom, tag] : tags)
        {
            mark(geom);
        }
    }

    // Highest dimension first, so a face or edge is only judged once nothing
    // above it can still be holding on to it.
    std::vector<int> deadEdges, deadFaces, deadVolumes;

    RemoveUnreachable<SpatialDomains::HexGeom>(graph, live, &deadVolumes);
    RemoveUnreachable<SpatialDomains::PrismGeom>(graph, live, &deadVolumes);
    RemoveUnreachable<SpatialDomains::PyrGeom>(graph, live, &deadVolumes);
    RemoveUnreachable<SpatialDomains::TetGeom>(graph, live, &deadVolumes);
    RemoveUnreachable<SpatialDomains::QuadGeom>(graph, live, &deadFaces);
    RemoveUnreachable<SpatialDomains::TriGeom>(graph, live, &deadFaces);
    RemoveUnreachable<SpatialDomains::SegGeom>(graph, live, &deadEdges);
    RemoveUnreachable<SpatialDomains::PointGeom>(graph, live);

    // Curves are filed under the id of the geometry they belong to.
    for (int id : deadEdges)
    {
        graph->GetCurvedEdges().erase(id);
    }
    for (int id : deadFaces)
    {
        graph->GetCurvedFaces().erase(id);
    }
    for (int id : deadVolumes)
    {
        graph->GetCurvedVolumes().erase(id);
    }

    // Finally the curvature nodes. These are owned by the graph and only ever
    // reached through a curve, so anything no surviving curve mentions has
    // gone with it.
    std::unordered_set<SpatialDomains::PointGeom *> usedNodes;
    for (auto *curveMap : {&graph->GetCurvedEdges(), &graph->GetCurvedFaces(),
                           &graph->GetCurvedVolumes()})
    {
        for (auto &[id, curve] : *curveMap)
        {
            for (auto *pt : curve->m_points)
            {
                usedNodes.insert(pt);
            }
        }
    }

    auto &nodes = graph->GetAllCurveNodes();
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
                       [&](const SpatialDomains::PointGeomUniquePtr &n) {
                           return usedNodes.find(n.get()) == usedNodes.end();
                       }),
        nodes.end());

    // Geometry has been freed, so any inter-module payload keyed on entity
    // pointers may now be holding dangling keys. Mark them all stale: a
    // consumer then sees no payload and falls back to whatever it would have
    // done had no producer run. Note that renumbering entities is
    // deliberately not an invalidation -- surviving it is the reason these
    // payloads are keyed on pointers rather than ids.
    m_mesh->GetContext().Invalidate();
}

void Module::ProcessVertices()
{
    // Determine mesh expansion dimension
    int expDim = m_mesh->m_meshGraph->GetMeshDimension();
    if (m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::PointGeom>())
    {
        expDim = 0;
    }
    if (m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::SegGeom>())
    {
        expDim = 1;
    }
    if (m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::TriGeom>() ||
        m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::QuadGeom>())
    {
        expDim = 2;
    }
    if (m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::HexGeom>() ||
        m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::PrismGeom>() ||
        m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::PyrGeom>() ||
        m_mesh->m_meshGraph->GetNumGeoms<SpatialDomains::TetGeom>())
    {
        expDim = 3;
    }
    m_mesh->m_meshGraph->SetMeshDimension(expDim);
}

/**
 * @brief Enumerate elements stored in Mesh::element.
 *
 * For all elements of equal dimension to the mesh dimension, we
 * enumerate sequentially. All other elements in the list should be of
 * lower dimension.
 */
void Module::ProcessElements()
{
    auto segGeoms = m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>();
    auto triGeoms = m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TriGeom>();
    auto quadGeoms =
        m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::QuadGeom>();
    auto hexGeoms = m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::HexGeom>();
    auto tetGeoms = m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::TetGeom>();
    auto pyrGeoms = m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PyrGeom>();
    auto prismGeoms =
        m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::PrismGeom>();

    for (auto geom : segGeoms)
    {
        geom.second->ResetLite();
    }

    for (auto geom : triGeoms)
    {
        geom.second->ResetLite();
    }

    for (auto geom : quadGeoms)
    {
        geom.second->ResetLite();
    }

    for (auto geom : hexGeoms)
    {
        geom.second->ResetLite();
    }

    for (auto geom : tetGeoms)
    {
        geom.second->ResetLite();
    }

    for (auto geom : pyrGeoms)
    {
        geom.second->ResetLite();
    }

    for (auto geom : prismGeoms)
    {
        geom.second->ResetLite();
    }
}

/**
 * @brief Generate a list of composites (groups of elements) from tag
 * IDs stored in mesh vertices/edges/faces/elements.
 *
 * Each element is assigned to a composite ID by an input module. We
 * then generate the composite objects and populate them.
 */
void Module::ProcessComposites()
{
    // For each element, check to see if a composite has been
    // created. If not, create a new composite. Otherwise, add the
    // element to the composite.
    int meshDim      = m_mesh->m_meshGraph->GetMeshDimension();
    auto &composites = m_mesh->m_meshGraph->GetComposites();
    composites.clear();
    for (int d = 0; d <= meshDim; ++d)
    {
        for (const auto &elmt : m_mesh->m_elementTags[d])
        {
            SpatialDomains::CompositeMap::iterator it;
            unsigned int compid = elmt.second;

            it = composites.find(compid);
            if (it == composites.end())
            {
                pair<SpatialDomains::CompositeMap::iterator, bool> testIns;
                testIns = composites.insert(
                    pair<unsigned int, SpatialDomains::CompositeSharedPtr>(
                        compid, std::make_shared<SpatialDomains::Composite>()));
                it = testIns.first;
            }
            else if (elmt.first->GetShapeType() !=
                         it->second->m_geomVec[0]->GetShapeType() &&
                     d != 2) // @TODO: The d !=2 avoids warning for composites
                             // of face elements
            {
                m_log(WARNING)
                    << "Different types of elements in same composite!" << endl
                    << " -> Composite uses "
                    << LibUtilities::ShapeTypeMap[it->second->m_geomVec[0]
                                                      ->GetShapeType()]
                    << endl
                    << " -> Element uses   "
                    << LibUtilities::ShapeTypeMap[elmt.first->GetShapeType()]
                    << endl
                    << "Have you specified physical volumes and surfaces?"
                    << endl;
            }

            it->second->m_geomVec.push_back(elmt.first);
        }
    }
    // Reorder every composite. m_elementTags is keyed on pointers, so the
    // loop above visits entities in an order that depends on where they
    // happen to have been allocated; without this the membership of a
    // composite is written out in a different order on every run. Sorting
    // only the composites of top dimension, as this used to, left the
    // boundary composites varying from run to run. Ties are broken on the
    // shape type because a two-dimensional composite can hold both triangles
    // and quadrilaterals.
    for (auto &comp : composites)
    {
        std::sort(
            comp.second->m_geomVec.begin(), comp.second->m_geomVec.end(),
            [](SpatialDomains::Geometry *a, SpatialDomains::Geometry *b) {
                return std::make_pair(a->GetGlobalID(),
                                      static_cast<int>(a->GetShapeType())) <
                       std::make_pair(b->GetGlobalID(),
                                      static_cast<int>(b->GetShapeType()));
            });
    }

    // @TODO: We currently lose domain information from input file here. This
    // assumes
    //        every composite that is of expansion dimension is a separate
    //        domain and sequentially numbered. So junks multi-composite domains
    //        & IDs.
    auto &domain = m_mesh->m_meshGraph->GetDomain();
    domain.clear();
    int cnt = 0;
    for (auto &[compID, comp] : composites)
    {
        if (comp->m_geomVec[0]->GetShapeDim() == meshDim)
        {
            domain[cnt++][compID] = comp;
        }
    }
}

/**
 * @brief Reorder node IDs so that prisms and tetrahedra are aligned
 * correctly.
 *
 * Orientation of prism lines (i.e. a large prism which has been split
 * into subprisms) cannot be guaranteed when elements are created
 * one-by-one, or when periodic boundary conditions are used. This
 * routine uses the following strategy:
 *
 *   - Destroy the existing node numbering.
 *   - Detect a line of prisms using the PrismLines routine.
 *   - For each line, renumber node IDs consistently so that highest ID
 *     per-element corresponds to the line of collapsed coordinate
 *     points.
 *   - New id key for all geom objects
 *   - Recreate every tet, prism and tetrahedron using the new ordering, use
 *     OrientPrism/OrientTet routines.
 *   - Reuse the edge curves for the high-order info
 */
void Module::ReorderPrisms(PerMap &perFaces)
{
    // Loop over elements and extract any that are prisms.
    int i, j, k;

    auto &m_graph = m_mesh->m_meshGraph;

    if (m_graph->GetMeshDimension() < 3)
    {
        return;
    }

    map<int, int> lines;
    set<int> prismsDone, tetsDone, pyrsDone;
    PerMap::iterator pIt;

    // Compile list of prisms, tets and pyramids, identified by geometry ID.
    for (auto &el : m_mesh->m_elementTags[3])
    {
        if (el.first->GetShapeType() == LibUtilities::ePrism)
        {
            prismsDone.insert(el.first->GetGlobalID());
        }
        else if (el.first->GetShapeType() == LibUtilities::eTetrahedron)
        {
            tetsDone.insert(el.first->GetGlobalID());
        }
        else if (el.first->GetShapeType() == LibUtilities::ePyramid)
        {
            pyrsDone.insert(el.first->GetGlobalID());
        }
    }

    // Face to Element to get neighbour el links
    m_graph->GetAllFaceToElMap().clear();
    for (auto &el : m_mesh->m_elementTags[3])
    {
        m_graph->PopulateFaceToElMap(
            static_cast<SpatialDomains::Geometry3D *>(el.first),
            el.first->GetNumFaces());
    }

    // Destroy existing node numbering.
    for (auto &vert : m_graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        vert.second->SetGlobalID(-1);
    }

    // Counter for new node IDs.
    int nodeId          = 0;
    int prismTris[2][3] = {{0, 1, 4}, {3, 2, 5}};

    // facesDone tracks face IDs inside prisms which have already been
    // aligned.
    std::unordered_set<int> facesDone;
    std::unordered_set<int>::iterator fIt[2], fIt2;

    // prism lines marching through the prism stacks
    // and give prism stack vertices new IDs
    set<int> prismsLeft = prismsDone;

    while (prismsLeft.size() > 0)
    {
        vector<SpatialDomains::Geometry *> line;

        // Call PrismLines to identify all prisms connected to
        // prismLeft.begin() and place them in line[].
        PrismLines(*prismsLeft.begin(), perFaces, prismsLeft, line);

        // Loop over each prism, figure out which line of vertices
        // contains the vertex with highest ID.
        for (i = 0; i < line.size(); ++i)
        {
            // Copy nodes from existing element.
            vector<SpatialDomains::PointGeom *> nodes;
            for (j = 0; j < 6; ++j)
            {
                nodes.push_back(line[i]->GetVertex(j));
            }

            // See if either face of this prism has been renumbered
            // already.
            SpatialDomains::Geometry2D *f[2] = {line[i]->GetFace(1),
                                                line[i]->GetFace(3)};

            fIt[0] = facesDone.find(f[0]->GetGlobalID());
            fIt[1] = facesDone.find(f[1]->GetGlobalID());

            // See if either of these faces is periodic. If it is, then
            // assign ids accordingly.
            for (j = 0; j < 2; ++j)
            {
                pIt = perFaces.find(f[j]->GetGlobalID());

                if (pIt == perFaces.end())
                {
                    continue;
                }

                fIt2 = facesDone.find(pIt->second.first->GetGlobalID());

                if (fIt[j] == facesDone.end() && fIt2 != facesDone.end())
                {
                    fIt[j] = fIt2;
                }
            }

            if (fIt[0] != facesDone.end() && fIt[1] != facesDone.end())
            {
                // Should not be the case that both faces have already
                // been renumbered.
                ASSERTL0(false, "Renumbering error!");
            }
            else if (fIt[0] == facesDone.end() && fIt[1] == facesDone.end())
            {
                // Renumber both faces.
                for (j = 0; j < 2; ++j)
                {
                    for (k = 0; k < 3; ++k)
                    {
                        SpatialDomains::PointGeom *n = nodes[prismTris[j][k]];
                        if (n->GetGlobalID() == -1)
                        {
                            n->SetGlobalID(nodeId++);
                        }
                    }
                }

                facesDone.insert(f[0]->GetGlobalID());
                facesDone.insert(f[1]->GetGlobalID());
            }
            else
            {
                // Renumber face. t identifies the face not yet
                // numbered, o identifies the other face.
                int t = fIt[0] == facesDone.end() ? 0 : 1;
                int o = (t + 1) % 2;
                ASSERTL1(fIt[o] != facesDone.end(), "Renumbering error");

                // Determine which of the three vertices on the 'other'
                // face corresponds to the highest ID - this signifies
                // the singular point of the line of prisms.
                std::vector<std::pair<int32_t, int32_t>> tmp;
                for (int j = 0; j < 3; ++j)
                {
                    tmp.push_back(std::make_pair(
                        j, nodes[prismTris[o][j]]->GetGlobalID()));
                }
                std::sort(tmp.begin(), tmp.end(),
                          [&](std::pair<int32_t, int32_t> a,
                              std::pair<int32_t, int32_t> b) {
                              return a.second < b.second;
                          });

                // Renumber this face so that highest ID matches.
                for (j = 0; j < 3; ++j)
                {
                    SpatialDomains::PointGeom *n =
                        nodes[prismTris[t][tmp[j].first]];
                    if (n->GetGlobalID() == -1)
                    {
                        n->SetGlobalID(nodeId++);
                    }
                }

                facesDone.insert(f[t]->GetGlobalID());
            }

            for (j = 0; j < 6; ++j)
            {
                ASSERTL1(nodes[j]->GetGlobalID() != -1, "Renumbering error");
            }
        }
    }

    int maxCouples = 3;
    // Loop over periodic faces, enumerate vertices.
    for (int flag = 0; flag < maxCouples; flag++)
    {
        for (pIt = perFaces.begin(); pIt != perFaces.end(); ++pIt)
        {
            SpatialDomains::Geometry2D *f2 = pIt->second.first;
            SpatialDomains::Geometry2D *f1 = perFaces[f2->GetGlobalID()].first;
            vector<int> perVerts           = pIt->second.second;
            int nVerts                     = perVerts.size() - 1;
            int coupleFlags                = perVerts[perVerts.size() - 1];

            if (coupleFlags != flag)
            {
                continue;
            }
            // Number periodic vertices first.
            for (j = 0; j < nVerts; ++j)
            {
                SpatialDomains::PointGeom *n1 = f1->GetVertex(j);
                SpatialDomains::PointGeom *n2 = f2->GetVertex(perVerts[j]);

                if (n1->GetGlobalID() == -1 && n2->GetGlobalID() == -1)
                {
                    n1->SetGlobalID(nodeId++);
                    n2->SetGlobalID(nodeId++);
                }
                else if (n1->GetGlobalID() != -1 && n2->GetGlobalID() != -1)
                {
                    continue;
                }
                else
                {
                    m_log(WARNING)
                        << "n1 " << n1->GetGlobalID() << " " << n1->x() << " "
                        << n1->y() << " " << n1->z()
                        << " n2=" << n2->GetGlobalID() << " " << n2->x() << " "
                        << n2->y() << " " << n2->z() << endl;
                    ASSERTL0(false, "Periodic face renumbering error");
                }
            }
        }
    }

    // Do tet vertices .
    set<int>::iterator it2;
    for (it2 = tetsDone.begin(); it2 != tetsDone.end(); ++it2)
    {
        SpatialDomains::Geometry *el = m_graph->GetGeometry3D(*it2);

        for (i = 0; i < 4; ++i)
        {
            if (el->GetVertex(i)->GetGlobalID() == -1)
            {
                el->GetVertex(i)->SetGlobalID(nodeId++);
            }
        }
    }

    // Enumerate rest of vertices (pyr, hex)
    for (auto &vert : m_graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        if (vert.second->GetGlobalID() == -1)
        {
            m_log(VERBOSE) << "Vertex that is not connected to Prism or Tet in "
                              "PerAlign id = nodeId++"
                           << endl;
            vert.second->SetGlobalID(nodeId++);
        }
    }

    // check for left vertices with no ID
    for (auto &vert : m_graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        if (vert.second->GetGlobalID() == -1)
        {
            m_log(FATAL) << "Vetex no ID " << endl;
        }
    }

    // Rebuld the vertices with the new ids and put into the geommap
    vector<int> vertIds;
    for (auto &vert : m_graph->GetGeomMap<SpatialDomains::PointGeom>())
    {
        vertIds.push_back(vert.first);
    }

    SpatialDomains::GeomMap<SpatialDomains::PointGeom> newVertGeoms;
    for (i = 0; i < vertIds.size(); ++i)
    {
        auto point =
            m_graph->ExtractGeom<SpatialDomains::PointGeom>(vertIds[i]);
        int newId           = point->GetGlobalID();
        newVertGeoms[newId] = std::move(point);
    }
    m_graph->SetGeomMap<SpatialDomains::PointGeom>(std::move(newVertGeoms));

    // Create the new edges with the new id key
    EdgeMap edgeSet;
    for (auto &e : m_mesh->m_edgeSet)
    {
        edgeSet[make_pair(e.second->GetVertex(0)->GetGlobalID(),
                          e.second->GetVertex(1)->GetGlobalID())] = e.second;
    }
    m_mesh->m_edgeSet = edgeSet;

    // Create the new faces triag and quad faces with the new vertids
    FaceMap faceSet;
    for (auto &f : m_mesh->m_faceSet)
    {
        std::array<int, 4> vids = {f.second->GetVertex(0)->GetGlobalID(),
                                   f.second->GetVertex(1)->GetGlobalID(),
                                   f.second->GetVertex(2)->GetGlobalID(), -1};
        if (f.second->GetNumVerts() == 4)
        {
            vids[3] = f.second->GetVertex(3)->GetGlobalID();
        }
        faceSet[vids] = f.second;
    }
    m_mesh->m_faceSet = faceSet;

    // Keep the composite to face ID as these will be rebuild in the end
    map<int, vector<int>> compIds;
    for (auto &comp : m_graph->GetComposites())
    {
        if (comp.second->m_geomVec[0]->GetShapeDim() == 3)
        {
            for (auto &geom : comp.second->m_geomVec)
            {
                compIds[comp.first].push_back(geom->GetGlobalID());
            }
        }
    }

    // Reorienting an element changes the vertex order also for triags.
    // Hence we keep the ID if necessary for later
    std::unordered_set<int> naiveTriIDs;
    for (auto &tri : m_graph->GetGeomMap<SpatialDomains::TriGeom>())
    {
        naiveTriIDs.insert(tri.first);
    }

    // Warning flag for face curvature informationß.
    bool warnCurvature = false;

    // Recreate prisms and tets: the factory reorients them using the
    // new vertex IDs, while their edges and faces are rebuilt above.
    vector<int> elmtsDone;
    elmtsDone.insert(elmtsDone.end(), prismsDone.begin(), prismsDone.end());
    elmtsDone.insert(elmtsDone.end(), pyrsDone.begin(), pyrsDone.end());
    elmtsDone.insert(elmtsDone.end(), tetsDone.begin(), tetsDone.end());

    for (auto &elId : elmtsDone)
    {
        SpatialDomains::Geometry *el  = m_graph->GetGeometry3D(elId);
        LibUtilities::ShapeType eType = el->GetShapeType();

        vector<SpatialDomains::PointGeom *> nodes;
        for (i = 0; i < el->GetNumVerts(); ++i)
        {
            nodes.push_back(el->GetVertex(i));
        }

        auto tagIt = m_mesh->m_elementTags[3].find(el);
        int tag    = tagIt->second;
        m_mesh->m_elementTags[3].erase(tagIt);

        if (eType == LibUtilities::ePrism)
        {
            m_graph->ExtractGeom<SpatialDomains::PrismGeom>(elId, true);
        }
        else if (eType == LibUtilities::ePyramid)
        {
            m_graph->ExtractGeom<SpatialDomains::PyrGeom>(elId, true);
        }
        else
        {
            m_graph->ExtractGeom<SpatialDomains::TetGeom>(elId, true);
        }

        ElmtConfig conf(eType, 1, false, false, true);
        ElmtIds ids;
        ids.elmt = elId;
        ids.edges.assign(9, NextEdgeId(m_graph));
        ids.faces.assign(5, NextFaceId(m_graph));

        // Create the new element
        SpatialDomains::Geometry *elNew = GetElementFactory().CreateInstance(
            conf.m_e, nodes, m_graph, m_mesh->m_edgeSet, m_mesh->m_faceSet,
            conf, nullptr, nullptr, &naiveTriIDs, &ids);
        m_mesh->m_elementTags[3][elNew] = tag;

        for (i = 0; i < elNew->GetNumFaces(); ++i)
        {
            SpatialDomains::Geometry2D *f = elNew->GetFace(i);
            if (naiveTriIDs.erase(f->GetGlobalID()) > 0 &&
                f->GetCurve() != nullptr)
            {
                warnCurvature = true;
            }
        }
    }

    if (warnCurvature)
    {
        m_log(WARNING)
            << "[ReorderPrisms] WARNING: Face intranal curvature will be "
               "dropped, but edge curvature retained."
            << endl;
    }

    for (auto &comp : compIds)
    {
        auto &geomVec = m_graph->GetComposites()[comp.first]->m_geomVec;
        for (i = 0; i < comp.second.size(); ++i)
        {
            geomVec[i] = m_graph->GetGeometry3D(comp.second[i]);
        }
    }

    // Rebuild the face to element map now that the elements have changed.
    m_graph->GetAllFaceToElMap().clear();
    for (auto &el : m_mesh->m_elementTags[3])
    {
        m_graph->PopulateFaceToElMap(
            static_cast<SpatialDomains::Geometry3D *>(el.first),
            el.first->GetNumFaces());
    }

    ProcessElements();
}

void Module::PrismLines(int prism, PerMap &perFaces, set<int> &prismsDone,
                        vector<SpatialDomains::Geometry *> &line)
{
    int i;
    set<int>::iterator it = prismsDone.find(prism);
    PerMap::iterator it2;

    if (it == prismsDone.end())
    {
        return;
    }

    auto &m_graph = m_mesh->m_meshGraph;

    // Remove this prism from the list.
    prismsDone.erase(it);
    line.push_back(m_graph->GetGeometry3D(prism));

    // Now find prisms connected to this one through a triangular face.
    for (i = 1; i <= 3; i += 2) // checks only face 1 and face 3
    {
        SpatialDomains::Geometry2D *f =
            m_graph->GetGeometry3D(prism)->GetFace(i);
        int nextId;

        // See if this face is periodic.
        it2 = perFaces.find(f->GetGlobalID());

        if (it2 != perFaces.end())
        {
            int id2 = it2->second.first->GetGlobalID();
            nextId  = m_graph->GetElementsFromFace(it2->second.first)
                         ->at(0)
                         .first->GetGlobalID();
            perFaces.erase(it2);
            perFaces.erase(id2);
            PrismLines(nextId, perFaces, prismsDone, line);
        }

        // Nothing else connected to this face.
        SpatialDomains::GeometryLinkSharedPtr elLink =
            m_graph->GetElementsFromFace(f);
        if (elLink->size() == 1)
        {
            continue;
        }

        nextId = elLink->at(0).first->GetGlobalID();
        if (nextId == prism)
        {
            nextId = elLink->at(1).first->GetGlobalID();
        }

        PrismLines(nextId, perFaces, prismsDone, line);
    }
}

/**
 * @brief Register a configuration option with a module.
 */
void Module::RegisterConfig(string key, string val)
{
    map<string, ConfigOption>::iterator it = m_config.find(key);
    if (it == m_config.end())
    {
        m_log(WARNING) << "WARNING: Unrecognised config option " << key
                       << ", proceeding anyway." << endl;
        return;
    }

    it->second.beenSet = true;

    if (it->second.isBool)
    {
        it->second.value = "1";
    }
    else
    {
        if (val.size() == 0)
        {
            it->second.value = it->second.defValue;
        }
        else
        {
            it->second.value = val;
        }
    }
}

/**
 * @brief Print out all configuration options for a module.
 */
void Module::PrintConfig()
{
    map<string, ConfigOption>::iterator it;

    if (m_config.size() == 0)
    {
        m_log << "No configuration options for this module." << endl;
        return;
    }

    for (it = m_config.begin(); it != m_config.end(); ++it)
    {
        m_log << setw(10) << it->first << ": " << it->second.desc << endl;
    }
}

/**
 * @brief Sets default configuration options for those which have not
 * been set.
 */
void Module::SetDefaults()
{
    map<string, ConfigOption>::iterator it;

    for (it = m_config.begin(); it != m_config.end(); ++it)
    {
        if (!it->second.beenSet)
        {
            it->second.value = it->second.defValue;
        }
    }
}

/**
 * @brief Print a brief summary of information.
 */
void InputModule::PrintSummary()
{
    // Compute the number of full-dimensional elements and boundary
    // elements.
    m_log(VERBOSE) << "Finished reading mesh." << endl;
    m_log(VERBOSE) << " - Element dimension        : "
                   << m_mesh->m_meshGraph->GetMeshDimension() << endl;
    m_log(VERBOSE) << " - Space dimension          : "
                   << m_mesh->m_meshGraph->GetSpaceDimension() << endl;
    m_log(VERBOSE) << " - No. of nodes             : "
                   << m_mesh->m_meshGraph->GetNvertices() << endl;
    m_log(VERBOSE) << " - No. of " << m_mesh->m_meshGraph->GetMeshDimension()
                   << "D elements       : "
                   << m_mesh->m_meshGraph->GetNumElements() << endl;
    m_log(VERBOSE) << " - No. of boundary elements : "
                   << m_mesh->GetNumBndryElements() << endl;
}

} // namespace Nektar::NekMesh
