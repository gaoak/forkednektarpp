////////////////////////////////////////////////////////////////////////////////
//
//  File: InputNek.cpp
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
//  Description: Nektar file format converter.
//
////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/MeshElements/HOAlignment.h>
#include <NekMesh/Module/SurfaceHints.h>
#include <boost/algorithm/string.hpp>

#include "InputNek.h"

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey InputNek::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "rea"), InputNek::create, "Reads Nektar rea file.");

InputNek::InputNek(MeshSharedPtr m) : InputModule(m)
{
    m_config["scalar"] = ConfigOption(
        true, "0", "If defined then assume input rea is for scalar problem");
}

InputNek::~InputNek()
{
}

/**
 * @brief Processes Nektar file format.
 *
 * Nektar sessions are defined by rea files, and contain sections defining a DNS
 * simulation in a specific order. The converter only reads mesh information,
 * curve information if it exists and boundary information.
 *
 * @param pFilename Filename of Nektar session file to read.
 */
void InputNek::Process()
{
    // Open the file stream.
    OpenStream();

    string line, word;
    int nParam, nElements, nCurves;
    int i, j, k, nodeCounter = 0;
    int nComposite = 0;
    LibUtilities::ShapeType elType;
    double vertex[3][8];
    map<LibUtilities::ShapeType, int> domainComposite;
    // Raw vertex coordinates per element, gathered by shape and turned into
    // shared PointGeoms below.
    typedef std::vector<std::array<NekDouble, 3>> CoordList;
    map<LibUtilities::ShapeType, vector<CoordList>> elNodes;

    // Nektar .rea meshes list each element's vertex coordinates and have no
    // unique node list, so vertices are shared by matching coordinates
    // exactly, as the reader's old node set did.
    std::map<std::array<NekDouble, 3>, SpatialDomains::PointGeom *>
        vertexLookup;

    // Elements in reordered id order; the curve and boundary condition
    // sections refer to elements by that index.
    std::vector<SpatialDomains::Geometry *> elements;

    // The vertices of each element in the order the file gave them. The
    // element factory reorients tetrahedra and prisms, so the element's own
    // vertex order is not the file's, and the curved side records are written
    // in the file's.
    std::vector<std::vector<SpatialDomains::PointGeom *>> fileNodes;
    map<LibUtilities::ShapeType, vector<int>> elIds;
    std::unordered_map<int, int> elMap;
    vector<LibUtilities::ShapeType> elmOrder;

    bool scalar = m_config["scalar"].as<bool>();

    // Set up vector of processing orders.
    elmOrder.push_back(LibUtilities::eSegment);
    elmOrder.push_back(LibUtilities::eTriangle);
    elmOrder.push_back(LibUtilities::eQuadrilateral);
    elmOrder.push_back(LibUtilities::ePrism);
    elmOrder.push_back(LibUtilities::ePyramid);
    elmOrder.push_back(LibUtilities::eTetrahedron);
    elmOrder.push_back(LibUtilities::eHexahedron);

    int expDim   = 0;
    int spaceDim = 0;

    m_log(VERBOSE) << "Reading Nektar .rea file '"
                   << m_config["infile"].as<string>() << "'" << endl;

    // -- Read in parameters.

    // Ignore first 3 lines. 4th line contains number of parameters.
    for (i = 0; i < 4; ++i)
    {
        getline(m_mshFile, line);
    }

    stringstream s(line);
    s >> nParam;

    for (i = 0; i < nParam; ++i)
    {
        string tmp1, tmp2;
        getline(m_mshFile, line);
        s.str(line);
        s >> tmp1 >> tmp2;
    }

    // -- Read in passive scalars (ignore)
    getline(m_mshFile, line);
    s.clear();
    s.str(line);
    s >> j;
    for (i = 0; i < j; ++i)
    {
        getline(m_mshFile, line);
    }

    // -- Read in logical switches (ignore)
    getline(m_mshFile, line);
    s.clear();
    s.str(line);
    s >> j;
    for (i = 0; i < j; ++i)
    {
        getline(m_mshFile, line);
    }

    // -- Read in mesh data.

    // First hunt for MESH tag
    bool foundMesh = false;
    while (!m_mshFile.eof())
    {
        getline(m_mshFile, line);
        if (line.find("MESH") != string::npos)
        {
            foundMesh = true;
            break;
        }
    }

    if (!foundMesh)
    {
        m_log(FATAL) << "Couldn't find MESH tag inside file." << endl;
    }

    // Now read in number of elements and space dimension.
    getline(m_mshFile, line);
    s.clear();
    s.str(line);
    s >> nElements >> expDim;
    spaceDim = expDim;

    m_mesh->m_meshGraph->SetMeshDimension(expDim);
    m_mesh->m_meshGraph->SetSpaceDimension(spaceDim);

    // The field names the file implies. The mesh no longer carries these, nor
    // the boundary conditions below, so they are kept locally: the conditions
    // still decide how boundary elements group into composites.
    //
    // @TODO: the conditions are therefore read, used to group composites and
    // then discarded, so a .rea file's boundary conditions no longer reach the
    // output. Restoring that needs somewhere for them to live.
    std::vector<std::string> fieldNames;
    fieldNames.push_back("u");
    if (!scalar)
    {
        fieldNames.push_back("v");
        if (spaceDim > 2)
        {
            fieldNames.push_back("w");
        }
        fieldNames.push_back("p");
    }
    m_log(WARNING) << "Nektar .rea boundary conditions and field definitions "
                   << "are not imported; only the geometry and its composites "
                   << "are." << endl;

    // Loop over and create elements.
    for (i = 0; i < nElements; ++i)
    {
        getline(m_mshFile, line);

        if (expDim == 2)
        {
            if (line.find("Qua") != string::npos ||
                line.find("qua") != string::npos)
            {
                elType = LibUtilities::eQuadrilateral;
            }
            else
            {
                // Default element type in 2D is triangle.
                elType = LibUtilities::eTriangle;
            }
        }
        else
        {
            if (line.find("Tet") != string::npos ||
                line.find("tet") != string::npos)
            {
                elType = LibUtilities::eTetrahedron;
            }
            else if (line.find("Hex") != string::npos ||
                     line.find("hex") != string::npos)
            {
                elType = LibUtilities::eHexahedron;
            }
            else if (line.find("Prism") != string::npos ||
                     line.find("prism") != string::npos)
            {
                elType = LibUtilities::ePrism;
            }
            else if (line.find("Pyr") != string::npos ||
                     line.find("pyr") != string::npos)
            {
                elType = LibUtilities::ePyramid;
            }
            else if (line.find("Qua") != string::npos ||
                     line.find("qua") != string::npos)
            {
                elType = LibUtilities::eQuadrilateral;
            }
            else
            {
                // Default element type in 2D is tetrahedron.
                elType = LibUtilities::eTetrahedron;
            }
        }

        // Read in number of vertices for element type.
        const int nNodes = GetNnodes(elType);

        for (j = 0; j < expDim; ++j)
        {
            getline(m_mshFile, line);
            s.clear();
            s.str(line);
            for (k = 0; k < nNodes; ++k)
            {
                s >> vertex[j][k];
            }
        }

        // Zero co-ordinates bigger than expansion dimension.
        for (j = expDim; j < 3; ++j)
        {
            for (k = 0; k < nNodes; ++k)
            {
                vertex[j][k] = 0.0;
            }
        }

        // Nektar meshes do not contain a unique list of nodes, so this
        // block constructs a unique set so that elements can be created
        // with unique nodes.
        CoordList nodeList;
        for (k = 0; k < nNodes; ++k)
        {
            nodeList.push_back({vertex[0][k], vertex[1][k], vertex[2][k]});
        }

        elNodes[elType].push_back(nodeList);
        elIds[elType].push_back(i);
    }

    int reorderedId = 0;
    nodeCounter     = 0;

    for (i = 0; i < elmOrder.size(); ++i)
    {
        LibUtilities::ShapeType elType = elmOrder[i];
        vector<CoordList> &tmp         = elNodes[elType];

        for (j = 0; j < tmp.size(); ++j)
        {
            vector<int> tags;
            auto compIt = domainComposite.find(elType);
            if (compIt == domainComposite.end())
            {
                tags.push_back(nComposite);
                domainComposite[elType] = nComposite;
                nComposite++;
            }
            else
            {
                tags.push_back(compIt->second);
            }

            elMap[elIds[elType][j]] = reorderedId++;

            vector<SpatialDomains::PointGeom *> nodeList(tmp[j].size());

            for (k = 0; k < tmp[j].size(); ++k)
            {
                auto &loc = tmp[j][k];
                auto it   = vertexLookup.find(loc);

                if (it != vertexLookup.end())
                {
                    nodeList[k] = it->second;
                }
                else
                {
                    nodeList[k] = m_mesh->m_meshGraph->CreatePointGeom(
                        3, nodeCounter++, loc[0], loc[1], loc[2]);
                    vertexLookup[loc] = nodeList[k];
                }
            }

            // Create linear element
            ElmtConfig conf(elType, 1, false, false);
            SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
                elType, nodeList, m_mesh->m_meshGraph, m_mesh->m_edgeSet,
                m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr, nullptr);
            m_mesh->m_elementTags[E->GetShapeDim()][E] = tags[0];
            elements.push_back(E);
            fileNodes.push_back(nodeList);
        }
    }

    // -- Read in curved data.
    getline(m_mshFile, line);
    if (line.find("CURVE") == string::npos)
    {
        m_log(FATAL) << "Cannot find curved side data." << endl;
    }

    // Read number of curves.
    getline(m_mshFile, line);
    s.clear();
    s.str(line);
    s >> nCurves;

    if (nCurves > 0)
    {
        string curveTag;

        for (i = 0; i < nCurves; ++i)
        {
            getline(m_mshFile, line);
            s.clear();
            s.str(line);
            s >> word;

            if (word == "File")
            {
                // Next line contains filename and curve tag.
                getline(m_mshFile, line);
                s.clear();
                s.str(line);
                s >> word >> curveTag;
                curveTags[curveTag] = make_pair(eFile, word);
            }
            else if (word == "Recon")
            {
                // Next line contains curve tag.
                getline(m_mshFile, line);
                s.clear();
                s.str(line);
                s >> word >> curveTag;
                curveTags[curveTag] = make_pair(eRecon, word);
            }
            else
            {
                m_log(FATAL) << "Unsupported curve type " << word << endl;
            }
        }

        // Load high order surface information.
        LoadHOSurfaces();

        // Read in curve information. First line should contain number
        // of curved sides.
        getline(m_mshFile, line);

        if (line.find("side") == string::npos)
        {
            m_log(FATAL) << "Unable to read number of curved sides" << endl;
        }

        int nCurvedSides;
        int faceId, elId;

        s.clear();
        s.str(line);
        s >> nCurvedSides;

        // Curved side records name a 3D element and one of its faces, in
        // the file's own numbering, together with the face's three vertices.
        // Rather than translate that face index into the element's -- the
        // factory reorients tetrahedra and prisms and does not report the
        // mapping -- the face and its edges are found from their vertices,
        // which is numbering-independent, and the curvature attached to those
        // shared geometries directly. That is where it belongs anyway: a face
        // curve is shared by both elements either side of it.
        static const int tetFaceVerts[4][3] = {
            {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};
        static const int prismFaceVerts[5][4] = {{0, 1, 2, 3},
                                                 {0, 1, 4, -1},
                                                 {1, 2, 5, 4},
                                                 {3, 2, 5, -1},
                                                 {0, 3, 5, 4}};

        for (i = 0; i < nCurvedSides; ++i)
        {
            getline(m_mshFile, line);
            s.clear();
            s.str(line);
            s >> faceId >> elId >> word;
            faceId--;
            elId = elMap[elId - 1];

            auto it = curveTags.find(word);
            if (it == curveTags.end())
            {
                m_log(FATAL) << "Unrecognised curve tag " << word
                             << " in curved lines" << endl;
            }

            // The element's three face vertices, in the file's ordering, so
            // that they line up with the ids or coordinates that follow.
            SpatialDomains::Geometry *el                 = elements[elId];
            std::vector<SpatialDomains::PointGeom *> &fn = fileNodes[elId];

            const int *fv = nullptr;
            if (el->GetShapeType() == LibUtilities::eTetrahedron)
            {
                ASSERTL0(faceId >= 0 && faceId < 4, "Bad tetrahedron face");
                fv = tetFaceVerts[faceId];
            }
            else if (el->GetShapeType() == LibUtilities::ePrism)
            {
                ASSERTL0(faceId >= 0 && faceId < 5, "Bad prism face");
                if (prismFaceVerts[faceId][3] != -1)
                {
                    m_log(FATAL) << "Curvature on a quadrilateral prism face "
                                 << "is not supported." << endl;
                }
                fv = prismFaceVerts[faceId];
            }
            else
            {
                m_log(FATAL) << "Curved sides are only supported on "
                             << "tetrahedra and prisms." << endl;
            }

            std::array<int, 4> faceKey = {fn[fv[0]]->GetGlobalID(),
                                          fn[fv[1]]->GetGlobalID(),
                                          fn[fv[2]]->GetGlobalID(), -1};

            auto faceIt = m_mesh->m_faceSet.find(faceKey);
            ASSERTL0(faceIt != m_mesh->m_faceSet.end(),
                     "Could not find the curved face among the element's "
                     "faces");
            SpatialDomains::Geometry2D *face = faceIt->second;

            if (it->second.first == eRecon)
            {
                // Spherigon reconstruction: the record is followed by the x,
                // then y, then z components of the true surface normal at each
                // of the face's vertices. Neither those nor the list of sides
                // to smooth are mesh data, so they are left in the mesh's
                // context for the spherigon module to pick up.
                //
                // The normals are taken to be listed in the file's own face
                // vertex order, fv[] above -- which is why no counterpart of
                // the rotation this module used to apply for odd prism faces
                // is needed. No test file in the tree uses a Recon side, so
                // that reading is unverified.
                std::vector<std::array<NekDouble, 3>> n(3);

                for (int d = 0; d < 3; ++d)
                {
                    getline(m_mshFile, line);
                    s.clear();
                    s.str(line);
                    for (int j = 0; j < 3; ++j)
                    {
                        s >> n[j][d];
                    }
                }

                auto &normals =
                    m_mesh->GetContext().Get<VertexNormals>().normals;

                for (int j = 0; j < 3; ++j)
                {
                    // First writer wins, as it did when this was keyed on
                    // vertex id: a vertex shared by several curved sides keeps
                    // the normal of the first one to mention it.
                    normals.emplace(fn[fv[j]], n[j]);
                }

                m_mesh->GetContext().Get<SpherigonSurfs>().surfs.push_back(
                    face);

                continue;
            }

            // High-order surface data read from the companion file.
            std::vector<int> vertId(3);
            s >> vertId[0] >> vertId[1] >> vertId[2];

            auto hoIt = hoData[word].find(HOSurfSharedPtr(new HOSurf(vertId)));

            if (hoIt == hoData[word].end())
            {
                m_log(FATAL) << "Unable to find high-order surface data "
                             << "for element id " << elId + 1 << endl;
            }

            if (face->GetCurve() != nullptr)
            {
                // Already curved from the neighbouring element.
                continue;
            }

            // Align the surface data to this face's own vertex ordering, then
            // reorder it from the surface file's layout into the nodal one
            // (vertices, then edges, then interior).
            std::vector<int> faceVertIds(3);
            for (int j = 0; j < 3; ++j)
            {
                faceVertIds[j] = face->GetVertex(j)->GetGlobalID();
            }

            // The ids the surface data is keyed by, in this face's order.
            std::vector<int> alignIds(3);
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    if (fn[fv[k]]->GetGlobalID() == faceVertIds[j])
                    {
                        alignIds[j] = vertId[k];
                        break;
                    }
                }
            }

            HOSurf surf = **hoIt;
            surf.Align(alignIds);

            const int Ntot = surf.surfVerts.size();
            const int N    = ((int)sqrt(8.0 * Ntot + 1.0) - 1) / 2;

            std::vector<SpatialDomains::PointGeom *> ordered(Ntot);
            for (int j = 0; j < Ntot; ++j)
            {
                ordered[hoMap[j]] = surf.surfVerts[j];
            }

            // Turn the coordinates into curvature nodes, which carry no
            // global ID of their own.
            // A copy per element: hoData is shared between the faces that
            // use a surface, so its carriers must not become mesh nodes.
            auto makeNode = [&](SpatialDomains::PointGeom *n) {
                auto node = ObjPoolManager<
                    SpatialDomains::PointGeom>::AllocateUniquePtr(3, -1,
                                                                  (*n)[0],
                                                                  (*n)[1],
                                                                  (*n)[2]);
                SpatialDomains::PointGeom *raw = node.get();
                m_mesh->m_meshGraph->GetAllCurveNodes().push_back(
                    std::move(node));
                return raw;
            };

            // Edge curves first, so the face curve can reuse their nodes.
            // An edge shared with another curved face must keep one set of
            // nodes, or the face curve and the edge curve disagree, so take
            // the existing nodes whenever the edge already has a curve.
            std::vector<std::vector<SpatialDomains::PointGeom *>> edgeNodes(3);
            for (int j = 0; j < 3; ++j)
            {
                // The nodal triangle runs edge j from face vertex j to j+1.
                SpatialDomains::PointGeom *v0 = face->GetVertex(j);
                SpatialDomains::PointGeom *v1 = face->GetVertex((j + 1) % 3);

                auto edgeIt = m_mesh->m_edgeSet.find(
                    std::make_pair(v0->GetGlobalID(), v1->GetGlobalID()));
                ASSERTL0(edgeIt != m_mesh->m_edgeSet.end(),
                         "Could not find a curved face's edge");
                SpatialDomains::SegGeom *edge = edgeIt->second;

                // A segment curve runs from its own first vertex to its
                // second, which may be the other way round to the face's.
                const bool forward = edge->GetVertex(0) == v0;

                SpatialDomains::Curve *existing = edge->GetCurve();

                if (existing != nullptr)
                {
                    ASSERTL0(existing->m_points.size() ==
                                 static_cast<size_t>(N),
                             "Curved edge shared between faces of different "
                             "order");

                    for (int k = 0; k < N - 2; ++k)
                    {
                        edgeNodes[j].push_back(
                            forward ? existing->m_points[k + 1]
                                    : existing->m_points[N - 2 - k]);
                    }
                    continue;
                }

                for (int k = 0; k < N - 2; ++k)
                {
                    edgeNodes[j].push_back(
                        makeNode(ordered[3 + j * (N - 2) + k]));
                }

                auto curve =
                    ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                        edge->GetGlobalID(),
                        LibUtilities::eGaussLobattoLegendre);

                curve->m_points.push_back(edge->GetVertex(0));
                for (int k = 0; k < N - 2; ++k)
                {
                    curve->m_points.push_back(
                        forward ? edgeNodes[j][k] : edgeNodes[j][N - 3 - k]);
                }
                curve->m_points.push_back(edge->GetVertex(1));

                SpatialDomains::Curve *curvePtr = curve.get();
                m_mesh->m_meshGraph->AddCurvedEdge(std::move(curve));
                edge->SetCurve(curvePtr);
            }

            // Now the face itself: vertices, the edge blocks in face order,
            // then the interior, which is the nodal triangle layout.
            auto faceCurve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    face->GetGlobalID(), LibUtilities::eNodalTriElec);

            for (int j = 0; j < 3; ++j)
            {
                faceCurve->m_points.push_back(face->GetVertex(j));
            }
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < N - 2; ++k)
                {
                    faceCurve->m_points.push_back(edgeNodes[j][k]);
                }
            }
            for (int j = 3 + 3 * (N - 2); j < Ntot; ++j)
            {
                faceCurve->m_points.push_back(makeNode(ordered[j]));
            }

            SpatialDomains::Curve *faceCurvePtr = faceCurve.get();
            m_mesh->m_meshGraph->AddCurvedFace(std::move(faceCurve));
            face->SetCurve(faceCurvePtr);
        }
    }

    // -- Process fluid boundary conditions.

    // Define a fairly horrendous map: key is the condition ID, the
    // value is a vector of pairs of composites and element
    // types. Essentially this map takes conditions -> composites for
    // each element type.
    map<int, vector<pair<int, LibUtilities::ShapeType>>> surfaceCompMap;

    // The mesh no longer stores boundary conditions, but they still decide
    // how boundary elements group into composites, so keep them locally for
    // that alone.
    ConditionMap conditions;

    // Face vertices in the file's numbering, for locating a boundary face
    // from its vertices.
    static const int bcTetFaceVerts[4][4] = {
        {0, 1, 2, -1}, {0, 1, 3, -1}, {1, 2, 3, -1}, {0, 2, 3, -1}};
    static const int bcPrismFaceVerts[5][4] = {
        {0, 1, 2, 3}, {0, 1, 4, -1}, {1, 2, 5, 4}, {3, 2, 5, -1}, {0, 3, 5, 4}};
    static const int bcHexFaceVerts[6][4] = {{0, 1, 2, 3}, {0, 1, 5, 4},
                                             {1, 2, 6, 5}, {3, 2, 6, 7},
                                             {0, 3, 7, 4}, {4, 5, 6, 7}};

    // Skip boundary conditions line.
    getline(m_mshFile, line);
    getline(m_mshFile, line);

    while (true)
    {
        getline(m_mshFile, line);

        // Break out of loop at end of boundary conditions section.
        if (line.find("*") != string::npos || m_mshFile.eof() ||
            line.length() == 0)
        {
            break;
        }

        // Read boundary type, element ID and face ID.
        char bcType;
        int elId, faceId;
        s.clear();
        s.str(line);
        s >> bcType >> elId >> faceId;
        faceId--;
        elId = elMap[elId - 1];

        vector<string> vals;
        vector<ConditionType> type;
        ConditionSharedPtr c = MemoryManager<Condition>::AllocateSharedPtr();

        SpatialDomains::Geometry *elm = elements[elId];

        // Ignore BCs for undefined edges/faces
        if ((elm->GetShapeDim() == 2 && faceId >= elm->GetNumEdges()) ||
            (elm->GetShapeDim() == 3 && faceId >= elm->GetNumFaces()))
        {
            continue;
        }

        // First character on each line describes type of BC. Currently
        // only support V, W, and O. In this switch statement we
        // construct the quantities needed to search for the condition.
        switch (bcType)
        {
            // Wall boundary.
            case 'W':
            {
                if (scalar)
                {
                    vals.push_back("0");
                    type.push_back(eDirichlet);
                }
                else
                {
                    for (i = 0; i < fieldNames.size() - 1; ++i)
                    {
                        vals.push_back("0");
                        type.push_back(eDirichlet);
                    }
                    // Set high-order boundary condition for wall.
                    vals.push_back("0");
                    type.push_back(eHOPCondition);
                }
                break;
            }

            // Velocity boundary condition (either constant or dependent
            // upon x,y,z).
            case 'V':
            case 'v':
            {
                if (scalar)
                {
                    getline(m_mshFile, line);
                    size_t p = line.find_first_of('=');
                    vals.push_back(
                        boost::algorithm::trim_copy(line.substr(p + 1)));
                    type.push_back(eDirichlet);
                }
                else
                {
                    for (i = 0; i < fieldNames.size() - 1; ++i)
                    {
                        getline(m_mshFile, line);
                        size_t p = line.find_first_of('=');
                        vals.push_back(
                            boost::algorithm::trim_copy(line.substr(p + 1)));
                        type.push_back(eDirichlet);
                    }
                    // Set high-order boundary condition for Dirichlet
                    // condition.
                    vals.push_back("0");
                    type.push_back(eHOPCondition);
                }
                break;
            }

            // Natural outflow condition (default value = 0.0?)
            case 'O':
            {
                if (scalar)
                {
                    vals.push_back("0");
                    type.push_back(eNeumann);
                }
                else
                {
                    for (i = 0; i < fieldNames.size(); ++i)
                    {
                        vals.push_back("0");
                        type.push_back(eNeumann);
                    }
                    // Set zero Dirichlet condition for outflow.
                    type[fieldNames.size() - 1] = eDirichlet;
                }
                break;
            }

            // Ignore unsupported BCs
            case 'P':
            case 'E':
            case 'F':
            case 'f':
            case 'N':
                continue;
                break;

            default:
                m_log(FATAL)
                    << "Unknown boundary condition type " << line[0] << endl;
        }

        // Populate condition information.
        c->field = fieldNames;
        c->type  = type;
        c->value = vals;

        // Now attempt to find this boundary condition inside
        // m_mesh->condition. This is currently a linear search and should
        // probably be made faster!
        bool found = false;
        auto it    = conditions.begin();
        for (; it != conditions.end(); ++it)
        {
            if (c == it->second)
            {
                found = true;
                break;
            }
        }

        int compTag, conditionId;
        SpatialDomains::Geometry *surfEl = nullptr;

        // Create element for face (3D) or segment (2D). At the moment
        // this is a bit of a hack since high-order nodes are not
        // copied, so some output modules (e.g. Gmsh) will not output
        // correctly.
        if (elm->GetShapeDim() == 3)
        {
            // Find the boundary face from its vertices in the file's
            // ordering, as with the curved sides above, rather than through
            // the element's own face numbering, which differs for the
            // reoriented shapes.
            const int *fv  = nullptr;
            int nFaceVerts = 0;

            switch (elm->GetShapeType())
            {
                case LibUtilities::eTetrahedron:
                    ASSERTL0(faceId < 4, "Bad tetrahedron face");
                    fv         = bcTetFaceVerts[faceId];
                    nFaceVerts = 3;
                    break;
                case LibUtilities::ePrism:
                    ASSERTL0(faceId < 5, "Bad prism face");
                    fv         = bcPrismFaceVerts[faceId];
                    nFaceVerts = (fv[3] == -1) ? 3 : 4;
                    break;
                case LibUtilities::eHexahedron:
                    ASSERTL0(faceId < 6, "Bad hexahedron face");
                    fv         = bcHexFaceVerts[faceId];
                    nFaceVerts = 4;
                    break;
                default:
                    m_log(FATAL) << "Unsupported 3D element shape for "
                                 << "boundary conditions." << endl;
                    break;
            }

            std::vector<SpatialDomains::PointGeom *> &fn = fileNodes[elId];
            vector<SpatialDomains::PointGeom *> nodeList;
            for (int v = 0; v < nFaceVerts; ++v)
            {
                nodeList.push_back(fn[fv[v]]);
            }

            // Looked up in the face and edge maps, so the face the parent
            // element already built is reused, curvature included.
            LibUtilities::ShapeType seg = (nFaceVerts == 3)
                                              ? LibUtilities::eTriangle
                                              : LibUtilities::eQuadrilateral;
            ElmtConfig conf(seg, 1, false, false, false,
                            LibUtilities::eGaussLobattoLegendre);
            surfEl = GetElementFactory().CreateInstance(
                seg, nodeList, m_mesh->m_meshGraph, m_mesh->m_edgeSet,
                m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr, nullptr);
        }
        else if (faceId < elm->GetNumEdges())
        {
            SpatialDomains::Geometry1D *f = elm->GetEdge(faceId);

            vector<SpatialDomains::PointGeom *> nodeList;
            nodeList.push_back(f->GetVertex(0));
            nodeList.push_back(f->GetVertex(1));

            ElmtConfig conf(LibUtilities::eSegment, 1, false, false, false,
                            LibUtilities::eGaussLobattoLegendre);
            surfEl = GetElementFactory().CreateInstance(
                LibUtilities::eSegment, nodeList, m_mesh->m_meshGraph,
                m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr, nullptr,
                nullptr, nullptr);
        }

        if (!surfEl)
        {
            continue;
        }

        LibUtilities::ShapeType surfElType = surfEl->GetShapeType();

        if (!found)
        {
            // If condition does not already exist, add to condition
            // list, create new composite tag and put inside
            // surfaceCompMap.
            conditionId = conditions.size();
            compTag     = nComposite;
            c->m_composite.push_back(compTag);
            conditions[conditionId] = c;

            surfaceCompMap[conditionId].push_back(
                pair<int, LibUtilities::ShapeType>(nComposite, surfElType));

            nComposite++;
        }
        else
        {
            // Otherwise find existing composite inside surfaceCompMap.
            auto it2 = surfaceCompMap.find(it->first);

            found = false;
            if (it2 == surfaceCompMap.end())
            {
                // This should never happen!
                m_log(FATAL) << "Unable to find condition!" << endl;
            }

            for (j = 0; j < it2->second.size(); ++j)
            {
                pair<int, LibUtilities::ShapeType> tmp = it2->second[j];
                if (tmp.second == surfElType)
                {
                    found   = true;
                    compTag = tmp.first;
                    break;
                }
            }

            // If no pairs were found, then this condition contains
            // multiple element types (i.e. both triangles and
            // quads). Create another composite for the new shape type
            // and insert into the map.
            if (!found)
            {
                it2->second.push_back(
                    pair<int, LibUtilities::ShapeType>(nComposite, surfElType));
                compTag = nComposite;
                conditions[it->first]->m_composite.push_back(compTag);
                nComposite++;
            }

            conditionId = it->first;
        }

        // Tag the boundary element with its composite.
        m_mesh->m_elementTags[surfEl->GetShapeDim()][surfEl] = compTag;
    }

    m_mshFile.reset();

    // -- Process rest of mesh. Edges and faces are built as the elements are
    // created, so only elements and composites are left to do.
    ProcessElements();
    ProcessComposites();
}

/**
 * Load high order surface information from hsf file.
 */
void InputNek::LoadHOSurfaces()
{
    int nodeId = m_mesh->GetNumTaggedEntities();

    for (auto &it : curveTags)
    {
        ifstream hsf;
        string line, fileName = it.second.second;
        size_t pos;
        int N, Nface, dot;

        if (it.second.first != eFile)
        {
            continue;
        }

        // Replace fro extension with hsf.
        dot      = fileName.find_last_of('.');
        fileName = fileName.substr(0, dot);
        fileName += ".hsf";

        // Open hsf file.
        hsf.open(fileName.c_str());
        if (!hsf.is_open())
        {
            m_log(FATAL) << "Could not open surface file " << fileName << endl;
        }

        // Read in header line; determine element order, number of faces
        // from this line.
        getline(hsf, line);
        pos = line.find("=");
        if (pos == string::npos)
        {
            m_log(FATAL) << "hsf header error: cannot read number of "
                         << "nodal points." << endl;
        }
        line = line.substr(pos + 1);
        stringstream ss(line);
        ss >> N;

        pos = line.find("=");
        if (pos == string::npos)
        {
            m_log(FATAL) << "hsf header error: cannot read number of "
                         << "faces." << endl;
        }
        line = line.substr(pos + 1);
        ss.clear();
        ss.str(line);
        ss >> Nface;

        int Ntot = N * (N + 1) / 2;

        // Skip a line, then read in r,s positions inside the next
        // comments.
        Array<OneD, NekDouble> r(Ntot), s(Ntot);
        getline(hsf, line);

        for (int i = 0; i < 2; ++i)
        {
            string word;

            getline(hsf, line);
            ss.clear();
            ss.str(line);
            ss >> word;

            if (word != "#")
            {
                m_log(FATAL) << "hsf header error: cannot read in "
                             << "r/s points" << endl;
            }

            for (int j = 0; j < Ntot; ++j)
            {
                ss >> (i == 0 ? r[j] : s[j]);
            }
        }

        // Generate electrostatic points so that re-mapping array can
        // be constructed.
        Array<OneD, NekDouble> rp(Ntot), sp(Ntot);
        LibUtilities::PointsKey elec(N, LibUtilities::eNodalTriElec);
        LibUtilities::PointsManager()[elec]->GetPoints(rp, sp);

        // Expensively construct remapping array nodemap. This will
        // map nodal ordering from hsf order to Nektar++ ordering
        // (i.e. vertices followed by edges followed by interior
        // points.)
        for (int i = 0; i < Ntot; ++i)
        {
            for (int j = 0; j < Ntot; ++j)
            {
                if (fabs(r[i] - rp[j]) < 1e-5 && fabs(s[i] - sp[j]) < 1e-5)
                {
                    hoMap[i] = j;
                    break;
                }
            }
        }

        // Skip variables line
        getline(hsf, line);

        // Read in nodal points for each face.
        map<int, vector<SpatialDomains::PointGeom *>> faceMap;
        for (int i = 0; i < Nface; ++i)
        {
            getline(hsf, line);
            vector<SpatialDomains::PointGeom *> faceNodes(Ntot);
            for (int j = 0; j < Ntot; ++j, ++nodeId)
            {
                double x, y, z;
                getline(hsf, line);
                ss.clear();
                ss.str(line);
                ss >> x >> y >> z;

                auto pt = ObjPoolManager<
                    SpatialDomains::PointGeom>::AllocateUniquePtr(3, nodeId, x,
                                                                  y, z);
                faceNodes[j] = pt.get();
                m_hoSurfNodes.push_back(std::move(pt));
            }
            // Skip over tecplot connectivity information.
            for (int j = 0; j < (N - 1) * (N - 1); ++j)
            {
                getline(hsf, line);
            }
            faceMap[i] = faceNodes;
        }

        // Finally, read in connectivity information to set up after
        // reading rea file.
        getline(hsf, line);
        for (int i = 0; i < Nface; ++i)
        {
            string tmp;
            int fid;
            vector<int> nodeIds(3);

            getline(hsf, line);
            ss.clear();
            ss.str(line);
            ss >> tmp >> fid >> nodeIds[0] >> nodeIds[1] >> nodeIds[2];

            if (tmp != "#")
            {
                m_log(FATAL)
                    << "Unable to read hsf connectivity information." << endl;
            }

            hoData[it.first].insert(
                HOSurfSharedPtr(new HOSurf(nodeIds, faceMap[i])));
        }

        hsf.close();
    }
}

/**
 * This routine aids the reading of Nektar files only; it returns the
 * number of nodes for a given entity typw.
 */
int InputNek::GetNnodes(LibUtilities::ShapeType InputNekEntity)
{
    int nNodes = 0;

    switch (InputNekEntity)
    {
        case LibUtilities::ePoint:
            nNodes = 1;
            break;
        case LibUtilities::eSegment:
            nNodes = 2;
            break;
        case LibUtilities::eTriangle:
            nNodes = 3;
            break;
        case LibUtilities::eQuadrilateral:
            nNodes = 4;
            break;
        case LibUtilities::eTetrahedron:
            nNodes = 4;
            break;
        case LibUtilities::ePyramid:
            nNodes = 5;
            break;
        case LibUtilities::ePrism:
            nNodes = 6;
            break;
        case LibUtilities::eHexahedron:
            nNodes = 8;
            break;
        default:
            m_log(FATAL) << "unknown Nektar element type" << endl;
    }

    return nNodes;
}
} // namespace Nektar::NekMesh
