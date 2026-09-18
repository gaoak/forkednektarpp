////////////////////////////////////////////////////////////////////////////////
//
//  File: InputSem.cpp
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
//  Description: Semtex session converter.
//
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/MeshElements/Element.h>

#include "InputSem.h"

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey InputSem::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "sem"), InputSem::create,
    "Reads Semtex session files.");

/**
 * @brief Initialises the InputSem class.
 */
InputSem::InputSem(MeshSharedPtr m) : InputModule(m)
{
}

InputSem::~InputSem()
{
}

/**
 * @brief Process a Semtex session file.
 *
 * Semtex files are defined by a tokenized markup format. We first
 * populate #sectionMap which stores the location of the various tags in
 * the session file so that they can be jumped to, since no ordering is
 * defined. The converter only requires the NODES and ELEMENTS sections
 * to exist, but can also read CURVES and SURFACES. High-order curves
 * rely on the meshfile session.msh to be created with the Semtex
 * utility meshpr first.
 *
 * @param pFilename Filename of Semtex session to read.
 */
void InputSem::Process()
{
    // Open the file stream.
    OpenStream();

    m_log(VERBOSE) << "Reading Semtex session file '"
                   << m_config["infile"].as<string>() << "'" << endl;

    // Read through input file and populate the section map.
    string fileContents, line, word;
    stringstream ss, ssFile;
    streampos linePos;

    sectionMap["NODES"]    = -1;
    sectionMap["ELEMENTS"] = -1;
    sectionMap["CURVES"]   = -1;
    sectionMap["SURFACES"] = -1;
    sectionMap["GROUPS"]   = -1;
    sectionMap["BCS"]      = -1;
    sectionMap["FIELDS"]   = -1;

    // We need to read entire file into a string and wrap around a stringstream,
    // since boost::io::filtered_stream does not support seeking with zlib.
    m_fileStream << m_mshFile.rdbuf();

    while (!m_fileStream.eof())
    {
        linePos = m_fileStream.tellg();
        getline(m_fileStream, line);
        ss.clear();
        ss.str(line);
        ss >> word;

        // Iterate over all tokens and see if section exists on this
        // line.
        for (auto &it : sectionMap)
        {
            if (word == "<" + it.first || word == "<" + it.first + ">")
            {
                sectionMap[it.first] = linePos;
            }
        }
    }

    // Clear eofbit and go back to the beginning of the file.
    m_fileStream.clear();
    m_fileStream.seekg(0);

    // Check that required sections exist in the file.
    if (sectionMap["NODES"] == std::streampos(-1))
    {
        m_log(FATAL) << "Unable to locate NODES section in session file."
                     << endl;
    }

    if (sectionMap["ELEMENTS"] == std::streampos(-1))
    {
        m_log(FATAL) << "Unable to locate ELEMENTS section in session file."
                     << endl;
    }

    if (sectionMap["SURFACES"] != std::streampos(-1))
    {
        if (sectionMap["BCS"] == std::streampos(-1))
        {
            m_log(FATAL) << "SURFACES section defined but BCS section not "
                         << "found." << endl;
        }

        if (sectionMap["GROUPS"] == std::streampos(-1))
        {
            m_log(FATAL) << "SURFACES section defined but GROUPS section not "
                         << "found." << endl;
        }

        if (sectionMap["FIELDS"] == std::streampos(-1))
        {
            m_log(FATAL) << "SURFACES section defined but FIELDS section not "
                         << "found." << endl;
        }
    }

    auto &graph = m_mesh->m_meshGraph;
    graph->SetMeshDimension(2);

    string tag;
    int start, end, nVertices, nEntities, nCurves, nSurf, nGroups;
    int id, i, j, k;
    vector<double> hoXData, hoYData;
    LibUtilities::ShapeType elType = LibUtilities::eQuadrilateral;
    ifstream homeshFile;

    // Begin by reading in list of nodes which define the linear
    // elements. The coordinates are buffered so that the space dimension is
    // known before any vertex is created with it.
    m_fileStream.seekg(sectionMap["NODES"]);
    getline(m_fileStream, line);
    ss.clear();
    ss.str(line);
    ss >> word;

    tag       = ss.str();
    start     = tag.find_first_of('=');
    end       = tag.find_first_of('>');
    nVertices = atoi(tag.substr(start + 1, end).c_str());

    vector<std::array<NekDouble, 3>> coords;
    coords.reserve(nVertices);

    int spaceDim = 2;

    i = id = 0;
    while (i < nVertices)
    {
        getline(m_fileStream, line);
        if (line.length() < 7)
        {
            continue;
        }
        ss.clear();
        ss.str(line);
        NekDouble x = 0, y = 0, z = 0;
        ss >> id >> x >> y >> z;

        if ((z * z) > 0.000001)
        {
            spaceDim = 3;
        }

        coords.push_back({x, y, z});
        ++i;
    }

    graph->SetSpaceDimension(spaceDim);

    // The nodes are owned here until the elements have been created, so that
    // the element factory can report which of them ended up as vertices and
    // which as curvature nodes; the two are given to the graph differently.
    // This follows InputGmsh, which has the same problem.
    std::map<int, SpatialDomains::PointGeomUniquePtr> loadedNodes;
    vector<SpatialDomains::PointGeom *> verts(nVertices);

    for (i = 0; i < nVertices; ++i)
    {
        loadedNodes[i] =
            ObjPoolManager<SpatialDomains::PointGeom>::AllocateUniquePtr(
                spaceDim, i, coords[i][0], coords[i][1], coords[i][2]);
        verts[i] = loadedNodes[i].get();
    }

    // Now read in the element vertices. The geometry is not created yet: the
    // CURVES section below may raise an element to high order, and building
    // it once is simpler than building it straight and replacing it.
    m_fileStream.seekg(sectionMap["ELEMENTS"]);
    getline(m_fileStream, line);
    ss.clear();
    ss.str(line);
    ss >> word;

    tag       = ss.str();
    start     = tag.find_first_of('=');
    end       = tag.find_first_of('>');
    nEntities = atoi(tag.substr(start + 1, end).c_str());

    vector<std::array<int, 4>> elmtVerts;
    elmtVerts.reserve(nEntities);

    i = id = 0;
    while (i < nEntities)
    {
        getline(m_fileStream, line);
        if (line.length() < 18)
        {
            continue;
        }

        // Read element node list
        ss.clear();
        ss.str(line);
        ss >> id >> word;

        std::array<int, 4> nodes = {0, 0, 0, 0};
        for (j = 0; j < 4; ++j)
        {
            int node = 0;
            ss >> node;
            nodes[j] = node - 1;
        }

        elmtVerts.push_back(nodes);
        ++i;
    }

    // Order of each element, and the curvature nodes it carries, filled in by
    // the CURVES section below. An order of one means a straight element.
    vector<int> elmtOrder(nEntities, 1);
    vector<vector<int>> elmtCurveNodes(nEntities);

    // Finally, process curves.
    if (sectionMap["CURVES"] != std::streampos(-1))
    {
        int np, nel, nodeId = nVertices;

        m_fileStream.seekg(sectionMap["CURVES"]);
        getline(m_fileStream, line);
        ss.clear();
        ss.str(line);
        ss >> word;

        tag     = ss.str();
        start   = tag.find_first_of('=');
        end     = tag.find_first_of('>');
        nCurves = atoi(tag.substr(start + 1, end).c_str());

        // Some session files have empty curves sections; if nCurves
        // is 0, no nead to load high order mesh file.
        if (nCurves > 0)
        {
            string fname    = m_config["infile"].as<string>();
            int ext         = fname.find_last_of('.');
            string meshfile = fname.substr(0, ext) + ".msh";

            homeshFile.open(meshfile.c_str());
            if (!homeshFile.is_open())
            {
                m_log(FATAL)
                    << "Cannot open or find mesh file: '" << meshfile << "'\n"
                    << "Make sure to run 'meshpr' on your session "
                    << "file first." << endl;
            }

            // Make sure we have matching header.
            getline(homeshFile, line);
            ss.clear();
            ss.str(line);
            ss >> np >> nel >> nel >> nel;

            if (nel != nEntities)
            {
                m_log(FATAL)
                    << "Number of elements mismatch in mesh file." << endl;
            }

            // Now read in all mesh data. This is horribly inefficient
            // since not all elements are curved, but it is the
            // easiest way of finding element data.
            hoXData.resize(nel * np * np);
            hoYData.resize(nel * np * np);

            for (j = 0; j < nel * np * np; ++j)
            {
                getline(homeshFile, line);
                ss.clear();
                ss.str(line);
                ss >> hoXData[j] >> hoYData[j];
            }

            homeshFile.close();
        }

        i = id = 0;
        while (i < nCurves)
        {
            getline(m_fileStream, line);
            if (line.length() < 18)
            {
                continue;
            }
            int elmt = 0, side = 0;
            ss.clear();
            ss.str(line);
            ss >> id >> elmt >> side >> word;
            id--;
            elmt--;

            if (word != "<SPLINE>" && word != "<ARC>")
            {
                m_log(FATAL) << "Unknown curve tag: " << word << endl;
            }

            if (elmt < 0 || elmt >= nEntities)
            {
                m_log(FATAL) << "Curve " << id + 1 << " names element "
                             << elmt + 1 << ", which does not exist." << endl;
            }

            // See if we have already retrieved high-order data
            // for this element; prevents unnecessary computation
            // for elements with multiple curves.
            if (elmtOrder[elmt] > 1)
            {
                ++i;
                continue;
            }

            // The mesh file holds the whole np by np grid of every element,
            // so one curve record is enough to raise the element to high
            // order on all four of its sides at once. Grid point (row, col)
            // of element e is at e*np*np + row*np + col, with row zero the
            // bottom side and column zero the left, which is the same
            // orientation as the element's local edges: edge 0 runs left to
            // right along the bottom, edge 1 up the right, edge 2 back along
            // the top and edge 3 down the left.
            vector<int> &curveNodes = elmtCurveNodes[elmt];
            const int elmtOffset    = elmt * np * np;

            for (side = 0; side < 4; ++side)
            {
                int offset = elmtOffset;
                int stride = 0;

                switch (side)
                {
                    case 0: // Bottom edge
                        offset += 0;
                        stride = 1;
                        break;
                    case 1: // Right edge
                        offset += np - 1;
                        stride = np;
                        break;
                    case 2: // Top edge
                        offset += np * np - 1;
                        stride = -1;
                        break;
                    case 3: // Left edge
                        offset += np * (np - 1);
                        stride = -np;
                        break;
                    default:
                        m_log(FATAL)
                            << "Unknown side for curve id " << id << endl;
                }

                for (j = 1; j < np - 1; ++j, ++nodeId)
                {
                    loadedNodes[nodeId] =
                        ObjPoolManager<SpatialDomains::PointGeom>::
                            AllocateUniquePtr(
                                spaceDim, nodeId, hoXData[offset + j * stride],
                                hoYData[offset + j * stride], 0.0);
                    curveNodes.push_back(nodeId);
                }
            }

            // Add internal points, row by row, which is the order
            // GetCurvedNodesQuad expects them in.
            for (j = 1; j < np - 1; ++j)
            {
                int offset = elmtOffset + j * np;
                for (k = 1; k < np - 1; ++k, ++nodeId)
                {
                    loadedNodes[nodeId] =
                        ObjPoolManager<SpatialDomains::PointGeom>::
                            AllocateUniquePtr(spaceDim, nodeId,
                                              hoXData[offset + k],
                                              hoYData[offset + k], 0.0);
                    curveNodes.push_back(nodeId);
                }
            }

            elmtOrder[elmt] = np - 1;

            ++i;
        }
    }

    // Create the elements. The factory sorts the node list into vertices,
    // edge curves and a face curve, sharing an edge curve between the two
    // elements either side of it, and records which nodes it used as what.
    std::set<int> vertIDs;
    std::unordered_set<int> curveNodeIDs;

    m_elements.clear();
    m_elements.reserve(nEntities);

    for (i = 0; i < nEntities; ++i)
    {
        vector<SpatialDomains::PointGeom *> nodeList;
        for (j = 0; j < 4; ++j)
        {
            nodeList.push_back(verts[elmtVerts[i][j]]);
        }
        for (int node : elmtCurveNodes[i])
        {
            nodeList.push_back(loadedNodes[node].get());
        }

        // Semtex writes its points at the Gauss-Lobatto-Legendre points of
        // the element, interior as well as edge.
        ElmtConfig conf(elType, elmtOrder[i], elmtOrder[i] > 1, false, true,
                        LibUtilities::eGaussLobattoLegendre,
                        LibUtilities::eGaussLobattoLegendre);

        SpatialDomains::Geometry *element = GetElementFactory().CreateInstance(
            elType, nodeList, graph, m_mesh->m_edgeSet, m_mesh->m_faceSet, conf,
            &vertIDs, &curveNodeIDs, nullptr, nullptr);

        m_mesh->m_elementTags[2][element] = 0;
        m_elements.push_back(element);
    }

    // Hand the vertices to the graph, renumbered contiguously in case the
    // session lists a node no element uses.
    std::vector<std::pair<int, SpatialDomains::PointGeomUniquePtr>> movedVerts;
    int contigVertID = 0;
    for (int node : vertIDs)
    {
        auto vert = std::move(loadedNodes[node]);
        vert->SetGlobalID(contigVertID);
        movedVerts.emplace_back(contigVertID, std::move(vert));
        contigVertID++;
    }
    graph->BulkAddGeom<SpatialDomains::PointGeom>(movedVerts);

    // Curvature nodes are owned by the graph but are not vertices and carry
    // no global id. Anything left in loadedNodes after this is a duplicate of
    // a curvature node on an edge shared by two curved elements, which the
    // factory generated once and shared, and is dropped with loadedNodes.
    for (int node : curveNodeIDs)
    {
        if (vertIDs.find(node) == vertIDs.end())
        {
            graph->GetAllCurveNodes().push_back(std::move(loadedNodes[node]));
        }
    }
    for (auto &node : graph->GetAllCurveNodes())
    {
        node->SetGlobalID(-1);
    }

    // Process surfaces if they exist. This is deliberately done after
    // curves to ensure high-order points are preserved.
    if (sectionMap["SURFACES"] != std::streampos(-1))
    {
        map<string, int> conditionMap;
        int maxTag = -1;

        m_log(WARNING) << "Semtex boundary conditions and field definitions "
                       << "are not imported; only the geometry and the "
                       << "composites its surface groups define are." << endl;

        // First read in list of groups, which defines each condition tag.
        m_fileStream.seekg(sectionMap["GROUPS"]);
        getline(m_fileStream, line);
        ss.clear();
        ss.str(line);
        ss >> word;

        tag     = ss.str();
        start   = tag.find_first_of('=');
        end     = tag.find_first_of('>');
        nGroups = atoi(tag.substr(start + 1, end).c_str());

        i = id = 0;
        while (i < nGroups)
        {
            getline(m_fileStream, line);
            ss.clear();
            ss.str(line);
            ss >> id >> tag;
            conditionMap[tag] = i++;
        }

        maxTag = i;

        // Finally read surface information.
        m_fileStream.seekg(sectionMap["SURFACES"]);
        getline(m_fileStream, line);
        ss.clear();
        ss.str(line);
        ss >> word;

        tag   = ss.str();
        start = tag.find_first_of('=');
        end   = tag.find_first_of('>');
        nSurf = atoi(tag.substr(start + 1, end).c_str());

        i = id = 0;
        int elmt, side;
        int periodicTagId = -1;

        set<pair<int, int>> visitedPeriodic;

        while (i < nSurf)
        {
            getline(m_fileStream, line);
            ss.clear();
            ss.str(line);
            ss >> id >> elmt >> side >> word;
            elmt--;
            side--;

            if (elmt < 0 || elmt >= nEntities || side < 0 || side > 3)
            {
                m_log(FATAL) << "Surface " << id << " names element "
                             << elmt + 1 << " side " << side + 1
                             << ", which does not exist." << endl;
            }

            if (word == "<P>")
            {
                // The two sides of a periodic pair go into a composite each.
                if (periodicTagId == -1)
                {
                    periodicTagId = maxTag;
                }

                int elmtB, sideB;

                ss >> elmtB >> sideB;
                elmtB--;
                sideB--;

                pair<int, int> c1(elmt, side);
                pair<int, int> c2(elmtB, sideB);

                if (visitedPeriodic.count(c1) == 0 &&
                    visitedPeriodic.count(c2) == 0)
                {
                    visitedPeriodic.insert(make_pair(elmtB, sideB));
                    visitedPeriodic.insert(make_pair(elmt, side));
                    insertEdge(elmt, side, periodicTagId + 1);
                    insertEdge(elmtB, sideB, periodicTagId + 2);
                }
            }
            else if (word == "<B>")
            {
                ss >> tag;
                insertEdge(elmt, side, conditionMap[tag] + 1);
            }
            else
            {
                m_log(FATAL) << "Unrecognised or unsupported tag: '" << word
                             << "'" << endl;
            }
            ++i;
        }
    }

    PrintSummary();

    ProcessElements();
    ProcessComposites();
}

void InputSem::insertEdge(int elmt, int side, int tagId)
{
    // A boundary element is now the tagged entity itself, so the quadrilateral
    // side just gains a tag. There is nothing to copy: it is the same SegGeom
    // the element holds, and it already carries whatever curve the CURVES
    // section gave it.
    auto *edge = static_cast<SpatialDomains::Geometry2D *>(m_elements[elmt])
                     ->GetEdge(side);

    m_mesh->m_elementTags[1][edge] = tagId;
}
} // namespace Nektar::NekMesh
