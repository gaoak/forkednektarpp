////////////////////////////////////////////////////////////////////////////////
//
//  File: InputNek5000.cpp
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

#include <LibUtilities/BasicUtils/HashUtils.hpp>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <NekMesh/MeshElements/Element.h>
#include <boost/algorithm/string.hpp>

#include "InputNek5000.h"

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey InputNek5000::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eInputModule, "rea5000"), InputNek5000::create,
    "Reads Nektar rea file.");

InputNek5000::InputNek5000(MeshSharedPtr m) : InputModule(m)
{
}

InputNek5000::~InputNek5000()
{
}

/**
 * @brief Processes Nek5000 file format.
 *
 * Nek5000 sessions are defined by rea files, and contain sections defining a
 * DNS simulation in a specific order. The converter only reads mesh
 * information, curve information if it exists and boundary information. The
 * format is similar to the rea format supported by #InputNek, but the layout is
 * sufficiently different that this module is separate.
 */
void InputNek5000::Process()
{
    // Open the file stream.
    OpenStream();

    string line, word;
    int nParam, nElements, nCurves;
    int i, j, k, nodeCounter = 0;
    int nComposite = 1;
    LibUtilities::ShapeType elType;
    double vertex[8][3];

    // Nek5000 meshes list each element's vertex coordinates and have no unique
    // node list, so vertices are shared by matching their coordinates exactly.
    std::map<std::array<NekDouble, 3>, SpatialDomains::PointGeom *>
        vertexLookup;

    // Elements in the order the file gives them: the curvature and boundary
    // condition sections below refer to elements by that index.
    std::vector<SpatialDomains::Geometry *> elements;

    int expDim   = 0;
    int spaceDim = 0;

    m_log(VERBOSE) << "Reading Nek5000 .rea file '"
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

    // The solver field names the file implies. The mesh no longer carries
    // these, nor the boundary conditions read below, so only the geometry and
    // its composites are imported.
    const int nVelocityFields = (spaceDim > 2) ? 3 : 2;
    m_log(WARNING) << "Nek5000 boundary conditions and field definitions are "
                   << "not imported; only the geometry and its composites are."
                   << endl;

    // Loop over and create elements.
    for (i = 0; i < nElements; ++i)
    {
        int nNodes;
        getline(m_mshFile, line);

        if (expDim == 2)
        {
            // - quad: 2 lines with x-coords, y-coords
            elType = LibUtilities::eQuadrilateral;
            nNodes = 4;
            for (j = 0; j < 2; ++j)
            {
                getline(m_mshFile, line);
                s.clear();
                s.str(line);
                for (k = 0; k < 4; ++k)
                {
                    s >> vertex[k][j];
                }
            }
        }
        else
        {
            // - hex: 3 lines with x/y/z-coords for base 4 nodes, then 3 more
            //   for upper 4 nodes
            elType = LibUtilities::eHexahedron;
            nNodes = 8;
            for (j = 0; j < 6; ++j)
            {
                getline(m_mshFile, line);
                s.clear();
                s.str(line);
                int offset = j > 2 ? 4 : 0;
                for (k = 0; k < 4; ++k)
                {
                    s >> vertex[offset + k][j % 3];
                }
            }
        }

        // Nek5000 meshes do not contain a unique list of nodes, so this block
        // constructs a unique set so that elements can be created with unique
        // nodes.
        vector<SpatialDomains::PointGeom *> nodeList(nNodes);
        for (k = 0; k < nNodes; ++k)
        {
            std::array<NekDouble, 3> loc = {vertex[k][0], vertex[k][1],
                                            vertex[k][2]};

            auto it = vertexLookup.find(loc);
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

        ElmtConfig conf(elType, 1, false, false);
        SpatialDomains::Geometry *E = GetElementFactory().CreateInstance(
            elType, nodeList, m_mesh->m_meshGraph, m_mesh->m_edgeSet,
            m_mesh->m_faceSet, conf, nullptr, nullptr, nullptr, nullptr);
        m_mesh->m_elementTags[E->GetShapeDim()][E] = 0;
        elements.push_back(E);
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

    Array<OneD, NekDouble> rp;
    int nq = 6;
    LibUtilities::PointsKey curveType(nq, LibUtilities::eGaussLobattoLegendre);
    LibUtilities::PointsManager()[curveType]->GetPoints(rp);

    // Map to reorder Nek5000 -> Nektar++ edge ordering. Nek5000 has the same
    // counter-clockwise ordering of edges/vertices; however the vertical
    // (i.e. t- or xi_3-direction) edges come last.
    int nek2nekedge[12] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7};

    // Map to reorder Nek5000 -> Nektar++ face ordering. Again we have the same
    // counter-clockwise ordering; however the 4 vertical faces of the hex are
    // first, followed by bottom face and then top face.
    int nek2nekface[6] = {1, 2, 3, 4, 0, 5};

    if (nCurves > 0)
    {
        for (i = 0; i < nCurves; ++i)
        {
            getline(m_mshFile, line);

            int elmt, side;
            NekDouble curveData[5];
            char curveType;

            if (nElements < 1000)
            {
                // side in first 3 characters, elmt in next 3
                s.str(line.substr(0, 3));
                s >> side;
                s.clear();
                s.str(line.substr(3, 3));
                s >> elmt;
                line = line.substr(6);
            }
            else if (nElements < 1000000)
            {
                // side in first 2 characters, elmt in next 6
                s.str(line.substr(0, 2));
                s >> side;
                s.clear();
                s.str(line.substr(2, 6));
                s >> elmt;
                line = line.substr(8);
            }
            else
            {
                // side in first 2 characters, elmt in next 12
                s.str(line.substr(0, 2));
                s >> side;
                s.clear();
                s.str(line.substr(2, 12));
                s >> elmt;
                line = line.substr(14);
            }

            s.clear();
            s.str(line);

            for (j = 0; j < 5; ++j)
            {
                s >> curveData[j];
            }
            s >> curveType;

            elmt--;
            side--;
            side = nek2nekedge[side];

            switch (curveType)
            {
                case 'C':
                {
                    // Apply circular curvature to edges. Nek5000 assumes that
                    // the curvature should be imposed in x-y planes and has no
                    // z-dependence. The following code is adapted from Semtex
                    // (src/mesh.C)
                    NekDouble radius = curveData[0];
                    int convexity    = radius < 0 ? -1 : 1;
                    radius           = fabs(radius);

                    SpatialDomains::Geometry *el = elements[elmt];
                    SpatialDomains::SegGeom *edge =
                        static_cast<SpatialDomains::SegGeom *>(
                            el->GetEdge(side));

                    SpatialDomains::PointGeom *n1 = edge->GetVertex(0);
                    SpatialDomains::PointGeom *n2 = edge->GetVertex(1);

                    if (fabs((*n1)[2] - (*n2)[2]) > 1e-8)
                    {
                        m_log(WARNING) << "detected curvature on edge that is "
                                       << "not located on x-y plane." << endl;
                    }

                    // The arc is imposed in the x-y plane and has no
                    // z-dependence, so work in 2D components throughout. The
                    // logic follows Semtex (src/mesh.C).
                    NekDouble p1x = (*n1)[0], p1y = (*n1)[1];
                    NekDouble p2x = (*n2)[0], p2y = (*n2)[1];

                    NekDouble midx = 0.5 * (p1x + p2x);
                    NekDouble midy = 0.5 * (p1y + p2y);
                    NekDouble dx = p2x - p1x, dy = p2y - p1y;
                    NekDouble l         = sqrt(dx * dx + dy * dy);
                    NekDouble semiangle = 0.0;

                    // Unit normal to the chord.
                    NekDouble nx = -dy / l, ny = dx / l;

                    if (2.0 * radius < l)
                    {
                        m_log(WARNING) << "Invalid curvature detected" << endl;
                    }
                    else
                    {
                        semiangle = asin(0.5 * l / radius);
                    }

                    // Element centroid, again in the x-y plane, used to pick
                    // which side of the chord the arc centre falls on.
                    NekDouble cx = 0.0, cy = 0.0;
                    const int nVerts = el->GetNumVerts();
                    for (int v = 0; v < nVerts; ++v)
                    {
                        cx += (*el->GetVertex(v))[0];
                        cy += (*el->GetVertex(v))[1];
                    }
                    cx /= (NekDouble)nVerts;
                    cy /= (NekDouble)nVerts;

                    NekDouble sign = (cx - midx) * nx + (cy - midy) * ny;
                    sign           = convexity * sign / fabs(sign);

                    NekDouble centrex =
                        midx + nx * (sign * cos(semiangle) * radius);
                    NekDouble centrey =
                        midy + ny * (sign * cos(semiangle) * radius);

                    NekDouble theta1 = atan2(p1y - centrey, p1x - centrex);
                    NekDouble theta2 = atan2(p2y - centrey, p2x - centrex);
                    NekDouble dtheta = theta2 - theta1;

                    if (fabs(dtheta) > 2.0 * semiangle + 1e-15)
                    {
                        dtheta += (dtheta < 0.0) ? 2.0 * M_PI : -2.0 * M_PI;
                    }

                    // Build the edge's curve: the two vertices with the arc
                    // points between them, in the order SegGeom expects.
                    auto curve = ObjPoolManager<SpatialDomains::Curve>::
                        AllocateUniquePtr(edge->GetGlobalID(),
                                          LibUtilities::eGaussLobattoLegendre);
                    curve->m_points.push_back(n1);

                    for (j = 1; j < nq - 1; ++j)
                    {
                        NekDouble phi = theta1 + dtheta * 0.5 * (rp[j] + 1.0);

                        // Curvature nodes carry no global ID of their own.
                        auto node = ObjPoolManager<SpatialDomains::PointGeom>::
                            AllocateUniquePtr(
                                3, -1, centrex + radius * cos(phi),
                                centrey + radius * sin(phi), (*n1)[2]);
                        curve->m_points.push_back(node.get());
                        m_mesh->m_meshGraph->GetAllCurveNodes().push_back(
                            std::move(node));
                    }

                    curve->m_points.push_back(n2);

                    SpatialDomains::Curve *curvePtr = curve.get();
                    m_mesh->m_meshGraph->AddCurvedEdge(std::move(curve));
                    edge->SetCurve(curvePtr);
                    break;
                }
                case 's':
                case 'S':
                case 'm':
                case 'M':
                    m_log(WARNING) << "Curve type '" << curveType << "' on "
                                   << "side " << side << " of element " << elmt
                                   << " is unsupported; will ignore." << endl;
                    break;
                default:
                    m_log(WARNING) << "Unknown curve type '" << curveType << "'"
                                   << " on side " << side << " of element "
                                   << elmt << "; will ignore." << endl;
                    break;
            }
        }
    }

    // Read boundary conditions.
    getline(m_mshFile, line);
    getline(m_mshFile, line);
    if (line.find("BOUNDARY") == string::npos)
    {
        m_log(FATAL) << "Cannot find boundary conditions." << endl;
    }

    // The mesh no longer stores boundary conditions, but they still decide
    // how boundary elements group into composites: elements sharing a
    // condition share a tag. So keep the conditions locally for that grouping
    // alone.
    ConditionMap conditions;

    std::unordered_set<pair<int, int>, HashOp> periodicIn;
    int periodicInId = -1, periodicOutId = -1;

    // Boundary conditions: should be precisely nElements * nFaces lines to
    // read.
    int lineCnt = 0;
    int perIn = 0, perOut = 0;

    while (m_mshFile.good())
    {
        getline(m_mshFile, line);

        // Found a new section. We don't support anything in the rea file beyond
        // this point so we'll just quit.
        if (line.find("*") != string::npos)
        {
            break;
        }

        ConditionSharedPtr c = MemoryManager<Condition>::AllocateSharedPtr();
        char bcType;
        int elmt, side;
        NekDouble data[5];

        // type in chars 0-3
        s.clear();
        s.str(line.substr(0, 4));
        s >> bcType;

        // Some lines have no boundary condition entries
        if (s.fail())
        {
            lineCnt++;
            continue;
        }

        if (nElements < 1000)
        {
            // elmt in chars 4-6, side in next 3
            s.clear();
            s.str(line.substr(4, 3));
            s >> elmt;
            s.clear();
            s.str(line.substr(7, 3));
            s >> side;
            line = line.substr(10);
        }
        else if (nElements < 100000)
        {
            // elmt in chars 4-8, side in next 1
            s.clear();
            s.str(line.substr(4, 5));
            s >> elmt;
            s.clear();
            s.str(line.substr(9, 1));
            s >> side;
            line = line.substr(10);
        }
        else if (nElements < 1000000)
        {
            // elmt in chars 4-9, no side
            s.clear();
            s.str(line.substr(4, 6));
            s >> elmt;
            side = lineCnt % (2 * expDim);
            line = line.substr(9);
        }
        else
        {
            // elmt in chars 4-15, no side
            s.clear();
            s.str(line.substr(4, 12));
            s >> elmt;
            side = lineCnt % (2 * expDim);
            line = line.substr(15);
        }

        s.clear();
        s.str(line);

        for (i = 0; i < 5; ++i)
        {
            s >> data[i];
        }

        // Our ordering starts from 0, not 1.
        --elmt;
        --side;

        // Increment lines read
        lineCnt++;

        SpatialDomains::Geometry *el = elements[elmt];

        std::string fields[] = {"u", "v", "w", "p"};

        switch (bcType)
        {
            case 'E':
                // Edge/face connectivity; ignore since we already have this, at
                // least for conformal meshes.
                continue;

            case 'W':
            {
                for (i = 0; i < nVelocityFields; ++i)
                {
                    c->field.push_back(fields[i]);
                    c->value.push_back("0");
                    c->type.push_back(eDirichlet);
                }

                // Set high-order boundary condition for wall.
                c->field.push_back(fields[3]);
                c->value.push_back("0");
                c->type.push_back(eHOPCondition);
                break;
            }

            case 'P':
            {
                // Determine periodic element and face.
                int perElmt = (int)(data[0] + 0.5) - 1;
                int perFace = (int)(data[1] + 0.5) - 1;

                bool setup = false;
                if (periodicInId == -1)
                {
                    periodicInId  = conditions.size();
                    periodicOutId = conditions.size() + 1;
                    setup         = true;
                }

                bool hasIn = periodicIn.find(make_pair(perElmt, perFace)) !=
                             periodicIn.end();

                if (hasIn)
                {
                    swap(periodicInId, periodicOutId);
                    perOut++;
                }
                else
                {
                    periodicIn.insert(make_pair(elmt, side));
                    perIn++;
                }

                std::string periodicInStr =
                    "[" + std::to_string(periodicInId) + "]";
                std::string periodicOutStr =
                    "[" + std::to_string(periodicOutId) + "]";

                for (i = 0; i < nVelocityFields; ++i)
                {
                    c->field.push_back(fields[i]);
                    c->value.push_back(periodicOutStr);
                    c->type.push_back(ePeriodic);
                }

                c->field.push_back(fields[3]);
                c->value.push_back(periodicOutStr);
                c->type.push_back(ePeriodic);

                if (setup)
                {
                    ConditionSharedPtr c2 =
                        MemoryManager<Condition>::AllocateSharedPtr();

                    c->m_composite.push_back(nComposite++);
                    c2->m_composite.push_back(nComposite++);

                    c2->field = c->field;
                    c2->type  = c->type;
                    for (i = 0; i < c->type.size(); ++i)
                    {
                        c2->value.push_back(periodicInStr);
                    }

                    conditions[periodicInId]  = c;
                    conditions[periodicOutId] = c2;
                }

                if (hasIn)
                {
                    swap(periodicInId, periodicOutId);
                }

                break;
            }

            default:
                continue;
        }

        int compTag;

        // Create the boundary element: a quadrilateral face in 3D, a segment
        // in 2D. Both are looked up in the edge and face maps first, so the
        // one the parent element already built is reused, and with it any
        // curvature -- unlike before, nothing needs copying across.
        SpatialDomains::Geometry *surfEl = nullptr;

        if (el->GetShapeDim() == 3)
        {
            SpatialDomains::Geometry2D *f = el->GetFace(nek2nekface[side]);

            vector<SpatialDomains::PointGeom *> nodeList;
            for (int v = 0; v < f->GetNumVerts(); ++v)
            {
                nodeList.push_back(f->GetVertex(v));
            }

            ElmtConfig conf(LibUtilities::eQuadrilateral, 1, false, false,
                            false, LibUtilities::eGaussLobattoLegendre);
            surfEl = GetElementFactory().CreateInstance(
                LibUtilities::eQuadrilateral, nodeList, m_mesh->m_meshGraph,
                m_mesh->m_edgeSet, m_mesh->m_faceSet, conf, nullptr, nullptr,
                nullptr, nullptr);
        }
        else
        {
            SpatialDomains::Geometry1D *f = el->GetEdge(side);

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

        // Find whether an identical condition has already been seen, so that
        // this element joins that composite rather than starting a new one.
        // A linear search, as before.
        bool found = false;
        for (auto &it : conditions)
        {
            if (c == it.second)
            {
                found = true;
                c     = it.second;
                break;
            }
        }

        if (!found)
        {
            compTag = nComposite;
            c->m_composite.push_back(compTag);
            conditions[conditions.size()] = c;
        }
        else
        {
            compTag = c->m_composite[0];
        }

        m_mesh->m_elementTags[surfEl->GetShapeDim()][surfEl] = compTag;
    }

    if (lineCnt != nElements * (expDim * 2))
    {
        m_log(WARNING) << "Boundary conditions may not have been correctly "
                       << "read from Nek5000 input file." << endl;
    }

    if (perIn != perOut)
    {
        m_log(WARNING) << "Number of periodic faces does not match." << endl;
    }

    m_mshFile.reset();

    // -- Process rest of mesh. Edges and faces are built as the elements are
    // created, so only elements and composites are left to do.
    ProcessElements();
    ProcessComposites();

    // Periodic composites used to be flagged so they were not reordered.
    // The mesh no longer keeps a composite map to carry that flag.
    if (periodicInId != -1)
    {
        m_log(WARNING) << "Periodic composites can no longer be marked as "
                       << "not reorderable; periodic boundaries may need "
                       << "realigning with the peralign module." << endl;
    }
}

} // namespace Nektar::NekMesh
