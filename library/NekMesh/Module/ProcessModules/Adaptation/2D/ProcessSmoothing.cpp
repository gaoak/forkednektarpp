////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessSmoothing.cpp
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

#include "ProcessSmoothing.h"
#include <NekMesh/MeshElements/Element.h>

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey ProcessSmoothing::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "smooth"), ProcessSmoothing::create,
        "Smooth the linear mesh.");

ProcessSmoothing::ProcessSmoothing(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["n"]     = ConfigOption(false, "2", "smoothing steps.");
    m_config["order"] = ConfigOption(false, "4", "order.");
    m_config["alpha"] =
        ConfigOption(false, "0.5", "smoothing parameter [0-1.0].");
    m_config["type"] =
        ConfigOption(false, "laplacian", "laplacian or centroidal smoothing");
}

ProcessSmoothing::~ProcessSmoothing()
{
}

void ProcessSmoothing::Process()
{
    m_log(VERBOSE) << "Smooth the mesh n = " << " times. " << endl;

    int N           = m_config["n"].as<int>();
    NekDouble alpha = m_config["alpha"].as<NekDouble>();
    string type     = m_config["type"].as<string>();

    SpatialDomains::MeshGraphSharedPtr graph = m_mesh->m_meshGraph;
    const int meshDim                        = graph->GetMeshDimension();

    // 0. Find boundary nodes && Find the adjacent vertices per vertex
    unordered_map<int, SpatialDomains::PointGeom *> fixedNodes;
    unordered_map<int, vector<SpatialDomains::PointGeom *>>
        vertexIDToAdjVertices;
    unordered_map<int, vector<SpatialDomains::Geometry *>> vertexIDToAdjElement;
    unordered_map<int, NekDouble> elementIDToArea;
    unordered_map<int, vector<NekDouble>> elementIDToCentroidXYZ;
    if (meshDim == 2)
    {
        for (auto [eid, edge] : graph->GetGeomMap<SpatialDomains::SegGeom>())
        {
            (void)eid; // unused
            SpatialDomains::PointGeom *n1 = edge->GetVertex(0);
            SpatialDomains::PointGeom *n2 = edge->GetVertex(1);

            if (graph->GetElementsFromEdge(edge).size() == 1 ||
                edge->GetCurve() != nullptr)
            {
                fixedNodes[n1->GetGlobalID()] = n1;
                fixedNodes[n2->GetGlobalID()] = n2;
            }

            vertexIDToAdjVertices[n1->GetGlobalID()].push_back(n2);
            vertexIDToAdjVertices[n2->GetGlobalID()].push_back(n1);
        }

        for (auto &[element, tag] : m_mesh->m_elementTags[2])
        {
            (void)tag; // unused
            for (int j = 0; j < element->GetNumVerts(); ++j)
            {
                vertexIDToAdjElement[element->GetVertex(j)->GetGlobalID()]
                    .push_back(element);
            }
        }
    }
    else
    {
        m_log(FATAL) << "3D not applied " << endl;
    }

    auto &vertexSet = graph->GetGeomMap<SpatialDomains::PointGeom>();

    m_log(VERBOSE) << " Free | Fixed | Overall Vertices" << endl;
    m_log(VERBOSE) << vertexSet.size() - fixedNodes.size() << " | "
                   << fixedNodes.size() << " | " << vertexSet.size() << endl;

    m_log(VERBOSE) << type << " Type Smoothing Selected" << endl;
    for (int i = 0; i < N; i++)
    {
        cout << i << endl;

        // 1. Laplacian smoothing
        if (type == "Laplacian" || type == "laplacian")
        {
            for (auto [vid, vertex] : vertexSet)
            {
                (void)vid; // unused
                auto it = fixedNodes.find(vertex->GetGlobalID());
                if (it != fixedNodes.end())
                {
                    // cout << "Boundary vertex " << endl;
                    continue;
                }

                // it 1.1 Calculate the "optimal " position
                const vector<SpatialDomains::PointGeom *> &LocaNodes =
                    vertexIDToAdjVertices[vertex->GetGlobalID()];
                vector<NekDouble> optimalXYZ(3, 0.0);
                for (auto vertexAdj : LocaNodes)
                {
                    optimalXYZ[0] += vertexAdj->x();
                    optimalXYZ[1] += vertexAdj->y();
                    optimalXYZ[2] += vertexAdj->z();
                }

                optimalXYZ[0] = optimalXYZ[0] / LocaNodes.size();
                optimalXYZ[1] = optimalXYZ[1] / LocaNodes.size();
                optimalXYZ[2] = optimalXYZ[2] / LocaNodes.size();

                // 1.2 Move the node to the optimila
                // vertex->x() = vertex->x() + alpha * (optimalXYZ[0] -
                // vertex->x()); vertex->y() = vertex->y() + alpha *
                // (optimalXYZ[1] - vertex->y()); vertex->z() = vertex->z() +
                // alpha * (optimalXYZ[2] - vertex->z());

                // 1.2 Move the node to the optimila
                vector<NekDouble> tmp(3);
                tmp[0] = vertex->x();
                tmp[1] = vertex->y();
                tmp[2] = vertex->z();

                vertex->UpdatePosition(
                    tmp[0] + alpha * (optimalXYZ[0] - tmp[0]),
                    tmp[1] + alpha * (optimalXYZ[1] - tmp[1]),
                    tmp[2] + alpha * (optimalXYZ[2] - tmp[2]));

                // 1.3 Check if all elements are valid
                if (IsNotValid(vertexIDToAdjElement[vertex->GetGlobalID()]))
                {
                    vertex->UpdatePosition(tmp[0], tmp[1], tmp[2]);
                    m_log(VERBOSE) << "reset element smoothing vertex = "
                                   << vertex->GetGlobalID() << endl;
                }
            }
        }
        else if (type == "Centroidal" || type == "centroidal")
        {
            // Calculate element area and centroid
            NekDouble area;
            vector<NekDouble> centroid(3);
            SpatialDomains::PointGeom *v0, *v1, *v2;
            for (auto &[element, tag] : m_mesh->m_elementTags[meshDim])
            {
                (void)tag; // unused
                if (element->GetShapeType() != LibUtilities::eTriangle)
                {
                    m_log(FATAL)
                        << "Smoothing implemented only for triangular elements"
                        << endl;
                }
                v0 = element->GetVertex(0);
                v1 = element->GetVertex(1);
                v2 = element->GetVertex(2);

                // Linear Area
                area = 0.5 * (v0->x() * (v1->y() - v2->y()) +
                              v1->x() * (v2->y() - v0->y()) +
                              v2->x() * (v0->y() - v1->y()));

                elementIDToArea[element->GetGlobalID()] = area;

                // Linear Centroid calculation
                centroid[0] = (v0->x() + v1->x() + v2->x()) / 3.0;
                centroid[1] = (v0->y() + v1->y() + v2->y()) / 3.0;
                centroid[2] = (v0->z() + v1->z() + v2->z()) / 3.0;

                elementIDToCentroidXYZ[element->GetGlobalID()] = centroid;
            }

            for (auto [vid, vertex] : vertexSet)
            {
                (void)vid; // unused
                auto it = fixedNodes.find(vertex->GetGlobalID());
                if (it != fixedNodes.end())
                {
                    continue;
                }
                // 1.1 Calculate the "optimal" position
                NekDouble sumArea = 0.0;
                vector<NekDouble> optimalXYZ(3, 0.0);
                const vector<SpatialDomains::Geometry *> &elementsAdj =
                    vertexIDToAdjElement[vertex->GetGlobalID()];
                vector<NekDouble> areaAdj(elementsAdj.size());
                vector<vector<NekDouble>> centroidAdj(elementsAdj.size());
                // access the centroid and the area
                for (int k = 0; k < elementsAdj.size(); k++)
                {
                    areaAdj[k] = elementIDToArea[elementsAdj[k]->GetGlobalID()];
                    centroidAdj[k] =
                        elementIDToCentroidXYZ[elementsAdj[k]->GetGlobalID()];
                }

                // optimal positions
                for (int k = 0; k < elementsAdj.size(); k++)
                {
                    optimalXYZ[0] += centroidAdj[k][0] * areaAdj[k];
                    optimalXYZ[1] += centroidAdj[k][1] * areaAdj[k];
                    optimalXYZ[2] += centroidAdj[k][2] * areaAdj[k];

                    sumArea += areaAdj[k];
                }

                optimalXYZ[0] = optimalXYZ[0] / sumArea;
                optimalXYZ[1] = optimalXYZ[1] / sumArea;
                optimalXYZ[2] = optimalXYZ[2] / sumArea;

                // 1.2 Move the node to the optimila
                // vertex->x() =
                //         vertex->x() + alpha * (optimalXYZ[0] - vertex->x());
                // vertex->y() = vertex->y() + alpha * (optimalXYZ[1] -
                // vertex->y()); vertex->z() = vertex->z() + alpha *
                // (optimalXYZ[2] - vertex->z());

                // 1.2 Move the node to the optimila
                vector<NekDouble> tmp(3);
                tmp[0] = vertex->x();
                tmp[1] = vertex->y();
                tmp[2] = vertex->z();

                vertex->UpdatePosition(
                    tmp[0] + alpha * (optimalXYZ[0] - tmp[0]),
                    tmp[1] + alpha * (optimalXYZ[1] - tmp[1]),
                    tmp[2] + alpha * (optimalXYZ[2] - tmp[2]));

                // 1.3 Check if all elements are valid
                if (IsNotValid(vertexIDToAdjElement[vertex->GetGlobalID()]))
                {
                    vertex->UpdatePosition(tmp[0], tmp[1], tmp[2]);
                    m_log(VERBOSE) << "reset element smoothing vertex = "
                                   << vertex->GetGlobalID() << endl;
                }
            }
        }
    }
}

bool ProcessSmoothing::IsNotValid(vector<SpatialDomains::Geometry *> &els)
{
    // short algebraic method to figure out the vailidy of elements
    // test the volume of tetrahedrons constituted by three sibling edges
    for (int i = 0; i < els.size(); i++)
    {
        vector<SpatialDomains::PointGeom *> ns(els[i]->GetNumVerts());
        for (int k = 0; k < ns.size(); ++k)
        {
            ns[k] = els[i]->GetVertex(k);
        }

        if (els[i]->GetShapeType() == LibUtilities::ePrism)
        {
            for (int j = 0; j < 6; j++)
            {
                NekDouble a2 = 0.5 * (1 + prismU1[j]);
                NekDouble b1 = 0.5 * (1 - prismV1[j]);
                NekDouble b2 = 0.5 * (1 + prismV1[j]);
                NekDouble c2 = 0.5 * (1 + prismW1[j]);
                NekDouble d  = 0.5 * (prismU1[j] + prismW1[j]);

                Array<OneD, NekDouble> jac(9, 0.0);

                jac[0] = -0.5 * b1 * ns[0]->x() + 0.5 * b1 * ns[1]->x() +
                         0.5 * b2 * ns[2]->x() - 0.5 * b2 * ns[3]->x();
                jac[1] = -0.5 * b1 * ns[0]->y() + 0.5 * b1 * ns[1]->y() +
                         0.5 * b2 * ns[2]->y() - 0.5 * b2 * ns[3]->y();
                jac[2] = -0.5 * b1 * ns[0]->z() + 0.5 * b1 * ns[1]->z() +
                         0.5 * b2 * ns[2]->z() - 0.5 * b2 * ns[3]->z();

                jac[3] = 0.5 * d * ns[0]->x() - 0.5 * a2 * ns[1]->x() +
                         0.5 * a2 * ns[2]->x() - 0.5 * d * ns[3]->x() -
                         0.5 * c2 * ns[4]->x() + 0.5 * c2 * ns[5]->x();
                jac[4] = 0.5 * d * ns[0]->y() - 0.5 * a2 * ns[1]->y() +
                         0.5 * a2 * ns[2]->y() - 0.5 * d * ns[3]->y() -
                         0.5 * c2 * ns[4]->y() + 0.5 * c2 * ns[5]->y();
                jac[5] = 0.5 * d * ns[0]->z() - 0.5 * a2 * ns[1]->z() +
                         0.5 * a2 * ns[2]->z() - 0.5 * d * ns[3]->z() -
                         0.5 * c2 * ns[4]->z() + 0.5 * c2 * ns[5]->z();

                jac[6] = -0.5 * b1 * ns[0]->x() - 0.5 * b2 * ns[3]->x() +
                         0.5 * b1 * ns[4]->x() + 0.5 * b2 * ns[5]->x();
                jac[7] = -0.5 * b1 * ns[0]->y() - 0.5 * b2 * ns[3]->y() +
                         0.5 * b1 * ns[4]->y() + 0.5 * b2 * ns[5]->y();
                jac[8] = -0.5 * b1 * ns[0]->z() - 0.5 * b2 * ns[3]->z() +
                         0.5 * b1 * ns[4]->z() + 0.5 * b2 * ns[5]->z();

                NekDouble jc = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7]) -
                               jac[3] * (jac[1] * jac[8] - jac[2] * jac[7]) +
                               jac[6] * (jac[1] * jac[5] - jac[2] * jac[4]);

                if (jc < NekConstants::kNekZeroTol)
                {
                    return true;
                }
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::ePyramid)
        {
            for (int j = 0; j < 4; j++)
            {
                Array<OneD, NekDouble> jac(9, 0.0);
                jac[0] = 0.5 * (ns[pyramidV1[j]]->x() - ns[pyramidV0[j]]->x());
                jac[1] = 0.5 * (ns[pyramidV1[j]]->y() - ns[pyramidV0[j]]->y());
                jac[2] = 0.5 * (ns[pyramidV1[j]]->z() - ns[pyramidV0[j]]->z());
                jac[3] = 0.5 * (ns[pyramidV2[j]]->x() - ns[pyramidV0[j]]->x());
                jac[4] = 0.5 * (ns[pyramidV2[j]]->y() - ns[pyramidV0[j]]->y());
                jac[5] = 0.5 * (ns[pyramidV2[j]]->z() - ns[pyramidV0[j]]->z());
                jac[6] = 0.5 * (ns[pyramidV3[j]]->x() - ns[pyramidV0[j]]->x());
                jac[7] = 0.5 * (ns[pyramidV3[j]]->y() - ns[pyramidV0[j]]->y());
                jac[8] = 0.5 * (ns[pyramidV3[j]]->z() - ns[pyramidV0[j]]->z());

                NekDouble jc = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7]) -
                               jac[3] * (jac[1] * jac[8] - jac[2] * jac[7]) +
                               jac[6] * (jac[1] * jac[5] - jac[2] * jac[4]);

                if (jc < NekConstants::kNekZeroTol)
                {
                    return true;
                }
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::eTetrahedron)
        {
            Array<OneD, NekDouble> jac(9, 0.0);

            jac[0] = 0.5 * (ns[1]->x() - ns[0]->x());
            jac[1] = 0.5 * (ns[1]->y() - ns[0]->y());
            jac[2] = 0.5 * (ns[1]->z() - ns[0]->z());
            jac[3] = 0.5 * (ns[2]->x() - ns[0]->x());
            jac[4] = 0.5 * (ns[2]->y() - ns[0]->y());
            jac[5] = 0.5 * (ns[2]->z() - ns[0]->z());
            jac[6] = 0.5 * (ns[3]->x() - ns[0]->x());
            jac[7] = 0.5 * (ns[3]->y() - ns[0]->y());
            jac[8] = 0.5 * (ns[3]->z() - ns[0]->z());

            NekDouble jc = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7]) -
                           jac[3] * (jac[1] * jac[8] - jac[2] * jac[7]) +
                           jac[6] * (jac[1] * jac[5] - jac[2] * jac[4]);

            if (jc < NekConstants::kNekZeroTol)
            {
                return true;
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::eTriangle)
        {
            for (int j = 0; j < 3; j++)
            {

                NekDouble a  = 0.5;                      // coeff for d/dksi
                NekDouble b1 = -1.0 / (2.0 * sqrt(3.0)); // coeff for d/deta
                NekDouble b2 = 1.0 / sqrt(3.0);

                Array<OneD, NekDouble> jac(4, 0.0);

                jac[0] = a * (ns[1]->x() - ns[0]->x());
                jac[1] = a * (ns[1]->y() - ns[0]->y());

                jac[2] = b1 * (ns[1]->x() + ns[0]->x()) + b2 * ns[2]->x();
                jac[3] = b1 * (ns[1]->y() + ns[0]->y()) + b2 * ns[2]->y();

                NekDouble jc = jac[0] * jac[3] - jac[1] * jac[2];

                if (jc < NekConstants::kNekZeroTol)
                {
                    return true;
                }
            }
        }
        else if (els[i]->GetShapeType() == LibUtilities::eQuadrilateral)
        {

            // There is a bug, checking in matlab
            for (int j = 0; j < 4; j++)
            {
                NekDouble a1 = 0.25 * (1 - quadV[j]); // coeff for d/dksi
                NekDouble a2 = 0.25 * (1 + quadV[j]);
                NekDouble b1 = 0.25 * (1 - quadU[j]); // coeff for d/deta
                NekDouble b2 = 0.25 * (1 + quadU[j]);

                Array<OneD, NekDouble> jac(4, 0.0);

                jac[0] = a1 * (ns[1]->x() - ns[0]->x()) +
                         a2 * (ns[2]->x() - ns[3]->x());
                jac[1] = a1 * (ns[1]->y() - ns[0]->y()) +
                         a2 * (ns[2]->y() - ns[3]->y());

                jac[2] = b1 * (ns[3]->x() - ns[0]->x()) +
                         b2 * (ns[2]->x() - ns[1]->x());
                jac[3] = b1 * (ns[3]->y() - ns[0]->y()) +
                         b2 * (ns[2]->y() - ns[1]->y());

                NekDouble jc = jac[0] * jac[3] - jac[1] * jac[2];

                if (jc < NekConstants::kNekZeroTol)
                {
                    cout << "jac<0 at j = " << j << endl;
                    cout << "   p1 = " << ns[0]->x() << ", " << ns[0]->y()
                         << endl;
                    cout << "   p2 = " << ns[1]->x() << ", " << ns[1]->y()
                         << endl;
                    cout << "   p3 = " << ns[2]->x() << ", " << ns[2]->y()
                         << endl;
                    cout << "   p4 = " << ns[3]->x() << ", " << ns[3]->y()
                         << endl;
                    return true;
                }
            }
        }
        else
        {
            m_log(FATAL) << "Only prisms, pyramids, tetrahedra, triangles and "
                            "quadrilaterals supported."
                         << endl;
        }
    }

    return false;
}

} // namespace Nektar::NekMesh
