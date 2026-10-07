////////////////////////////////////////////////////////////////////////////////
//
//  File: ElUtil.cpp
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
//  Description: Calculate jacobians of elements.
//
////////////////////////////////////////////////////////////////////////////////

#include "ElUtil.h"
#include "ProcessVarOpti.h"
#include <SpatialDomains/CADSystem/CADAssociation.h>

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_set>

using namespace std;

namespace Nektar::NekMesh
{

std::mutex mtx2;

ElUtil::ElUtil(SpatialDomains::Geometry *e, DerivUtilSharedPtr d,
               ResidualSharedPtr r, int n, int o)
{
    m_el        = e;
    m_derivUtil = d;
    m_res       = r;
    m_mode      = n;
    m_order     = o;
    m_dim       = m_el->GetShapeDim();
    m_radapt    = false;

    // The optimiser works directly on the node coordinates, so hold a pointer
    // into each one rather than a copy.
    std::vector<SpatialDomains::PointGeom *> ns = GetCurvedNodes(m_el);
    m_nNodes                                    = ns.size();
    nodes.resize(size_t(m_nNodes) * m_dim);
    for (int i = 0; i < m_nNodes; ++i)
    {
        for (int j = 0; j < m_dim; ++j)
        {
            nodes[i * m_dim + j] = &(*ns[i])[j];
        }
    }

    // The node list has to hold each node once: NodeId() picks out the column
    // of the derivative operator belonging to the node being moved, so a list
    // that repeats one node silently differentiates with respect to the wrong
    // one.
    std::unordered_set<SpatialDomains::PointGeom *> seen(ns.begin(), ns.end());
    ASSERTL0(seen.size() == ns.size(),
             "Element " + std::to_string(m_el->GetGlobalID()) +
                 " has a repeated node in its node list.");

    MappingIdealToRef();
}

void ElUtil::MappingIdealToRef()
{
    if (m_el->GetShapeType() == LibUtilities::eQuadrilateral)
    {
        LibUtilities::PointsKey pkey1(m_mode, LibUtilities::eNodalQuadElec);
        LibUtilities::PointsKey pkey2(m_mode + m_order,
                                      LibUtilities::eNodalQuadElec);

        Array<OneD, NekDouble> u1(m_derivUtil->ptsStd), v1(m_derivUtil->ptsStd),
            u2(m_derivUtil->pts), v2(m_derivUtil->pts);

        LibUtilities::PointsManager()[pkey1]->GetPoints(u1, v1);
        LibUtilities::PointsManager()[pkey2]->GetPoints(u2, v2);

        vector<vector<NekDouble>> xyz(4);
        std::vector<SpatialDomains::PointGeom *> ns(m_el->GetNumVerts());
        for (int i = 0; i < m_el->GetNumVerts(); ++i)
        {
            ns[i] = m_el->GetVertex(i);
        }
        for (int i = 0; i < 4; i++)
        {
            vector<NekDouble> x(3);
            x[0]   = (*ns[i])[0];
            x[1]   = (*ns[i])[1];
            x[2]   = (*ns[i])[2];
            xyz[i] = x;
        }

        for (int i = 0; i < m_derivUtil->ptsStd; ++i)
        {
            NekDouble a1 = 0.5 * (1 - u1[i]);
            NekDouble a2 = 0.5 * (1 + u1[i]);
            NekDouble b1 = 0.5 * (1 - v1[i]);
            NekDouble b2 = 0.5 * (1 + v1[i]);

            DNekMat J(2, 2, 1.0, eFULL);

            J(0, 0) = -0.5 * b1 * xyz[0][0] + 0.5 * b1 * xyz[1][0] +
                      0.5 * b2 * xyz[2][0] - 0.5 * b2 * xyz[3][0];
            J(1, 0) = -0.5 * b1 * xyz[0][1] + 0.5 * b1 * xyz[1][1] +
                      0.5 * b2 * xyz[2][1] - 0.5 * b2 * xyz[3][1];

            J(0, 1) = -0.5 * a1 * xyz[0][0] - 0.5 * a2 * xyz[1][0] +
                      0.5 * a2 * xyz[2][0] + 0.5 * a1 * xyz[3][0];
            J(1, 1) = -0.5 * a1 * xyz[0][1] - 0.5 * a2 * xyz[1][1] +
                      0.5 * a2 * xyz[2][1] + 0.5 * a1 * xyz[3][1];

            J.Invert();

            vector<NekDouble> r(10, 0.0); // store det in 10th entry

            r[9] = 1.0 / (J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0));

            r[0] = J(0, 0);
            r[1] = J(1, 0);
            r[2] = 0.0;
            r[3] = J(0, 1);
            r[4] = J(1, 1);
            r[5] = 0.0;
            r[6] = 0.0;
            r[7] = 0.0;
            r[8] = 0.0;
            m_mapsStd.push_back(r);
        }

        for (int i = 0; i < m_derivUtil->pts; ++i)
        {
            NekDouble a1 = 0.5 * (1 - u2[i]);
            NekDouble a2 = 0.5 * (1 + u2[i]);
            NekDouble b1 = 0.5 * (1 - v2[i]);
            NekDouble b2 = 0.5 * (1 + v2[i]);

            DNekMat J(2, 2, 1.0, eFULL);

            J(0, 0) = -0.5 * b1 * xyz[0][0] + 0.5 * b1 * xyz[1][0] +
                      0.5 * b2 * xyz[2][0] - 0.5 * b2 * xyz[3][0];
            J(1, 0) = -0.5 * b1 * xyz[0][1] + 0.5 * b1 * xyz[1][1] +
                      0.5 * b2 * xyz[2][1] - 0.5 * b2 * xyz[3][1];

            J(0, 1) = -0.5 * a1 * xyz[0][0] - 0.5 * a2 * xyz[1][0] +
                      0.5 * a2 * xyz[2][0] + 0.5 * a1 * xyz[3][0];
            J(1, 1) = -0.5 * a1 * xyz[0][1] - 0.5 * a2 * xyz[1][1] +
                      0.5 * a2 * xyz[2][1] + 0.5 * a1 * xyz[3][1];

            J.Invert();

            vector<NekDouble> r(10, 0.0); // store det in 10th entry

            r[9] = 1.0 / (J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0));

            r[0] = J(0, 0);
            r[1] = J(1, 0);
            r[2] = 0.0;
            r[3] = J(0, 1);
            r[4] = J(1, 1);
            r[5] = 0.0;
            r[6] = 0.0;
            r[7] = 0.0;
            r[8] = 0.0;
            m_maps.push_back(r);
        }
    }
    else if (m_el->GetShapeType() == LibUtilities::eTriangle)
    {
        DNekMat J(2, 2, 0.0);
        J(0, 0) = (*nodes[1 * m_dim + 0] - *nodes[0 * m_dim + 0]);
        J(1, 0) = (*nodes[1 * m_dim + 1] - *nodes[0 * m_dim + 1]);
        J(0, 1) = (*nodes[2 * m_dim + 0] - *nodes[0 * m_dim + 0]);
        J(1, 1) = (*nodes[2 * m_dim + 1] - *nodes[0 * m_dim + 1]);

        J.Invert();

        DNekMat R(2, 2, 0.0);
        R(0, 0) = 2.0;
        R(1, 1) = 2.0;

        J = J * R;

        vector<NekDouble> r(10, 0.0); // store det in 10th entry

        r[9] = 1.0 / (J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0));
        r[0] = J(0, 0);
        r[1] = J(1, 0);
        r[3] = J(0, 1);
        r[4] = J(1, 1);

        // The mapping does not vary over a triangle, so one matrix stands
        // for every point and the stride below is zero. Holding a copy per
        // integration point was most of what the optimiser spent its memory
        // on.
        m_maps.push_back(r);
        m_mapsStd.push_back(r);
        m_constantMap = true;
    }
    else if (m_el->GetShapeType() == LibUtilities::eTetrahedron)
    {
        DNekMat J(3, 3, 0.0);
        J(0, 0) = (*nodes[1 * m_dim + 0] - *nodes[0 * m_dim + 0]);
        J(1, 0) = (*nodes[1 * m_dim + 1] - *nodes[0 * m_dim + 1]);
        J(2, 0) = (*nodes[1 * m_dim + 2] - *nodes[0 * m_dim + 2]);
        J(0, 1) = (*nodes[2 * m_dim + 0] - *nodes[0 * m_dim + 0]);
        J(1, 1) = (*nodes[2 * m_dim + 1] - *nodes[0 * m_dim + 1]);
        J(2, 1) = (*nodes[2 * m_dim + 2] - *nodes[0 * m_dim + 2]);
        J(0, 2) = (*nodes[3 * m_dim + 0] - *nodes[0 * m_dim + 0]);
        J(1, 2) = (*nodes[3 * m_dim + 1] - *nodes[0 * m_dim + 1]);
        J(2, 2) = (*nodes[3 * m_dim + 2] - *nodes[0 * m_dim + 2]);

        J.Invert();

        DNekMat R(3, 3, 0.0);
        R(0, 0) = 2.0;
        R(1, 1) = 2.0;
        R(2, 2) = 2.0;

        J = J * R;

        vector<NekDouble> r(10, 0.0); // store det in 10th entry

        r[9] = 1.0 / (J(0, 0) * (J(1, 1) * J(2, 2) - J(2, 1) * J(1, 2)) -
                      J(0, 1) * (J(1, 0) * J(2, 2) - J(2, 0) * J(1, 2)) +
                      J(0, 2) * (J(1, 0) * J(2, 1) - J(2, 0) * J(1, 1)));

        r[0] = J(0, 0);
        r[1] = J(1, 0);
        r[2] = J(2, 0);
        r[3] = J(0, 1);
        r[4] = J(1, 1);
        r[5] = J(2, 1);
        r[6] = J(0, 2);
        r[7] = J(1, 2);
        r[8] = J(2, 2);

        // See the triangle above: one matrix stands for the whole element.
        m_maps.push_back(r);
        m_mapsStd.push_back(r);
        m_constantMap = true;
    }
    else if (m_el->GetShapeType() == LibUtilities::ePrism)
    {
        LibUtilities::PointsKey pkey1(m_mode, LibUtilities::eNodalPrismElec);
        LibUtilities::PointsKey pkey2(m_mode + m_order,
                                      LibUtilities::eNodalPrismSPI);
        Array<OneD, NekDouble> u1, v1, u2, v2, w1, w2;
        LibUtilities::PointsManager()[pkey1]->GetPoints(u1, v1, w1);
        LibUtilities::PointsManager()[pkey2]->GetPoints(u2, v2, w2);

        vector<vector<NekDouble>> xyz(6);
        std::vector<SpatialDomains::PointGeom *> ns(m_el->GetNumVerts());
        for (int i = 0; i < m_el->GetNumVerts(); ++i)
        {
            ns[i] = m_el->GetVertex(i);
        }
        for (int i = 0; i < 6; i++)
        {
            vector<NekDouble> x(3);
            x[0]   = (*ns[i])[0];
            x[1]   = (*ns[i])[1];
            x[2]   = (*ns[i])[2];
            xyz[i] = x;
        }

        for (int i = 0; i < m_derivUtil->ptsStd; ++i)
        {
            NekDouble a2 = 0.5 * (1 + u1[i]);
            NekDouble b1 = 0.5 * (1 - v1[i]);
            NekDouble b2 = 0.5 * (1 + v1[i]);
            NekDouble c2 = 0.5 * (1 + w1[i]);
            NekDouble d  = 0.5 * (u1[i] + w1[i]);

            DNekMat J(3, 3, 1.0, eFULL);

            J(0, 0) = -0.5 * b1 * xyz[0][0] + 0.5 * b1 * xyz[1][0] +
                      0.5 * b2 * xyz[2][0] - 0.5 * b2 * xyz[3][0];
            J(1, 0) = -0.5 * b1 * xyz[0][1] + 0.5 * b1 * xyz[1][1] +
                      0.5 * b2 * xyz[2][1] - 0.5 * b2 * xyz[3][1];
            J(2, 0) = -0.5 * b1 * xyz[0][2] + 0.5 * b1 * xyz[1][2] +
                      0.5 * b2 * xyz[2][2] - 0.5 * b2 * xyz[3][2];

            J(0, 1) = 0.5 * d * xyz[0][0] - 0.5 * a2 * xyz[1][0] +
                      0.5 * a2 * xyz[2][0] - 0.5 * d * xyz[3][0] -
                      0.5 * c2 * xyz[4][0] + 0.5 * c2 * xyz[5][0];
            J(1, 1) = 0.5 * d * xyz[0][1] - 0.5 * a2 * xyz[1][1] +
                      0.5 * a2 * xyz[2][1] - 0.5 * d * xyz[3][1] -
                      0.5 * c2 * xyz[4][1] + 0.5 * c2 * xyz[5][1];
            J(2, 1) = 0.5 * d * xyz[0][2] - 0.5 * a2 * xyz[1][2] +
                      0.5 * a2 * xyz[2][2] - 0.5 * d * xyz[3][2] -
                      0.5 * c2 * xyz[4][2] + 0.5 * c2 * xyz[5][2];

            J(0, 2) = -0.5 * b1 * xyz[0][0] - 0.5 * b2 * xyz[3][0] +
                      0.5 * b1 * xyz[4][0] + 0.5 * b2 * xyz[5][0];
            J(1, 2) = -0.5 * b1 * xyz[0][1] - 0.5 * b2 * xyz[3][1] +
                      0.5 * b1 * xyz[4][1] + 0.5 * b2 * xyz[5][1];
            J(2, 2) = -0.5 * b1 * xyz[0][2] - 0.5 * b2 * xyz[3][2] +
                      0.5 * b1 * xyz[4][2] + 0.5 * b2 * xyz[5][2];

            J.Invert();

            vector<NekDouble> r(10, 0.0); // store det in 10th entry

            r[9] = 1.0 / (J(0, 0) * (J(1, 1) * J(2, 2) - J(2, 1) * J(1, 2)) -
                          J(0, 1) * (J(1, 0) * J(2, 2) - J(2, 0) * J(1, 2)) +
                          J(0, 2) * (J(1, 0) * J(2, 1) - J(2, 0) * J(1, 1)));

            r[0] = J(0, 0);
            r[1] = J(1, 0);
            r[2] = J(2, 0);
            r[3] = J(0, 1);
            r[4] = J(1, 1);
            r[5] = J(2, 1);
            r[6] = J(0, 2);
            r[7] = J(1, 2);
            r[8] = J(2, 2);
            m_mapsStd.push_back(r);
        }
        for (int i = 0; i < m_derivUtil->pts; ++i)
        {
            NekDouble a2 = 0.5 * (1 + u2[i]);
            NekDouble b1 = 0.5 * (1 - v2[i]);
            NekDouble b2 = 0.5 * (1 + v2[i]);
            NekDouble c2 = 0.5 * (1 + w2[i]);
            NekDouble d  = 0.5 * (u2[i] + w2[i]);

            DNekMat J(3, 3, 1.0, eFULL);

            J(0, 0) = -0.5 * b1 * xyz[0][0] + 0.5 * b1 * xyz[1][0] +
                      0.5 * b2 * xyz[2][0] - 0.5 * b2 * xyz[3][0];
            J(1, 0) = -0.5 * b1 * xyz[0][1] + 0.5 * b1 * xyz[1][1] +
                      0.5 * b2 * xyz[2][1] - 0.5 * b2 * xyz[3][1];
            J(2, 0) = -0.5 * b1 * xyz[0][2] + 0.5 * b1 * xyz[1][2] +
                      0.5 * b2 * xyz[2][2] - 0.5 * b2 * xyz[3][2];

            J(0, 1) = 0.5 * d * xyz[0][0] - 0.5 * a2 * xyz[1][0] +
                      0.5 * a2 * xyz[2][0] - 0.5 * d * xyz[3][0] -
                      0.5 * c2 * xyz[4][0] + 0.5 * c2 * xyz[5][0];
            J(1, 1) = 0.5 * d * xyz[0][1] - 0.5 * a2 * xyz[1][1] +
                      0.5 * a2 * xyz[2][1] - 0.5 * d * xyz[3][1] -
                      0.5 * c2 * xyz[4][1] + 0.5 * c2 * xyz[5][1];
            J(2, 1) = 0.5 * d * xyz[0][2] - 0.5 * a2 * xyz[1][2] +
                      0.5 * a2 * xyz[2][2] - 0.5 * d * xyz[3][2] -
                      0.5 * c2 * xyz[4][2] + 0.5 * c2 * xyz[5][2];

            J(0, 2) = -0.5 * b1 * xyz[0][0] - 0.5 * b2 * xyz[3][0] +
                      0.5 * b1 * xyz[4][0] + 0.5 * b2 * xyz[5][0];
            J(1, 2) = -0.5 * b1 * xyz[0][1] - 0.5 * b2 * xyz[3][1] +
                      0.5 * b1 * xyz[4][1] + 0.5 * b2 * xyz[5][1];
            J(2, 2) = -0.5 * b1 * xyz[0][2] - 0.5 * b2 * xyz[3][2] +
                      0.5 * b1 * xyz[4][2] + 0.5 * b2 * xyz[5][2];

            J.Invert();

            vector<NekDouble> r(10, 0.0); // store det in 10th entry

            r[9] = 1.0 / (J(0, 0) * (J(1, 1) * J(2, 2) - J(2, 1) * J(1, 2)) -
                          J(0, 1) * (J(1, 0) * J(2, 2) - J(2, 0) * J(1, 2)) +
                          J(0, 2) * (J(1, 0) * J(2, 1) - J(2, 0) * J(1, 1)));

            r[0] = J(0, 0);
            r[1] = J(1, 0);
            r[2] = J(2, 0);
            r[3] = J(0, 1);
            r[4] = J(1, 1);
            r[5] = J(2, 1);
            r[6] = J(0, 2);
            r[7] = J(1, 2);
            r[8] = J(2, 2);
            m_maps.push_back(r);
        }
    }
    else if (m_el->GetShapeType() == LibUtilities::eHexahedron)
    {
        LibUtilities::PointsKey pkey1(m_mode, LibUtilities::eNodalHexElec);
        LibUtilities::PointsKey pkey2(m_mode + m_order,
                                      LibUtilities::eNodalHexElec);
        Array<OneD, NekDouble> u1, v1, w1, u2, v2, w2;

        LibUtilities::PointsManager()[pkey1]->GetPoints(u1, v1, w1);
        LibUtilities::PointsManager()[pkey2]->GetPoints(u2, v2, w2);

        std::array<std::array<NekDouble, 3>, 8> xyz;
        for (int i = 0; i < 8; i++)
        {
            SpatialDomains::PointGeom *vert = m_el->GetVertex(i);
            for (int j = 0; j < 3; ++j)
            {
                xyz[i][j] = (*vert)[j];
            }
        }

        // The trilinear mapping of the straight-sided element, written out
        // from the standard element's vertex coordinates rather than by hand:
        // the shape function of vertex i is the product over the three
        // directions of (1 + sign * xi) / 2, where sign is the vertex's own
        // coordinate in that direction. Spelling the twenty-four derivative
        // terms out one by one is what left this mapping with one vertex
        // written twice and another not at all, which made it singular.
        static const NekDouble sgn[8][3] = {
            {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0},
            {-1.0, 1.0, -1.0},  {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
            {1.0, 1.0, 1.0},    {-1.0, 1.0, 1.0}};

        auto idealMap = [&xyz](NekDouble xi0, NekDouble xi1, NekDouble xi2) {
            const NekDouble xi[3] = {xi0, xi1, xi2};

            DNekMat J(3, 3, 0.0, eFULL);

            for (int v = 0; v < 8; ++v)
            {
                for (int e = 0; e < 3; ++e)
                {
                    // d(shape function)/d(xi_e)
                    NekDouble dphi = 0.5 * sgn[v][e];
                    for (int d = 0; d < 3; ++d)
                    {
                        if (d != e)
                        {
                            dphi *= 0.5 * (1.0 + sgn[v][d] * xi[d]);
                        }
                    }

                    for (int c = 0; c < 3; ++c)
                    {
                        J(c, e) += dphi * xyz[v][c];
                    }
                }
            }

            J.Invert();

            std::vector<NekDouble> r(10, 0.0); // store det in 10th entry

            r[9] = 1.0 / (J(0, 0) * (J(1, 1) * J(2, 2) - J(2, 1) * J(1, 2)) -
                          J(0, 1) * (J(1, 0) * J(2, 2) - J(2, 0) * J(1, 2)) +
                          J(0, 2) * (J(1, 0) * J(2, 1) - J(2, 0) * J(1, 1)));

            for (int e = 0; e < 3; ++e)
            {
                for (int c = 0; c < 3; ++c)
                {
                    r[e * 3 + c] = J(c, e);
                }
            }

            return r;
        };

        for (int i = 0; i < m_derivUtil->ptsStd; ++i)
        {
            m_mapsStd.push_back(idealMap(u1[i], v1[i], w1[i]));
        }
        for (int i = 0; i < m_derivUtil->pts; ++i)
        {
            m_maps.push_back(idealMap(u2[i], v2[i], w2[i]));
        }
    }
    else
    {
        ASSERTL0(false, "not coded");
    }

    // Flatten into the arrays the optimiser reads. A constant mapping is
    // stored once and read with a stride of zero.
    mapStride    = m_constantMap ? 0 : 10;
    mapStdStride = mapStride;

    maps.resize(m_maps.size() * 10);
    mapsStd.resize(m_mapsStd.size() * 10);
    for (size_t i = 0; i < m_maps.size(); ++i)
    {
        std::copy(m_maps[i].begin(), m_maps[i].end(), &maps[i * 10]);
    }
    for (size_t i = 0; i < m_mapsStd.size(); ++i)
    {
        std::copy(m_mapsStd[i].begin(), m_mapsStd[i].end(), &mapsStd[i * 10]);
    }

    // The unscaled originals are only wanted if an r-adaptation scaling is
    // later applied, and UpdateMapping() takes its own copy then. Until that
    // happens the arrays above are the originals, and a second copy of them
    // is the single largest thing this class would hold.
    m_maps.clear();
    m_maps.shrink_to_fit();
    m_mapsStd.clear();
    m_mapsStd.shrink_to_fit();
}

void ElUtil::Evaluate()
{
    NekDouble mx2 = -1.0 * numeric_limits<double>::max();
    NekDouble mn2 = numeric_limits<double>::max();

    ASSERTL0(m_nNodes == m_derivUtil->ptsStd, "node count wrong");
    const int nNodes = m_nNodes;

    if (m_dim == 2)
    {
        std::vector<NekDouble> X(nNodes), Y(nNodes);
        for (int j = 0; j < nNodes; j++)
        {
            X[j] = *nodes[j * m_dim];
            Y[j] = *nodes[j * m_dim + 1];
        }

        std::vector<NekDouble> x1i(nNodes), y1i(nNodes);
        std::vector<NekDouble> x2i(nNodes), y2i(nNodes);

        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[0].GetRawPtr(), nNodes, &X[0], 1, 0.0,
                    &x1i[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[0].GetRawPtr(), nNodes, &Y[0], 1, 0.0,
                    &y1i[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[1].GetRawPtr(), nNodes, &X[0], 1, 0.0,
                    &x2i[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[1].GetRawPtr(), nNodes, &Y[0], 1, 0.0,
                    &y2i[0], 1.0);

        for (int j = 0; j < nNodes; j++)
        {
            NekDouble jacDet = x1i[j] * y2i[j] - x2i[j] * y1i[j];
            mx2              = max(mx2, jacDet / mapsStd[j * mapStdStride + 9]);
            mn2              = min(mn2, jacDet / mapsStd[j * mapStdStride + 9]);
        }
    }
    else if (m_dim == 3)
    {
        std::vector<NekDouble> X(nNodes), Y(nNodes), Z(nNodes);
        for (int j = 0; j < nNodes; j++)
        {
            X[j] = *nodes[j * m_dim];
            Y[j] = *nodes[j * m_dim + 1];
            Z[j] = *nodes[j * m_dim + 2];
        }
        std::vector<NekDouble> x1i2(nNodes), y1i2(nNodes), z1i2(nNodes);
        std::vector<NekDouble> x2i2(nNodes), y2i2(nNodes), z2i2(nNodes);
        std::vector<NekDouble> x3i2(nNodes), y3i2(nNodes), z3i2(nNodes);

        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[0].GetRawPtr(), nNodes, &X[0], 1, 0.0,
                    &x1i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[0].GetRawPtr(), nNodes, &Y[0], 1, 0.0,
                    &y1i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[0].GetRawPtr(), nNodes, &Z[0], 1, 0.0,
                    &z1i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[1].GetRawPtr(), nNodes, &X[0], 1, 0.0,
                    &x2i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[1].GetRawPtr(), nNodes, &Y[0], 1, 0.0,
                    &y2i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[1].GetRawPtr(), nNodes, &Z[0], 1, 0.0,
                    &z2i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[2].GetRawPtr(), nNodes, &X[0], 1, 0.0,
                    &x3i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[2].GetRawPtr(), nNodes, &Y[0], 1, 0.0,
                    &y3i2[0], 1.0);
        Blas::Dgemv('N', nNodes, nNodes, 1.0,
                    m_derivUtil->VdmDStd[2].GetRawPtr(), nNodes, &Z[0], 1, 0.0,
                    &z3i2[0], 1.0);

        for (int j = 0; j < nNodes; j++)
        {
            NekDouble jacDet =
                x1i2[j] * (y2i2[j] * z3i2[j] - z2i2[j] * y3i2[j]) -
                x2i2[j] * (y1i2[j] * z3i2[j] - z1i2[j] * y3i2[j]) +
                x3i2[j] * (y1i2[j] * z2i2[j] - z1i2[j] * y2i2[j]);

            mx2 = max(mx2, jacDet / mapsStd[j * mapStdStride + 9]);
            mn2 = min(mn2, jacDet / mapsStd[j * mapStdStride + 9]);
        }
    }

    mtx2.lock();
    if (mn2 < 0)
    {
        m_res->startInv++;
    }
    m_res->worstJac = min(m_res->worstJac, (mn2 / mx2));
    mtx2.unlock();

    m_scaledJac = (mn2 / mx2);
}

/**
 * @brief Find the smallest Jacobian of this element at the integration
 * points, and reduce it into the residual's mesh-wide minimum.
 *
 * These are the points the energy functional is integrated on, so this is the
 * Jacobian the Jacobian regularisation has to cope with. Evaluate() above
 * measures the element's quality at its nodes instead, which is a different
 * question and a different set of points.
 *
 * Reads the derivatives CalcDeriv() leaves behind, so it has to follow it.
 */
void ElUtil::CalcMinJac()
{
    NekDouble mn  = numeric_limits<double>::max();
    const int pts = m_derivUtil->pts;

    ASSERTL0(deriv.size() == size_t(m_dim) * m_dim * pts,
             "CalcDeriv must run before CalcMinJac");

    // Zeroed once rather than per point: only the leading m_dim by m_dim
    // block is written below, and in two dimensions the determinant does not
    // read the rest -- but the compiler cannot see that, and warns.
    NekDouble jac[3][3] = {};

    for (int j = 0; j < pts; j++)
    {
        for (int d = 0; d < m_dim; ++d)
        {
            for (int c = 0; c < m_dim; ++c)
            {
                jac[c][d] = deriv[(d * m_dim + c) * pts + j];
            }
        }

        NekDouble jacDet =
            m_dim == 2
                ? jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
                : jac[0][0] * (jac[1][1] * jac[2][2] - jac[2][1] * jac[1][2]) -
                      jac[0][1] *
                          (jac[1][0] * jac[2][2] - jac[2][0] * jac[1][2]) +
                      jac[0][2] *
                          (jac[1][0] * jac[2][1] - jac[2][0] * jac[1][1]);

        // The ideal mapping's determinant turns det(grad phi_M) into the J
        // that the functional sees. The three-dimensional case used to leave
        // this out, which left the regularisation parameter scaled by the
        // size of the elements.
        mn = min(mn, jacDet / maps[j * mapStride + 9]);
    }

    mtx2.lock();
    m_res->minJac = min(m_res->minJac, mn);
    mtx2.unlock();
}

void ElUtil::MoveNode(int id, const NekDouble *offset, int dim)
{
    const int pts = m_derivUtil->pts;

    for (int d = 0; d < dim; ++d)
    {
        const NekDouble *basis = m_derivUtil->VdmD[d].GetRawPtr() + id * pts;

        for (int c = 0; c < dim; ++c)
        {
            NekDouble *out = &deriv[(d * dim + c) * pts];
            for (int k = 0; k < pts; ++k)
            {
                out[k] += offset[c] * basis[k];
            }
        }
    }
}

void ElUtil::CalcDeriv()
{
    const int nNodes = m_nNodes;
    const int pts    = m_derivUtil->pts;

    deriv.resize(size_t(m_dim) * m_dim * pts);

    // Gathered coordinates, one per thread: this runs on every element of
    // every thread, once per iteration.
    static thread_local std::vector<NekDouble> X;
    X.resize(size_t(m_dim) * nNodes);

    for (int c = 0; c < m_dim; ++c)
    {
        for (int j = 0; j < nNodes; j++)
        {
            X[c * nNodes + j] = *nodes[j * m_dim + c];
        }
    }

    for (int d = 0; d < m_dim; ++d)
    {
        Blas::Dgemm('N', 'N', pts, m_dim, nNodes, 1.0,
                    m_derivUtil->VdmD[d].GetRawPtr(), pts, &X[0], nNodes, 0.0,
                    &deriv[size_t(d) * m_dim * pts], pts);
    }
}

bool ElUtil::PreUpdateMapping(AdaptCurveVector &adaptCurves, NekDouble scale,
                              NekDouble rad, SpatialDomains::MeshGraph *graph)
{
    ASSERTL0(m_dim > 1, "Adaption for mesh dim < 2 not implemented.");

    std::vector<SpatialDomains::CADCurveSharedPtr> radaptCurves;

    // Bounding boxes are {xmin, ymin, zmin, xmax, ymax, zmax}. The box is
    // held by the locator rather than the geometry, so one is built here and
    // dropped again; this runs once per element, not per search.
    auto el_bb =
        SpatialDomains::GeometryLocator::Create(m_el)->GetBoundingBox();

    for (auto &curve : adaptCurves)
    {
        auto &curve_bb = curve.second;

        bool x_outside = (el_bb[3] <= curve_bb[0]) || (el_bb[0] >= curve_bb[3]);
        bool y_outside = (el_bb[4] <= curve_bb[1]) || (el_bb[1] >= curve_bb[4]);
        bool z_outside = (m_dim > 2) ? ((el_bb[5] <= curve_bb[2]) ||
                                        (el_bb[2] >= curve_bb[5]))
                                     : false;

        if (!(x_outside || y_outside || z_outside))
        {
            radaptCurves.push_back(curve.first);
        }
    }

    if (!radaptCurves.empty())
    {
        SetScalingFromInput(scale, rad, radaptCurves, graph);
        return true;
    }
    else
    {
        return false;
    }
}

void ElUtil::UpdateMapping()
{
    NekDouble scaling = 1.0;
    if (m_interp.GetInField())
    {
        if (!m_interpField)
        {
            Array<OneD, Array<OneD, NekDouble>> centre(m_dim + 1);
            for (int i = 0; i < m_dim + 1; ++i)
            {
                centre[i] = Array<OneD, NekDouble>(1, 0.0);
            }

            vector<string> fieldNames;
            fieldNames.push_back("");

            map<LibUtilities::PtsInfo, int> ptsInfo =
                LibUtilities::NullPtsInfoMap;

            m_interpField =
                MemoryManager<LibUtilities::PtsField>::AllocateSharedPtr(
                    m_dim, fieldNames, centre, ptsInfo);
        }

        const int nVerts = m_el->GetNumVerts();

        vector<NekDouble> centre(m_dim, 0.0);
        for (int i = 0; i < nVerts; ++i)
        {
            SpatialDomains::PointGeom *v = m_el->GetVertex(i);
            centre[0] += (*v)[0];
            centre[1] += (*v)[1];
            if (m_dim > 2)
            {
                centre[2] += (*v)[2];
            }
        }

        m_interpField->SetPointVal(0, 0, centre[0] / nVerts);
        m_interpField->SetPointVal(1, 0, centre[1] / nVerts);
        if (m_dim > 2)
        {
            m_interpField->SetPointVal(2, 0, centre[2] / nVerts);
        }

        m_interp.CalcWeights(m_interp.GetInField(), m_interpField, true);
        m_interp.Interpolate(m_interp.GetInField(), m_interpField);

        scaling = m_interpField->GetPointVal(m_dim, 0);
    }

    // r-adaption scale manually set
    else if (m_radapt)
    {
        if (m_adapt_radius) // Using elements with a node within a radius from
                            // the curve
        {
            [&] {
                for (int i = 0; i < m_el->GetNumVerts(); ++i)
                {
                    SpatialDomains::PointGeom *vert = m_el->GetVertex(i);
                    std::array<NekDouble, 3> x;
                    x[0] = (*vert)[0];
                    x[1] = (*vert)[1];
                    x[2] = (*vert)[2];
                    for (auto &curve : m_adaptcurves)
                    {
                        if (curve->GetMinDistance(x) < m_adapt_radius)
                        {
                            scaling = m_adapt_scale;
                            return; // return lambda function to break loop
                        }
                    }
                }
            }();
        }
        // Using only elements with a node on the curve. Without CAD there are
        // no curves to sit on, so the scaling stays at its default.
        else if (m_graph->HasCAD())
        {
            [&] {
                for (int i = 0; i < m_el->GetNumVerts(); ++i)
                {
                    for (auto &curve : m_graph->GetCADAssociation()->GetCurves(
                             m_el->GetVertex(i)))
                    {
                        auto it = std::find(m_adaptcurves.begin(),
                                            m_adaptcurves.end(), curve);
                        if (it != m_adaptcurves.end())
                        {
                            scaling = m_adapt_scale;
                            return; // return lambda function to break loop
                        }
                    }
                }
            }();
        }
    }

    if (scaling == 1.0 && m_orig.empty())
    {
        // Never scaled, so the arrays already hold the ideal mapping.
        return;
    }

    // The first scaling has to keep the unscaled mapping to work from, since
    // each later one is applied to that rather than compounded.
    if (m_orig.empty())
    {
        m_orig    = maps;
        m_origStd = mapsStd;
    }

    ASSERTL0(m_dim > 1, "Scaling for mesh dim < 2 not implemented.");
    const NekDouble det_scaling =
        m_dim == 2 ? scaling * scaling : scaling * scaling * scaling;

    for (size_t i = 0; i < m_orig.size(); i += 10)
    {
        for (int j = 0; j < 9; ++j)
        {
            maps[i + j] = m_orig[i + j] / scaling;
        }
        maps[i + 9] = m_orig[i + 9] * det_scaling;
    }
    for (size_t i = 0; i < m_origStd.size(); i += 10)
    {
        for (int j = 0; j < 9; ++j)
        {
            mapsStd[i + j] = m_origStd[i + j] / scaling;
        }
        mapsStd[i + 9] = m_origStd[i + 9] * det_scaling;
    }
}

ElUtilJob *ElUtil::GetJob(bool update)
{
    return new ElUtilJob(this, update);
}

ElUtilJob *ElUtil::GetAdaptJob(AdaptCurveVector &adaptCurves, NekDouble scale,
                               NekDouble rad, SpatialDomains::MeshGraph *graph)
{
    return new ElUtilJob(this, adaptCurves, scale, rad, graph);
}

} // namespace Nektar::NekMesh
