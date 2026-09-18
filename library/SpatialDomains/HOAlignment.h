////////////////////////////////////////////////////////////////////////////////
//
//  File: HOAlignment.h
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
//  Description: HO aligment routines.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SPATIALDOMAINS_HOALIGNMENT
#define NEKTAR_SPATIALDOMAINS_HOALIGNMENT

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <SpatialDomains/PointGeom.h>
#include <array>
#include <map>

namespace Nektar::SpatialDomains
{
/**
 * @brief A lightweight struct for dealing with high-order triangle
 * alignment.
 *
 * The logic underlying these routines is taken from the original Nektar
 * code.
 */
template <typename T> struct HOTriangle
{
    HOTriangle(std::vector<int> pVertId, std::vector<T> pSurfVerts)
        : vertId(pVertId), surfVerts(pSurfVerts)
    {
    }
    HOTriangle(std::vector<int> pVertId) : vertId(pVertId)
    {
    }

    /// The triangle vertex IDs
    std::vector<int> vertId;

    /// The triangle surface vertices -- templated so that this can
    /// either be nodes or IDs.
    std::vector<T> surfVerts;

    /**
     * @brief Rotates the triangle of data points inside #surfVerts
     * counter-clockwise nrot times.
     *
     * @param nrot Number of times to rotate triangle.
     */
    void Rotate(int nrot)
    {
        int n, i, j, cnt;
        int np = ((int)sqrt(8.0 * surfVerts.size() + 1.0) - 1) / 2;
        std::vector<T> tmp(np * np);

        for (n = 0; n < nrot; ++n)
        {
            for (cnt = i = 0; i < np; ++i)
            {
                for (j = 0; j < np - i; ++j, cnt++)
                {
                    tmp[i * np + j] = surfVerts[cnt];
                }
            }
            for (cnt = i = 0; i < np; ++i)
            {
                for (j = 0; j < np - i; ++j, cnt++)
                {
                    surfVerts[cnt] = tmp[(np - 1 - i - j) * np + i];
                }
            }
        }
    }

    /**
     * @brief Reflect data points inside #surfVerts.
     *
     * This applies a mapping essentially doing the following
     * reordering:
     *
     * 9          9
     * 7 8    ->  8 7
     * 4 5 6      6 5 4
     * 0 1 2 3    3 2 1 0
     */
    void Reflect()
    {
        int i, j, cnt;
        int np = ((int)sqrt(8.0 * surfVerts.size() + 1.0) - 1) / 2;
        std::vector<T> tmp(np * np);

        for (cnt = i = 0; i < np; ++i)
        {
            for (j = 0; j < np - i; ++j, cnt++)
            {
                tmp[i * np + np - i - 1 - j] = surfVerts[cnt];
            }
        }

        for (cnt = i = 0; i < np; ++i)
        {
            for (j = 0; j < np - i; ++j, cnt++)
            {
                surfVerts[cnt] = tmp[i * np + j];
            }
        }
    }

    /**
     * @brief Align this surface to a given vertex ID.
     */
    void Align(std::vector<int> vertId)
    {
        if (vertId[0] == this->vertId[0])
        {
            if (vertId[1] == this->vertId[1] || vertId[1] == this->vertId[2])
            {
                if (vertId[1] == this->vertId[2])
                {
                    Rotate(1);
                    Reflect();
                }
            }
        }
        else if (vertId[0] == this->vertId[1])
        {
            if (vertId[1] == this->vertId[0] || vertId[1] == this->vertId[2])
            {
                if (vertId[1] == this->vertId[0])
                {
                    Reflect();
                }
                else
                {
                    Rotate(2);
                }
            }
        }
        else if (vertId[0] == this->vertId[2])
        {
            if (vertId[1] == this->vertId[0] || vertId[1] == this->vertId[1])
            {
                if (vertId[1] == this->vertId[1])
                {
                    Rotate(2);
                    Reflect();
                }
                else
                {
                    Rotate(1);
                }
            }
        }
    }
};

/**
 * @brief A lightweight struct for dealing with high-order quadrilateral
 * alignment.
 */
template <typename T> struct HOQuadrilateral
{
    HOQuadrilateral(std::vector<int> pVertId, std::vector<T> pSurfVerts)
        : vertId(pVertId), surfVerts(pSurfVerts)
    {
    }

    HOQuadrilateral(std::vector<int> pVertId) : vertId(pVertId)
    {
    }

    /// The quadrilateral vertex IDs
    std::vector<int> vertId;

    /// The quadrilateral surface vertices -- templated so that this can either
    /// be nodes or IDs.
    std::vector<T> surfVerts;

    void ReverseX()
    {
        int np = (int)(sqrt((NekDouble)surfVerts.size()) + 0.5);
        for (int i = 0; i < np; ++i)
        {
            for (int j = 0; j < np / 2; ++j)
            {
                std::swap(surfVerts[i * np + j],
                          surfVerts[i * np + np - j - 1]);
            }
        }
    }

    void ReverseY()
    {
        int np = (int)(sqrt((NekDouble)surfVerts.size()) + 0.5);
        // Reverse y direction
        for (int j = 0; j < np; ++j)
        {
            for (int i = 0; i < np / 2; ++i)
            {
                std::swap(surfVerts[i * np + j],
                          surfVerts[(np - i - 1) * np + j]);
            }
        }
    }

    void Transpose()
    {
        int np = (int)(sqrt((NekDouble)surfVerts.size()) + 0.5);
        std::vector<T> tmp(surfVerts.size());

        for (int i = 0; i < np; ++i)
        {
            for (int j = 0; j < np; ++j)
            {
                tmp[i * np + j] = surfVerts[j * np + i];
            }
        }

        surfVerts = tmp;
    }

    /**
     * @brief Align this surface to a given vertex ID.
     */
    void Align(std::vector<int> vertId)
    {

        int vmap[4] = {-1, -1, -1, -1};

        // Determine which vertices map to vertId
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (this->vertId[j] == vertId[i])
                {
                    vmap[i] = j;
                    break;
                }
            }

            ASSERTL1(vmap[i] != -1,
                     "Could not determine mapping between vertex IDs");
        }

        StdRegions::Orientation orient = StdRegions::eNoOrientation;

        if (vmap[1] == (vmap[0] + 1) % 4)
        {
            switch (vmap[0])
            {
                case 0:
                    orient = StdRegions::eDir1FwdDir1_Dir2FwdDir2;
                    break;
                case 1:
                    orient = StdRegions::eDir1BwdDir2_Dir2FwdDir1;
                    break;
                case 2:
                    orient = StdRegions::eDir1BwdDir1_Dir2BwdDir2;
                    break;
                case 3:
                    orient = StdRegions::eDir1FwdDir2_Dir2BwdDir1;
                    break;
            }
        }
        else
        {
            switch (vmap[0])
            {
                case 0:
                    orient = StdRegions::eDir1FwdDir2_Dir2FwdDir1;
                    break;
                case 1:
                    orient = StdRegions::eDir1BwdDir1_Dir2FwdDir2;
                    break;
                case 2:
                    orient = StdRegions::eDir1BwdDir2_Dir2BwdDir1;
                    break;
                case 3:
                    orient = StdRegions::eDir1FwdDir1_Dir2BwdDir2;
                    break;
            }
        }

        if (orient == StdRegions::eDir1FwdDir2_Dir2FwdDir1 ||
            orient == StdRegions::eDir1BwdDir2_Dir2FwdDir1 ||
            orient == StdRegions::eDir1FwdDir2_Dir2BwdDir1 ||
            orient == StdRegions::eDir1BwdDir2_Dir2BwdDir1)
        {
            Transpose();
        }

        if (orient == StdRegions::eDir1BwdDir1_Dir2FwdDir2 ||
            orient == StdRegions::eDir1BwdDir1_Dir2BwdDir2 ||
            orient == StdRegions::eDir1FwdDir2_Dir2BwdDir1 ||
            orient == StdRegions::eDir1BwdDir2_Dir2BwdDir1)
        {
            ReverseX();
        }

        if (orient == StdRegions::eDir1FwdDir1_Dir2BwdDir2 ||
            orient == StdRegions::eDir1BwdDir1_Dir2BwdDir2 ||
            orient == StdRegions::eDir1BwdDir2_Dir2FwdDir1 ||
            orient == StdRegions::eDir1BwdDir2_Dir2BwdDir1)
        {
            ReverseY();
        }
    }
};

/**
 * @brief The interior nodes of a high-order tetrahedron, and the vertices they
 * are positioned against.
 *
 * The counterpart of HOTriangle for the inside of an element rather than a
 * face. A tetrahedron's nodes are a nodal distribution rather than a grid, so
 * there is no axis to rotate about and no pair of directions to reflect; what
 * there is instead is each node's barycentric position, how much of each of the
 * four vertices it is made of. Those weights belong to the vertices themselves
 * rather than to any numbering of them, so relabelling the vertices is enough
 * to say where every node goes.
 *
 * Nothing here evaluates a point. Aligning is a permutation of the nodes
 * already held, so a node read from a file keeps the position the file gave it.
 */
template <typename T> struct HOTetrahedron
{
    /**
     * @param pVertId   The four vertex IDs these nodes are positioned against.
     * @param pVolVerts The nodes, in the ordering a nodal tetrahedron puts
     *                  them in.
     * @param pNumPts   Points along an edge of the element the nodes came
     *                  from, when @p pVolVerts holds every node of it. Leave
     *                  this out when it holds only the interior: the two
     *                  cannot be told apart by counting, since the interior of
     *                  one order has as many nodes as the whole of another.
     */
    HOTetrahedron(std::vector<int> pVertId, std::vector<T> pVolVerts,
                  int pNumPts = 0)
        : vertId(pVertId), volVerts(pVolVerts), numPts(pNumPts)
    {
    }
    HOTetrahedron(std::vector<int> pVertId) : vertId(pVertId)
    {
    }

    /// The tetrahedron vertex IDs.
    std::vector<int> vertId;

    /// The interior nodes, in the ordering a nodal tetrahedron of this order
    /// puts them in -- templated so that this can be either nodes or IDs.
    std::vector<T> volVerts;

    /// Points along an edge of the element, or zero if #volVerts is only its
    /// interior and the order should be worked out from that.
    int numPts = 0;

    /**
     * @brief Reorder #volVerts into the frame in which the vertices are
     * numbered as @p vertId.
     *
     * @param vertId The vertex IDs to align to, in the desired order.
     */
    void Align(std::vector<int> vertId)
    {
        // One node sits at the centre and any number below that is no node at
        // all; either way there is nothing a relabelling could move.
        if (volVerts.size() < 2)
        {
            return;
        }

        // Which of the stored vertices each of the wanted ones is.
        std::array<int, 4> vmap = {-1, -1, -1, -1};
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (vertId[i] == this->vertId[j])
                {
                    vmap[i] = j;
                    break;
                }
            }
            ASSERTL0(vmap[i] != -1, "Tetrahedron vertices do not match.");
        }

        const int nInt = volVerts.size();

        // Recover the order. Told the element's own size, the nodes are all of
        // it; otherwise they are the interior of an element with np points
        // along each edge, of which there are (np-4)(np-3)(np-2)/6.
        int np = numPts;
        if (np == 0)
        {
            np = 4;
            while ((np - 4) * (np - 3) * (np - 2) / 6 < nInt)
            {
                ++np;
            }
            ASSERTL0((np - 4) * (np - 3) * (np - 2) / 6 == nInt,
                     "Number of interior nodes is not a tetrahedral number.");
        }

        const int nTot = np * (np + 1) * (np + 2) / 6;
        ASSERTL0(nInt <= nTot, "More nodes than the element can hold.");

        // Where in the distribution the nodes we were given start: the
        // interior is its tail, and a whole element is all of it.
        const int first = nTot - nInt;

        LibUtilities::PointsKey pKey(np, LibUtilities::eNodalTetEvenlySpaced);
        Array<OneD, NekDouble> px, py, pz;
        LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

        // Barycentric weights of a node, as whole numbers of 1/(np-1) so that
        // they can be compared exactly.
        auto weights = [&](int m) {
            std::array<int, 4> w;
            w[1] = (int)std::lround(0.5 * (1.0 + px[m]) * (np - 1));
            w[2] = (int)std::lround(0.5 * (1.0 + py[m]) * (np - 1));
            w[3] = (int)std::lround(0.5 * (1.0 + pz[m]) * (np - 1));
            w[0] = (np - 1) - w[1] - w[2] - w[3];
            return w;
        };
        auto key = [np](const std::array<int, 4> &w) {
            return ((w[0] * np + w[1]) * np + w[2]) * np + w[3];
        };

        std::map<int, int> slotOf;
        for (int s = 0; s < nInt; ++s)
        {
            slotOf[key(weights(first + s))] = s;
        }

        std::vector<T> tmp(nInt);
        for (int s = 0; s < nInt; ++s)
        {
            // The same mix of the same four corners, with the labels moved.
            std::array<int, 4> want = weights(first + s), have{};
            for (int i = 0; i < 4; ++i)
            {
                have[vmap[i]] = want[i];
            }

            auto it = slotOf.find(key(have));
            ASSERTL0(it != slotOf.end(), "Could not place a tetrahedron node.");
            tmp[s] = volVerts[it->second];
        }

        volVerts = tmp;
    }
};

/**
 * @brief The nodes of a high-order prism, and the vertices they sit against.
 *
 * A prism is a triangle extruded along an axis, and the only way one is ever
 * reoriented is by turning that triangle: NekMesh's OrientPrism leaves the
 * element unrotated or spins it one way or the other, and either way both
 * triangular faces turn together and the axis is left alone. So a node is
 * placed by how much of each corner of the triangle it is made of, carried
 * along at whatever height up the axis it already had.
 *
 * As with HOTetrahedron nothing is evaluated: aligning permutes the nodes that
 * are already held.
 */
template <typename T> struct HOPrism
{
    /**
     * @param pVertId   The six vertex IDs, in the standard element's order.
     * @param pVolVerts Every node of the element, in the ordering a nodal
     *                  prism puts them in.
     * @param pNumPts   Points along an edge of the element.
     */
    HOPrism(std::vector<int> pVertId, std::vector<T> pVolVerts, int pNumPts)
        : vertId(pVertId), volVerts(pVolVerts), numPts(pNumPts)
    {
    }

    /// The prism vertex IDs.
    std::vector<int> vertId;

    /// The nodes, in nodal prism ordering.
    std::vector<T> volVerts;

    /// Points along an edge of the element.
    int numPts;

    /**
     * @brief Turn the triangular cross-section @p nrot thirds of a turn.
     */
    void Rotate(int nrot)
    {
        nrot = ((nrot % 3) + 3) % 3;
        if (nrot == 0 || volVerts.empty())
        {
            return;
        }

        const int np = numPts;
        ASSERTL0(np * (np + 1) / 2 * np == (int)volVerts.size(),
                 "Number of nodes is not a prism number.");

        LibUtilities::PointsKey pKey(np, LibUtilities::eNodalPrismEvenlySpaced);
        Array<OneD, NekDouble> px, py, pz;
        LibUtilities::PointsManager()[pKey]->GetPoints(px, py, pz);

        // Where a node sits: how much of each corner of the triangular
        // cross-section it is made of, and how far along the axis it is. The
        // triangle lies in the first and third directions, the axis is the
        // second.
        auto place = [&](int m) {
            std::array<int, 4> w;
            w[1] = (int)std::lround(0.5 * (1.0 + px[m]) * (np - 1));
            w[2] = (int)std::lround(0.5 * (1.0 + pz[m]) * (np - 1));
            w[0] = (np - 1) - w[1] - w[2];
            w[3] = (int)std::lround(0.5 * (1.0 + py[m]) * (np - 1));
            return w;
        };
        auto key = [np](const std::array<int, 4> &w) {
            return ((w[0] * np + w[1]) * np + w[2]) * np + w[3];
        };

        std::map<int, int> slotOf;
        for (size_t m = 0; m < volVerts.size(); ++m)
        {
            slotOf[key(place(m))] = m;
        }

        std::vector<T> tmp(volVerts.size());
        for (size_t m = 0; m < volVerts.size(); ++m)
        {
            // Turning the triangle moves each corner's share round to the
            // next; the height up the axis is untouched.
            std::array<int, 4> want = place(m), have{};
            for (int c = 0; c < 3; ++c)
            {
                have[(c + nrot) % 3] = want[c];
            }
            have[3] = want[3];

            auto it = slotOf.find(key(have));
            ASSERTL0(it != slotOf.end(), "Could not place a prism node.");
            tmp[m] = volVerts[it->second];
        }

        volVerts = tmp;
    }

    /**
     * @brief Turn the prism so that its vertices read as @p vertId.
     */
    void Align(std::vector<int> vertId)
    {
        // The corners of the triangular face, in the standard element's
        // numbering. Whichever of them the wanted first vertex is says how far
        // the triangle has been turned.
        static const int triVerts[3] = {0, 1, 4};

        for (int r = 0; r < 3; ++r)
        {
            if (vertId[0] == this->vertId[triVerts[r]])
            {
                Rotate(r);
                return;
            }
        }

        ASSERTL0(false, "Prism vertices do not match.");
    }
};

} // namespace Nektar::SpatialDomains
#endif
