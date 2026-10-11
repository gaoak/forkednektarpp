////////////////////////////////////////////////////////////////////////////////
//
//  File: Octant.h
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
//  Description: class of individal octree octants
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_OCTREE_OCTANT_H
#define NEKTAR_MESHUTILS_OCTREE_OCTANT_H

#include <cmath>

#include <NekMesh/Octree/SourcePoint.hpp>

#include <LibUtilities/Memory/NekMemoryManager.hpp>

namespace Nektar::NekMesh
{

/**
 * @brief enumeration of the 6 faces of a cube/octant
 */
enum OctantFace
{
    eUp,
    eDown,
    eForward,
    eBack,
    eLeft,
    eRight
};

/**
 * @brief enumeration of the possible locations of the octree with respect to
 * the CAD
 */
enum OctantLocation
{
    eInside,
    eOutside,
    eOnBoundary,
    eUnknown
};

class Octant; // have to forward declare the class for the sharedptr
typedef std::shared_ptr<Octant> OctantSharedPtr;

/**
 * @brief this class contains the infomration and methods for individal octants
 * in the Octree
 */
class Octant
{
public:
    friend class MemoryManager<Octant>;

    /**
     * @brief Defualt constructor
     */
    Octant(int i, Octant *p, std::array<OctantFace, 3> dir);

    /**
     * @brief constructor for master octant
     */
    Octant(int i, NekDouble x, NekDouble y, NekDouble z, NekDouble dx,
           const std::vector<const SourcePoint *> &splist);

    /**
     * @brief Subdivide the octant
     */
    void Subdivide(Octant *p, int &numoct);

    /**
     * @brief Recursive method which gets a list of the leaf octants
     *        Moves down levels if octant is not a leaf
     */
    void CompileLeaves(std::vector<OctantSharedPtr> &Octants)
    {
        for (int i = 0; i < 8; i++)
        {
            if (m_children[i]->IsLeaf())
            {
                Octants.push_back(m_children[i]);
            }
            else
            {
                m_children[i]->CompileLeaves(Octants);
            }
        }
    }

    /// Get the Id of the octant
    int GetId() const
    {
        return m_id;
    }

    /**
     * @brief Get the location of the center of the octant
     */
    const std::array<NekDouble, 3> &GetLoc() const
    {
        return m_loc;
    }

    /**
     * @brief Get the octants half dimension
     */
    NekDouble DX()
    {
        return m_hd;
    }

    /**
     * @brief Get the list of curvature points that are within this octant.
     */
    const std::vector<const SourcePoint *> &GetSPList() const
    {
        return m_localSPList;
    }

    const SourcePoint *GetABoundPoint()
    {
        const SourcePoint *ret = nullptr;
        bool found             = false;
        for (int i = 0; i < m_localSPList.size(); i++)
        {
            if (m_localSPList[i]->GetType() == eCBoundary ||
                m_localSPList[i]->GetType() == ePBoundary)
            {
                ret   = m_localSPList[i];
                found = true;
                break;
            }
        }
        ASSERTL0(found, "failed to find point");
        return ret;
    }

    /**
     * @brief Get the number of points in the octants cp list
     */
    int NumCurvePoint()
    {
        return m_localSPList.size();
    }

    /**
     * @brief Get the number of valid cp points in the octants list
     */
    int NumValidCurvePoint()
    {
        return m_numValidPoints;
    }

    /**
     * @brief Set the value for delta for this octant
     */
    void SetDelta(NekDouble d)
    {
        m_delta.first  = true;
        m_delta.second = d;
    }

    /**
     * @brief Get value of delta
     */
    bool HasDelta()
    {
        return m_delta.first;
    }

    /**
     * @brief Get value of delta
     */
    NekDouble GetDelta()
    {
        ASSERTL0(m_delta.first, "Tried to acsess delta of octant"
                                "which has not been set");

        return m_delta.second;
    }

    /**
     * @brief Set the children of this octant
     */
    void SetChildren(std::array<OctantSharedPtr, 8> c)
    {
        m_children = c;
    }

    /**
     * @brief Get whether the octant is a leaf or not
     */
    bool IsLeaf()
    {
        return m_leaf;
    }

    /**
     * @brief Get the far dimension in a given direction f
     */
    NekDouble FX(OctantFace f)
    {
        switch (f)
        {
            case eUp:
                return m_loc[1] + m_hd;
                break;
            case eDown:
                return m_loc[1] - m_hd;
                break;
            case eForward:
                return m_loc[0] + m_hd;
                break;
            case eBack:
                return m_loc[0] - m_hd;
                break;
            case eLeft:
                return m_loc[2] + m_hd;
                break;
            case eRight:
                return m_loc[2] - m_hd;
                break;
        }
        return 0.0;
    }

    /**
     * @brief Remove a neigbour from this octants list
     */
    void RemoveNeigbour(int id, OctantFace f);

    void SetNeigbour(Octant *o, OctantFace f)
    {
        m_neigbours[f].push_back(o);
    }

    /**
     * @brief Get the neighbours of this octant.
     */
    const std::array<std::vector<Octant *>, 6> &GetNeigbours() const
    {
        return m_neigbours;
    }

    /**
     * @brief Get whether this octants needs to divide based on the curvature
     * sampling
     */
    bool NeedDivide()
    {
        return m_needToDivide;
    }

    /**
     * @brief Get the distance from this octant to another
     */
    NekDouble Distance(Octant *o)
    {
        auto loc = o->GetLoc();
        return sqrt((loc[0] - m_loc[0]) * (loc[0] - m_loc[0]) +
                    (loc[1] - m_loc[1]) * (loc[1] - m_loc[1]) +
                    (loc[2] - m_loc[2]) * (loc[2] - m_loc[2]));
    }

    /**
     * @brief Get whether a value of delta has been set or not
     */
    bool IsDeltaKnown()
    {
        return m_delta.first;
    }

    /**
     * @brief set the location of the octant with respect to the CAD
     */
    void SetLocation(OctantLocation l)
    {
        m_location = l;
    }

    /**
     * @brief Get the location of the octant with respect to the geometry
     */
    OctantLocation GetLocation()
    {
        return m_location;
    }

    /**
     * @brief Get a specific child of this octant
     */
    Octant *GetChild(int q) const
    {
        return m_children[q].get();
    }

    int GetNumBoundary()
    {
        return m_numBoundaryPoints;
    }

private:
    /// id
    int m_id;
    /// leaf identifer
    bool m_leaf;
    /// Parent octant.
    ///
    /// @note This is not owned here; the tree is owned from the root down
    /// through #m_children.
    Octant *m_parent;
    /// list of child ids
    std::array<OctantSharedPtr, 8> m_children;
    /// x,y,z location of the octant
    std::array<NekDouble, 3> m_loc;
    /// half dimension of the octant
    NekDouble m_hd;
    /// The sizing points inside this octant, pointing into the octree's own
    /// list.
    std::vector<const SourcePoint *> m_localSPList;
    /// number of valid cp points
    int m_numValidPoints;
    int m_numBoundaryPoints;
    /// mesh sizing parameter
    std::pair<bool, NekDouble> m_delta;
    /// idenify if division is needed
    bool m_needToDivide; // asume no need to divide
    /// idenify if delta has ben set
    OctantLocation m_location;
    /// Neighbours across each of the six faces, indexed by OctantFace, not
    /// owned.
    std::array<std::vector<Octant *>, 6> m_neigbours;
};

} // namespace Nektar::NekMesh

#endif
