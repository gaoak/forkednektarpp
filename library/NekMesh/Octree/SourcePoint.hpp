////////////////////////////////////////////////////////////////////////////////
//
//  File: SourcePoint.hpp
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
//  Description: the sizing points the octree is built from
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_MESHUTILS_OCTREE_SOURCEPOINT_H
#define NEKTAR_MESHUTILS_OCTREE_SOURCEPOINT_H

#include <array>

#include <LibUtilities/BasicUtils/SharedArray.hpp>

namespace Nektar::NekMesh
{

enum SPType
{
    eCBoundary,  // on a curved boundary
    eRCBoundary, // on a curved boundary with refinement
    ePBoundary,  // on a planar boundary (R=inf)
    eSrcPoint    // source point
};

/**
 * @brief A sizing point for octree construction.
 *
 * Designed to be kept deliberately small, since they may exist across a very
 * large octree. This class is laid out so that it is exactly 64 bytes.
 */
class SourcePoint
{
public:
    /// A point on a curved boundary, with the element size its curvature asks
    /// for.
    SourcePoint(int surf, std::array<NekDouble, 2> uv,
                std::array<NekDouble, 3> loc, NekDouble delta)
        : m_loc(loc), m_uv(uv), m_delta(delta), m_sid(surf), m_type(eCBoundary)
    {
    }

    /// As above, with an additional refinement size.
    SourcePoint(int surf, std::array<NekDouble, 2> uv,
                std::array<NekDouble, 3> loc, NekDouble delta, NekDouble rdelta)
        : m_loc(loc), m_uv(uv), m_delta(delta), m_rdelta(rdelta), m_sid(surf),
          m_type(eRCBoundary)
    {
    }

    /// A point on a planar boundary, whose curvature asks for nothing.
    SourcePoint(int surf, std::array<NekDouble, 2> uv,
                std::array<NekDouble, 3> loc)
        : m_loc(loc), m_uv(uv), m_sid(surf), m_type(ePBoundary)
    {
    }

    /// A point away from any surface, carrying only a refinement size.
    SourcePoint(std::array<NekDouble, 3> loc, NekDouble rdelta)
        : m_loc(loc), m_rdelta(rdelta), m_type(eSrcPoint)
    {
    }

    SPType GetType() const
    {
        return m_type;
    }

    const std::array<NekDouble, 3> &GetLoc() const
    {
        return m_loc;
    }

    /// Zero for a kind that carries no element size; ask HasDelta() first.
    NekDouble GetDelta() const
    {
        return m_delta;
    }

    /// Zero for a kind that carries no refinement size; ask HasRDelta().
    NekDouble GetRDelta() const
    {
        return m_rdelta;
    }

    void SetDelta(NekDouble delta)
    {
        m_delta = delta;
    }

    void SetRDelta(NekDouble rdelta)
    {
        m_rdelta = rdelta;
    }

    /// The surface this point sits on and where on it. Meaningless for
    /// eSrcPoint, which sits on no surface, and left untouched for it.
    void GetCAD(int &surf, std::array<NekDouble, 2> &uv) const
    {
        if (m_type == eSrcPoint)
        {
            return;
        }

        surf = m_sid;
        uv   = m_uv;
    }

    bool HasDelta() const
    {
        return m_type == eCBoundary || m_type == eRCBoundary;
    }

    bool HasRDelta() const
    {
        return m_type == eRCBoundary || m_type == eSrcPoint;
    }

    bool Isboundary() const
    {
        return m_type == eCBoundary || m_type == ePBoundary ||
               m_type == eRCBoundary;
    }

private:
    /// x,y,z location
    std::array<NekDouble, 3> m_loc;
    /// position on the surface this point sits on
    std::array<NekDouble, 2> m_uv = {0.0, 0.0};
    /// element size the curvature here asks for
    NekDouble m_delta = 0.0;
    /// element size a refinement region here asks for
    NekDouble m_rdelta = 0.0;
    /// id of the surface this point sits on
    int m_sid = 0;
    /// which of the four kinds this is
    SPType m_type;
};

static_assert(sizeof(SourcePoint) == 64,
              "SourcePoint is held one per curvature sample, of which a fine "
              "octree has tens of millions, so its size is worth noticing.");

} // namespace Nektar::NekMesh

#endif
