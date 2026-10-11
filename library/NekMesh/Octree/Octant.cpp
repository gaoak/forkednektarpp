////////////////////////////////////////////////////////////////////////////////
//
//  File: Octant.cpp
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
//  Description: octant object methods.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <NekMesh/Octree/Octant.h>

using namespace std;
namespace Nektar::NekMesh
{

inline OctantFace GetReverseFace(OctantFace f)
{
    switch (f)
    {
        case eUp:
            return eDown;
        case eDown:
            return eUp;
        case eForward:
            return eBack;
        case eBack:
            return eForward;
        case eLeft:
            return eRight;
        case eRight:
            return eLeft;
    }

    return eUp;
}

Octant::Octant(int i, Octant *p, std::array<OctantFace, 3> dir)
    : m_id(i), m_parent(p)
{
    // initialise variables to defualt states
    m_leaf              = true;
    m_needToDivide      = false;
    m_numValidPoints    = 0;
    m_numBoundaryPoints = 0;
    m_delta             = pair<bool, NekDouble>(false, 0.0);
    NekDouble maxDif    = 0;
    NekDouble minDif    = numeric_limits<double>::max();
    m_location          = eUnknown;

    // build empty neigbour map

    // pull information from parent
    auto parentloc = m_parent->GetLoc();
    m_hd           = m_parent->DX() / 2.0;

    if (dir[0] == eForward)
    {
        m_loc[0] = parentloc[0] + m_hd;
    }
    else
    {
        m_loc[0] = parentloc[0] - m_hd;
    }
    if (dir[1] == eUp)
    {
        m_loc[1] = parentloc[1] + m_hd;
    }
    else
    {
        m_loc[1] = parentloc[1] - m_hd;
    }
    if (dir[2] == eLeft)
    {
        m_loc[2] = parentloc[2] + m_hd;
    }
    else
    {
        m_loc[2] = parentloc[2] - m_hd;
    }

    const vector<const SourcePoint *> &SourcePointList = m_parent->GetSPList();

    // setup complete

    // look over the curvature point list provided by the parent,
    // firstly look to see if it is in the new octant and if so
    // add it to the conserdation of the delta specification
    for (int i = 0; i < SourcePointList.size(); i++)
    {
        auto cploc = SourcePointList[i]->GetLoc();
        if (!(cploc[0] > m_loc[0] + m_hd || cploc[0] < m_loc[0] - m_hd ||
              cploc[1] > m_loc[1] + m_hd || cploc[1] < m_loc[1] - m_hd ||
              cploc[2] > m_loc[2] + m_hd || cploc[2] < m_loc[2] - m_hd))
        {
            m_localSPList.push_back(SourcePointList[i]);

            if (SourcePointList[i]->HasDelta())
            {
                if (SourcePointList[i]->GetDelta() > maxDif)
                {
                    maxDif = SourcePointList[i]->GetDelta();
                }
                if (SourcePointList[i]->GetDelta() < minDif)
                {
                    minDif = SourcePointList[i]->GetDelta();
                }
                m_numValidPoints++;
            }

            if (SourcePointList[i]->HasRDelta())
            {
                if (SourcePointList[i]->GetRDelta() < minDif)
                {
                    minDif = SourcePointList[i]->GetRDelta();
                }
            }

            if (SourcePointList[i]->Isboundary())
            {
                m_numBoundaryPoints++;
            }
        }
    }

    // if it has valid points delta can be calculated
    if (NumValidCurvePoint() > 0)
    {
        // geometrically octant should subdivide
        if (maxDif / minDif > 1.5)
        {
            m_needToDivide = true;
        }

        SetDelta(minDif);

        // encourage subdivision to keep spec smooth
        if (GetDelta() < 5.0 * DX())
        {
            m_needToDivide = true;
        }
    }

    if (GetNumBoundary() > 0)
    {
        m_location = eOnBoundary;
    }
}

// constructor for the master octant
Octant::Octant(int i, NekDouble x, NekDouble y, NekDouble z, NekDouble dx,
               const vector<const SourcePoint *> &splist)
    : m_id(i), m_hd(dx)
{

    // initialise variables to defualt states
    m_leaf           = true;
    m_needToDivide   = true;
    m_numValidPoints = 0;
    m_delta          = pair<bool, NekDouble>(false, 0.0);
    m_loc            = {x, y, z};

    m_localSPList = splist;

    for (int i = 0; i < m_localSPList.size(); i++)
    {
        if (m_localSPList[i]->HasDelta())
        {
            m_numValidPoints++;
        }
    }

    m_location = eOnBoundary;
}

void Octant::Subdivide(Octant *p, int &numoct)
{
    ASSERTL0(m_leaf, "octant must be a leaf for subdivision");

    m_leaf = false; // set as not leaf and make children

    // need to loop over all neigbours and remove this octant from their lists
    for (int i = 0; i < 6; i++)
    {
        OctantFace f = static_cast<OctantFace>(i);

        // By reference, but still through operator[]: the empty entry it
        // inserts for a face with no neighbours is meaningful further on,
        // where an empty list is what says the face is on the wall of the
        // octree. Nothing here changes this octant's own lists -- the call
        // below edits the neighbour's -- so there is nothing to copy for.
        vector<Octant *> &os = m_neigbours[f];
        for (int j = 0; j < os.size(); j++)
        {
            os[j]->RemoveNeigbour(GetId(), GetReverseFace(f));
        }
    }

    std::array<OctantSharedPtr, 8> children;

    for (int i = 0; i < 8; i++)
    {
        // set up x,y,z ordering of the 8 octants
        std::array<OctantFace, 3> dir;
        if (i < 4)
        {
            dir[0] = eForward;
            if (i < 2)
            {
                dir[1] = eUp;
            }
            else
            {
                dir[1] = eDown;
            }
            if (i == 0 || i == 2)
            {
                dir[2] = eLeft;
            }
            else
            {
                dir[2] = eRight;
            }
        }
        else
        {
            dir[0] = eBack;
            if (i < 6)
            {
                dir[1] = eUp;
            }
            else
            {
                dir[1] = eDown;
            }
            if (i == 4 || i == 6)
            {
                dir[2] = eLeft;
            }
            else
            {
                dir[2] = eRight;
            }
        }

        children[i] = std::shared_ptr<Octant>(new Octant(numoct++, p, dir));
    }

    SetChildren(children);

    // this set of neibours are based on the children of the octant, only covers
    // three sides
    children[0]->SetNeigbour(children[1].get(), eRight);
    children[0]->SetNeigbour(children[4].get(), eBack);
    children[0]->SetNeigbour(children[2].get(), eDown);

    children[1]->SetNeigbour(children[0].get(), eLeft);
    children[1]->SetNeigbour(children[5].get(), eBack);
    children[1]->SetNeigbour(children[3].get(), eDown);

    children[2]->SetNeigbour(children[3].get(), eRight);
    children[2]->SetNeigbour(children[6].get(), eBack);
    children[2]->SetNeigbour(children[0].get(), eUp);

    children[3]->SetNeigbour(children[2].get(), eLeft);
    children[3]->SetNeigbour(children[7].get(), eBack);
    children[3]->SetNeigbour(children[1].get(), eUp);

    children[4]->SetNeigbour(children[5].get(), eRight);
    children[4]->SetNeigbour(children[0].get(), eForward);
    children[4]->SetNeigbour(children[6].get(), eDown);

    children[5]->SetNeigbour(children[4].get(), eLeft);
    children[5]->SetNeigbour(children[1].get(), eForward);
    children[5]->SetNeigbour(children[7].get(), eDown);

    children[6]->SetNeigbour(children[7].get(), eRight);
    children[6]->SetNeigbour(children[2].get(), eForward);
    children[6]->SetNeigbour(children[4].get(), eUp);

    children[7]->SetNeigbour(children[6].get(), eLeft);
    children[7]->SetNeigbour(children[3].get(), eForward);
    children[7]->SetNeigbour(children[5].get(), eUp);

    // need to obtain the remaning information from the parents neigbours
    // (m_neigbours)
    // consider top face
    if (m_neigbours[eUp].size() == 1)
    {
        children[0]->SetNeigbour(m_neigbours[eUp][0], eUp);
        children[1]->SetNeigbour(m_neigbours[eUp][0], eUp);
        children[4]->SetNeigbour(m_neigbours[eUp][0], eUp);
        children[5]->SetNeigbour(m_neigbours[eUp][0], eUp);
        m_neigbours[eUp][0]->SetNeigbour(children[0].get(), eDown);
        m_neigbours[eUp][0]->SetNeigbour(children[1].get(), eDown);
        m_neigbours[eUp][0]->SetNeigbour(children[4].get(), eDown);
        m_neigbours[eUp][0]->SetNeigbour(children[5].get(), eDown);
    }
    else if (m_neigbours[eUp].size() == 4)
    {
        children[0]->SetNeigbour(m_neigbours[eUp][0], eUp); // 2
        children[1]->SetNeigbour(m_neigbours[eUp][1], eUp); // 3
        children[4]->SetNeigbour(m_neigbours[eUp][2], eUp); // 6
        children[5]->SetNeigbour(m_neigbours[eUp][3], eUp); // 7
        m_neigbours[eUp][0]->SetNeigbour(children[0].get(), eDown);
        m_neigbours[eUp][1]->SetNeigbour(children[1].get(), eDown);
        m_neigbours[eUp][2]->SetNeigbour(children[4].get(), eDown);
        m_neigbours[eUp][3]->SetNeigbour(children[5].get(), eDown);
    }
    else if (m_neigbours[eUp].size() != 0)
    {
    }

    if (m_neigbours[eDown].size() == 1)
    {
        children[2]->SetNeigbour(m_neigbours[eDown][0], eDown);
        children[3]->SetNeigbour(m_neigbours[eDown][0], eDown);
        children[6]->SetNeigbour(m_neigbours[eDown][0], eDown);
        children[7]->SetNeigbour(m_neigbours[eDown][0], eDown);
        m_neigbours[eDown][0]->SetNeigbour(children[2].get(), eUp);
        m_neigbours[eDown][0]->SetNeigbour(children[3].get(), eUp);
        m_neigbours[eDown][0]->SetNeigbour(children[6].get(), eUp);
        m_neigbours[eDown][0]->SetNeigbour(children[7].get(), eUp);
    }
    else if (m_neigbours[eDown].size() == 4)
    {
        children[2]->SetNeigbour(m_neigbours[eDown][0], eDown); // 0
        children[3]->SetNeigbour(m_neigbours[eDown][1], eDown); // 1
        children[6]->SetNeigbour(m_neigbours[eDown][2], eDown); // 4
        children[7]->SetNeigbour(m_neigbours[eDown][3], eDown); // 5
        m_neigbours[eDown][0]->SetNeigbour(children[2].get(), eUp);
        m_neigbours[eDown][1]->SetNeigbour(children[3].get(), eUp);
        m_neigbours[eDown][2]->SetNeigbour(children[6].get(), eUp);
        m_neigbours[eDown][3]->SetNeigbour(children[7].get(), eUp);
    }
    else if (m_neigbours[eDown].size() != 0)
    {
    }

    if (m_neigbours[eForward].size() == 1)
    {
        children[0]->SetNeigbour(m_neigbours[eForward][0], eForward);
        children[1]->SetNeigbour(m_neigbours[eForward][0], eForward);
        children[2]->SetNeigbour(m_neigbours[eForward][0], eForward);
        children[3]->SetNeigbour(m_neigbours[eForward][0], eForward);
        m_neigbours[eForward][0]->SetNeigbour(children[0].get(), eBack);
        m_neigbours[eForward][0]->SetNeigbour(children[1].get(), eBack);
        m_neigbours[eForward][0]->SetNeigbour(children[2].get(), eBack);
        m_neigbours[eForward][0]->SetNeigbour(children[3].get(), eBack);
    }
    else if (m_neigbours[eForward].size() == 4)
    {
        children[0]->SetNeigbour(m_neigbours[eForward][0], eForward); // 4
        children[1]->SetNeigbour(m_neigbours[eForward][1], eForward); // 5
        children[2]->SetNeigbour(m_neigbours[eForward][2], eForward); // 6
        children[3]->SetNeigbour(m_neigbours[eForward][3], eForward); // 7
        m_neigbours[eForward][0]->SetNeigbour(children[0].get(), eBack);
        m_neigbours[eForward][1]->SetNeigbour(children[1].get(), eBack);
        m_neigbours[eForward][2]->SetNeigbour(children[2].get(), eBack);
        m_neigbours[eForward][3]->SetNeigbour(children[3].get(), eBack);
    }
    else if (m_neigbours[eForward].size() != 0)
    {
    }

    if (m_neigbours[eBack].size() == 1)
    {
        children[4]->SetNeigbour(m_neigbours[eBack][0], eBack);
        children[5]->SetNeigbour(m_neigbours[eBack][0], eBack);
        children[6]->SetNeigbour(m_neigbours[eBack][0], eBack);
        children[7]->SetNeigbour(m_neigbours[eBack][0], eBack);
        m_neigbours[eBack][0]->SetNeigbour(children[4].get(), eForward);
        m_neigbours[eBack][0]->SetNeigbour(children[5].get(), eForward);
        m_neigbours[eBack][0]->SetNeigbour(children[6].get(), eForward);
        m_neigbours[eBack][0]->SetNeigbour(children[7].get(), eForward);
    }
    else if (m_neigbours[eBack].size() == 4)
    {
        children[4]->SetNeigbour(m_neigbours[eBack][0], eBack); // 0
        children[5]->SetNeigbour(m_neigbours[eBack][1], eBack); // 1
        children[6]->SetNeigbour(m_neigbours[eBack][2], eBack); // 2
        children[7]->SetNeigbour(m_neigbours[eBack][3], eBack); // 3
        m_neigbours[eBack][0]->SetNeigbour(children[4].get(), eForward);
        m_neigbours[eBack][1]->SetNeigbour(children[5].get(), eForward);
        m_neigbours[eBack][2]->SetNeigbour(children[6].get(), eForward);
        m_neigbours[eBack][3]->SetNeigbour(children[7].get(), eForward);
    }
    else if (m_neigbours[eBack].size() != 0)
    {
    }

    if (m_neigbours[eLeft].size() == 1)
    {
        children[0]->SetNeigbour(m_neigbours[eLeft][0], eLeft);
        children[2]->SetNeigbour(m_neigbours[eLeft][0], eLeft);
        children[4]->SetNeigbour(m_neigbours[eLeft][0], eLeft);
        children[6]->SetNeigbour(m_neigbours[eLeft][0], eLeft);
        m_neigbours[eLeft][0]->SetNeigbour(children[0].get(), eRight);
        m_neigbours[eLeft][0]->SetNeigbour(children[2].get(), eRight);
        m_neigbours[eLeft][0]->SetNeigbour(children[4].get(), eRight);
        m_neigbours[eLeft][0]->SetNeigbour(children[6].get(), eRight);
    }
    else if (m_neigbours[eLeft].size() == 4)
    {
        children[0]->SetNeigbour(m_neigbours[eLeft][0], eLeft); // 1
        children[2]->SetNeigbour(m_neigbours[eLeft][1], eLeft); // 3
        children[4]->SetNeigbour(m_neigbours[eLeft][2], eLeft); // 5
        children[6]->SetNeigbour(m_neigbours[eLeft][3], eLeft); // 7
        m_neigbours[eLeft][0]->SetNeigbour(children[0].get(), eRight);
        m_neigbours[eLeft][1]->SetNeigbour(children[2].get(), eRight);
        m_neigbours[eLeft][2]->SetNeigbour(children[4].get(), eRight);
        m_neigbours[eLeft][3]->SetNeigbour(children[6].get(), eRight);
    }
    else if (m_neigbours[eLeft].size() != 0)
    {
    }

    if (m_neigbours[eRight].size() == 1)
    {
        children[1]->SetNeigbour(m_neigbours[eRight][0], eRight);
        children[3]->SetNeigbour(m_neigbours[eRight][0], eRight);
        children[5]->SetNeigbour(m_neigbours[eRight][0], eRight);
        children[7]->SetNeigbour(m_neigbours[eRight][0], eRight);
        m_neigbours[eRight][0]->SetNeigbour(children[1].get(), eLeft);
        m_neigbours[eRight][0]->SetNeigbour(children[3].get(), eLeft);
        m_neigbours[eRight][0]->SetNeigbour(children[5].get(), eLeft);
        m_neigbours[eRight][0]->SetNeigbour(children[7].get(), eLeft);
    }
    else if (m_neigbours[eRight].size() == 4)
    {
        children[1]->SetNeigbour(m_neigbours[eRight][0], eRight); // 0
        children[3]->SetNeigbour(m_neigbours[eRight][1], eRight); // 2
        children[5]->SetNeigbour(m_neigbours[eRight][2], eRight); // 4
        children[7]->SetNeigbour(m_neigbours[eRight][3], eRight); // 6
        m_neigbours[eRight][0]->SetNeigbour(children[1].get(), eLeft);
        m_neigbours[eRight][1]->SetNeigbour(children[3].get(), eLeft);
        m_neigbours[eRight][2]->SetNeigbour(children[5].get(), eLeft);
        m_neigbours[eRight][3]->SetNeigbour(children[7].get(), eLeft);
    }
    else if (m_neigbours[eRight].size() != 0)
    {
    }

    // The children have taken the points they need, and nothing reads an
    // octant's point list once it has stopped being a leaf -- the only reader
    // is the child constructor above, and the neighbour lists everything else
    // works through hold leaves only.
    m_localSPList.clear();
    m_localSPList.shrink_to_fit();
}

void Octant::RemoveNeigbour(int id, OctantFace f)
{
    vector<Octant *> &neigbours = m_neigbours[f];

    neigbours.erase(
        std::remove_if(neigbours.begin(), neigbours.end(),
                       [id](const Octant *o) { return o->GetId() == id; }),
        neigbours.end());
}

} // namespace Nektar::NekMesh
