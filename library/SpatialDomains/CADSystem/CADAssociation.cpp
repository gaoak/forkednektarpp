////////////////////////////////////////////////////////////////////////////////
//
//  File: CADAssociation.cpp
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
//  Description: Association between mesh entities and the CAD they lie on.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <SpatialDomains/Curve.hpp>
#include <SpatialDomains/Geometry.h>

namespace Nektar::SpatialDomains
{

namespace
{
/// Returned by GetLinks() for an entity with no association, so that callers
/// may iterate without first testing for presence.
const std::vector<CADLink> g_noLinks;
} // namespace

const std::vector<CADLink> &CADAssociation::GetLinks(const Geometry *geom) const
{
    auto it = m_links.find(geom);
    return it == m_links.end() ? g_noLinks : it->second;
}

std::vector<CADLink> CADAssociation::GetLinks(const Geometry *geom,
                                              CADType::cadType t) const
{
    std::vector<CADLink> ret;
    for (auto &link : GetLinks(geom))
    {
        if (link.Type() == t)
        {
            ret.push_back(link);
        }
    }
    return ret;
}

size_t CADAssociation::Count(const Geometry *geom, CADType::cadType t) const
{
    size_t cnt = 0;
    for (auto &link : GetLinks(geom))
    {
        cnt += (link.Type() == t) ? 1 : 0;
    }
    return cnt;
}

const CADLink *CADAssociation::GetLink(const Geometry *geom, CADType::cadType t,
                                       int cadId) const
{
    for (auto &link : GetLinks(geom))
    {
        if (link.Type() == t && link.Id() == cadId)
        {
            return &link;
        }
    }
    return nullptr;
}

const CADLink *CADAssociation::GetLink(const Geometry *geom,
                                       CADType::cadType t) const
{
    for (auto &link : GetLinks(geom))
    {
        if (link.Type() == t)
        {
            return &link;
        }
    }
    return nullptr;
}

CADSurfSharedPtr CADAssociation::GetSurf(const Geometry *geom) const
{
    const CADLink *link = GetLink(geom, CADType::eSurf);
    return link == nullptr ? CADSurfSharedPtr()
                           : std::static_pointer_cast<CADSurf>(link->obj);
}

CADCurveSharedPtr CADAssociation::GetCurve(const Geometry *geom) const
{
    const CADLink *link = GetLink(geom, CADType::eCurve);
    return link == nullptr ? CADCurveSharedPtr()
                           : std::static_pointer_cast<CADCurve>(link->obj);
}

CADVertSharedPtr CADAssociation::GetVert(const Geometry *geom) const
{
    const CADLink *link = GetLink(geom, CADType::eVert);
    return link == nullptr ? CADVertSharedPtr()
                           : std::static_pointer_cast<CADVert>(link->obj);
}

std::vector<CADSurfSharedPtr> CADAssociation::GetSurfs(
    const Geometry *geom) const
{
    std::vector<CADSurfSharedPtr> ret;
    for (auto &link : GetLinks(geom))
    {
        if (link.Type() == CADType::eSurf)
        {
            ret.push_back(std::static_pointer_cast<CADSurf>(link.obj));
        }
    }
    return ret;
}

std::vector<CADCurveSharedPtr> CADAssociation::GetCurves(
    const Geometry *geom) const
{
    std::vector<CADCurveSharedPtr> ret;
    for (auto &link : GetLinks(geom))
    {
        if (link.Type() == CADType::eCurve)
        {
            ret.push_back(std::static_pointer_cast<CADCurve>(link.obj));
        }
    }
    return ret;
}

std::array<NekDouble, 2> CADAssociation::GetSurfUV(const Geometry *geom,
                                                   int cadId) const
{
    const CADLink *link = GetLink(geom, CADType::eSurf, cadId);
    ASSERTL0(link != nullptr,
             "The desired CADSurf is not associated with this entity.");
    return link->param;
}

NekDouble CADAssociation::GetCurveT(const Geometry *geom, int cadId) const
{
    const CADLink *link = GetLink(geom, CADType::eCurve, cadId);
    ASSERTL0(link != nullptr,
             "The desired CADCurve is not associated with this entity.");
    return link->param[0];
}

void CADAssociation::Add(const Geometry *geom, CADLink link)
{
    ASSERTL1(link.obj != nullptr, "Cannot associate a null CAD object.");

    auto &links = m_links[geom];
    for (auto &existing : links)
    {
        // One link per distinct CAD object: overwrite the parametric position
        // if this CAD object is already present.
        if (existing.Type() == link.Type() && existing.Id() == link.Id())
        {
            existing = std::move(link);
            return;
        }
    }
    links.push_back(std::move(link));
}

void CADAssociation::Set(const Geometry *geom, CADLink link)
{
    ASSERTL1(link.obj != nullptr, "Cannot associate a null CAD object.");

    CADType::cadType t = link.Type();
    auto &links        = m_links[geom];

    links.erase(std::remove_if(links.begin(), links.end(),
                               [t](const CADLink &existing) {
                                   return existing.Type() == t;
                               }),
                links.end());
    links.push_back(std::move(link));
}

void CADAssociation::Remove(Geometry *geom)
{
    m_links.erase(geom);

    // Curvature nodes are owned by the curve attached to this geometry, so they
    // die with it; drop their associations too. Nothing else holds them, so a
    // caller could not reasonably be expected to find them itself. Note that
    // Geometry::GetCurve() is fatal for anything without a shape dimension.
    if (geom->GetShapeDim() < 1)
    {
        return;
    }

    if (Curve *curve = geom->GetCurve(); curve != nullptr)
    {
        for (auto *pt : curve->m_points)
        {
            m_links.erase(pt);
        }
    }
}

void CADAssociation::ClearVertLinks(CADType::cadType t)
{
    for (auto it = m_links.begin(); it != m_links.end();)
    {
        if (it->first->GetShapeDim() > 0)
        {
            ++it;
            continue;
        }

        auto &links = it->second;
        links.erase(std::remove_if(
                        links.begin(), links.end(),
                        [t](const CADLink &link) { return link.Type() == t; }),
                    links.end());

        it = links.empty() ? m_links.erase(it) : std::next(it);
    }
}

} // namespace Nektar::SpatialDomains
