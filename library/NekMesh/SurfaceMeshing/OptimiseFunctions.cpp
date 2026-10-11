////////////////////////////////////////////////////////////////////////////////
//
//  File: OptimiseFunctions.cpp
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
//  Description: the objectives the high-order surface mesher minimises.
//  functions and gradients for optimisation
////////////////////////////////////////////////////////////////////////////////

#include <NekMesh/SurfaceMeshing/OptimiseFunctions.h>

using namespace std;

namespace Nektar::NekMesh
{

namespace
{

NekDouble Dot(const std::array<NekDouble, 3> &a,
              const std::array<NekDouble, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<NekDouble, 3> Take(const std::array<NekDouble, 3> &a,
                              const std::array<NekDouble, 3> &b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

/// The free variables with the fixed ends put back on, @p perNode entries at
/// each end.
vector<NekDouble> WithEnds(const vector<NekDouble> &interior,
                           const vector<NekDouble> &begin,
                           const vector<NekDouble> &end)
{
    vector<NekDouble> all;
    all.reserve(begin.size() + interior.size() + end.size());
    all.insert(all.end(), begin.begin(), begin.end());
    all.insert(all.end(), interior.begin(), interior.end());
    all.insert(all.end(), end.begin(), end.end());
    return all;
}

} // namespace

Objective CurveEdgeObjective(const SpatialDomains::CADCurveSharedPtr &curve,
                             NekDouble tBegin, NekDouble tEnd,
                             const std::vector<NekDouble> &z)
{
    const size_t nFree = z.size() - 2;
    auto bounds        = curve->GetBounds();

    Objective obj;
    obj.lower = vector<NekDouble>(nFree, bounds[0]);
    obj.upper = vector<NekDouble>(nFree, bounds[1]);
    obj.scale = vector<NekDouble>(nFree, bounds[1] - bounds[0]);

    obj.value = [curve, tBegin, tEnd, z](const vector<NekDouble> &xi) {
        vector<NekDouble> t = WithEnds(xi, {tBegin}, {tEnd});

        // Each position is wanted by the interval before it and the one
        // after, so evaluate the curve once per node rather than once per
        // interval end.
        vector<std::array<NekDouble, 3>> p(t.size());
        for (size_t i = 0; i < t.size(); ++i)
        {
            p[i] = curve->P(t[i]);
        }

        NekDouble ret = 0.0;
        for (size_t i = 0; i < p.size() - 1; ++i)
        {
            auto dis = Take(p[i + 1], p[i]);
            ret += Dot(dis, dis) / (z[i + 1] - z[i]);
        }
        return ret;
    };

    obj.gradient = [curve, tBegin, tEnd, z](const vector<NekDouble> &xi) {
        vector<NekDouble> t = WithEnds(xi, {tBegin}, {tEnd});

        vector<std::array<NekDouble, 3>> r(t.size()), dr(t.size());
        for (size_t i = 0; i < t.size(); ++i)
        {
            auto d2 = curve->D2(t[i]);
            r[i]    = {d2[0], d2[1], d2[2]};
            dr[i]   = {d2[3], d2[4], d2[5]};
        }

        vector<NekDouble> J(t.size() - 2);
        for (size_t i = 0; i < J.size(); ++i)
        {
            J[i] =
                2.0 / (z[i + 1] - z[i]) * Dot(dr[i + 1], Take(r[i + 1], r[i])) -
                2.0 / (z[i + 2] - z[i + 1]) *
                    Dot(dr[i + 1], Take(r[i + 2], r[i + 1]));
        }
        return J;
    };

    return obj;
}

Objective SurfaceEdgeObjective(const SpatialDomains::CADSurfSharedPtr &surf,
                               std::array<NekDouble, 2> uvBegin,
                               std::array<NekDouble, 2> uvEnd,
                               const std::vector<NekDouble> &z)
{
    const size_t nFree = 2 * (z.size() - 2);
    auto bounds        = surf->GetBounds();

    Objective obj;
    obj.lower.resize(nFree);
    obj.upper.resize(nFree);

    // u and v alternate, and have their own bounds.
    for (size_t i = 0; i < nFree; ++i)
    {
        obj.lower[i] = (i % 2 == 0) ? bounds[0] : bounds[2];
        obj.upper[i] = (i % 2 == 0) ? bounds[1] : bounds[3];
    }

    // Tested on the gradient as it stands, which is what this has always
    // done; the scaled form the curve objective uses would want a different
    // tolerance here.
    obj.scale = vector<NekDouble>(nFree, 1.0);

    auto positions = [surf](const vector<NekDouble> &uv) {
        vector<std::array<NekDouble, 2>> out(uv.size() / 2);
        for (size_t i = 0; i < out.size(); ++i)
        {
            out[i] = {uv[i * 2], uv[i * 2 + 1]};
        }
        return out;
    };

    obj.value = [surf, uvBegin, uvEnd, z,
                 positions](const vector<NekDouble> &xi) {
        vector<NekDouble> all =
            WithEnds(xi, {uvBegin[0], uvBegin[1]}, {uvEnd[0], uvEnd[1]});
        auto uv = positions(all);

        vector<std::array<NekDouble, 3>> p(uv.size());
        for (size_t i = 0; i < uv.size(); ++i)
        {
            p[i] = surf->P(uv[i]);
        }

        NekDouble ret = 0.0;
        for (size_t i = 0; i < p.size() - 1; ++i)
        {
            auto dis = Take(p[i + 1], p[i]);
            ret += Dot(dis, dis) / (z[i + 1] - z[i]);
        }
        return ret;
    };

    obj.gradient = [surf, uvBegin, uvEnd, z,
                    positions](const vector<NekDouble> &xi) {
        vector<NekDouble> all =
            WithEnds(xi, {uvBegin[0], uvBegin[1]}, {uvEnd[0], uvEnd[1]});
        auto uv = positions(all);

        vector<std::array<NekDouble, 3>> r(uv.size()), dru(uv.size()),
            drv(uv.size());
        for (size_t i = 0; i < uv.size(); ++i)
        {
            auto d1 = surf->D1(uv[i]);
            r[i]    = {d1[0], d1[1], d1[2]};
            dru[i]  = {d1[3], d1[4], d1[5]};
            drv[i]  = {d1[6], d1[7], d1[8]};
        }

        vector<NekDouble> J(2 * (uv.size() - 2));
        for (size_t i = 0; i < uv.size() - 2; ++i)
        {
            J[2 * i + 0] = 2.0 / (z[i + 1] - z[i]) *
                               Dot(dru[i + 1], Take(r[i + 1], r[i])) +
                           2.0 / (z[i + 2] - z[i + 1]) *
                               Dot(dru[i + 1], Take(r[i + 1], r[i + 2]));
            J[2 * i + 1] = 2.0 / (z[i + 1] - z[i]) *
                               Dot(drv[i + 1], Take(r[i + 1], r[i])) +
                           2.0 / (z[i + 2] - z[i + 1]) *
                               Dot(drv[i + 1], Take(r[i + 1], r[i + 2]));
        }
        return J;
    };

    return obj;
}

} // namespace Nektar::NekMesh
