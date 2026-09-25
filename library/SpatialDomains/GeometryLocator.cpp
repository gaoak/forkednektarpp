////////////////////////////////////////////////////////////////////////////////
//
//  File: GeometryLocator.cpp
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
//  Description: Inverse mapping x -> xi for a geometry.
//
////////////////////////////////////////////////////////////////////////////////

#include <iomanip>

#include <SpatialDomains/GeomFactors.h>
#include <SpatialDomains/Geometry.h>
#include <SpatialDomains/Geometry1D.h>
#include <SpatialDomains/Geometry2D.h>
#include <SpatialDomains/Geometry3D.h>
#include <SpatialDomains/GeometryLocator.h>
#include <SpatialDomains/PointGeom.h>
#include <SpatialDomains/SegGeom.h>

namespace Nektar::SpatialDomains
{

GeometryLocator::GeometryLocator(Geometry *geom) : m_geom(geom)
{
    ASSERTL0(m_geom != nullptr, "A locator needs a geometry to search.");
}

/**
 * @brief Populate the cache of quantities derived from the geometry.
 *
 * Separate from the constructor because it dispatches virtually: the
 * inverse-coefficient and straight-edged forms differ by dimension.
 */
void GeometryLocator::Build()
{
    m_straightEdge = m_geom->HasLinearXmap() ? 1 : 0;

    // Classifying the geometry is exactly the test of whether the nonlinear
    // isoparametric coefficients vanish, so this yields both at once.
    if (m_geom->CalcGeomType(m_iso) == eRegular)
    {
        v_CalculateInverseIsoParam();
    }
    else if (m_straightEdge)
    {
        v_PreSolveStraightEdge();
    }
}

GeometryLocatorUniquePtr GeometryLocator::Create(Geometry *geom)
{
    ASSERTL0(geom != nullptr, "A locator needs a geometry to search.");

    GeometryLocatorUniquePtr loc;
    switch (geom->GetShapeDim())
    {
        case 1:
            loc = std::make_unique<GeometryLocator1D>(geom);
            break;
        case 2:
            loc = std::make_unique<GeometryLocator2D>(geom);
            break;
        case 3:
            loc = std::make_unique<GeometryLocator3D>(geom);
            break;
        default:
            NEKERROR(ErrorUtil::efatal,
                     "No inverse mapping for this shape dimension.");
            return loc;
    }

    loc->Build();
    return loc;
}

/**
 * @brief Determine whether the element contains a Cartesian coordinate.
 *
 * @see GeometryLocator::ContainsPoint
 */
bool GeometryLocator::ContainsPoint(
    const Array<OneD, const NekDouble> &gloCoord, NekDouble tol)
{
    Array<OneD, NekDouble> locCoord(m_geom->GetCoordim(), 0.0);
    NekDouble dist;
    return ContainsPoint(gloCoord, locCoord, tol, dist);
}

/**
 * @copydoc GeometryLocator::ContainsPoint
 */
bool GeometryLocator::ContainsPoint(
    const Array<OneD, const NekDouble> &gloCoord,
    Array<OneD, NekDouble> &locCoord, NekDouble tol)
{
    NekDouble dist;
    return ContainsPoint(gloCoord, locCoord, tol, dist);
}

/**
 * @brief Determine the local collapsed coordinates corresponding to a given
 * Cartesian coordinate.
 *
 * For curvilinear and non-affine elements this is a nonlinear optimisation
 * problem solved by Newton iteration, so it can be expensive.
 *
 * @param coords   Input Cartesian global coordinates
 * @param Lcoords  Corresponding local coordinates
 *
 * @return Distance between the obtained coordinates and the provided ones.
 */
NekDouble GeometryLocator::GetLocCoords(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    return v_GetLocCoords(coords, Lcoords);
}

NekDouble GeometryLocator::FindDistance(const Array<OneD, const NekDouble> &xs,
                                        Array<OneD, NekDouble> &xi)
{
    return v_FindDistance(xs, xi);
}

/**
 * @brief Generates the bounding box for the element.
 *
 * For regular elements, the vertices are sufficient to define the extent of
 * the bounding box. For non-regular elements, the extremes of the quadrature
 * point coordinates are used. A 10% margin is added around this computed
 * region to account for convex hull elements where the true extent of the
 * element may extend slightly beyond the quadrature points.
 */
std::array<NekDouble, 6> GeometryLocator::GetBoundingBox()
{

    if (m_hasBoundingBox)
    {
        return {{m_boundingBox[0], m_boundingBox[1], m_boundingBox[2],
                 m_boundingBox[3], m_boundingBox[4], m_boundingBox[5]}};
    }
    // NekDouble minx, miny, minz, maxx, maxy, maxz;
    Array<OneD, NekDouble> min(3), max(3);

    // Always get vertexes min/max
    PointGeom *p = m_geom->GetVertex(0);
    Array<OneD, NekDouble> x(3, 0.0);
    p->GetCoords(x[0], x[1], x[2]);
    for (int j = 0; j < 3; ++j)
    {
        min[j] = x[j];
        max[j] = x[j];
    }
    for (int i = 1; i < m_geom->GetNumVerts(); ++i)
    {
        p = m_geom->GetVertex(i);
        p->GetCoords(x[0], x[1], x[2]);
        for (int j = 0; j < 3; ++j)
        {
            min[j] = (x[j] < min[j] ? x[j] : min[j]);
            max[j] = (x[j] > max[j] ? x[j] : max[j]);
        }
    }
    // If element is deformed loop over quadrature points
    NekDouble marginFactor = NekConstants::kGeomFactorsTol;
    if (m_geom->CalcGeomType() != eRegular)
    {
        marginFactor = 0.1;
        const int nq = m_geom->GetXmap()->GetTotPoints();
        Array<OneD, Array<OneD, NekDouble>> xvec(3);
        for (int j = 0; j < 3; ++j)
        {
            xvec[j] = Array<OneD, NekDouble>(nq, 0.0);
        }
        for (int j = 0; j < m_geom->GetCoordim(); ++j)
        {
            m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(j), xvec[j]);
        }
        for (int j = 0; j < 3; ++j)
        {
            for (int i = 0; i < nq; ++i)
            {
                min[j] = (xvec[j][i] < min[j] ? xvec[j][i] : min[j]);
                max[j] = (xvec[j][i] > max[j] ? xvec[j][i] : max[j]);
            }
        }
    }
    // Add margin to bounding box, in order to
    // return the nearest element
    for (int j = 0; j < 3; ++j)
    {
        NekDouble margin =
            marginFactor * (max[j] - min[j]) + NekConstants::kFindDistanceMin;
        min[j] -= margin;
        max[j] += margin;
    }

    // save bounding box
    m_hasBoundingBox = true;
    for (int j = 0; j < 3; ++j)
    {
        m_boundingBox[j]     = min[j];
        m_boundingBox[j + 3] = max[j];
    }
    // Return bounding box
    return {{min[0], min[1], min[2], max[0], max[1], max[2]}};
}

void GeometryLocator::ClearBoundingBox()
{
    m_hasBoundingBox = false;
}

/**
 * @brief A fast and robust check if a given global coord is outside of a
 * deformed element. For regular elements, this check is unnecessary.
 *
 * @param coords   Input Cartesian global coordinates
 *
 * @return 1 is inside of the element.
 *         0 maybe inside
 *        -1 outside of the element
 */
int GeometryLocator::PreliminaryCheck(
    const Array<OneD, const NekDouble> &gloCoord)
{
    // bounding box check
    if (!MinMaxCheck(gloCoord))
    {
        return -1;
    }

    // regular element check
    if (m_geom->CalcGeomType() == eRegular)
    {
        return 0;
    }

    // All left check for straight edges/plane surfaces
    return v_AllLeftCheck(gloCoord);
}

/**
 * @brief Check if given global coord is within the BoundingBox of the element.
 *
 * @param coords   Input Cartesian global coordinates
 *
 * @return True if within distance or False otherwise.
 */
bool GeometryLocator::MinMaxCheck(const Array<OneD, const NekDouble> &gloCoord)
{
    // Validation checks
    ASSERTL1(gloCoord.size() >= m_geom->GetCoordim(),
             "Expects number of global coordinates supplied to be greater than "
             "or equal to the mesh dimension.");

    std::array<NekDouble, 6> minMax = GetBoundingBox();
    for (int i = 0; i < m_geom->GetCoordim(); ++i)
    {
        if ((gloCoord[i] < minMax[i]) || (gloCoord[i] > minMax[i + 3]))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Clamp local coords to be within standard regions [-1, 1]^dim.
 *
 * @param Lcoords  Corresponding local coordinates
 */
bool GeometryLocator::ClampLocCoords(Array<OneD, NekDouble> &locCoord, int dim,
                                     NekDouble tol)
{
    // Validation checks
    ASSERTL1(locCoord.size() >= dim, "Expects local coordinates to be same or "
                                     "larger than shape dimension.");

    // If out of range clamp locCoord to be within [-1,1]^dim
    // since any larger value will be very oscillatory if
    // called by 'returnNearestElmt' option in
    // ExpList::GetExpIndex
    bool clamp = false;
    for (int i = 0; i < dim; ++i)
    {
        if (!std::isfinite(locCoord[i]))
        {
            locCoord[i] = 0.;
            clamp       = true;
        }
        else if (locCoord[i] < -(1. + tol))
        {
            locCoord[i] = -(1. + tol);
            clamp       = true;
        }
        else if (locCoord[i] > (1. + tol))
        {
            locCoord[i] = 1. + tol;
            clamp       = true;
        }
    }
    return clamp;
}

/**
 * @copydoc Geometry::ContainsPoint(
 *     const Array<OneD, const NekDouble> &, Array<OneD, NekDouble> &,
 *     NekDouble, NekDouble&)
 * dist is assigned value for curved elements
 */
bool GeometryLocator::ContainsPoint(
    const Array<OneD, const NekDouble> &gloCoord,
    Array<OneD, NekDouble> &locCoord, NekDouble tol, NekDouble &dist)
{
    int inside = PreliminaryCheck(gloCoord);
    if (inside == -1)
    {
        dist = std::numeric_limits<double>::max();
        return false;
    }
    dist = GetLocCoords(gloCoord, locCoord);
    if (inside == 1)
    {
        dist = 0.;
        return true;
    }
    else
    {
        Array<OneD, NekDouble> eta(m_geom->GetShapeDim(), 0.);
        m_geom->GetXmap()->LocCoordToLocCollapsed(locCoord, eta);
        if (ClampLocCoords(eta, m_geom->GetShapeDim(), tol))
        {
            if (m_geom->CalcGeomType() == eRegular)
            {
                dist = std::numeric_limits<double>::max();
            }
            return false;
        }
        return 3 != m_geom->GetCoordim() ||
               (LibUtilities::eTriangle != m_geom->GetShapeType() &&
                LibUtilities::eQuadrilateral != m_geom->GetShapeType()) ||
               dist <= tol;
    }
}

/**
 * @copydoc Geometry::GetLocCoords()
 */
NekDouble GeometryLocator::v_GetLocCoords(
    [[maybe_unused]] const Array<OneD, const NekDouble> &coords,
    [[maybe_unused]] Array<OneD, NekDouble> &Lcoords)
{
    NEKERROR(ErrorUtil::efatal,
             "This function is only valid for expansion type geometries");
    return 0.0;
}

NekDouble GeometryLocator::v_FindDistance(
    [[maybe_unused]] const Array<OneD, const NekDouble> &xs,
    [[maybe_unused]] Array<OneD, NekDouble> &xi)
{
    NEKERROR(ErrorUtil::efatal,
             "This function has not been defined for this geometry");
    return false;
}

int GeometryLocator::v_AllLeftCheck(
    [[maybe_unused]] const Array<OneD, const NekDouble> &gloCoord)
{
    return 0;
}

/**
 * @copydoc GeometryLocator::v_CalculateInverseIsoParam()
 *
 * Nothing to do by default. Build() drives this for every shape, but only
 * the 2D and 3D inverse maps are expressed through an inverted coefficient
 * matrix; a segment inverts \f$\chi\f$ in closed form straight from its
 * quadrature points.
 *
 * @see Geometry1D::v_GetLocCoords
 */
void GeometryLocator::v_CalculateInverseIsoParam()
{
}

/**
 * @copydoc GeometryLocator::v_PreSolveStraightEdge()
 *
 * Only the quadrilateral has a straight-edged solve to prepare for.
 */
void GeometryLocator::v_PreSolveStraightEdge()
{
}

NekDouble GeometryLocator1D::v_GetLocCoords(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    NekDouble dist = std::numeric_limits<double>::max();
    m_geom->FillGeom();

    // calculate local coordinate for coord
    if (m_geom->CalcGeomType() == eRegular)
    {
        NekDouble len = 0.0;
        NekDouble xi  = 0.0;

        const int npts = m_geom->GetXmap()->GetTotPoints();
        Array<OneD, NekDouble> pts(npts);

        for (int i = 0; i < m_geom->GetCoordim(); ++i)
        {
            m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(i), pts);
            len += (pts[npts - 1] - pts[0]) * (pts[npts - 1] - pts[0]);
            xi += (coords[i] - pts[0]) * (pts[npts - 1] - pts[0]);
        }
        xi = xi / len;
        if (xi < 0.)
        {
            dist = -xi * sqrt(len);
        }
        else if (xi > 1.)
        {
            dist = (xi - 1.) * sqrt(len);
        }

        Lcoords[0] = 2. * xi - 1.0;
    }
    else
    {
        NEKERROR(ErrorUtil::efatal,
                 "inverse mapping must be set up to use this call");
    }
    return dist;
}

NekDouble GeometryLocator1D::v_FindDistance(
    const Array<OneD, const NekDouble> &xs, Array<OneD, NekDouble> &xiOut)
{
    GeomType Gtype = m_geom->CalcGeomType();
    if (Gtype == eRegular)
    {
        xiOut = Array<OneD, NekDouble>(1, 0.0);

        GetLocCoords(xs, xiOut);
        ClampLocCoords(xiOut, m_geom->GetShapeDim());

        Array<OneD, NekDouble> gloCoord(m_geom->GetCoordim());
        NekDouble tmp = 0;
        for (int i = 0; i < m_geom->GetCoordim(); ++i)
        {
            gloCoord[i] = m_geom->GetCoord(i, xiOut);
            tmp += (xs[i] - gloCoord[i]) * (xs[i] - gloCoord[i]);
        }

        return sqrt(tmp);
    }
    // If deformed edge then the inverse mapping is non-linear so need to
    // numerically solve for the local coordinate
    else if (Gtype == eDeformed)
    {
        Array<OneD, NekDouble> xi(1, 0.0);

        // Armijo constants:
        // https://en.wikipedia.org/wiki/Backtracking_line_search
        const NekDouble c1 = 1e-4, c2 = 0.9;

        int dim = m_geom->GetCoordim();
        int nq  = m_geom->GetXmap()->GetTotPoints();

        Array<OneD, Array<OneD, NekDouble>> x(dim), xder(dim), xder2(dim);
        // Get x,y,z phys values from coefficients
        for (int i = 0; i < dim; ++i)
        {
            x[i]     = Array<OneD, NekDouble>(nq);
            xder[i]  = Array<OneD, NekDouble>(nq);
            xder2[i] = Array<OneD, NekDouble>(nq);

            m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(i), x[i]);
        }

        NekDouble fx_prev = std::numeric_limits<NekDouble>::max();

        // Minimisation loop (Quasi-newton method)
        for (int i = 0; i < NekConstants::kNewtonIterations; ++i)
        {
            // Compute the objective function, f(x_k) and its derivatives
            Array<OneD, NekDouble> xc(dim);
            Array<OneD, std::array<NekDouble, 3>> xc_der(dim);
            Array<OneD, std::array<NekDouble, 6>> xc_der2(dim);
            NekDouble fx = 0, fxp = 0, fxp2 = 0, xcDiff = 0;
            for (int j = 0; j < dim; ++j)
            {
                xc[j] = m_geom->GetXmap()->PhysEvaluate(xi, x[j], xc_der[j],
                                                        xc_der2[j]);

                xcDiff = xc[j] - xs[j];
                // Objective function is the distance to the search point
                fx += xcDiff * xcDiff;
                fxp += xc_der[j][0] * xcDiff;
                fxp2 += xc_der2[j][0] * xcDiff + xc_der[j][0] * xc_der[j][0];
            }

            fxp *= 2;
            fxp2 *= 2;

            // Check for convergence
            if (std::abs(fx - fx_prev) < 1e-12)
            {
                fx_prev = fx;
                break;
            }
            else
            {
                fx_prev = fx;
            }

            NekDouble gamma = 1.0;
            bool conv       = false;

            // Search direction: Newton's method
            NekDouble pk = -fxp / fxp2;

            // Perform backtracking line search
            while (gamma > 1e-10)
            {
                Array<OneD, NekDouble> xi_pk(1);
                xi_pk[0] = xi[0] + pk * gamma;

                if (xi_pk[0] < -1.0 || xi_pk[0] > 1.0)
                {
                    gamma /= 2.0;
                    continue;
                }

                Array<OneD, NekDouble> xc_pk(dim);
                Array<OneD, std::array<NekDouble, 3>> xc_der_pk(dim);
                NekDouble fx_pk = 0, fxp_pk = 0, xc_pkDiff = 0;
                for (int j = 0; j < dim; ++j)
                {
                    xc_pk[j] = m_geom->GetXmap()->PhysEvaluate(xi_pk, x[j],
                                                               xc_der_pk[j]);

                    xc_pkDiff = xc_pk[j] - xs[j];
                    fx_pk += xc_pkDiff * xc_pkDiff;
                    fxp_pk += xc_der_pk[j][0] * xc_pkDiff;
                }

                fxp_pk *= 2;

                // Check Wolfe conditions using Armijo constants
                // https://en.wikipedia.org/wiki/Wolfe_conditions
                if ((fx_pk - (fx + c1 * gamma * pk * fxp)) <
                        std::numeric_limits<NekDouble>::epsilon() &&
                    (-pk * fxp_pk + c2 * pk * fxp) <
                        std::numeric_limits<NekDouble>::epsilon())
                {
                    conv = true;
                    break;
                }

                gamma /= 2.0;
            }

            if (!conv)
            {
                break;
            }

            xi[0] += gamma * pk;
        }

        xiOut = xi;
        return sqrt(fx_prev);
    }
    else
    {
        NEKERROR(ErrorUtil::efatal, "Geometry type unknown");
    }

    return -1.0;
}

void GeometryLocator2D::SolveStraightEdgeQuad(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    auto &isoParam = m_iso.m_coeff;
    int i0 = 0, i1 = 1, j1 = 0, j2 = 1;
    if (m_straightEdge & 2)
    {
        i0 = 1;
        i1 = 0;
    }
    if (m_straightEdge & 4)
    {
        j1 = 1;
        j2 = 0;
    }
    NekDouble beta  = isoParam[1][2];
    NekDouble gamma = isoParam[1][3];
    NekDouble tty =
        (coords[i1] - gamma * coords[i0] - isoParam[1][0]) * isoParam[1][1];
    NekDouble denom   = 1. / (isoParam[0][2] + isoParam[0][3] * tty);
    NekDouble epsilon = -isoParam[0][3] * beta * denom;
    NekDouble h = (isoParam[0][0] + isoParam[0][1] * tty - coords[i0]) * denom;
    Lcoords[j2] = -h / (0.5 + sqrt(0.25 - epsilon * h));
    Lcoords[j1] = -beta * Lcoords[j2] + tty;
}

void GeometryLocator2D::NewtonIterationForLocCoord(
    const Array<OneD, const NekDouble> &coords,
    const Array<OneD, const NekDouble> &ptsx,
    const Array<OneD, const NekDouble> &ptsy, Array<OneD, NekDouble> &Lcoords,
    NekDouble &dist)
{
    // Maximum iterations for convergence
    const int MaxIterations = NekConstants::kNewtonIterations;
    // |x-xp|^2 < EPSILON  error    tolerance
    const NekDouble Tol = 1.e-8;
    // |r,s|    > LcoordDIV stop   the search
    const NekDouble LcoordDiv = 15.0;

    LibUtilities::PointsKeyVector ptsKeys = m_geom->GetXmap()->GetPointsKeys();
    Array<OneD, const NekDouble> Jac =
        m_geom->GenGeomFactors(ptsKeys)->GetJac();

    NekDouble ScaledTol =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    ScaledTol *= Tol;

    NekDouble xmap, ymap, F1, F2;
    NekDouble derx_1, derx_2, dery_1, dery_2, jac;

    // save intiial guess for later reference if required.
    NekDouble init0 = Lcoords[0], init1 = Lcoords[1];

    Array<OneD, NekDouble> DxD1(ptsx.size());
    Array<OneD, NekDouble> DxD2(ptsx.size());
    Array<OneD, NekDouble> DyD1(ptsx.size());
    Array<OneD, NekDouble> DyD2(ptsx.size());

    // Ideally this will be stored in m_geomfactors
    m_geom->GetXmap()->PhysDeriv(ptsx, DxD1, DxD2);
    m_geom->GetXmap()->PhysDeriv(ptsy, DyD1, DyD2);

    int cnt = 0;
    Array<OneD, DNekMatSharedPtr> I(2);
    Array<OneD, NekDouble> eta(2);

    F1 = F2         = 2000; // Starting value of Function
    NekDouble resid = sqrt(F1 * F1 + F2 * F2);
    while (cnt++ < MaxIterations)
    {
        //  evaluate lagrange interpolant at Lcoords
        m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, eta);
        I[0] = m_geom->GetXmap()->GetBasis(0)->GetI(eta);
        I[1] = m_geom->GetXmap()->GetBasis(1)->GetI(eta + 1);

        // calculate the global point `corresponding to Lcoords
        xmap = m_geom->GetXmap()->PhysEvaluate(I, ptsx);
        ymap = m_geom->GetXmap()->PhysEvaluate(I, ptsy);

        F1 = coords[0] - xmap;
        F2 = coords[1] - ymap;

        if (F1 * F1 + F2 * F2 < ScaledTol)
        {
            resid = sqrt(F1 * F1 + F2 * F2);
            break;
        }

        // Interpolate derivative metric at Lcoords
        derx_1 = m_geom->GetXmap()->PhysEvaluate(I, DxD1);
        derx_2 = m_geom->GetXmap()->PhysEvaluate(I, DxD2);
        dery_1 = m_geom->GetXmap()->PhysEvaluate(I, DyD1);
        dery_2 = m_geom->GetXmap()->PhysEvaluate(I, DyD2);

        jac = dery_2 * derx_1 - dery_1 * derx_2;

        // use analytical inverse of derivitives which are
        // also similar to those of metric factors.
        Lcoords[0] =
            Lcoords[0] +
            (dery_2 * (coords[0] - xmap) - derx_2 * (coords[1] - ymap)) / jac;

        Lcoords[1] =
            Lcoords[1] +
            (-dery_1 * (coords[0] - xmap) + derx_1 * (coords[1] - ymap)) / jac;

        if (!(std::isfinite(Lcoords[0]) && std::isfinite(Lcoords[1])))
        {
            dist = 1e16;
            std::ostringstream ss;
            ss << "nan or inf found in NewtonIterationForLocCoord in element "
               << m_geom->GetGlobalID();
            WARNINGL1(false, ss.str());
            return;
        }
        if (fabs(Lcoords[0]) > LcoordDiv || fabs(Lcoords[1]) > LcoordDiv)
        {
            break; // lcoords have diverged so stop iteration
        }
    }

    m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, eta);
    if (ClampLocCoords(eta, m_geom->GetShapeDim(), 0.))
    {
        I[0] = m_geom->GetXmap()->GetBasis(0)->GetI(eta);
        I[1] = m_geom->GetXmap()->GetBasis(1)->GetI(eta + 1);
        // calculate the global point corresponding to Lcoords
        xmap = m_geom->GetXmap()->PhysEvaluate(I, ptsx);
        ymap = m_geom->GetXmap()->PhysEvaluate(I, ptsy);
        F1   = coords[0] - xmap;
        F2   = coords[1] - ymap;
        dist = sqrt(F1 * F1 + F2 * F2);
    }
    else
    {
        dist = 0.;
    }

    if (cnt >= MaxIterations)
    {
        Array<OneD, NekDouble> collCoords(2);
        m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, collCoords);

        // if coordinate is inside element dump error!
        if ((collCoords[0] >= -1.0 && collCoords[0] <= 1.0) &&
            (collCoords[1] >= -1.0 && collCoords[1] <= 1.0))
        {
            std::ostringstream ss;

            ss << "Reached MaxIterations (" << MaxIterations
               << ") in Newton iteration ";
            ss << "Init value (" << std::setprecision(4) << init0 << ","
               << init1 << ","
               << ") ";
            ss << "Fin  value (" << Lcoords[0] << "," << Lcoords[1] << ","
               << ") ";
            ss << "Resid = " << resid
               << " Tolerance = " << std::sqrt(ScaledTol);

            WARNINGL1(cnt < MaxIterations, ss.str());
        }
    }
}

NekDouble GeometryLocator2D::v_GetLocCoords(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    NekDouble dist = std::numeric_limits<double>::max();
    Array<OneD, NekDouble> tmpcoords(2);
    tmpcoords[0]   = coords[m_iso.m_manifold[0]];
    tmpcoords[1]   = coords[m_iso.m_manifold[1]];
    GeomType Gtype = m_geom->CalcGeomType();
    if (Gtype == eRegular)
    {
        tmpcoords[0] -= m_iso.m_coeff[0][0];
        tmpcoords[1] -= m_iso.m_coeff[1][0];
        Lcoords[0] = m_invIsoParam[0][0] * tmpcoords[0] +
                     m_invIsoParam[0][1] * tmpcoords[1];
        Lcoords[1] = m_invIsoParam[1][0] * tmpcoords[0] +
                     m_invIsoParam[1][1] * tmpcoords[1];
    }
    else if (m_straightEdge)
    {
        SolveStraightEdgeQuad(tmpcoords, Lcoords);
    }
    else if (Gtype == eDeformed)
    {
        m_geom->FillGeom();
        // Determine nearest point of coords  to values in m_geom->GetXmap()
        int npts = m_geom->GetXmap()->GetTotPoints();
        Array<OneD, NekDouble> ptsx(npts), ptsy(npts);
        Array<OneD, NekDouble> tmpx(npts), tmpy(npts);

        // Determine 3D manifold orientation
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(m_iso.m_manifold[0]),
                                    ptsx);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(m_iso.m_manifold[1]),
                                    ptsy);

        Array<OneD, NekDouble> eta(2, 0.);
        m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, eta);
        ClampLocCoords(eta, m_geom->GetShapeDim(), 0.);

        m_geom->GetXmap()->LocCollapsedToLocCoord(eta, Lcoords);

        // Perform newton iteration to find local coordinates
        NewtonIterationForLocCoord(tmpcoords, ptsx, ptsy, Lcoords, dist);
    }
    if (m_geom->GetCoordim() == 3)
    {
        Array<OneD, NekDouble> eta(2, 0.), xi(2, 0.);
        m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, eta);
        ClampLocCoords(eta, m_geom->GetShapeDim(), 0.);
        m_geom->GetXmap()->LocCollapsedToLocCoord(eta, xi);
        int npts = m_geom->GetXmap()->GetTotPoints();
        Array<OneD, NekDouble> ptsz(npts);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(m_iso.m_manifold[2]),
                                    ptsz);
        NekDouble z = m_geom->GetXmap()->PhysEvaluate(xi, ptsz) -
                      coords[m_iso.m_manifold[2]];
        if (Gtype == eDeformed)
        {
            dist = sqrt(z * z + dist * dist);
        }
        else
        {
            dist = fabs(z);
        }
    }
    return dist;
}

NekDouble GeometryLocator2D::v_FindDistance(
    const Array<OneD, const NekDouble> &xs, Array<OneD, NekDouble> &xiOut)
{
    GeomType Gtype = m_geom->CalcGeomType();
    if (Gtype == eRegular)
    {
        xiOut = Array<OneD, NekDouble>(2, 0.0);

        GetLocCoords(xs, xiOut);
        ClampLocCoords(xiOut, m_geom->GetShapeDim());

        Array<OneD, NekDouble> gloCoord(3);
        gloCoord[0] = m_geom->GetCoord(0, xiOut);
        gloCoord[1] = m_geom->GetCoord(1, xiOut);
        gloCoord[2] = m_geom->GetCoord(2, xiOut);

        return sqrt((xs[0] - gloCoord[0]) * (xs[0] - gloCoord[0]) +
                    (xs[1] - gloCoord[1]) * (xs[1] - gloCoord[1]) +
                    (xs[2] - gloCoord[2]) * (xs[2] - gloCoord[2]));
    }
    // If deformed edge then the inverse mapping is non-linear so need to
    // numerically solve for the local coordinate
    else if (Gtype == eDeformed)
    {
        // Choose starting based on closest quad
        Array<OneD, NekDouble> xi(2, 0.0), eta(2, 0.0);
        m_geom->GetXmap()->LocCollapsedToLocCoord(eta, xi);

        // Armijo constants:
        // https://en.wikipedia.org/wiki/Backtracking_line_search
        const NekDouble c1 = 1e-4, c2 = 0.9;

        int nq = m_geom->GetXmap()->GetTotPoints();

        Array<OneD, NekDouble> x(nq), y(nq), z(nq);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(0), x);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(1), y);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(2), z);

        Array<OneD, NekDouble> xderxi1(nq, 0.0), yderxi1(nq, 0.0),
            zderxi1(nq, 0.0), xderxi2(nq, 0.0), yderxi2(nq, 0.0),
            zderxi2(nq, 0.0), xderxi1xi1(nq, 0.0), yderxi1xi1(nq, 0.0),
            zderxi1xi1(nq, 0.0), xderxi1xi2(nq, 0.0), yderxi1xi2(nq, 0.0),
            zderxi1xi2(nq, 0.0), xderxi2xi1(nq, 0.0), yderxi2xi1(nq, 0.0),
            zderxi2xi1(nq, 0.0), xderxi2xi2(nq, 0.0), yderxi2xi2(nq, 0.0),
            zderxi2xi2(nq, 0.0);

        // Get first & second derivatives & partial derivatives of x,y,z values
        std::array<NekDouble, 3> xc_derxi, yc_derxi, zc_derxi;

        m_geom->GetXmap()->PhysDeriv(x, xderxi1, xderxi2);
        m_geom->GetXmap()->PhysDeriv(y, yderxi1, yderxi2);
        m_geom->GetXmap()->PhysDeriv(z, zderxi1, zderxi2);

        m_geom->GetXmap()->PhysDeriv(xderxi1, xderxi1xi1, xderxi1xi2);
        m_geom->GetXmap()->PhysDeriv(yderxi1, yderxi1xi1, yderxi1xi2);
        m_geom->GetXmap()->PhysDeriv(zderxi1, zderxi1xi1, zderxi1xi2);

        m_geom->GetXmap()->PhysDeriv(yderxi2, yderxi2xi1, yderxi2xi2);
        m_geom->GetXmap()->PhysDeriv(xderxi2, xderxi2xi1, xderxi2xi2);
        m_geom->GetXmap()->PhysDeriv(zderxi2, zderxi2xi1, zderxi2xi2);

        // Minimisation loop (Quasi-newton method)
        NekDouble fx_prev = std::numeric_limits<NekDouble>::max();
        for (int i = 0; i < NekConstants::kNewtonIterations; ++i)
        {
            // Compute the objective function, f(x_k) and its derivatives
            NekDouble xc = m_geom->GetXmap()->PhysEvaluate(xi, x, xc_derxi);
            NekDouble yc = m_geom->GetXmap()->PhysEvaluate(xi, y, yc_derxi);
            NekDouble zc = m_geom->GetXmap()->PhysEvaluate(xi, z, zc_derxi);

            NekDouble xc_derxi1xi1 =
                m_geom->GetXmap()->PhysEvaluate(xi, xderxi1xi1);
            NekDouble yc_derxi1xi1 =
                m_geom->GetXmap()->PhysEvaluate(xi, yderxi1xi1);
            NekDouble zc_derxi1xi1 =
                m_geom->GetXmap()->PhysEvaluate(xi, zderxi1xi1);

            NekDouble xc_derxi1xi2 =
                m_geom->GetXmap()->PhysEvaluate(xi, xderxi1xi2);
            NekDouble yc_derxi1xi2 =
                m_geom->GetXmap()->PhysEvaluate(xi, yderxi1xi2);
            NekDouble zc_derxi1xi2 =
                m_geom->GetXmap()->PhysEvaluate(xi, zderxi1xi2);

            NekDouble xc_derxi2xi2 =
                m_geom->GetXmap()->PhysEvaluate(xi, xderxi2xi2);
            NekDouble yc_derxi2xi2 =
                m_geom->GetXmap()->PhysEvaluate(xi, yderxi2xi2);
            NekDouble zc_derxi2xi2 =
                m_geom->GetXmap()->PhysEvaluate(xi, zderxi2xi2);

            // Objective function is the distance to the search point
            NekDouble xdiff = xc - xs[0];
            NekDouble ydiff = yc - xs[1];
            NekDouble zdiff = zc - xs[2];

            NekDouble fx = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;

            NekDouble fx_derxi1 = 2.0 * xdiff * xc_derxi[0] +
                                  2.0 * ydiff * yc_derxi[0] +
                                  2.0 * zdiff * zc_derxi[0];

            NekDouble fx_derxi2 = 2.0 * xdiff * xc_derxi[1] +
                                  2.0 * ydiff * yc_derxi[1] +
                                  2.0 * zdiff * zc_derxi[1];

            NekDouble fx_derxi1xi1 =
                2.0 * xdiff * xc_derxi1xi1 + 2.0 * xc_derxi[0] * xc_derxi[0] +
                2.0 * ydiff * yc_derxi1xi1 + 2.0 * yc_derxi[0] * yc_derxi[0] +
                2.0 * zdiff * zc_derxi1xi1 + 2.0 * zc_derxi[0] * zc_derxi[0];

            NekDouble fx_derxi1xi2 =
                2.0 * xdiff * xc_derxi1xi2 + 2.0 * xc_derxi[1] * xc_derxi[0] +
                2.0 * ydiff * yc_derxi1xi2 + 2.0 * yc_derxi[1] * yc_derxi[0] +
                2.0 * zdiff * zc_derxi1xi2 + 2.0 * zc_derxi[1] * zc_derxi[0];

            NekDouble fx_derxi2xi2 =
                2.0 * xdiff * xc_derxi2xi2 + 2.0 * xc_derxi[1] * xc_derxi[1] +
                2.0 * ydiff * yc_derxi2xi2 + 2.0 * yc_derxi[1] * yc_derxi[1] +
                2.0 * zdiff * zc_derxi2xi2 + 2.0 * zc_derxi[1] * zc_derxi[1];

            // Jacobian
            NekDouble jac[2];
            jac[0] = fx_derxi1;
            jac[1] = fx_derxi2;

            // Inverse of 2x2 hessian
            NekDouble hessInv[2][2];

            NekDouble det =
                1 / (fx_derxi1xi1 * fx_derxi2xi2 - fx_derxi1xi2 * fx_derxi1xi2);
            hessInv[0][0] = det * fx_derxi2xi2;
            hessInv[0][1] = det * -fx_derxi1xi2;
            hessInv[1][0] = det * -fx_derxi1xi2;
            hessInv[1][1] = det * fx_derxi1xi1;

            // Check for convergence
            if (abs(fx - fx_prev) < 1e-12)
            {
                fx_prev = fx;
                break;
            }
            else
            {
                fx_prev = fx;
            }

            NekDouble gamma = 1.0;
            bool conv       = false;

            // Search direction: Newton's method
            NekDouble pk[2];
            pk[0] = -(hessInv[0][0] * jac[0] + hessInv[1][0] * jac[1]);
            pk[1] = -(hessInv[0][1] * jac[0] + hessInv[1][1] * jac[1]);

            // Backtracking line search
            while (gamma > 1e-10)
            {
                Array<OneD, NekDouble> xi_pk(2);
                xi_pk[0] = xi[0] + pk[0] * gamma;
                xi_pk[1] = xi[1] + pk[1] * gamma;

                Array<OneD, NekDouble> eta_pk(2, 0.0);
                m_geom->GetXmap()->LocCoordToLocCollapsed(xi_pk, eta_pk);

                if (eta_pk[0] <
                        (-1 - std::numeric_limits<NekDouble>::epsilon()) ||
                    eta_pk[0] >
                        (1 + std::numeric_limits<NekDouble>::epsilon()) ||
                    eta_pk[1] <
                        (-1 - std::numeric_limits<NekDouble>::epsilon()) ||
                    eta_pk[1] > (1 + std::numeric_limits<NekDouble>::epsilon()))
                {
                    gamma /= 2.0;
                    continue;
                }

                std::array<NekDouble, 3> xc_pk_derxi, yc_pk_derxi, zc_pk_derxi;

                NekDouble xc_pk =
                    m_geom->GetXmap()->PhysEvaluate(xi_pk, x, xc_pk_derxi);
                NekDouble yc_pk =
                    m_geom->GetXmap()->PhysEvaluate(xi_pk, y, yc_pk_derxi);
                NekDouble zc_pk =
                    m_geom->GetXmap()->PhysEvaluate(xi_pk, z, zc_pk_derxi);

                NekDouble xc_pk_diff = xc_pk - xs[0];
                NekDouble yc_pk_diff = yc_pk - xs[1];
                NekDouble zc_pk_diff = zc_pk - xs[2];

                NekDouble fx_pk = xc_pk_diff * xc_pk_diff +
                                  yc_pk_diff * yc_pk_diff +
                                  zc_pk_diff * zc_pk_diff;

                NekDouble fx_pk_derxi1 = 2.0 * xc_pk_diff * xc_pk_derxi[0] +
                                         2.0 * yc_pk_diff * yc_pk_derxi[0] +
                                         2.0 * zc_pk_diff * zc_pk_derxi[0];

                NekDouble fx_pk_derxi2 = 2.0 * xc_pk_diff * xc_pk_derxi[1] +
                                         2.0 * yc_pk_diff * yc_pk_derxi[1] +
                                         2.0 * zc_pk_diff * zc_pk_derxi[1];

                // Check Wolfe conditions using Armijo constants
                // https://en.wikipedia.org/wiki/Wolfe_conditions
                NekDouble tmp  = pk[0] * fx_derxi1 + pk[1] * fx_derxi2;
                NekDouble tmp2 = pk[0] * fx_pk_derxi1 + pk[1] * fx_pk_derxi2;
                if ((fx_pk - (fx + c1 * gamma * tmp)) <
                        std::numeric_limits<NekDouble>::epsilon() &&
                    (-tmp2 - (-c2 * tmp)) <
                        std::numeric_limits<NekDouble>::epsilon())
                {
                    conv = true;
                    break;
                }

                gamma /= 2.0;
            }

            if (!conv)
            {
                break;
            }

            xi[0] += gamma * pk[0];
            xi[1] += gamma * pk[1];
        }

        xiOut = xi;
        return sqrt(fx_prev);
    }
    else
    {
        NEKERROR(ErrorUtil::efatal, "Geometry type unknown");
    }

    return -1.0;
}

void GeometryLocator2D::v_CalculateInverseIsoParam()
{
    NekDouble Jac = m_iso.m_coeff[0][1] * m_iso.m_coeff[1][2] -
                    m_iso.m_coeff[1][1] * m_iso.m_coeff[0][2];
    Jac = 1. / Jac;
    // a12, -a02, -a11, a01
    m_invIsoParam[0][0] = m_iso.m_coeff[1][2] * Jac;
    m_invIsoParam[0][1] = -m_iso.m_coeff[0][2] * Jac;
    m_invIsoParam[1][0] = -m_iso.m_coeff[1][1] * Jac;
    m_invIsoParam[1][1] = m_iso.m_coeff[0][1] * Jac;
}

int GeometryLocator2D::v_AllLeftCheck(
    const Array<OneD, const NekDouble> &gloCoord)
{
    int nc = 1, d0 = m_iso.m_manifold[0], d1 = m_iso.m_manifold[1];
    if (!m_edgeNormalBuilt)
    {
        m_edgeNormalBuilt = true;
        std::array<std::array<NekDouble, 3>, 2> x;
        m_geom->GetVertex(0)->GetCoords(x[0][0], x[0][1], x[0][2]);
        int i0 = 1, i1 = 0, direction = 1;
        for (size_t i = 0; i < m_geom->GetNumVerts(); ++i)
        {
            i0 ^= 1;
            i1 ^= 1;
            m_geom->GetVertex((i + 1) % m_geom->GetNumVerts())
                ->GetCoords(x[i1][0], x[i1][1], x[i1][2]);
            if (m_geom->GetEdge(i)->GetXmap()->GetBasis(0)->GetNumModes() > 2)
            {
                continue;
            }
            m_hasEdgeNormal[i] = true;
            m_edgeNormal[i][0] = x[i0][d1] - x[i1][d1];
            m_edgeNormal[i][1] = x[i1][d0] - x[i0][d0];
        }
        if (m_geom->GetCoordim() == 3)
        {
            for (size_t i = 0; i < m_geom->GetNumVerts(); ++i)
            {
                if (m_hasEdgeNormal[i])
                {
                    m_geom->GetVertex(i)->GetCoords(x[0][0], x[0][1], x[0][2]);
                    m_geom->GetVertex((i + 2) % m_geom->GetNumVerts())
                        ->GetCoords(x[1][0], x[1][1], x[1][2]);
                    if (m_edgeNormal[i][0] * (x[1][d0] - x[0][d0]) <
                        m_edgeNormal[i][1] * (x[0][d1] - x[1][d1]))
                    {
                        direction = -1;
                    }
                    break;
                }
            }
        }
        if (direction == -1)
        {
            for (size_t i = 0; i < m_geom->GetNumVerts(); ++i)
            {
                if (m_hasEdgeNormal[i])
                {
                    m_edgeNormal[i][0] = -m_edgeNormal[i][0];
                    m_edgeNormal[i][1] = -m_edgeNormal[i][1];
                }
            }
        }
    }

    std::array<NekDouble, 3> vertex;
    for (size_t i = 0; i < m_geom->GetNumVerts(); ++i)
    {
        int i1 = (i + 1) % m_geom->GetNumVerts();
        if (m_geom->GetVertex(i)->GetGlobalID() <
            m_geom->GetVertex(i1)->GetGlobalID())
        {
            m_geom->GetVertex(i)->GetCoords(vertex[0], vertex[1], vertex[2]);
        }
        else
        {
            m_geom->GetVertex(i1)->GetCoords(vertex[0], vertex[1], vertex[2]);
        }
        if (!m_hasEdgeNormal[i])
        {
            nc = 0; // not sure
            continue;
        }
        if (m_edgeNormal[i][0] * (gloCoord[d0] - vertex[d0]) <
            m_edgeNormal[i][1] * (vertex[d1] - gloCoord[d1]))
        {
            return -1; // outside
        }
    }
    // 3D manifold needs to check the distance
    if (m_geom->GetCoordim() == 3)
    {
        nc = 0;
    }
    // nc: 1 (side element), 0 (maybe inside), -1 (outside)
    return nc;
}

void GeometryLocator2D::v_PreSolveStraightEdge()
{
    int i0, i1, j1, j2;
    if (fabs(m_iso.m_coeff[0][3]) >= fabs(m_iso.m_coeff[1][3]))
    {
        i0 = 0;
        i1 = 1;
    }
    else
    {
        i1 = 0;
        i0 = 1;
        m_straightEdge |= 2;
    }
    NekDouble gamma = m_iso.m_coeff[i1][3] / m_iso.m_coeff[i0][3];
    std::vector<NekDouble> c(3);
    for (int i = 0; i < 3; ++i)
    {
        c[i] = m_iso.m_coeff[i1][i] - gamma * m_iso.m_coeff[i0][i];
    }
    if (fabs(c[1]) >= fabs(c[2]))
    {
        j1 = 1;
        j2 = 2;
    }
    else
    {
        j1 = 2;
        j2 = 1;
        m_straightEdge |= 4;
    }
    NekDouble beta = c[j2] / c[j1];
    if (i0 == 1)
    {
        m_iso.m_coeff[0] = m_iso.m_coeff[1];
    }
    if (j1 == 2)
    {
        NekDouble temp       = m_iso.m_coeff[0][j1];
        m_iso.m_coeff[0][j1] = m_iso.m_coeff[0][j2];
        m_iso.m_coeff[0][j2] = temp;
    }
    m_iso.m_coeff[0][2] -= m_iso.m_coeff[0][1] * beta;
    m_iso.m_coeff[1][0] = c[0];
    m_iso.m_coeff[1][1] = 1. / c[j1];
    m_iso.m_coeff[1][2] = beta;
    m_iso.m_coeff[1][3] = gamma;
}

void GeometryLocator3D::NewtonIterationForLocCoord(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    auto &isoParam = m_iso.m_coeff;
    const int nIso = m_iso.m_nCoeff;

    // maximum iterations for convergence
    const int MaxIterations = 51;
    // |x-xp|^2 < EPSILON  error tolerance
    const NekDouble Tol = 1.e-8;
    // |r,s|    > LcoordDIV stop the search
    const NekDouble LcoordDiv = 15.0;

    LibUtilities::PointsKeyVector ptsKeys = m_geom->GetXmap()->GetPointsKeys();
    Array<OneD, const NekDouble> Jac =
        m_geom->GenGeomFactors(ptsKeys)->GetJac();

    NekDouble ScaledTol =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    ScaledTol *= Tol * Tol;

    NekDouble xmap, ymap, zmap, res;
    int cnt = 0;
    Array<OneD, NekDouble> var(8, 1.);
    Array<OneD, NekDouble> deriv(9);
    Array<OneD, NekDouble> tmp(3);
    while (cnt++ < MaxIterations)
    {
        var[1] = Lcoords[0];
        var[2] = Lcoords[1];
        var[3] = Lcoords[2];
        var[4] = Lcoords[0] * Lcoords[1];
        var[5] = Lcoords[1] * Lcoords[2];
        var[6] = Lcoords[0] * Lcoords[2];
        var[7] = var[4] * Lcoords[2];
        // calculate the global point corresponding to Lcoords
        xmap = Vmath::Dot(nIso, var.data(), 1, isoParam[0].data(), 1);
        ymap = Vmath::Dot(nIso, var.data(), 1, isoParam[1].data(), 1);
        zmap = Vmath::Dot(nIso, var.data(), 1, isoParam[2].data(), 1);

        tmp[0] = coords[0] - xmap;
        tmp[1] = coords[1] - ymap;
        tmp[2] = coords[2] - zmap;

        res = tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2];
        if (res < ScaledTol)
        {
            break;
        }

        // Interpolate derivative metric at Lcoords (ddx1, ddx2, ddx3)
        deriv[0] = isoParam[0][1] + isoParam[0][4] * Lcoords[1];
        deriv[1] = isoParam[0][2] + isoParam[0][4] * Lcoords[0];
        deriv[2] = isoParam[0][3];
        deriv[3] = isoParam[1][1] + isoParam[1][4] * Lcoords[1];
        deriv[4] = isoParam[1][2] + isoParam[1][4] * Lcoords[0];
        deriv[5] = isoParam[1][3];
        deriv[6] = isoParam[2][1] + isoParam[2][4] * Lcoords[1];
        deriv[7] = isoParam[2][2] + isoParam[2][4] * Lcoords[0];
        deriv[8] = isoParam[2][3];
        if (nIso >= 6)
        {
            deriv[1] += isoParam[0][5] * Lcoords[2];
            deriv[2] += isoParam[0][5] * Lcoords[1];
            deriv[4] += isoParam[1][5] * Lcoords[2];
            deriv[5] += isoParam[1][5] * Lcoords[1];
            deriv[7] += isoParam[2][5] * Lcoords[2];
            deriv[8] += isoParam[2][5] * Lcoords[1];
        }
        if (nIso >= 8)
        {
            deriv[0] += isoParam[0][6] * Lcoords[2] + isoParam[0][7] * var[5];
            deriv[1] += isoParam[0][7] * var[6];
            deriv[2] += isoParam[0][6] * Lcoords[0] + isoParam[0][7] * var[4];
            deriv[3] += isoParam[1][6] * Lcoords[2] + isoParam[1][7] * var[5];
            deriv[4] += isoParam[1][7] * var[6];
            deriv[5] += isoParam[1][6] * Lcoords[0] + isoParam[1][7] * var[4];
            deriv[6] += isoParam[2][6] * Lcoords[2] + isoParam[2][7] * var[5];
            deriv[7] += isoParam[2][7] * var[6];
            deriv[8] += isoParam[2][6] * Lcoords[0] + isoParam[2][7] * var[4];
        }
        DNekMatSharedPtr mat =
            MemoryManager<DNekMat>::AllocateSharedPtr(3, 3, 0, eFULL);
        Vmath::Vcopy(9, deriv, 1, mat->GetPtr(), 1);
        mat->Invert();
        Lcoords[0] += Vmath::Dot(3, mat->GetPtr(), 1, tmp, 1);
        Lcoords[1] += Vmath::Dot(3, mat->GetPtr() + 3, 1, tmp, 1);
        Lcoords[2] += Vmath::Dot(3, mat->GetPtr() + 6, 1, tmp, 1);

        if (!(std::isfinite(Lcoords[0]) && std::isfinite(Lcoords[1]) &&
              std::isfinite(Lcoords[2])) ||
            fabs(Lcoords[0]) > LcoordDiv || fabs(Lcoords[1]) > LcoordDiv ||
            fabs(Lcoords[2]) > LcoordDiv)
        {
            std::ostringstream ss;
            ss << "Iteration has diverged in NewtonIterationForLocCoord in "
                  "element "
               << m_geom->GetGlobalID();
            WARNINGL1(false, ss.str());
            return;
        }
    }

    if (cnt >= MaxIterations)
    {
        std::ostringstream ss;

        ss << "Reached MaxIterations (" << MaxIterations
           << ") in Newton iteration ";

        WARNINGL1(cnt < MaxIterations, ss.str());
    }
}

void GeometryLocator3D::NewtonIterationForLocCoord(
    const Array<OneD, const NekDouble> &coords,
    const Array<OneD, const NekDouble> &ptsx,
    const Array<OneD, const NekDouble> &ptsy,
    const Array<OneD, const NekDouble> &ptsz, Array<OneD, NekDouble> &Lcoords,
    NekDouble &dist)
{
    // maximum iterations for convergence
    const int MaxIterations = NekConstants::kNewtonIterations;
    // |x-xp|^2 < EPSILON  error tolerance
    const NekDouble Tol = 1.e-8;
    // |r,s|    > LcoordDIV stop the search
    const NekDouble LcoordDiv = 15.0;

    LibUtilities::PointsKeyVector ptsKeys = m_geom->GetXmap()->GetPointsKeys();
    Array<OneD, const NekDouble> Jac =
        m_geom->GenGeomFactors(ptsKeys)->GetJac();

    NekDouble ScaledTol =
        Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
    ScaledTol *= Tol;

    NekDouble xmap, ymap, zmap, F1, F2, F3;

    NekDouble derx_1, derx_2, derx_3, dery_1, dery_2, dery_3, derz_1, derz_2,
        derz_3, jac;

    // save intiial guess for later reference if required.
    NekDouble init0 = Lcoords[0], init1 = Lcoords[1], init2 = Lcoords[2];

    Array<OneD, NekDouble> DxD1(ptsx.size());
    Array<OneD, NekDouble> DxD2(ptsx.size());
    Array<OneD, NekDouble> DxD3(ptsx.size());
    Array<OneD, NekDouble> DyD1(ptsx.size());
    Array<OneD, NekDouble> DyD2(ptsx.size());
    Array<OneD, NekDouble> DyD3(ptsx.size());
    Array<OneD, NekDouble> DzD1(ptsx.size());
    Array<OneD, NekDouble> DzD2(ptsx.size());
    Array<OneD, NekDouble> DzD3(ptsx.size());

    // Ideally this will be stored in m_geomfactors
    m_geom->GetXmap()->PhysDeriv(ptsx, DxD1, DxD2, DxD3);
    m_geom->GetXmap()->PhysDeriv(ptsy, DyD1, DyD2, DyD3);
    m_geom->GetXmap()->PhysDeriv(ptsz, DzD1, DzD2, DzD3);

    int cnt = 0;
    Array<OneD, DNekMatSharedPtr> I(3);
    Array<OneD, NekDouble> eta(3);

    F1 = F2 = F3    = 2000; // Starting value of Function
    NekDouble resid = sqrt(F1 * F1 + F2 * F2 + F3 * F3);
    while (cnt++ < MaxIterations)
    {
        //  evaluate lagrange interpolant at Lcoords
        m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, eta);
        I[0] = m_geom->GetXmap()->GetBasis(0)->GetI(eta);
        I[1] = m_geom->GetXmap()->GetBasis(1)->GetI(eta + 1);
        I[2] = m_geom->GetXmap()->GetBasis(2)->GetI(eta + 2);

        // calculate the global point `corresponding to Lcoords
        xmap = m_geom->GetXmap()->PhysEvaluate(I, ptsx);
        ymap = m_geom->GetXmap()->PhysEvaluate(I, ptsy);
        zmap = m_geom->GetXmap()->PhysEvaluate(I, ptsz);

        F1 = coords[0] - xmap;
        F2 = coords[1] - ymap;
        F3 = coords[2] - zmap;

        if (F1 * F1 + F2 * F2 + F3 * F3 < ScaledTol)
        {
            resid = sqrt(F1 * F1 + F2 * F2 + F3 * F3);
            break;
        }

        // Interpolate derivative metric at Lcoords
        derx_1 = m_geom->GetXmap()->PhysEvaluate(I, DxD1);
        derx_2 = m_geom->GetXmap()->PhysEvaluate(I, DxD2);
        derx_3 = m_geom->GetXmap()->PhysEvaluate(I, DxD3);
        dery_1 = m_geom->GetXmap()->PhysEvaluate(I, DyD1);
        dery_2 = m_geom->GetXmap()->PhysEvaluate(I, DyD2);
        dery_3 = m_geom->GetXmap()->PhysEvaluate(I, DyD3);
        derz_1 = m_geom->GetXmap()->PhysEvaluate(I, DzD1);
        derz_2 = m_geom->GetXmap()->PhysEvaluate(I, DzD2);
        derz_3 = m_geom->GetXmap()->PhysEvaluate(I, DzD3);

        jac = derx_1 * (dery_2 * derz_3 - dery_3 * derz_2) -
              derx_2 * (dery_1 * derz_3 - dery_3 * derz_1) +
              derx_3 * (dery_1 * derz_2 - dery_2 * derz_1);

        // use analytical inverse of derivitives which are also similar to
        // those of metric factors.
        Lcoords[0] =
            Lcoords[0] +
            ((dery_2 * derz_3 - dery_3 * derz_2) * (coords[0] - xmap) -
             (derx_2 * derz_3 - derx_3 * derz_2) * (coords[1] - ymap) +
             (derx_2 * dery_3 - derx_3 * dery_2) * (coords[2] - zmap)) /
                jac;

        Lcoords[1] =
            Lcoords[1] -
            ((dery_1 * derz_3 - dery_3 * derz_1) * (coords[0] - xmap) -
             (derx_1 * derz_3 - derx_3 * derz_1) * (coords[1] - ymap) +
             (derx_1 * dery_3 - derx_3 * dery_1) * (coords[2] - zmap)) /
                jac;

        Lcoords[2] =
            Lcoords[2] +
            ((dery_1 * derz_2 - dery_2 * derz_1) * (coords[0] - xmap) -
             (derx_1 * derz_2 - derx_2 * derz_1) * (coords[1] - ymap) +
             (derx_1 * dery_2 - derx_2 * dery_1) * (coords[2] - zmap)) /
                jac;

        if (!(std::isfinite(Lcoords[0]) && std::isfinite(Lcoords[1]) &&
              std::isfinite(Lcoords[2])))
        {
            dist = 1e16;
            std::ostringstream ss;
            ss << "nan or inf found in NewtonIterationForLocCoord in element "
               << m_geom->GetGlobalID();
            WARNINGL1(false, ss.str());
            return;
        }
        if (fabs(Lcoords[0]) > LcoordDiv || fabs(Lcoords[1]) > LcoordDiv ||
            fabs(Lcoords[2]) > LcoordDiv)
        {
            break; // lcoords have diverged so stop iteration
        }
    }

    m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, eta);
    if (ClampLocCoords(eta, m_geom->GetShapeDim(), 0.))
    {
        I[0] = m_geom->GetXmap()->GetBasis(0)->GetI(eta);
        I[1] = m_geom->GetXmap()->GetBasis(1)->GetI(eta + 1);
        I[2] = m_geom->GetXmap()->GetBasis(2)->GetI(eta + 2);
        // calculate the global point corresponding to Lcoords
        xmap = m_geom->GetXmap()->PhysEvaluate(I, ptsx);
        ymap = m_geom->GetXmap()->PhysEvaluate(I, ptsy);
        zmap = m_geom->GetXmap()->PhysEvaluate(I, ptsz);
        F1   = coords[0] - xmap;
        F2   = coords[1] - ymap;
        F3   = coords[2] - zmap;
        dist = sqrt(F1 * F1 + F2 * F2 + F3 * F3);
    }
    else
    {
        dist = 0.;
    }

    if (cnt >= MaxIterations)
    {
        Array<OneD, NekDouble> collCoords(3);
        m_geom->GetXmap()->LocCoordToLocCollapsed(Lcoords, collCoords);

        // if coordinate is inside element dump error!
        if ((collCoords[0] >= -1.0 && collCoords[0] <= 1.0) &&
            (collCoords[1] >= -1.0 && collCoords[1] <= 1.0) &&
            (collCoords[2] >= -1.0 && collCoords[2] <= 1.0))
        {
            std::ostringstream ss;

            ss << "Reached MaxIterations (" << MaxIterations
               << ") in Newton iteration ";
            ss << "Init value (" << std::setprecision(4) << init0 << ","
               << init1 << "," << init2 << ") ";
            ss << "Fin  value (" << Lcoords[0] << "," << Lcoords[1] << ","
               << Lcoords[2] << ") ";
            ss << "Resid = " << resid
               << " Tolerance = " << std::sqrt(ScaledTol);

            WARNINGL1(cnt < MaxIterations, ss.str());
        }
    }
}

NekDouble GeometryLocator3D::v_GetLocCoords(
    const Array<OneD, const NekDouble> &coords, Array<OneD, NekDouble> &Lcoords)
{
    NekDouble dist = std::numeric_limits<double>::max();
    Array<OneD, NekDouble> tmpcoords(3);
    tmpcoords[0] = coords[0];
    tmpcoords[1] = coords[1];
    tmpcoords[2] = coords[2];
    if (m_geom->CalcGeomType() == eRegular)
    {
        tmpcoords[0] -= m_iso.m_coeff[0][0];
        tmpcoords[1] -= m_iso.m_coeff[1][0];
        tmpcoords[2] -= m_iso.m_coeff[2][0];
        Lcoords[0] = m_invIsoParam[0][0] * tmpcoords[0] +
                     m_invIsoParam[0][1] * tmpcoords[1] +
                     m_invIsoParam[0][2] * tmpcoords[2];
        Lcoords[1] = m_invIsoParam[1][0] * tmpcoords[0] +
                     m_invIsoParam[1][1] * tmpcoords[1] +
                     m_invIsoParam[1][2] * tmpcoords[2];
        Lcoords[2] = m_invIsoParam[2][0] * tmpcoords[0] +
                     m_invIsoParam[2][1] * tmpcoords[1] +
                     m_invIsoParam[2][2] * tmpcoords[2];
    }
    else if (m_straightEdge && m_iso.m_nCoeff > 0)
    {
        // Straight-sided, and the shape has a polynomial form of its mapping
        // to iterate on. A pyramid with a non-parallelogram base has not.
        ClampLocCoords(Lcoords, m_geom->GetShapeDim(), 0.);
        NewtonIterationForLocCoord(tmpcoords, Lcoords);
    }
    else
    {
        m_geom->FillGeom();
        // Determine nearest point of coords  to values in m_geom->GetXmap()
        int npts = m_geom->GetXmap()->GetTotPoints();
        Array<OneD, NekDouble> ptsx(npts), ptsy(npts), ptsz(npts);
        Array<OneD, NekDouble> tmp1(npts), tmp2(npts);

        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(0), ptsx);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(1), ptsy);
        m_geom->GetXmap()->BwdTrans(m_geom->GetCoeffs(2), ptsz);

        const Array<OneD, const NekDouble> za = m_geom->GetXmap()->GetPoints(0);
        const Array<OneD, const NekDouble> zb = m_geom->GetXmap()->GetPoints(1);
        const Array<OneD, const NekDouble> zc = m_geom->GetXmap()->GetPoints(2);

        // guess the first local coords based on nearest point
        Vmath::Sadd(npts, -coords[0], ptsx, 1, tmp1, 1);
        Vmath::Vmul(npts, tmp1, 1, tmp1, 1, tmp1, 1);
        Vmath::Sadd(npts, -coords[1], ptsy, 1, tmp2, 1);
        Vmath::Vvtvp(npts, tmp2, 1, tmp2, 1, tmp1, 1, tmp1, 1);
        Vmath::Sadd(npts, -coords[2], ptsz, 1, tmp2, 1);
        Vmath::Vvtvp(npts, tmp2, 1, tmp2, 1, tmp1, 1, tmp1, 1);

        int min_i = Vmath::Imin(npts, tmp1, 1);

        // Get Local coordinates
        int qa = za.size(), qb = zb.size();

        Array<OneD, NekDouble> eta(3, 0.);
        eta[2] = zc[min_i / (qa * qb)];
        min_i  = min_i % (qa * qb);
        eta[1] = zb[min_i / qa];
        eta[0] = za[min_i % qa];
        m_geom->GetXmap()->LocCollapsedToLocCoord(eta, Lcoords);

        // Perform newton iteration to find local coordinates
        NewtonIterationForLocCoord(coords, ptsx, ptsy, ptsz, Lcoords, dist);
    }
    return dist;
}

void GeometryLocator3D::v_CalculateInverseIsoParam()
{
    DNekMatSharedPtr mat =
        MemoryManager<DNekMat>::AllocateSharedPtr(3, 3, 0, eFULL);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {

            mat->SetValue(i, j, m_iso.m_coeff[i][j + 1]);
        }
    }
    mat->Invert();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            m_invIsoParam[i][j] = mat->GetValue(i, j);
        }
    }
}

} // namespace Nektar::SpatialDomains
