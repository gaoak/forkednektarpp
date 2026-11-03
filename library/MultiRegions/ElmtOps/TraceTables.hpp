///////////////////////////////////////////////////////////////////////////////
//
// File: TraceTables.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: The interpolation tables, weights and collocation flags the
// trace lifting operators fetch at construction.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file TraceTables.hpp
 * @brief The tables an operator lifting an element's traces onto its volume
 * reads, fetched in one place.
 *
 * @details
 * The trace inner product (IProductWRTPhysTrace) and the lift-and-project
 * operator (PhysLiftProject), which applies its lift, read one element
 * block's faces through the same tables: the end-point interpolant of each
 * direction onto its traces, then the in-face interpolant of each face pair
 * onto its trace rule, and the trace rules' weights, with the flags saying
 * where an interpolant is the identity. FetchTraceTables() fetches them from
 * the data warehouse in the order the kernels index them.
 */

#pragma once

#include <vector>

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LocalRegions/Expansion.h>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief The trace tables of one element block.
 *
 * The face pairs are numbered `(dim - 1) * f + d` over the faces' normal
 * direction @c f and their tangential directions @c d, the representative
 * trace of normal direction @c f being `dim - 1 - f`, as the trace kernels
 * number them. A segment has no face pairs.
 *
 * @tparam TTable  Table entry type: the scalar, or the SIMD vector the
 *                 kernels broadcast across the lanes.
 */
template <typename TTable> struct TraceTables
{
    /// Volume quadrature points per direction.
    std::vector<unsigned int> nm;
    /// The end-point interpolant of each direction, \f$h_p(\pm 1)\f$ on its
    /// traces, then the in-face interpolant of each face pair onto its trace
    /// rule.
    std::vector<const TTable *> B;
    /// The trace quadrature weights of each face pair.
    std::vector<const TTable *> W;
    /// Trace quadrature points of each face pair.
    std::vector<unsigned int> nq;
    /// Whether each direction's volume rule contains its end points, in which
    /// case its end-point interpolant is a delta.
    std::vector<bool> endPtsCollocated;
    /// Whether each face pair's trace rule is the volume rule of its
    /// direction, in which case its in-face interpolant is the identity.
    std::vector<bool> isCollocated;
};

/**
 * @brief Fetch the trace tables of the block whose representative expansion
 * is @p exp from @p warehouse.
 *
 * A direction with one trace takes its end-point interpolant onto the
 * Gauss-Legendre-with-M points, whose first point is \f$\xi = -1\f$; a
 * direction with two onto the Gauss-Lobatto points, both ends.
 *
 * @tparam TTable    Table entry type.
 * @tparam MemSpace  Memory space the tables are fetched to.
 *
 * @param   exp        Representative expansion of the block.
 * @param   warehouse  Warehouse the tables are fetched from.
 */
template <typename TTable, typename MemSpace>
TraceTables<TTable> FetchTraceTables(
    const LocalRegions::ExpansionSharedPtr &exp,
    const LibUtilities::NekDataWarehouseSharedPtr &warehouse)
{
    TraceTables<TTable> t;
    const auto shape        = exp->DetShapeType();
    const unsigned int ndim = exp->GetShapeDimension();

    for (unsigned int dir = 0; dir < ndim; ++dir)
    {
        const auto bkey = exp->GetBasis(dir)->GetBasisKey();
        t.nm.push_back(exp->GetNumPoints(dir));

        const unsigned int ntrace =
            LibUtilities::ShapeTypeNumTraceInDir[shape][dir];
        const LibUtilities::PointsType endPtsType =
            ntrace == 1 ? LibUtilities::eGaussLegendreWithM
                        : LibUtilities::eGaussLobattoLegendre;
        t.B.push_back(warehouse->template GetData<MemSpace>(
            LibUtilities::BasisDataKey<TTable>(bkey, LibUtilities::eInterp,
                                               ntrace, endPtsType)));

        const unsigned int nEndPts =
            LibUtilities::PointsTypeNumEndPts[bkey.GetPointsType()];
        ASSERTL1(nEndPts == 0 || nEndPts == ntrace,
                 "Trace and number of end points in PointsType differ");
        t.endPtsCollocated.push_back(nEndPts != 0);
    }

    const auto expPtsKeys = exp->GetPointsKeys();
    for (unsigned int f = 0; f < ndim; ++f)
    {
        const unsigned int tr = ndim - 1 - f;
        for (unsigned int d = 0; d + 1 < ndim; ++d)
        {
            const unsigned int dir = exp->GetGeom()->GetDir(tr, d);
            const auto trBKey      = exp->GetTraceBasisKey(tr, d);
            const auto trPtsKey    = trBKey.GetPointsKey();
            const unsigned int nq  = trPtsKey.GetNumPoints();

            t.isCollocated.push_back(expPtsKeys[dir] == trPtsKey);
            t.nq.push_back(nq);
            t.B.push_back(warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TTable>(
                    exp->GetBasis(dir)->GetBasisKey(), LibUtilities::eInterp,
                    nq, trPtsKey.GetPointsType())));
            t.W.push_back(warehouse->template GetData<MemSpace>(
                LibUtilities::BasisDataKey<TTable>(trBKey,
                                                   LibUtilities::eWeights)));
        }
    }
    return t;
}

} // namespace Nektar::MultiRegions::detail
