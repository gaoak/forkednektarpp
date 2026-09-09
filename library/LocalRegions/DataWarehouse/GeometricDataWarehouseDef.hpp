///////////////////////////////////////////////////////////////////////////////
//
// File: GeometricDataWarehouseDef.hpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::LocalRegions
{

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const WeightsKey<TData> &weightsKey)
{
    // Fetch data from key.
    const auto block_idx        = weightsKey.m_block_idx;
    const auto interleave_width = weightsKey.m_interleave_width;

    // Fetch expansion.
    auto coll   = m_collections[block_idx].GetExpVector();
    auto expPtr = coll[0];

    // Allocate memory and get pointer.
    const auto memsize = interleave_width * expPtr->GetTotPoints();
    auto weights       = LibUtilities::MemoryRegion<TData>(memsize);

    auto wptr = weights.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    auto dimension = expPtr->GetShapeDimension();

    auto W = expPtr->GetQuadratureWeights();

    switch (dimension)
    {
        case 1:
        {
            auto nq0 = expPtr->GetNumPoints(0);
            for (unsigned i = 0; i < nq0; ++i)
            {
                for (unsigned wd = 0; wd < interleave_width; ++wd)
                {
                    wptr[i * interleave_width + wd] = W[0][i];
                }
            }
        }
        break;
        case 2:
        {
            auto nq0 = expPtr->GetNumPoints(0);
            auto nq1 = expPtr->GetNumPoints(1);

            for (unsigned j = 0; j < nq1; ++j)
            {
                for (unsigned i = 0; i < nq0; ++i)
                {
                    auto offset = (j * nq0 + i) * interleave_width;
                    for (unsigned wd = 0; wd < interleave_width; ++wd)
                    {
                        wptr[offset + wd] = W[1][j] * W[0][i];
                    }
                }
            }
        }
        break;
        case 3:
        {
            auto nq0  = expPtr->GetNumPoints(0);
            auto nq1  = expPtr->GetNumPoints(1);
            auto nq2  = expPtr->GetNumPoints(2);
            auto nq01 = nq0 * nq1;

            for (unsigned k = 0; k < nq2; ++k)
            {
                for (unsigned j = 0; j < nq1; ++j)
                {
                    auto w12 = W[1][j] * W[2][k];
                    for (unsigned i = 0; i < nq0; ++i)
                    {
                        auto offset =
                            (k * nq01 + j * nq0 + i) * interleave_width;
                        for (unsigned wd = 0; wd < interleave_width; ++wd)
                        {
                            wptr[offset + wd] = w12 * W[0][i];
                        }
                    }
                }
            }
        }
        break;
    }
    return weights;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const JacobianKey<TData> &jacobianKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = jacobianKey.m_block_idx;
    const auto interleave_width = jacobianKey.m_interleave_width;

    // Fetch expansion.
    auto coll               = m_collections[block_idx].GetExpVector();
    auto expPtr             = coll[0];
    const auto num_elements = coll.size();
    const bool isDeformed =
        (expPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Allocate memory and get pointer.
    const auto memsize = (isDeformed) ? num_elmt_groups * interleave_width *
                                            expPtr->GetTotPoints()
                                      : num_elmt_groups * interleave_width;
    auto jac           = LibUtilities::MemoryRegion<TData>(memsize);
    auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Deformed geometry.
    if (isDeformed)
    {
        Array<OneD, Array<OneD, double>> jacArray(interleave_width);

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over interleave width
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
                // Check for padding
                if (el < num_elements)
                {
                    jacArray[i] = coll[el]->GetGeomFactors()->GetJac();
                }
                else
                {
                    jacArray[i] =
                        Array<OneD, double>(expPtr->GetTotPoints(), 0.0);
                }
            }

            // Save interleaved data.
            for (unsigned int pt = 0; pt < expPtr->GetTotPoints(); ++pt)
            {
                for (unsigned i = 0; i < interleave_width; ++i)
                {
                    *(jacptr++) = jacArray[i][pt];
                }
            }
        }

        return jac;
    }
    // Regular geometry.
    else
    {
        auto jac    = LibUtilities::MemoryRegion<TData>(num_elmt_groups *
                                                        interleave_width);
        auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over interleave width
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
                // Check for padding
                if (el < num_elements)
                {
                    auto &auxJac = coll[el]->GetGeomFactors()->GetJac();
                    jacptr[el]   = auxJac[0];
                }
                else
                {
                    jacptr[el] = 0.0;
                }
            }
        }

        return jac;
    }
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const DerivFactorKey<TData> &derivFactorKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = derivFactorKey.m_block_idx;
    const auto interleave_width = derivFactorKey.m_interleave_width;
    const auto transpose        = derivFactorKey.m_transpose;

    // Fetch expansion.
    auto coll               = m_collections[block_idx].GetExpVector();
    auto expPtr             = coll[0];
    const auto num_elements = coll.size();
    const bool isDeformed =
        (expPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);
    const auto nDim   = expPtr->GetShapeDimension();
    const auto nCoord = expPtr->GetCoordim();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto range1 = transpose ? nDim * nCoord : expPtr->GetTotPoints();
    const auto range2 = transpose ? expPtr->GetTotPoints() : nDim * nCoord;

    // Allocate memory and get pointer.
    const auto memsize =
        (isDeformed) ? num_elmt_groups * interleave_width *
                           expPtr->GetTotPoints() * nDim * nCoord
                     : num_elmt_groups * interleave_width * nDim * nCoord;
    auto df    = LibUtilities::MemoryRegion<TData>(memsize);
    auto dfptr = df.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Deformed geometry.
    if (isDeformed)
    {
        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over component 1.
            for (unsigned int index1 = 0; index1 < range1; ++index1)
            {
                // Loop over component 2.
                for (unsigned int index2 = 0; index2 < range2; ++index2)
                {
                    // Loop over interleave width.
                    for (unsigned int i = 0; i < interleave_width; ++i)
                    {
                        // Check for padding.
                        if (el + i < num_elements)
                        {
                            const auto d  = transpose ? index1 : index2;
                            const auto pt = transpose ? index2 : index1;

                            auto &tmp = coll[el + i]
                                            ->GetGeomFactors()
                                            ->GetDerivFactors();
                            *(dfptr++) = tmp[d][pt];
                        }
                        else
                        {
                            *(dfptr++) = 0.0;
                        }
                    }
                }
            }

            el += interleave_width;
        }

        return df;
    }
    // Regular geometry.
    else
    {
        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over component.
            for (unsigned int d = 0; d < nDim * nCoord; ++d)
            {
                // Loop over interleave width.
                for (unsigned int i = 0; i < interleave_width; ++i)
                {
                    // Check for padding.
                    if (el + i < num_elements)
                    {
                        auto &tmp =
                            coll[el + i]->GetGeomFactors()->GetDerivFactors();
                        *(dfptr++) = tmp[d][0];
                    }
                    else
                    {
                        *(dfptr++) = 0.0;
                    }
                }
            }

            el += interleave_width;
        }

        return df;
    }
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const CoordKey<TData> &coordKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = coordKey.m_block_idx;
    const auto interleave_width = coordKey.m_interleave_width;
    const auto transpose        = coordKey.m_transpose;

    // Fetch expansion.
    auto coll               = m_collections[block_idx].GetExpVector();
    auto expPtr             = coll[0];
    const auto num_elements = coll.size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nDim   = expPtr->GetCoordim();
    const auto range1 = transpose ? nDim : expPtr->GetTotPoints();
    const auto range2 = transpose ? expPtr->GetTotPoints() : nDim;

    // Allocate memory and get pointer.
    const auto memsize =
        num_elmt_groups * interleave_width * expPtr->GetTotPoints() * nDim;
    auto crds   = LibUtilities::MemoryRegion<TData>(memsize);
    auto crdptr = crds.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over chunks.
    for (size_t chunk = 0, el = 0, crd_id = 0; chunk < num_elmt_groups; ++chunk)
    {
        // Loop over component or points
        for (unsigned index1 = 0; index1 < range1; ++index1)
        {
            // Loop over points or components
            for (unsigned index2 = 0; index2 < range2; ++index2)
            {
                // Loop over interleave width
                for (unsigned i = 0; i < interleave_width; ++i, ++crd_id)
                {
                    // Check for padding
                    if (el + i < num_elements)
                    {
                        const auto d  = transpose ? index1 : index2;
                        const auto pt = transpose ? index2 : index1;

                        auto tmp       = coll[el + i]->GetCoords();
                        crdptr[crd_id] = tmp[d][pt];
                    }
                    else
                    {
                        crdptr[crd_id] = 0.0;
                    }
                }
            }
        }

        el += interleave_width;
    }

    return crds;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned> GeometricDataCreator::Create(
    const OrientKey<TData> &orientKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = orientKey.m_block_idx;
    const auto interleave_width = orientKey.m_interleave_width;

    // Fetch expansion.
    auto coll               = m_collections[block_idx].GetExpVector();
    auto expPtr             = coll[0];
    const auto num_elements = coll.size();
    const auto nedge        = expPtr->GetGeom()->GetNumEdges();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Allocate memory and get pointer.
    const auto memsize = num_elmt_groups * interleave_width * nedge;
    auto orients       = LibUtilities::MemoryRegion<unsigned int>(memsize);
    auto orientptr =
        orients.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over chunks.
    for (size_t chunk = 0, el = 0, orient_id = 0; chunk < num_elmt_groups;
         ++chunk)
    {
        // Loop over component or points
        for (unsigned int ed = 0; ed < nedge; ++ed)
        {
            // Loop over interleave width
            for (unsigned i = 0; i < interleave_width; ++i, ++orient_id)
            {
                // Check for padding
                if (el + i < num_elements)
                {
                    auto orient          = coll[el + i]->GetTraceOrient(ed);
                    orientptr[orient_id] = static_cast<unsigned int>(orient);
                }
                else
                {
                    orientptr[orient_id] =
                        static_cast<unsigned>(StdRegions::eForwards);
                }
            }
        }

        el += interleave_width;
    }

    return orients;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned> GeometricDataCreator::Create(
    const TraceToElmtMapKey<TData> &traceToElmtMapKey)
{
    // Fetch data from key.
    const auto block_idx = traceToElmtMapKey.m_block_idx;

    // Get the reference element.
    // Note the traceToElemtMap is equivalent for all elements within a block.
    auto coll        = m_collections[block_idx].GetExpVector();
    auto expPtr      = coll[0];
    const auto nedge = expPtr->GetGeom()->GetNumEdges();

    // Total nummodes for trace
    auto nmTrace = 0;
    std::vector<unsigned int> nmEdge;
    for (int i = 0; i < nedge; i++)
    {
        nmTrace += expPtr->GetTraceNcoeffs(i);
        nmEdge.push_back(expPtr->GetTraceNcoeffs(i));
    }

    // Allocate memory and get pointer.
    const auto memsize = nmTrace;
    auto map           = LibUtilities::MemoryRegion<unsigned>(memsize);
    auto mapptr = map.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over component or points
    for (unsigned int ed = 0; ed < nedge; ++ed)
    {
        // Allocate temporary arrays for map and sign
        Array<OneD, unsigned> tmpMap(nmEdge[ed], (unsigned)0);
        Array<OneD, int> tmpSign(nmEdge[ed], 1);

        // Extract map and sign
        expPtr->GetTraceToElementMap(ed, tmpMap, tmpSign);

        // Assign map to memory region
        for (unsigned nm = 0; nm < nmEdge[ed]; ++nm)
        {
            *(mapptr++) = tmpMap[nm];
        }
    }

    return map;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<int> GeometricDataCreator::Create(
    const TraceToElmtSignKey<TData> &traceToElmtSignKey)
{
    // Fetch data from key.
    const auto block_idx = traceToElmtSignKey.m_block_idx;

    // Get the reference element.
    // Note the sign is equivalent for all elements within a block.
    auto coll        = m_collections[block_idx].GetExpVector();
    auto expPtr      = coll[0];
    const auto nedge = expPtr->GetGeom()->GetNumEdges();

    // Total nummodes for trace
    auto nmTrace = 0;
    for (int ed = 0; ed < nedge; ed++)
    {
        nmTrace += expPtr->GetTraceNcoeffs(ed);
    }

    // Allocate memory and get pointer.
    const auto memsize = nmTrace;
    auto sign          = LibUtilities::MemoryRegion<int>(memsize);
    auto signptr = sign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over number of edges

    for (unsigned int ed = 0; ed < nedge; ++ed)
    {
        // Allocate temporary arrays for map and sign
        Array<OneD, unsigned int> tmpMap(expPtr->GetTraceNcoeffs(ed),
                                         (unsigned)0);
        Array<OneD, int> tmpSign(expPtr->GetTraceNcoeffs(ed), 1);

        // Extract map and sign
        expPtr->GetTraceToElementMap(ed, tmpMap, tmpSign);

        // Assign map to memory region
        for (unsigned int nm = 0; nm < expPtr->GetTraceNcoeffs(ed); ++nm)
        {
            *(signptr++) = tmpSign[nm];
        }
    }

    return sign;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<unsigned> GeometricDataCreator::Create(
    const InteriorMapKey<TData> &interiorMapKey)
{
    // Fetch data from key.
    const auto block_idx = interiorMapKey.m_block_idx;

    // Get the reference element.
    // Note the interior map is equivalent for all elements within a block.
    auto coll   = m_collections[block_idx].GetExpVector();
    auto expPtr = coll[0];

    // Get size of interior map
    auto numInteriorCoeffs = expPtr->GetNcoeffs() - expPtr->NumBndryCoeffs();

    // Allocate memory and get pointer.
    const auto memsize = numInteriorCoeffs;
    auto map           = LibUtilities::MemoryRegion<unsigned>(memsize);
    auto mapptr = map.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Allocate temporary arrays for map and sign
    Array<OneD, unsigned> tmpMap(numInteriorCoeffs);

    // Extract map
    expPtr->GetInteriorMap(tmpMap);

    // Assign map to memory region
    for (unsigned nm = 0; nm < numInteriorCoeffs; ++nm)
    {
        *(mapptr++) = tmpMap[nm];
    }

    return map;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const JacobianTraceKey<TData> &jacobianTraceKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = jacobianTraceKey.m_block_idx;
    const auto interleave_width = jacobianTraceKey.m_interleave_width;

    auto coll               = m_collections[block_idx].GetExpVector();
    auto expPtr             = coll[0];
    const auto num_elements = coll.size();
    // Assume non-deformed edge for now.
    const bool isDeformed = false;
    // const bool isDeformed =
    //    (expPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);
    const auto nedge = expPtr->GetGeom()->GetNumEdges();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Allocate memory.
    auto nTraceJacPoints = 0;
    for (int ed = 0; ed < nedge; ed++)
    {
        nTraceJacPoints += expPtr->GetTraceExp(ed)->GetTotPoints();
    }

    const auto memsize =
        (isDeformed) ? num_elmt_groups * interleave_width * nTraceJacPoints
                     : num_elmt_groups * interleave_width * nedge;
    auto jac    = LibUtilities::MemoryRegion<TData>(memsize);
    auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Deformed geometry.
    if (isDeformed)
    {
        Array<OneD, Array<OneD, double>> jacArray(interleave_width);

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over edges.
            for (unsigned int ed = 0; ed < nedge; ++ed)
            {
                for (unsigned int i = 0; i < interleave_width; ++i)
                {
                    // Check for padding.
                    if (el + i < num_elements)
                    {
                        // Get trace expansion.
                        auto traceExp = coll[el + i]->GetTraceExp(ed);

                        // Get trace Jacobian.
                        jacArray[i] = traceExp->GetGeomFactors()->GetJac();
                    }
                    else
                    {
                        jacArray[i] = Array<OneD, double>(
                            expPtr->GetTraceExp(ed)->GetTotPoints(), 0.0);
                    }
                }

                // Save interleaved data.
                for (unsigned int pt = 0;
                     pt < expPtr->GetTraceExp(ed)->GetTotPoints(); ++pt)
                {
                    for (unsigned int i = 0; i < interleave_width; ++i)
                    {
                        *(jacptr++) = jacArray[i][pt];
                    }
                }
            }

            el += interleave_width;
        }

        return jac;
    }
    // Regular geometry.
    else
    {
        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over edges.
            for (unsigned int ed = 0; ed < nedge; ++ed)
            {
                for (unsigned int i = 0; i < interleave_width; ++i)
                {
                    if (el + i < num_elements)
                    {
                        // Get trace expansion.
                        auto traceExp = coll[el + i]->GetTraceExp(ed);

                        // Get trace Jacobian.
                        auto &auxJac = traceExp->GetGeomFactors()->GetJac();
                        *(jacptr++)  = auxJac[0];
                    }
                    else
                    {
                        *(jacptr++) = 0.0;
                    }
                }
            }

            el += interleave_width;
        }

        return jac;
    }
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const JacobianLocTraceKey<TData> &jacobianLocTraceKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = jacobianLocTraceKey.m_block_idx;
    const auto interleave_width = jacobianLocTraceKey.m_interleave_width;

    auto coll       = m_collections[block_idx].GetExpVector();
    auto expPtr     = coll[0];
    const auto nDim = expPtr->GetShapeDimension();

    const auto num_elements = coll.size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // get total trace size
    LibUtilities::ShapeType shape = expPtr->DetShapeType();

    // if element deformed treat trace group as deformed.
    bool isDeformed =
        (expPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);

    // caculate the total number of quadrature points
    std::vector<unsigned> nTracePts;
    unsigned nTraceJacPoints = 0;
    for (unsigned dir = 0; dir < nDim; ++dir)
    {
        auto ntraces = LibUtilities::ShapeTypeNumTraceInDir[shape][dir];

        // first trace id
        auto traceId = LibUtilities::ShapeTypeTraceIDInDir[shape][dir][0];

        auto npts = expPtr->GetLocTraceExp(traceId)->GetTotPoints();
        nTracePts.push_back(npts);

        if (isDeformed)
        {
            nTraceJacPoints += npts * ntraces;
        }
        else
        {
            nTraceJacPoints += ntraces;
        }
    }

    const auto memsize = num_elmt_groups * interleave_width * nTraceJacPoints;
    auto jac           = LibUtilities::MemoryRegion<TData>(memsize);
    auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over chunks.
    for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
    {
        for (unsigned i = 0; i < interleave_width; ++i, ++el)
        {
            unsigned offset = 0;
            for (unsigned dir = 0; dir < nDim; ++dir)
            {
                auto ntraces = LibUtilities::ShapeTypeNumTraceInDir[shape][dir];

                // Loop over traces
                for (unsigned n = 0; n < ntraces; ++n)
                {
                    auto traceId =
                        LibUtilities::ShapeTypeTraceIDInDir[shape][dir][n];

                    // Index for edge id
                    if (el < num_elements)
                    {
                        // Get trace expansion
                        auto traceExp = coll[el]->GetLocTraceExp(traceId);

                        // Get trace Jacobian
                        auto jacArray = traceExp->GetGeomFactors()->GetJac();

                        if (isDeformed)
                        {
                            auto TraceDeformed =
                                (traceExp->GetGeomFactors()->GetGtype() ==
                                 SpatialDomains::eDeformed);

                            if (TraceDeformed)
                            {
                                for (unsigned pt = 0; pt < nTracePts[dir]; ++pt)
                                {
                                    jacptr[(offset + pt) * interleave_width +
                                           i] = jacArray[pt];
                                }
                            }
                            else
                            {
                                for (unsigned pt = 0; pt < nTracePts[dir]; ++pt)
                                {
                                    jacptr[(offset + pt) * interleave_width +
                                           i] = jacArray[0];
                                }
                            }
                            offset += nTracePts[dir];
                        }
                        else
                        {
                            jacptr[offset * interleave_width + i] = jacArray[0];
                            offset++;
                        }
                    }
                    else
                    {
                        if (isDeformed)
                        {
                            for (unsigned pt = 0; pt < nTracePts[dir]; ++pt)
                            {
                                jacptr[(offset + pt) * interleave_width + i] =
                                    0.0;
                            }
                            offset += nTracePts[dir];
                        }
                        else
                        {
                            jacptr[offset * interleave_width + i] = 0.0;
                            offset++;
                        }
                    }
                }
            }
            ASSERTL1(offset == nTraceJacPoints,
                     "Tot Jacobian points not correct");
        }
        jacptr += interleave_width * nTraceJacPoints;
    }
    return jac;
}
/**
   Convert the derivative factors from cartesian format to local
   collapsed coordinates in direction 'd'. Inputs include
   - dfactors: Cartesian deriv factors times the normals
   - shape: the shape type
   - traceId: legacy trace id
   - eta: the three collapsed coordinates of this trace point. The
     coordinate normal to the trace is fixed at -1 or +1 and the tangential
     ones are the trace expansion's own abscissae, which for a collapsed
     shape are not the element's abscissae: a modified triangle carries
     Gauss-Radau points in direction 1 but hands its edges 1 and 2 a
     Gauss-Lobatto basis with one extra point.
   - pt: local point in trace.
 */
double CollCoordDfactor(Array<OneD, Array<OneD, Array<OneD, double>>> &dfactors,
                        LibUtilities::ShapeType shape, const unsigned traceId,
                        const double *eta, const unsigned d, const unsigned pt)
{
    double returnval = -99;

    switch (shape)
    {
        // no modification required for Seg,Quad,Hex
        case LibUtilities::Seg:
        case LibUtilities::Quad:
        case LibUtilities::Hex:
            returnval = dfactors[d][traceId][pt];
            break;
        case LibUtilities::Tri:
        case LibUtilities::NodalTri:
        {
            if (d == 0)
            {
                // dfactors[0] + (1+eta_0)/2 dfactors[1].
                // The 2/(1-eta_1) of the chain rule is left to the operator.
                returnval = dfactors[0][traceId][pt] +
                            0.5 * (1.0 + eta[0]) * dfactors[1][traceId][pt];
            }
            else
            {
                returnval = dfactors[d][traceId][pt];
            }
            break;
        }
        case LibUtilities::Prism:
        case LibUtilities::NodalPrism:
        {
            if (d == 0)
            {
                // dfactors[0] + (1+eta_0)/2 dfactors[2].
                // The 2/(1-eta_2) of the chain rule is left to the operator.
                returnval = dfactors[0][traceId][pt] +
                            0.5 * (1.0 + eta[0]) * dfactors[2][traceId][pt];
            }
            else // dir = 1,2 are standard factors
            {
                returnval = dfactors[d][traceId][pt];
            }
            break;
        }
        case LibUtilities::Pyr:
        {
            if (d == 0)
            {
                // dfactors[0] + (1+eta_0)/2 dfactors[2].
                // The 2/(1-eta_2) of the chain rule is left to the operator.
                returnval = dfactors[0][traceId][pt] +
                            0.5 * (1.0 + eta[0]) * dfactors[2][traceId][pt];
            }
            else if (d == 1)
            {
                // dfactors[1] + (1+eta_1)/2 dfactors[2].
                // The 2/(1-eta_2) of the chain rule is left to the operator.
                returnval = dfactors[1][traceId][pt] +
                            0.5 * (1.0 + eta[1]) * dfactors[2][traceId][pt];
            }
            else
            {
                returnval = dfactors[d][traceId][pt];
            }
            break;
        }
        case LibUtilities::Tet:
        case LibUtilities::NodalTet:
        {
            if (d == 0)
            {
                // dfactors[0] + (1+eta_0)/2 (dfactors[1] + dfactors[2]).
                // Both 2/(1-eta_1) and 2/(1-eta_2) of the chain rule are
                // left to the operator.
                returnval =
                    dfactors[0][traceId][pt] +
                    0.5 * (1.0 + eta[0]) *
                        (dfactors[1][traceId][pt] + dfactors[2][traceId][pt]);
            }
            else if (d == 1)
            {
                // dfactors[1] + (1+eta_1)/2 dfactors[2].
                // The 2/(1-eta_2) of the chain rule is left to the operator.
                returnval = dfactors[1][traceId][pt] +
                            0.5 * (1.0 + eta[1]) * dfactors[2][traceId][pt];
            }
            else
            {
                returnval = dfactors[d][traceId][pt];
            }
            break;
        }
        default:
            NEKERROR(ErrorUtil::efatal, "Shape type not know or not set up");
            break;
    }

    return returnval;
}

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> GeometricDataCreator::Create(
    const JacNormGeomFactorLocTraceKey<TData> &jacnormgeomfacLocTraceKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = jacnormgeomfacLocTraceKey.m_block_idx;
    const auto interleave_width = jacnormgeomfacLocTraceKey.m_interleave_width;
    const auto facDir           = jacnormgeomfacLocTraceKey.m_dir;

    auto collVec                  = m_collections[block_idx].GetExpVector();
    auto locExpPtr                = collVec[0];
    const auto nDim               = locExpPtr->GetShapeDimension();
    LibUtilities::ShapeType shape = locExpPtr->DetShapeType();

    const auto num_elements = collVec.size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // A collapsed shape needs a factor per trace point whatever its geometry
    // type: eta_0 varies along edge 0 of a triangle, so the regular part of
    // the chain rule varies with it. The operator makes the same decision, so
    // the two agree on the layout of this array.
    bool isDeformed = TraceDerivFactorsArePointwise(
        shape,
        locExpPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);

    // A segment is handled on its own. Its traces are the two vertices, and
    // a 0D trace is a point: no trace quadrature, no orientation, and a unit
    // trace jacobian. None of the trace expansion machinery below applies -
    // GetStdTraceExp does not even exist for a Seg - and all that is left is
    // the normal derivative factor at each vertex, which
    // Expansion1D::v_NormalTraceDerivFactors returns as factors[trace][0].
    if (nDim == 1)
    {
        const auto nTrace  = LibUtilities::ShapeTypeNumTraces[shape];
        const auto memsize = num_elmt_groups * interleave_width * nTrace;

        auto jacngf = LibUtilities::MemoryRegion<TData>(memsize);
        auto ptr = jacngf.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        Array<OneD, Array<OneD, Array<OneD, double>>> dfac(3);

        // No collapsed direction in 1D, so eta is only here to satisfy the
        // shared signature; the Seg branch of CollCoordDfactor ignores it.
        const double eta[3] = {-1.0, -1.0, -1.0};

        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            for (unsigned i = 0; i < interleave_width; ++i, ++el)
            {
                if (el < num_elements)
                {
                    auto Exp = collVec[el];
                    if (facDir < 0)
                    {
                        Exp->NormalTraceDerivFactors(dfac[0], dfac[1], dfac[2]);
                    }
                    else
                    {
                        Exp->TraceDerivFactors(facDir, dfac[0], dfac[1],
                                               dfac[2]);
                    }

                    for (unsigned n = 0; n < nTrace; ++n)
                    {
                        ptr[n * interleave_width + i] = static_cast<TData>(
                            CollCoordDfactor(dfac, shape, n, eta, 0, 0));
                    }
                }
                else
                {
                    // Padding lane.
                    for (unsigned n = 0; n < nTrace; ++n)
                    {
                        ptr[n * interleave_width + i] = 0.0;
                    }
                }
            }
            ptr += interleave_width * nTrace;
        }

        return jacngf;
    }

    // caculate the total number of quadrature points
    std::vector<unsigned> nTracePts;
    unsigned nTraceJacPoints = 0;
    for (unsigned dir = 0; dir < nDim; ++dir)
    {
        auto ntraces = LibUtilities::ShapeTypeNumTraceInDir[shape][dir];

        // first trace id
        auto traceId = LibUtilities::ShapeTypeTraceIDInDir[shape][dir][0];

        auto npts = locExpPtr->GetStdTraceExp(traceId)->GetTotPoints();
        nTracePts.push_back(npts);

        if (isDeformed)
        {
            nTraceJacPoints += npts * ntraces;
        }
        else
        {
            nTraceJacPoints += ntraces;
        }
    }

    // The nDim components are stored one whole component after the other,
    // so the stride from one component to the next is the size of a single
    // component and not the size of the entire region.
    const auto compStride =
        num_elmt_groups * interleave_width * nTraceJacPoints;
    const auto memsize = compStride * nDim;
    auto jacngf        = LibUtilities::MemoryRegion<TData>(memsize);
    auto jacngfptr =
        jacngf.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    Array<OneD, Array<OneD, Array<OneD, double>>> dfactors(3);

    // Collapsed coordinates along each trace, taken from the trace's own
    // expansion. These are not in general the element's abscissae: a
    // modified triangle carries Gauss-Radau points in direction 1 but hands
    // its edges 1 and 2 a Gauss-Lobatto basis with one extra point, so
    // indexing the element's array by a trace point index reads both the
    // wrong values and, on the last point, off the end.
    //
    // Indexed by legacy trace id, then by trace direction.
    std::vector<std::vector<Array<OneD, const double>>> ztr(
        LibUtilities::ShapeTypeNumTraces[shape]);

    for (unsigned dir = 0; dir < nDim; ++dir)
    {
        auto ntraces = LibUtilities::ShapeTypeNumTraceInDir[shape][dir];
        for (unsigned n = 0; n < ntraces; ++n)
        {
            auto traceId = LibUtilities::ShapeTypeTraceIDInDir[shape][dir][n];
            auto stdExp  = locExpPtr->GetStdTraceExp(traceId);

            for (unsigned k = 0; k + 1 < nDim; ++k)
            {
                ztr[traceId].push_back(stdExp->GetBasis(k)->GetZ());
            }
        }
    }

    // Fill the collapsed coordinates of trace point pt on trace traceId,
    // whose normal direction is normDir and which sits at eta = +/- 1 there.
    auto SetEta = [&](const LocalRegions::ExpansionSharedPtr &Exp,
                      const unsigned traceId, const unsigned normDir,
                      const bool atPlusOne, const unsigned pt, double *eta) {
        eta[0] = eta[1] = eta[2] = -1.0;
        eta[normDir]             = atPlusOne ? 1.0 : -1.0;

        if (nDim == 2)
        {
            eta[Exp->GetGeom()->GetDir(traceId, 0)] = ztr[traceId][0][pt];
        }
        else if (nDim == 3)
        {
            const auto n0                           = ztr[traceId][0].size();
            eta[Exp->GetGeom()->GetDir(traceId, 0)] = ztr[traceId][0][pt % n0];
            eta[Exp->GetGeom()->GetDir(traceId, 1)] = ztr[traceId][1][pt / n0];
        }
    };

    // Loop over chunks.
    for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
    {
        for (unsigned i = 0; i < interleave_width; ++i, ++el)
        {
            // The deriv factors are element dependent, so they have to be
            // regenerated for every element in the chunk rather than once
            // per chunk from the chunk's first element.
            if (el < num_elements)
            {
                auto Exp = collVec[el];

                // get hold of d0factors dxi_0/dx nx + dxi_0/dy ny ,
                // d1factors = dxi_1/dx nx + dxi_1/dy ny etc on traces in
                // legacy format - or, for facDir >= 0, the uncontracted
                // dxi_e/dx_facDir of that one Cartesian direction, feeding
                // the vector-input lift. The chain-rule combination and the
                // trace Jacobian below are direction-blind, so everything
                // downstream is shared.
                if (facDir < 0)
                {
                    Exp->NormalTraceDerivFactors(dfactors[0], dfactors[1],
                                                 dfactors[2]);
                }
                else
                {
                    Exp->TraceDerivFactors(facDir, dfactors[0], dfactors[1],
                                           dfactors[2]);
                }
            }

            unsigned offset = 0;
            for (unsigned dir = 0; dir < nDim; ++dir)
            {
                auto ntraces = LibUtilities::ShapeTypeNumTraceInDir[shape][dir];

                // Loop over traces
                for (unsigned n = 0; n < ntraces; ++n)
                {
                    auto traceId =
                        LibUtilities::ShapeTypeTraceIDInDir[shape][dir][n];

                    // Index for edge id
                    if (el < num_elements)
                    {
                        auto Exp = collVec[el];

                        // Trace geometry comes from the aligned expansion, so
                        // it is laid out in the global trace frame; the
                        // standard trace expansion supplies the element local
                        // extents.
                        auto alignedExp = Exp->GetAlignedTraceExp(traceId);
                        auto stdExp     = Exp->GetStdTraceExp(traceId);

                        // Get trace Jacobian
                        auto jacArray = alignedExp->GetGeomFactors()->GetJac();

                        if (isDeformed)
                        {
                            auto TraceDeformed =
                                (alignedExp->GetGeomFactors()->GetGtype() ==
                                 SpatialDomains::eDeformed);

                            if (TraceDeformed)
                            {
                                // The deriv factors are already in the element
                                // local frame, so scatter the Jacobian into
                                // the same frame before combining the two
                                // (global -> local, hence Forwards = false).
                                //
                                // A 2D element needs this just as much as a
                                // 3D one: its edges 2 and 3 run against the
                                // local coordinate, so a backwards oriented
                                // edge has to be reversed.
                                Array<OneD, double> jacLoc;
                                const auto orient =
                                    Exp->GetTraceOrient(traceId);
                                const auto identity =
                                    (nDim == 3)
                                        ? StdRegions::eDir1FwdDir1_Dir2FwdDir2
                                        : StdRegions::eForwards;

                                if (orient != identity)
                                {
                                    jacLoc =
                                        Array<OneD, double>(nTracePts[dir]);
                                    Exp->ReOrientTracePhysVals(
                                        orient, jacArray, jacLoc,
                                        stdExp->GetNumPoints(0),
                                        (nDim == 3) ? stdExp->GetNumPoints(1)
                                                    : 1,
                                        false);
                                }
                                else
                                {
                                    jacLoc = jacArray;
                                }

                                for (unsigned pt = 0; pt < nTracePts[dir]; ++pt)
                                {
                                    double eta[3];
                                    SetEta(Exp, traceId, dir, n == 1, pt, eta);

                                    for (unsigned d = 0; d < nDim; ++d)
                                    {
                                        double fac = CollCoordDfactor(
                                            dfactors, shape, traceId, eta, d,
                                            pt);

                                        jacngfptr[(offset + pt) *
                                                      interleave_width +
                                                  i + d * compStride] =
                                            jacLoc[pt] * fac;
                                    }
                                }
                            }
                            else
                            {
                                for (unsigned pt = 0; pt < nTracePts[dir]; ++pt)
                                {
                                    double eta[3];
                                    SetEta(Exp, traceId, dir, n == 1, pt, eta);

                                    for (unsigned d = 0; d < nDim; ++d)
                                    {
                                        double fac = CollCoordDfactor(
                                            dfactors, shape, traceId, eta, d,
                                            pt);

                                        jacngfptr[(offset + pt) *
                                                      interleave_width +
                                                  i + d * compStride] =
                                            jacArray[0] * fac;
                                    }
                                }
                            }
                            offset += nTracePts[dir];
                        }
                        else
                        {
                            // A regular trace carries one factor, so the
                            // collapsed correction is evaluated at the first
                            // trace point.
                            double eta[3];
                            SetEta(Exp, traceId, dir, n == 1, 0, eta);

                            for (unsigned d = 0; d < nDim; ++d)
                            {
                                double fac = CollCoordDfactor(
                                    dfactors, shape, traceId, eta, d, 0);
                                jacngfptr[offset * interleave_width + i +
                                          d * compStride] = jacArray[0] * fac;
                            }
                            offset++;
                        }
                    }
                    else
                    {
                        if (isDeformed)
                        {
                            for (unsigned pt = 0; pt < nTracePts[dir]; ++pt)
                            {
                                for (unsigned d = 0; d < nDim; ++d)
                                {
                                    jacngfptr[(offset + pt) * interleave_width +
                                              i + d * compStride] = 0.0;
                                }
                            }
                            offset += nTracePts[dir];
                        }
                        else
                        {
                            for (unsigned d = 0; d < nDim; ++d)
                            {
                                jacngfptr[offset * interleave_width + i +
                                          d * compStride] = 0.0;
                            }
                            offset++;
                        }
                    }
                }
            }
            ASSERTL1(offset == nTraceJacPoints,
                     "Tot Jacobian points not correct");
        }
        jacngfptr += interleave_width * nTraceJacPoints;
    }
    return jacngf;
}

} // namespace Nektar::LocalRegions
