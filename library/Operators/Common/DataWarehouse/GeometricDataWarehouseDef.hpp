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

#include <Operators/Common/DataWarehouse/GeometricDataWarehouse.hpp>
#include <Operators/Field/Block.hpp>
#include <Operators/Field/Field.hpp>

#include <MultiRegions/ExpList.h>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

template <typename MemSpace, typename TData>
MemoryRegion<TData> GeometricDataCreator::Create(
    const WeightsKey<TData> &weightsKey)
{
    // Fetch data from key.
    const auto block_idx        = weightsKey.m_block_idx;
    const auto interleave_width = weightsKey.m_interleave_width;

    // Fetch expansion.
    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    // Allocate memory and get pointer.
    const auto memsize = interleave_width * expPtr->GetTotPoints();
    auto weights       = MemoryRegion<TData>(memsize);

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
MemoryRegion<TData> GeometricDataCreator::Create(
    const JacobianKey<TData> &jacobianKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = jacobianKey.m_block_idx;
    const auto interleave_width = jacobianKey.m_interleave_width;

    // Fetch expansion.
    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto exp_idx      = expPtr->GetElmtId();
    const bool isDeformed =
        (expPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Allocate memory and get pointer.
    const auto memsize = (isDeformed) ? num_elmt_groups * interleave_width *
                                            expPtr->GetTotPoints()
                                      : num_elmt_groups * interleave_width;
    auto jac           = MemoryRegion<TData>(memsize);
    auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Deformed geometry.
    if (isDeformed)
    {
        Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            // Loop over interleave width
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
                // Check for padding
                if (el < num_elements)
                {
                    jacArray[i] = this->m_expansionList->GetExp(exp_idx + el)
                                      ->GetGeomFactors()
                                      ->GetJac();
                }
                else
                {
                    jacArray[i] =
                        Array<OneD, NekDouble>(expPtr->GetTotPoints(), 0.0);
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
        auto jac    = MemoryRegion<TData>(num_elmt_groups * interleave_width);
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
                    auto &auxJac = this->m_expansionList->GetExp(exp_idx + el)
                                       ->GetGeomFactors()
                                       ->GetJac();
                    jacptr[el] = auxJac[0];
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
MemoryRegion<TData> GeometricDataCreator::Create(
    const DerivFactorKey<TData> &derivFactorKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = derivFactorKey.m_block_idx;
    const auto interleave_width = derivFactorKey.m_interleave_width;
    const auto transpose        = derivFactorKey.m_transpose;

    // Fetch expansion.
    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto exp_idx      = expPtr->GetElmtId();
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
    auto df    = MemoryRegion<TData>(memsize);
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

                            auto &tmp =
                                this->m_expansionList->GetExp(exp_idx + el + i)
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
                            this->m_expansionList->GetExp(exp_idx + el + i)
                                ->GetGeomFactors()
                                ->GetDerivFactors();
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
MemoryRegion<TData> GeometricDataCreator::Create(
    const CoordKey<TData> &coordKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = coordKey.m_block_idx;
    const auto interleave_width = coordKey.m_interleave_width;
    const auto transpose        = coordKey.m_transpose;

    // Fetch expansion.
    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto exp_idx      = expPtr->GetElmtId();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    const auto nDim   = expPtr->GetCoordim();
    const auto range1 = transpose ? nDim : expPtr->GetTotPoints();
    const auto range2 = transpose ? expPtr->GetTotPoints() : nDim;

    // Allocate memory and get pointer.
    const auto memsize =
        num_elmt_groups * interleave_width * expPtr->GetTotPoints() * nDim;
    auto crds   = MemoryRegion<TData>(memsize);
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

                        auto tmp =
                            this->m_expansionList->GetExp(exp_idx + el + i)
                                ->GetCoords();
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
MemoryRegion<unsigned> GeometricDataCreator::Create(
    const OrientKey<TData> &orientKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = orientKey.m_block_idx;
    const auto interleave_width = orientKey.m_interleave_width;

    // Fetch expansion.
    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto exp_idx      = expPtr->GetElmtId();
    const auto nedge        = expPtr->GetGeom()->GetNumEdges();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Allocate memory and get pointer.
    const auto memsize = num_elmt_groups * interleave_width * nedge;
    auto orients       = MemoryRegion<unsigned int>(memsize);
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
                    auto orient = m_expansionList->GetExp(exp_idx + el + i)
                                      ->GetTraceOrient(ed);
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
MemoryRegion<unsigned> GeometricDataCreator::Create(
    const TraceToElmtMapKey<TData> &traceToElmtMapKey)
{
    // Fetch data from key.
    const auto block_idx = traceToElmtMapKey.m_block_idx;

    // Get the reference element.
    // Note the traceToElemtMap is equivalent for all elements within a block.
    auto coll        = GetCollection(m_expansionList, block_idx);
    auto expPtr      = coll.GetExpVector()[0];
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
    auto map           = MemoryRegion<unsigned>(memsize);
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
MemoryRegion<int> GeometricDataCreator::Create(
    const TraceToElmtSignKey<TData> &traceToElmtSignKey)
{
    // Fetch data from key.
    const auto block_idx = traceToElmtSignKey.m_block_idx;

    // Get the reference element.
    // Note the sign is equivalent for all elements within a block.
    auto coll        = GetCollection(m_expansionList, block_idx);
    auto expPtr      = coll.GetExpVector()[0];
    const auto nedge = expPtr->GetGeom()->GetNumEdges();

    // Total nummodes for trace
    auto nmTrace = 0;
    for (int ed = 0; ed < nedge; ed++)
    {
        nmTrace += expPtr->GetTraceNcoeffs(ed);
    }

    // Allocate memory and get pointer.
    const auto memsize = nmTrace;
    auto sign          = MemoryRegion<int>(memsize);
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
MemoryRegion<unsigned> GeometricDataCreator::Create(
    const InteriorMapKey<TData> &interiorMapKey)
{
    // Fetch data from key.
    const auto block_idx = interiorMapKey.m_block_idx;

    // Get the reference element.
    // Note the interior map is equivalent for all elements within a block.
    auto coll   = GetCollection(m_expansionList, block_idx);
    auto expPtr = coll.GetExpVector()[0];

    // Get size of interior map
    auto numInteriorCoeffs = expPtr->GetNcoeffs() - expPtr->NumBndryCoeffs();

    // Allocate memory and get pointer.
    const auto memsize = numInteriorCoeffs;
    auto map           = MemoryRegion<unsigned>(memsize);
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
MemoryRegion<TData> GeometricDataCreator::Create(
    const JacobianTraceKey<TData> &jacobianTraceKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    // Fetch data from key.
    const auto block_idx        = jacobianTraceKey.m_block_idx;
    const auto interleave_width = jacobianTraceKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto num_elements = coll.GetExpVector().size();
    const auto exp_idx      = expPtr->GetElmtId();
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
    auto jac    = MemoryRegion<TData>(memsize);
    auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Deformed geometry.
    if (isDeformed)
    {
        Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);

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
                        auto traceExp =
                            m_expansionList->GetExp(exp_idx + el + i)
                                ->GetTraceExp(ed);

                        // Get trace Jacobian.
                        jacArray[i] = traceExp->GetGeomFactors()->GetJac();
                    }
                    else
                    {
                        jacArray[i] = Array<OneD, NekDouble>(
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
                        auto traceExp =
                            m_expansionList->GetExp(exp_idx + el + i)
                                ->GetTraceExp(ed);

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
MemoryRegion<TData> GeometricDataCreator::Create(
    const JacobianLocTraceKey<TData> &jacobianLocTraceKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = jacobianLocTraceKey.m_block_idx;
    const auto interleave_width = jacobianLocTraceKey.m_interleave_width;

    auto coll          = GetCollection(m_expansionList, block_idx);
    auto expPtr        = coll.GetExpVector()[0];
    const auto exp_idx = expPtr->GetElmtId();
    const auto nDim    = expPtr->GetShapeDimension();

    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Get number of traces on element
    const auto locExpPtr = m_expansionList->GetExp(exp_idx);

    // get total trace size
    LibUtilities::ShapeType shape = expPtr->DetShapeType();

    // if element deformed treat trace group as deformed.
    bool isDeformed =
        (locExpPtr->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed);

    // caculate the total number of quadrature points
    std::vector<unsigned> nTracePts;
    unsigned nTraceJacPoints = 0;
    for (unsigned dir = 0; dir < nDim; ++dir)
    {
        auto ntraces = LibUtilities::ShapeTypeNumTraceInDir[shape][dir];

        // first trace id
        auto traceId = LibUtilities::ShapeTypeTraceIDInDir[shape][dir][0];

        auto npts = locExpPtr->GetLocTraceExp(traceId)->GetTotPoints();
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
    auto jac           = MemoryRegion<TData>(memsize);
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
                        auto traceExp = m_expansionList->GetExp(exp_idx + el)
                                            ->GetLocTraceExp(traceId);

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

} // namespace Nektar::Operators
