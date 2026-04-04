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
    const JacobianKey<TData> &jacobianKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = jacobianKey.m_block_idx;
    const auto interleave_width = jacobianKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto exp_idx      = expPtr->GetElmtId();
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Deformed geometry.
    if (this->m_expansionList->GetExp(exp_idx)->GetGeomFactors()->GetGtype() ==
        SpatialDomains::eDeformed)
    {
        const auto memsize =
            num_elmt_groups * interleave_width * expPtr->GetTotPoints();
        Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);
        auto jac    = MemoryRegion<TData>(memsize);
        auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
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

            for (unsigned int pt = 0; pt < expPtr->GetTotPoints(); ++pt)
            {
                for (unsigned int i = 0; i < interleave_width; ++i)
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
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
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

    const auto block_idx        = derivFactorKey.m_block_idx;
    const auto interleave_width = derivFactorKey.m_interleave_width;
    const auto transpose        = derivFactorKey.m_transpose;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto exp_idx      = expPtr->GetElmtId();
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;
    const auto nDim   = expPtr->GetShapeDimension();
    const auto nCoord = expPtr->GetCoordim();

    const auto range1 = transpose ? nDim * nCoord : expPtr->GetTotPoints();
    const auto range2 = transpose ? expPtr->GetTotPoints() : nDim * nCoord;

    // Deformed geometry.
    if (this->m_expansionList->GetExp(exp_idx)->GetGeomFactors()->GetGtype() ==
        SpatialDomains::eDeformed)
    {
        // Allocate memory and get pointer.
        const auto memsize = num_elmt_groups * interleave_width *
                             expPtr->GetTotPoints() * nDim * nCoord;
        auto df    = MemoryRegion<TData>(memsize);
        auto dfptr = df.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            for (unsigned int index1 = 0; index1 < range1; ++index1)
            {
                for (unsigned int index2 = 0; index2 < range2; ++index2)
                {
                    for (unsigned int i = 0; i < interleave_width; ++i)
                    {
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
        auto df    = MemoryRegion<TData>(num_elmt_groups * interleave_width *
                                      nDim * nCoord);
        auto dfptr = df.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            for (unsigned int d = 0; d < nDim * nCoord; ++d)
            {
                for (unsigned int i = 0; i < interleave_width; ++i)
                {
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

    const auto block_idx        = coordKey.m_block_idx;
    const auto interleave_width = coordKey.m_interleave_width;
    const auto transpose        = coordKey.m_transpose;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto exp_idx      = expPtr->GetElmtId();
    const auto num_elements = coll.GetExpVector().size();
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
        for (unsigned int index1 = 0; index1 < range1; ++index1)
        {
            // Loop over points or components
            for (unsigned int index2 = 0; index2 < range2; ++index2)
            {
                // Loop over interleave width
                for (unsigned int i = 0; i < interleave_width; ++i, ++crd_id)
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
MemoryRegion<unsigned int> GeometricDataCreator::Create(
    const OrientKey<TData> &orientKey)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    const auto block_idx        = orientKey.m_block_idx;
    const auto interleave_width = orientKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto exp_idx      = expPtr->GetElmtId();
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Get number of edges on trace
    const auto locExpPtr = m_expansionList->GetExp(exp_idx);
    const auto range1    = locExpPtr->GetGeom()->GetNumEdges();

    // Allocate memory and get pointer.
    const auto memsize = num_elmt_groups * interleave_width * range1;
    auto orients       = MemoryRegion<unsigned int>(memsize);
    auto orientptr =
        orients.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over chunks.
    for (size_t chunk = 0, el = 0, orient_id = 0; chunk < num_elmt_groups;
         ++chunk)
    {
        // Loop over component or points
        for (unsigned int index1 = 0; index1 < range1; ++index1)
        {
            // Loop over interleave width
            for (unsigned int i = 0; i < interleave_width; ++i, ++orient_id)
            {
                // Check for padding
                if (el + i < num_elements)
                {
                    const auto ed = index1;

                    auto tmp = m_expansionList->GetExp(exp_idx + el + i)
                                   ->GetTraceOrient(ed);
                    orientptr[orient_id] = static_cast<unsigned int>(tmp);
                }
                else
                {
                    orientptr[orient_id] =
                        static_cast<unsigned int>(StdRegions::eForwards);
                }
            }
        }

        el += interleave_width;
    }

    return orients;
}

template <typename MemSpace, typename TData>
MemoryRegion<unsigned int> GeometricDataCreator::Create(
    const TraceToElmtMapKey<TData> &traceToElmtMapKey)
{
    // Get the reference element.
    // Note the traceToElemtMap is equivalent for all elements within a block.
    const auto block_idx = traceToElmtMapKey.m_block_idx;
    auto coll            = GetCollection(m_expansionList, block_idx);
    auto expPtr          = coll.GetExpVector()[0];
    const auto exp_idx   = expPtr->GetElmtId();

    // Get number of edges on trace
    const auto locExpPtr = m_expansionList->GetExp(exp_idx);
    const auto range1    = locExpPtr->GetGeom()->GetNumEdges();

    // Total nummodes for trace
    auto nmTrace = 0;
    std::vector<unsigned int> nmEdge;
    for (int i = 0; i < range1; i++)
    {
        nmTrace += expPtr->GetTraceNcoeffs(i);
        nmEdge.push_back(expPtr->GetTraceNcoeffs(i));
    }

    // Allocate memory and get pointer.
    const auto memsize = nmTrace;
    auto map           = MemoryRegion<unsigned int>(memsize);
    auto mapptr = map.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over component or points
    for (unsigned int index1 = 0; index1 < range1; ++index1)
    {
        // Index for edge id
        const auto ed = index1;

        // nummodes for this edge
        auto exp = m_expansionList->GetExp(exp_idx);

        // Allocate temporary arrays for map and sign
        Array<OneD, unsigned int> tmpMap(nmEdge[ed], (unsigned)0);
        Array<OneD, int> tmpSign(nmEdge[ed], 1);

        // Extract map and sign
        exp->GetTraceToElementMap(ed, tmpMap, tmpSign);

        // Assign map to memory region
        for (unsigned int nm = 0; nm < nmEdge[ed]; ++nm)
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
    // Get the reference element.
    // Note the sign is equivalent for all elements within a block.
    const auto block_idx = traceToElmtSignKey.m_block_idx;
    auto coll            = GetCollection(m_expansionList, block_idx);
    auto expPtr          = coll.GetExpVector()[0];
    const auto exp_idx   = expPtr->GetElmtId();

    // Get number of edges on trace
    const auto locExpPtr = m_expansionList->GetExp(exp_idx);
    const auto range1    = locExpPtr->GetGeom()->GetNumEdges();

    // Total nummodes for trace
    auto nmTrace = 0;
    std::vector<unsigned int> nmEdge;
    for (int i = 0; i < range1; i++)
    {
        nmTrace += expPtr->GetTraceNcoeffs(i);
        nmEdge.push_back(expPtr->GetTraceNcoeffs(i));
    }

    // Allocate memory and get pointer.
    const auto memsize = nmTrace;
    auto sign          = MemoryRegion<int>(memsize);
    auto signptr = sign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Loop over number of edges
    for (unsigned int index1 = 0; index1 < range1; ++index1)
    {
        // Index for edge id
        const auto ed = index1;

        // nummodes for this edge
        auto exp = m_expansionList->GetExp(exp_idx);

        // Allocate temporary arrays for map and sign
        Array<OneD, unsigned int> tmpMap(nmEdge[ed], (unsigned)0);
        Array<OneD, int> tmpSign(nmEdge[ed], 1);

        // Extract map and sign
        exp->GetTraceToElementMap(ed, tmpMap, tmpSign);

        // Assign map to memory region
        for (unsigned int nm = 0; nm < nmEdge[ed]; ++nm)
        {
            *(signptr++) = tmpSign[nm];
        }
    }

    return sign;
}

template <typename MemSpace, typename TData>
MemoryRegion<unsigned int> GeometricDataCreator::Create(
    const InteriorMapKey<TData> &interiorMapKey)
{
    // Get the reference element.
    // Note the interior map is equivalent for all elements within a block.
    const auto block_idx = interiorMapKey.m_block_idx;
    auto coll            = GetCollection(m_expansionList, block_idx);
    auto expPtr          = coll.GetExpVector()[0];
    const auto exp_idx   = expPtr->GetElmtId();

    // Get size of interior map
    auto numInteriorCoeffs = expPtr->GetNcoeffs() - expPtr->NumBndryCoeffs();

    // Allocate memory and get pointer.
    const auto memsize = numInteriorCoeffs;
    auto map           = MemoryRegion<unsigned int>(memsize);
    auto mapptr = map.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // nummodes for this edge
    auto exp = m_expansionList->GetExp(exp_idx);

    // Allocate temporary arrays for map and sign
    Array<OneD, unsigned int> tmpMap(numInteriorCoeffs);

    // Extract map
    exp->GetInteriorMap(tmpMap);

    // Assign map to memory region
    for (unsigned int nm = 0; nm < numInteriorCoeffs; ++nm)
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

    const auto block_idx        = jacobianTraceKey.m_block_idx;
    const auto interleave_width = jacobianTraceKey.m_interleave_width;

    auto coll               = GetCollection(m_expansionList, block_idx);
    auto expPtr             = coll.GetExpVector()[0];
    const auto exp_idx      = expPtr->GetElmtId();
    const auto num_elements = coll.GetExpVector().size();
    const auto num_elmt_groups =
        ((num_elements + vector_width - 1) / vector_width) * vector_width /
        interleave_width;

    // Get number of edges on trace
    const auto locExpPtr = m_expansionList->GetExp(exp_idx);
    const auto range1    = locExpPtr->GetGeom()->GetNumEdges();

    // Check if any edge is deformed
    auto nTraceJacPoints   = 0;
    bool isAnyEdgeDeformed = false;
    std::vector<bool> isDeformedEdge;
    std::vector<unsigned int> bndPoints;
    for (int ed = 0; ed < range1; ed++)
    {
        isDeformedEdge.push_back(
            locExpPtr->GetTraceExp(ed)->GetGeomFactors()->GetGtype() ==
            SpatialDomains::eDeformed);
        isAnyEdgeDeformed    = isAnyEdgeDeformed || isDeformedEdge[ed];
        auto tracePhysPoints = locExpPtr->GetTraceExp(ed)->GetTotPoints();
        if (isDeformedEdge[ed])
        {
            nTraceJacPoints += tracePhysPoints;
            bndPoints.push_back(tracePhysPoints);
        }
        else
        {
            nTraceJacPoints += 1;
            bndPoints.push_back(1);
        }
    }

    const auto memsize = num_elmt_groups * interleave_width * nTraceJacPoints;
    Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);
    auto jac    = MemoryRegion<TData>(memsize);
    auto jacptr = jac.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Deformed geometry.
    if (isAnyEdgeDeformed)
    {
        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
                // Loop over component or points
                for (unsigned int index1 = 0; index1 < range1; ++index1)
                {
                    // Index for edge id
                    const auto ed = index1;
                    if (el < num_elements)
                    {
                        // Get trace expansion
                        auto traceExp = m_expansionList->GetExp(exp_idx + el)
                                            ->GetTraceExp(ed);

                        // Get trace Jacobian
                        jacArray[i] = traceExp->GetGeomFactors()->GetJac();
                    }
                    else
                    {
                        jacArray[i] = Array<OneD, NekDouble>(bndPoints[i], 0.0);
                    }

                    for (unsigned int pt = 0; pt < bndPoints[i]; ++pt)
                    {
                        *(jacptr++) = jacArray[i][pt];
                    }
                }
            }
        }

        return jac;
    }
    // Regular geometry.
    else
    {
        // Loop over chunks.
        for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
        {
            for (unsigned int i = 0; i < interleave_width; ++i, ++el)
            {
                // Loop over component or points
                for (unsigned int index1 = 0; index1 < range1; ++index1)
                {
                    // Index for edge id
                    const auto ed = index1;

                    if (el < num_elements)
                    {
                        // Get trace expansion
                        auto traceExp = m_expansionList->GetExp(exp_idx + el)
                                            ->GetTraceExp(ed);

                        // Get trace Jacobian
                        auto &auxJac = traceExp->GetGeomFactors()->GetJac();
                        jacptr[el * range1 + ed] = auxJac[0];
                    }
                    else
                    {
                        jacptr[el * range1 + ed] = 0.0;
                    }
                }
            }
        }

        return jac;
    }
}

} // namespace Nektar::Operators
