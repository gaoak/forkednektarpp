///////////////////////////////////////////////////////////////////////////////
//
// File: OperatorHelper.hpp
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

#include "Field/MemoryRegion.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <MultiRegions/ExpList.h>

namespace Nektar::Operators
{

enum BasisDataType
{
    BASIS_UNKNOWN_DATA          = 0,
    BASIS_BASIS_DATA            = 1,
    BASIS_BASIS_DERIVATIVE_DATA = 2,
    BASIS_WEIGHT_DATA           = 3,
    BASIS_POINT_DATA            = 4,
    BASIS_DERIVATIVE_DATA       = 5,
};

template <typename TData>
using BasisDataMap = std::map<LibUtilities::BasisKey, MemoryRegion<TData>>;

/**
 * @brief Helper function to copy Basis data from an Array<OneD, TDataIn> to
 * a typed MemoryRegion.
 *
 * @param basis - Basis data.
 * @param basisDataType - Basis data to copy.
 *
 * @return MemoryRegion<TDataOut>
 */
template <typename MemSpace, typename TDataIn, typename TDataOut = TDataIn>
MemoryRegion<TDataOut> GetBasisData(
    const LibUtilities::BasisSharedPtr &basis, BasisDataType basisDataType,
    size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
{
    switch (basisDataType)
    {
        case BASIS_BASIS_DATA:
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetBdata(), alignment);

        case BASIS_BASIS_DERIVATIVE_DATA:
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetDbdata(), alignment);

        case BASIS_WEIGHT_DATA:
        {
            auto ndata = basis->GetW().size();
            Array<OneD, TDataIn> wTmp(ndata);

            if (basis->GetPointsType() == LibUtilities::eGaussRadauMAlpha1Beta0)
            {
                Vmath::Smul(ndata, 0.5, basis->GetW().get(), 1, wTmp.get(), 1);
            }
            else if (basis->GetPointsType() ==
                     LibUtilities::eGaussRadauMAlpha2Beta0)
            {
                Vmath::Smul(ndata, 0.25, basis->GetW().get(), 1, wTmp.get(), 1);
            }
            else
            {
                Vmath::Vcopy(ndata, basis->GetW().get(), 1, wTmp.get(), 1);
            }

            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                wTmp, alignment);
        }

        case BASIS_POINT_DATA:
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetZ(), alignment);

        case BASIS_DERIVATIVE_DATA:
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetD()->GetPtr(), alignment);

        default:
            NEKERROR(ErrorUtil::efatal, "invalid basis data requested.");
            return MemoryRegion<TDataOut>::template create<MemSpace>(0,
                                                                     alignment);
            break;
    }
}

/**
 * @brief Helper function to copy Basis data from the expansionList to
 * a MemoryRegionHost.
 *
 * The MemoryRegionHost is placed into a map that uses the BasisKey as
 * the key.
 *
 * @param expansionList - The expanision list which contains the basis data.
 * @param basisDataType - Basis data to copy.
 *
 * @return BasisDataMap<TDataOut>
 */
template <typename MemSpace, typename TDataIn, typename TDataOut = TDataIn>
BasisDataMap<TDataOut> GetBasisData(
    const MultiRegions::ExpListSharedPtr &expansionList,
    BasisDataType basisDataType,
    size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
{
    // Initialize the data map.
    BasisDataMap<TDataOut> basisDataMap;

    // Loop over the elements of expansionList.
    size_t nDim = expansionList->GetShapeDimension();

    for (size_t i = 0; i < expansionList->GetNumElmts(); ++i)
    {
        const auto expPtr = expansionList->GetExp(i);

        // Fetch basiskeys of the current element.
        for (size_t d = 0; d < nDim; d++)
        {
            LibUtilities::BasisKey basisKey =
                expPtr->GetBasis(d)->GetBasisKey();

            // If necessary copy the basis data in the map.
            if (basisDataMap.find(basisKey) == basisDataMap.end())
            {
                basisDataMap[basisKey] =
                    GetBasisData<MemSpace, TDataIn, TDataOut>(
                        expPtr->GetBasis(d), basisDataType, alignment);
            }
        }
    }

    return basisDataMap;
}

} // namespace Nektar::Operators
