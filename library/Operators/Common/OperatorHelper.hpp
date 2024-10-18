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

#include "Operators/Field/Field.hpp"

namespace Nektar::Operators
{

enum BasisDataType
{
    eUnknown         = 0,
    eBasis           = 1, // was BASIS_BASIS_DATA
    eBasisDerivative = 2, // Was BASIS_BASIS_DERIVATIVE_DATA
    eWeights         = 3, // Was BASIS_WEIGHT_DATA
    eZeros           = 4, // Was BASIS_POINT_DATA
    eDerivative      = 5, // Was BASIS_DERIVATIVE_DATA
    eHalfMultOnePlusZero,
    eTwoOverOneMinusZero

};

template <typename TData>
using BasisDataMap = std::map<LibUtilities::BasisKey, MemoryRegion<TData>>;

size_t GetGeometricFactorSize(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::vector<BlockAttributes> &blocks);

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
        case eBasis:
        {
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetBdata(), alignment);
            break;
        }
        case eBasisDerivative:
        {
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetDbdata(), alignment);
            break;
        }
        case eWeights:
        {
            auto ndata = basis->GetW().size();
            Array<OneD, TDataIn> wTmp(ndata);

            switch (basis->GetBasisType())
            {
                case LibUtilities::eModified_B:
                case LibUtilities::eOrtho_B:
                {

                    if (basis->GetPointsType() ==
                        LibUtilities::eGaussRadauMAlpha1Beta0)
                    {
                        Vmath::Smul(ndata, 0.5, basis->GetW().get(), 1,
                                    wTmp.get(), 1);
                    }
                    else
                    {
                        const Array<OneD, NekDouble> z = basis->GetZ();
                        const Array<OneD, NekDouble> w = basis->GetW();
                        for (int i = 0; i < ndata; ++i)
                        {
                            wTmp[i] = 0.5 * (1 - z[i]) * w[i];
                        }
                    }
                }
                break;
                case LibUtilities::eModified_C:
                case LibUtilities::eModifiedPyr_C:
                case LibUtilities::eOrtho_C:
                case LibUtilities::eOrthoPyr_C:
                {
                    if (basis->GetPointsType() ==
                        LibUtilities::eGaussRadauMAlpha2Beta0)
                    {
                        Vmath::Smul(ndata, 0.25, basis->GetW().get(), 1,
                                    wTmp.get(), 1);
                    }
                    else if (basis->GetPointsType() ==
                             LibUtilities::eGaussRadauMAlpha1Beta0)
                    {
                        const Array<OneD, NekDouble> z = basis->GetZ();
                        const Array<OneD, NekDouble> w = basis->GetW();
                        for (int i = 0; i < ndata; ++i)
                        {
                            wTmp[i] = 0.25 * (1 - z[i]) * w[i];
                        }
                    }
                    else
                    {
                        const Array<OneD, NekDouble> z = basis->GetZ();
                        const Array<OneD, NekDouble> w = basis->GetW();
                        for (int i = 0; i < ndata; ++i)
                        {
                            wTmp[i] = 0.25 * (1 - z[i]) * (1 - z[i]) * w[i];
                        }
                    }
                    break;
                    default:
                    {
                        Vmath::Vcopy(ndata, basis->GetW().get(), 1, wTmp.get(),
                                     1);
                    }
                }
            }

            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                wTmp, alignment);
        }
        break;
        case eZeros:
        {
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetZ(), alignment);
        }
        break;
        case eDerivative:
        {
            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                basis->GetD()->GetPtr(), alignment);
        }
        break;
        case eHalfMultOnePlusZero:
        {
            const auto z = basis->GetZ();
            Array<OneD, TDataIn> Tmp(z.size());

            for (int i = 0; i < z.size(); ++i)
            {
                Tmp[i] = 0.5 * (1.0 + z[i]);
            }

            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                Tmp, alignment);
        }
        break;
        case eTwoOverOneMinusZero:
        {
            const auto z = basis->GetZ();
            auto n       = z.size();
            Array<OneD, TDataIn> Tmp(n);

            for (int i = 0; i < n; ++i)
            {
                Tmp[i] = 2 / (1.0 - z[i]);
            }

            return MemoryRegion<TDataOut>::template fromArray<MemSpace,
                                                              TDataIn>(
                Tmp, alignment);
        }
        break;
        default:
            NEKERROR(ErrorUtil::efatal, "invalid basis data requested.");
            return MemoryRegion<TDataOut>::template create<MemSpace>(0,
                                                                     alignment);
            break;
    }
}

/**
 * @brief Helper function to copy Basis data from the expansionList
 * to a MemoryRegionHost.
 *
 * The MemoryRegionHost is placed into a map that uses the BasisKey
 * as the key.
 *
 * @param expansionList - The expanision list which contains the
 * basis data.
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

template <typename TData>
std::shared_ptr<std::vector<TData>> SetJacobian(
    const MultiRegions::ExpListSharedPtr &expansionList, size_t jacSize,
    std::vector<BlockAttributes> &blocks)
{
    // Allocate memory for the jacobian
    std::vector<TData> jac;
    jac.resize(jacSize);

    size_t exp_id = 0;
    size_t jac_id = 0;

    for (size_t blk = 0; blk < blocks.size(); ++blk)
    {
        size_t interleave_width = blocks[blk].interleave_width;
        size_t num_elements     = blocks[blk].num_elements;
        size_t num_elmt_groups  = blocks[blk].GetNumElmtGroups();

        auto expPtr = expansionList->GetExp(exp_id);

        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            Array<OneD, Array<OneD, NekDouble>> jacArray(interleave_width);

            for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
            {
                for (size_t i = 0; i < interleave_width; ++i, ++el)
                {
                    if (el < num_elements)
                    {
                        jacArray[i] = expansionList->GetExp(exp_id++)
                                          ->GetMetricInfo()
                                          ->GetJac(expPtr->GetPointsKeys());
                    }
                    else
                    {
                        jacArray[i] =
                            Array<OneD, NekDouble>(expPtr->GetTotPoints(), 0.0);
                    }
                }

                for (size_t pt = 0; pt < expPtr->GetTotPoints(); ++pt)
                {
                    for (size_t i = 0; i < interleave_width; ++i, ++jac_id)
                    {
                        jac[jac_id] = jacArray[i][pt];
                    }
                }
            }
        }
        else // regular geometry
        {
            for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
            {
                for (size_t i = 0; i < interleave_width; ++i, ++el, ++jac_id)
                {
                    if (el < num_elements)
                    {
                        auto &auxJac = expansionList->GetExp(exp_id++)
                                           ->GetMetricInfo()
                                           ->GetJac(expPtr->GetPointsKeys());
                        jac[jac_id] = auxJac[0];
                    }
                    else
                    {
                        jac[jac_id] = 0.0;
                    }
                }
            }
        }
    }

    return MemoryManager<std::vector<TData>>::AllocateSharedPtr(jac);
}

template <typename TData>
std::shared_ptr<std::vector<TData>> SetDerivativeFactor(
    const MultiRegions::ExpListSharedPtr &expansionList, size_t dfSize,
    std::vector<BlockAttributes> &blocks, bool transpose = false)
{
    // Allocate memory for the derivative factor
    size_t nDim   = expansionList->GetShapeDimension();
    size_t nCoord = expansionList->GetCoordim(0);

    std::vector<TData> derivFac;
    derivFac.resize(nDim * nCoord * dfSize);

    size_t exp_id = 0;
    size_t df_id  = 0;

    for (size_t blk = 0; blk < blocks.size(); ++blk)
    {
        size_t interleave_width = blocks[blk].interleave_width;
        size_t num_elements     = blocks[blk].num_elements;
        size_t num_elmt_groups  = blocks[blk].GetNumElmtGroups();
        auto expPtr             = expansionList->GetExp(exp_id);

        size_t range1 = transpose ? nDim * nCoord : expPtr->GetTotPoints();
        size_t range2 = transpose ? expPtr->GetTotPoints() : nDim * nCoord;

        if (expPtr->GetMetricInfo()->GetGtype() == SpatialDomains::eDeformed)
        {
            for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
            {
                for (size_t index1 = 0; index1 < range1; ++index1)
                {
                    for (size_t index2 = 0; index2 < range2; ++index2)
                    {
                        for (size_t i = 0; i < interleave_width; ++i, ++df_id)
                        {
                            if (el + i < num_elements)
                            {
                                size_t d  = transpose ? index1 : index2;
                                size_t pt = transpose ? index2 : index1;

                                auto &df = expansionList->GetExp(exp_id + i)
                                               ->GetMetricInfo()
                                               ->GetDerivFactors(
                                                   expPtr->GetPointsKeys());
                                derivFac[df_id] = df[d][pt];
                            }
                            else
                            {
                                derivFac[df_id] = 0.0;
                            }
                        }
                    }
                }

                if (el < num_elements)
                {
                    exp_id += std::min(interleave_width, num_elements - el);
                }
                el += interleave_width;
            }
        }
        else
        {
            for (size_t chunk = 0, el = 0; chunk < num_elmt_groups; ++chunk)
            {
                for (size_t d = 0; d < nDim * nCoord; ++d)
                {
                    for (size_t i = 0; i < interleave_width; ++i, ++df_id)
                    {
                        if (el + i < num_elements)
                        {
                            auto &df =
                                expansionList->GetExp(exp_id + i)
                                    ->GetMetricInfo()
                                    ->GetDerivFactors(expPtr->GetPointsKeys());
                            derivFac[df_id] = df[d][0];
                        }
                        else
                        {
                            derivFac[df_id] = 0.0;
                        }
                    }
                }

                if (el < num_elements)
                {
                    exp_id += std::min(interleave_width, num_elements - el);
                }
                el += interleave_width;
            }
        }
    }

    return MemoryManager<std::vector<NekDouble>>::AllocateSharedPtr(derivFac);
}

} // namespace Nektar::Operators
