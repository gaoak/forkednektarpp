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

#ifdef NEKTAR_ENABLE_CUDA
#include "MemoryRegionCUDA.hpp"
#endif
#include <MultiRegions/ExpList.h>

namespace Nektar::Operators
{

static size_t GetCUDAGridSize(size_t ndata, size_t blockSize)
{
    return (ndata + blockSize - 1) / blockSize;
}

template <typename TData>
using DataMap =
    std::map<std::vector<LibUtilities::BasisKey>, std::vector<TData *>>;

template <typename TData>
DataMap<TData> GetBasisDataCUDA(
    const MultiRegions::ExpListSharedPtr &expansionList)
{
    // Initialize data map.
    DataMap<TData> basis;

    // Initialize basiskey.
    std::vector<LibUtilities::BasisKey> basisKeys(3,
                                                  LibUtilities::NullBasisKey);

    // Loop over the elements of expansionList.
    size_t nDim = expansionList->GetShapeDimension();
    for (size_t i = 0; i < expansionList->GetNumElmts(); ++i)
    {
        auto const expPtr = expansionList->GetExp(i);

        // Fetch basiskeys of current element.
        for (size_t d = 0; d < nDim; d++)
        {
            basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        // Copy data to basis, if necessary.
        if (basis.find(basisKeys) == basis.end())
        {
            basis[basisKeys] = std::vector<TData *>(nDim, 0);
            for (size_t d = 0; d < nDim; d++)
            {
                auto ndata      = expPtr->GetBasis(d)->GetBdata().size();
                auto hostPtr    = expPtr->GetBasis(d)->GetBdata().get();
                auto &devicePtr = basis[basisKeys][d];
                cudaMalloc((void **)&devicePtr, sizeof(TData) * ndata);
                cudaMemcpy(devicePtr, hostPtr, sizeof(TData) * ndata,
                           cudaMemcpyHostToDevice);
            }
        }
    }
    return basis;
}

template <typename TData>
DataMap<TData> GetDeriveBasisDataCUDA(
    const MultiRegions::ExpListSharedPtr &expansionList)
{
    // Initialize data map.
    DataMap<TData> dbasis;

    // Initialize basiskey.
    std::vector<LibUtilities::BasisKey> basisKeys(3,
                                                  LibUtilities::NullBasisKey);

    // Loop over the elements of expansionList.
    size_t nDim = expansionList->GetShapeDimension();
    for (size_t i = 0; i < expansionList->GetNumElmts(); ++i)
    {
        auto const expPtr = expansionList->GetExp(i);

        // Fetch basiskeys of current element.
        for (size_t d = 0; d < nDim; d++)
        {
            basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        // Copy data to dbasis, if necessary.
        if (dbasis.find(basisKeys) == dbasis.end())
        {
            dbasis[basisKeys] = std::vector<TData *>(nDim, 0);
            for (size_t d = 0; d < nDim; d++)
            {
                auto ndata      = expPtr->GetBasis(d)->GetDbdata().size();
                auto hostPtr    = expPtr->GetBasis(d)->GetDbdata().get();
                auto &devicePtr = dbasis[basisKeys][d];
                cudaMalloc((void **)&devicePtr, sizeof(TData) * ndata);
                cudaMemcpy(devicePtr, hostPtr, sizeof(TData) * ndata,
                           cudaMemcpyHostToDevice);
            }
        }
    }
    return dbasis;
}

template <typename TData>
DataMap<TData> GetWeightDataCUDA(
    const MultiRegions::ExpListSharedPtr &expansionList)
{
    // Initialize data map.
    DataMap<TData> weight;

    // Initialize basiskey.
    std::vector<LibUtilities::BasisKey> basisKeys(3,
                                                  LibUtilities::NullBasisKey);

    // Loop over the elements of expansionList.
    size_t nDim = expansionList->GetShapeDimension();
    for (size_t i = 0; i < expansionList->GetNumElmts(); ++i)
    {
        auto const expPtr = expansionList->GetExp(i);

        // Fetch basiskeys of current element.
        for (size_t d = 0; d < nDim; d++)
        {
            basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        // Copy data to weight, if necessary.
        if (weight.find(basisKeys) == weight.end())
        {
            weight[basisKeys] = std::vector<TData *>(nDim, 0);
            for (size_t d = 0; d < nDim; d++)
            {
                auto ndata = expPtr->GetBasis(d)->GetW().size();
                Array<OneD, TData> w(ndata);
                if (expPtr->GetBasis(d)->GetPointsType() ==
                    LibUtilities::eGaussRadauMAlpha1Beta0)
                {
                    Vmath::Smul(ndata, 0.5, expPtr->GetBasis(d)->GetW().get(),
                                1, w.get(), 1);
                }
                else if (expPtr->GetBasis(d)->GetPointsType() ==
                         LibUtilities::eGaussRadauMAlpha2Beta0)
                {
                    Vmath::Smul(ndata, 0.25, expPtr->GetBasis(d)->GetW().get(),
                                1, w.get(), 1);
                }
                else
                {
                    Vmath::Vcopy(ndata, expPtr->GetBasis(d)->GetW().get(), 1,
                                 w.get(), 1);
                }
                auto hostPtr    = w.get();
                auto &devicePtr = weight[basisKeys][d];
                cudaMalloc((void **)&devicePtr, sizeof(TData) * ndata);
                cudaMemcpy(devicePtr, hostPtr, sizeof(TData) * ndata,
                           cudaMemcpyHostToDevice);
            }
        }
    }
    return weight;
}

template <typename TData>
DataMap<TData> GetPointDataCUDA(
    const MultiRegions::ExpListSharedPtr &expansionList)
{
    // Initialize data map.
    DataMap<TData> points;

    // Initialize basiskey.
    std::vector<LibUtilities::BasisKey> basisKeys(3,
                                                  LibUtilities::NullBasisKey);

    // Loop over the elements of expansionList.
    size_t nDim = expansionList->GetShapeDimension();
    for (size_t i = 0; i < expansionList->GetNumElmts(); ++i)
    {
        auto const expPtr = expansionList->GetExp(i);

        // Fetch basiskeys of current element.
        for (size_t d = 0; d < nDim; d++)
        {
            basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        // Copy data to points, if necessary.
        if (points.find(basisKeys) == points.end())
        {
            points[basisKeys] = std::vector<TData *>(nDim, 0);
            for (size_t d = 0; d < nDim; d++)
            {
                auto ndata      = expPtr->GetBasis(d)->GetZ().size();
                auto hostPtr    = expPtr->GetBasis(d)->GetZ().get();
                auto &devicePtr = points[basisKeys][d];
                cudaMalloc((void **)&devicePtr, sizeof(TData) * ndata);
                cudaMemcpy(devicePtr, hostPtr, sizeof(TData) * ndata,
                           cudaMemcpyHostToDevice);
            }
        }
    }
    return points;
}

template <typename TData>
DataMap<TData> GetDerivativeDataCUDA(
    const MultiRegions::ExpListSharedPtr &expansionList)
{
    // Initialize data map.
    DataMap<TData> derivative;

    // Initialize basiskey.
    std::vector<LibUtilities::BasisKey> basisKeys(3,
                                                  LibUtilities::NullBasisKey);

    // Loop over the elements of expansionList.
    size_t nDim = expansionList->GetShapeDimension();
    for (size_t i = 0; i < expansionList->GetNumElmts(); ++i)
    {
        auto const expPtr = expansionList->GetExp(i);

        // Fetch basiskeys of current element.
        for (size_t d = 0; d < nDim; d++)
        {
            basisKeys[d] = expPtr->GetBasis(d)->GetBasisKey();
        }

        // Copy data to derivative, if necessary.
        if (derivative.find(basisKeys) == derivative.end())
        {
            derivative[basisKeys] = std::vector<TData *>(nDim, 0);
            for (size_t d = 0; d < nDim; d++)
            {
                auto ndata      = expPtr->GetBasis(d)->GetD()->GetPtr().size();
                auto hostPtr    = expPtr->GetBasis(d)->GetD()->GetPtr().get();
                auto &devicePtr = derivative[basisKeys][d];
                cudaMalloc((void **)&devicePtr, sizeof(TData) * ndata);
                cudaMemcpy(devicePtr, hostPtr, sizeof(TData) * ndata,
                           cudaMemcpyHostToDevice);
            }
        }
    }
    return derivative;
}

template <typename TData> void DeallocateDataCUDA(DataMap<TData> &dataMap)
{
    for (auto &data : dataMap)
    {
        for (size_t i = 0; i < data.second.size(); i++)
        {
            cudaFree(data.second[i]);
        }
    }
}

} // namespace Nektar::Operators
