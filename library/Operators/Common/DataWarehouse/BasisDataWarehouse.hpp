///////////////////////////////////////////////////////////////////////////////
//
// File: BasisDataWarehouse.hpp
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

#include "Operators/Common/DataWarehouse/NekDataWarehouse.hpp"

#include <LibUtilities/Foundations/Basis.h>

namespace Nektar::Operators
{

enum BasisDataType
{
    eNoBasis,
    eBasis,
    eBasisDerivative,
    eDerivative,
    eInterp,
    eInterpTranspose,
    eWeights,
    eZeros,
    eHalfMultOnePlusZero,
    eTwoOverOneMinusZero
};

class BasisDataCreator;

template <typename TData> class BasisDataKey : public BaseKey
{
    friend class BasisDataCreator;

public:
    using creator = BasisDataCreator;
    typedef TData value_type;

    ~BasisDataKey() override = default;

    BasisDataKey(const LibUtilities::BasisKey &basisKey,
                 const BasisDataType basisDataType, const unsigned int npts = 0,
                 const LibUtilities::PointsType toPointsType =
                     LibUtilities::eNoPointsType)
        : m_basisKey(basisKey), m_basisDataType(basisDataType), m_npts(npts),
          m_toPointsType(toPointsType)
    {
        hash_combine(m_hash, m_basisKey.GetNumModes(),
                     m_basisKey.GetBasisType(),
                     m_basisKey.GetPointsKey().GetNumPoints(),
                     m_basisKey.GetPointsKey().GetPointsType(),
                     m_basisKey.GetPointsKey().GetFactor(), m_basisDataType,
                     m_toPointsType, typeid(value_type).name(), "BasisKey");
    }

private:
    LibUtilities::BasisKey m_basisKey;
    BasisDataType m_basisDataType;
    unsigned int m_npts;
    LibUtilities::PointsType m_toPointsType;
};

class BasisDataCreator : public DataCreatorClass
{
public:
    ~BasisDataCreator() override = default;

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const BasisDataKey<TData> &basisDataKey);

    inline static const std::string m_name = "BasisDataCreator";
};

} // namespace Nektar::Operators
