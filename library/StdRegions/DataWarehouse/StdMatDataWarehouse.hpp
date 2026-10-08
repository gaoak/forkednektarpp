///////////////////////////////////////////////////////////////////////////////
//
// File: StdMatDataWarehouse.hpp
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

#include <LibUtilities/BasicUtils/DataWarehouse/NekDataWarehouse.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/Foundations/Basis.h>

namespace Nektar::StdRegions
{

enum StdMatType
{
    eNoStdMat                            = 0,
    eBwdTransStdMat                      = 1,
    eBwdTransStdMatTranspose             = 2,
    ePhysDerivStdMat                     = 3,
    ePhysDerivStdMatTranspose            = 4,
    eDerivStdMat                         = 5,
    eDerivStdMatTranspose                = 6,
    eIProductWRTBaseStdMat               = 7,
    eIProductWRTBaseStdMatTranspose      = 8,
    eIProductWRTDerivBaseStdMat          = 9,
    eIProductWRTDerivBaseStdMatTranspose = 10,
    ePhysInterpStdMat                    = 11,
    ePhysInterpStdMatTranspose           = 12,
    eMassStdMat                          = 13,
    eMassStdMatTranspose                 = 14,
    eInvMassStdMat                       = 15,
    eInvMassStdMatTranspose              = 16,
    eNodalToModal                        = 17,
    eNodalToModalTranspose               = 18,
    eModalToNodal                        = 19,
    eInvMassInteriorStdMat               = 20,
    eInvMassInteriorStdMatTranspose      = 21,
    // Galerkin projection from a fine physical grid to the native grid.
    eGalerkinProjectStdMat          = 22,
    eGalerkinProjectStdMatTranspose = 23,
};

class StdMatDataCreator;

template <typename TData> class StdMatKey : public LibUtilities::BaseKey
{
    friend class StdMatDataCreator;

public:
    using creator = StdMatDataCreator;
    typedef TData value_type;

    ~StdMatKey() override = default;

    StdMatKey(
        const std::vector<LibUtilities::BasisKey> basisKeys,
        const LibUtilities::ShapeType shapeType, const StdMatType stdMatType,
        const LibUtilities::PointsType nodalType = LibUtilities::eNoPointsType,
        const std::vector<unsigned int> nq = std::vector<unsigned int>(3, 1))
        : m_basisKeys(basisKeys), m_shapeType(shapeType),
          m_stdMatType(stdMatType), m_nodalType(nodalType), m_nq(nq)
    {
        if (m_basisKeys.size() > 2)
        {
            hash_combine(m_hash, m_basisKeys[2].GetNumModes(),
                         m_basisKeys[2].GetBasisType(),
                         m_basisKeys[2].GetPointsKey().GetNumPoints(),
                         m_basisKeys[2].GetPointsKey().GetPointsType(),
                         m_basisKeys[2].GetPointsKey().GetFactor(), m_nq[2]);
        }
        if (m_basisKeys.size() > 1)
        {
            hash_combine(m_hash, m_basisKeys[1].GetNumModes(),
                         m_basisKeys[1].GetBasisType(),
                         m_basisKeys[1].GetPointsKey().GetNumPoints(),
                         m_basisKeys[1].GetPointsKey().GetPointsType(),
                         m_basisKeys[1].GetPointsKey().GetFactor(), m_nq[1]);
        }
        hash_combine(
            m_hash, m_basisKeys[0].GetNumModes(), m_basisKeys[0].GetBasisType(),
            m_basisKeys[0].GetPointsKey().GetNumPoints(),
            m_basisKeys[0].GetPointsKey().GetPointsType(),
            m_basisKeys[0].GetPointsKey().GetFactor(), m_nq[0], m_shapeType,
            m_stdMatType, m_nodalType, typeid(value_type).name(), "StdMatKey");
    }

private:
    std::vector<LibUtilities::BasisKey> m_basisKeys;
    LibUtilities::ShapeType m_shapeType;
    StdMatType m_stdMatType;
    LibUtilities::PointsType m_nodalType;
    std::vector<unsigned int> m_nq;
};

class StdMatDataCreator : public LibUtilities::DataCreatorClass
{
public:
    ~StdMatDataCreator() override = default;

    template <typename MemSpace, typename TData>
    LibUtilities::MemoryRegion<TData> Create(const StdMatKey<TData> &stdMatKey);

    inline static const std::string m_name = "StdMatDataCreator";
};

} // namespace Nektar::StdRegions
