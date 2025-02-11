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

#include <StdRegions/StdHexExp.h>
#include <StdRegions/StdNodalPrismExp.h>
#include <StdRegions/StdNodalTetExp.h>
#include <StdRegions/StdNodalTriExp.h>
#include <StdRegions/StdPointExp.h>
#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdPyrExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdSegExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>

#include "Operators/Common/NekDataWarehouse.hpp"

namespace Nektar::Operators
{

enum StdMatType
{
    eNoStdMat                    = 0,
    eBwdTransStdMat              = 1,
    ePhysDerivStdMat             = 2,
    eIProductWRTBaseStdMat       = 3,
    eIProductWRTDerivBaseStdMat  = 4,
    eMultiplyByElmtInvMassStdMat = 5,
};

class StdMatDataCreator;

template <typename TData> class StdMatKey : public BaseKey
{
public:
    using creator = StdMatDataCreator;

    ~StdMatKey() override = default;
    StdMatKey(const std::vector<LibUtilities::BasisKey> basisKeys,
              const LibUtilities::ShapeType shapeType,
              const StdMatType stdMatType)
        : m_basisKeys(basisKeys), m_shapeType(shapeType),
          m_stdMatType(stdMatType)
    {
        if (m_basisKeys.size() > 2)
        {
            hash_combine(m_hash, m_basisKeys[2].GetNumModes(),
                         m_basisKeys[2].GetBasisType(),
                         m_basisKeys[2].GetPointsKey().GetNumPoints(),
                         m_basisKeys[2].GetPointsKey().GetPointsType(),
                         m_basisKeys[2].GetPointsKey().GetFactor());
        }
        if (m_basisKeys.size() > 1)
        {
            hash_combine(m_hash, m_basisKeys[1].GetNumModes(),
                         m_basisKeys[1].GetBasisType(),
                         m_basisKeys[1].GetPointsKey().GetNumPoints(),
                         m_basisKeys[1].GetPointsKey().GetPointsType(),
                         m_basisKeys[1].GetPointsKey().GetFactor());
        }
        hash_combine(m_hash, m_basisKeys[0].GetNumModes(),
                     m_basisKeys[0].GetBasisType(),
                     m_basisKeys[0].GetPointsKey().GetNumPoints(),
                     m_basisKeys[0].GetPointsKey().GetPointsType(),
                     m_basisKeys[0].GetPointsKey().GetFactor(), m_shapeType,
                     m_stdMatType, m_name);
    }

    std::vector<LibUtilities::BasisKey> m_basisKeys;
    LibUtilities::ShapeType m_shapeType;
    StdMatType m_stdMatType;
    typedef TData m_data_type;

private:
    inline static const std::string m_name = "StdMatKey";
};

class StdMatDataCreator : public DataCreatorClass
{
public:
    ~StdMatDataCreator() override = default;

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const StdMatKey<TData> &stdMatKey,
                               const unsigned int alignment)
    {
        using namespace Nektar::LibUtilities;
        using namespace Nektar::StdRegions;

        const auto shapeType  = stdMatKey.m_shapeType;
        const auto bkey       = stdMatKey.m_basisKeys;
        const auto stdMatType = stdMatKey.m_stdMatType;

        StdExpansion *stdExp = nullptr;

        switch (shapeType)
        {
            case ePoint:
            {
                stdExp = new StdPointExp(bkey[0]);
                break;
            }
            case eSegment:
            {
                stdExp = new StdSegExp(bkey[0]);
                break;
            }
            case eTriangle:
            {
                // stdExp = nodaltype != eNoPointsType
                //             ? new StdNodalTriExp(bkey[0], bkey[1], nodaltype)
                //             : new StdTriExp(bkey[0], bkey[1]);
                stdExp = new StdTriExp(bkey[0], bkey[1]);
                break;
            }
            case eQuadrilateral:
            {
                stdExp = new StdQuadExp(bkey[0], bkey[1]);
                break;
            }
            case eTetrahedron:
            {
                // stdExp = nodaltype != eNoPointsType
                //              ? new StdNodalTetExp(bkey[0], bkey[1], bkey[2],
                //                                   nodaltype)
                //              : new StdTetExp(bkey[0], bkey[1], bkey[2]);
                stdExp = new StdTetExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            case ePyramid:
            {
                stdExp = new StdPyrExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            case ePrism:
            {
                // stdExp = nodaltype != eNoPointsType
                //              ? new StdNodalPrismExp(bkey[0], bkey[1],
                //              bkey[2],
                //                                     nodaltype)
                //              : new StdPrismExp(bkey[0], bkey[1], bkey[2]);
                stdExp = new StdPrismExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            case eHexahedron:
            {
                stdExp = new StdHexExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            default:
                break;
        }

        const auto dimension = stdExp->GetShapeDimension();
        const auto nqTot     = stdExp->GetTotPoints();
        const auto nmTot     = stdExp->GetNcoeffs();

        switch (stdMatType)
        {
            case eBwdTransStdMat:
            {
                Array<OneD, NekDouble> tmp(nmTot), t;
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (size_t i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->BwdTrans(tmp, t = mat + i * nqTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    mat, alignment);
            }
            break;
            case ePhysDerivStdMat:
            {
                Array<OneD, NekDouble> tmp(nqTot), t;
                Array<OneD, NekDouble> mat(dimension * nqTot * nqTot);
                for (size_t d = 0; d < dimension; ++d)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        stdExp->PhysDeriv(
                            d, tmp, t = mat + d * nqTot * nqTot + i * nqTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    mat, alignment);
            }
            break;
            case eIProductWRTBaseStdMat:
            {
                Array<OneD, NekDouble> tmp(nqTot), t;
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (size_t i = 0; i < nqTot; ++i)
                {
                    Vmath::Zero(nqTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->IProductWRTBase(tmp, t = mat + i * nmTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    mat, alignment);
            }
            break;
            case eIProductWRTDerivBaseStdMat:
            {
                Array<OneD, NekDouble> tmp(nqTot), t;
                Array<OneD, NekDouble> mat(dimension * nmTot * nqTot);
                for (size_t d = 0; d < dimension; ++d)
                {
                    for (size_t i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        stdExp->IProductWRTDerivBase(
                            d, tmp, t = mat + d * nmTot * nqTot + i * nmTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    mat, alignment);
            }
            break;
            case eMultiplyByElmtInvMassStdMat:
            {
                Nektar::StdRegions::StdMatrixKey mkey(
                    StdRegions::eInvMass, stdExp->DetShapeType(), *stdExp);
                const auto &InvMass = stdExp->GetStdMatrix(mkey);
                Array<OneD, NekDouble> mat(InvMass->GetStorageSize());
                std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, mat.data());

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    mat, alignment);
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid StdMat requested.");
                return MemoryRegion<TData>::template Create<MemSpace>(
                    0, alignment);
                break;
        }
    }

    inline static const std::string m_name = "StdMatDataCreator";
};

} // namespace Nektar::Operators
