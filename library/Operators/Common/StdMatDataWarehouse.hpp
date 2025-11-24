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

#include <LibUtilities/Foundations/Interp.h>
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

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

enum StdMatType
{
    eNoStdMat                            = 0,
    eBwdTransStdMat                      = 1,
    eBwdTransStdMatTranspose             = 2,
    ePhysDerivStdMat                     = 3,
    ePhysDerivStdMatTranspose            = 4,
    eIProductWRTBaseStdMat               = 5,
    eIProductWRTBaseStdMatTranspose      = 6,
    eIProductWRTDerivBaseStdMat          = 7,
    eIProductWRTDerivBaseStdMatTranspose = 8,
    ePhysInterpStdMat                    = 9,
    ePhysInterpStdMatTranspose           = 10,
    eMassStdMat                          = 11,
    eMassStdMatTranspose                 = 12,
    eInvMassStdMat                       = 13,
    eInvMassStdMatTranspose              = 14,
};

class StdMatDataCreator;

template <typename TData> class StdMatKey : public BaseKey
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

class StdMatDataCreator : public DataCreatorClass
{
public:
    ~StdMatDataCreator() override = default;

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const StdMatKey<TData> &stdMatKey)
    {
        using namespace Nektar::LibUtilities;
        using namespace Nektar::StdRegions;

        const auto shapeType  = stdMatKey.m_shapeType;
        const auto bkey       = stdMatKey.m_basisKeys;
        const auto stdMatType = stdMatKey.m_stdMatType;
        const auto nodaltype  = stdMatKey.m_nodalType;
        const auto &nq        = stdMatKey.m_nq;

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
                stdExp = new StdTriExp(bkey[0], bkey[1]);
                break;
            }
            case eNodalTri:
            {
                stdExp = new StdNodalTriExp(bkey[0], bkey[1], nodaltype);
                break;
            }
            case eQuadrilateral:
            {
                stdExp = new StdQuadExp(bkey[0], bkey[1]);
                break;
            }
            case eTetrahedron:
            {
                stdExp = new StdTetExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            case eNodalTet:
            {
                stdExp =
                    new StdNodalTetExp(bkey[0], bkey[1], bkey[2], nodaltype);
                break;
            }
            case ePyramid:
            {
                stdExp = new StdPyrExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            case ePrism:
            {
                stdExp = new StdPrismExp(bkey[0], bkey[1], bkey[2]);
                break;
            }
            case eNodalPrism:
            {
                stdExp =
                    new StdNodalPrismExp(bkey[0], bkey[1], bkey[2], nodaltype);
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
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->BwdTrans(tmp, t = mat + i * nqTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eBwdTransStdMatTranspose:
            {
                Array<OneD, NekDouble> tmp(nmTot), t(nqTot);
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->BwdTrans(tmp, t);
                    // copy to mat with stride nmTot
                    Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case ePhysDerivStdMat:
            {
                Array<OneD, NekDouble> tmp(nqTot), t;
                Array<OneD, NekDouble> mat(dimension * nqTot * nqTot);
                for (unsigned int d = 0; d < dimension; ++d)
                {
                    for (unsigned int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        stdExp->PhysDeriv(
                            d, tmp, t = mat + d * nqTot * nqTot + i * nqTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case ePhysDerivStdMatTranspose:
            {
                Array<OneD, NekDouble> tmp(nqTot), t(nqTot);
                Array<OneD, NekDouble> mat(dimension * nqTot * nqTot);
                for (unsigned int d = 0; d < dimension; ++d)
                {
                    for (unsigned int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        stdExp->PhysDeriv(d, tmp, t);
                        // copy to mat with stride nqTot
                        Vmath::Vcopy(nqTot, &t[0], 1,
                                     &mat[i + d * nqTot * nqTot], nqTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eIProductWRTBaseStdMat:
            {
                Array<OneD, NekDouble> tmp(nqTot), t;
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (unsigned int i = 0; i < nqTot; ++i)
                {
                    Vmath::Zero(nqTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->IProductWRTBase(tmp, t = mat + i * nmTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eIProductWRTBaseStdMatTranspose:
            {
                Array<OneD, NekDouble> tmp(nqTot), t(nmTot);
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (unsigned int i = 0; i < nqTot; ++i)
                {
                    Vmath::Zero(nqTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->IProductWRTBase(tmp, t);
                    // copy to mat with stride nqTot
                    Vmath::Vcopy(nmTot, &t[0], 1, &mat[i], nqTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eIProductWRTDerivBaseStdMat:
            {
                Array<OneD, NekDouble> tmp(nqTot), t;
                Array<OneD, NekDouble> mat(dimension * nmTot * nqTot);
                for (unsigned int d = 0; d < dimension; ++d)
                {
                    for (unsigned int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        stdExp->IProductWRTDerivBase(
                            d, tmp, t = mat + d * nmTot * nqTot + i * nmTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eIProductWRTDerivBaseStdMatTranspose:
            {
                Array<OneD, NekDouble> tmp(nqTot), t(nmTot);
                Array<OneD, NekDouble> mat(dimension * nmTot * nqTot);
                for (unsigned int d = 0; d < dimension; ++d)
                {
                    for (unsigned int i = 0; i < nqTot; ++i)
                    {
                        Vmath::Zero(nqTot, tmp, 1);
                        tmp[i] = 1.0;
                        stdExp->IProductWRTDerivBase(d, tmp, t);
                        // copy to mat with stride nqTot
                        Vmath::Vcopy(nmTot, &t[0], 1,
                                     &mat[i + d * nmTot * nqTot], nqTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case ePhysInterpStdMat:
            {
                const auto nmTot = stdExp->GetTotPoints();
                const auto nqTot =
                    std::accumulate(nq.begin(), nq.end(), 1, std::multiplies());
                Array<OneD, NekDouble> tmp(nmTot), t;
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;

                    if (stdExp->GetShapeDimension() == 1)
                    {
                        // In keys
                        const LibUtilities::PointsKey &inkey0 =
                            stdExp->GetBasis(0)->GetPointsKey();

                        // Out keys
                        const LibUtilities::PointsKey outkey0(
                            nq[0], stdExp->GetBasis(0)->GetPointsType());

                        LibUtilities::Interp1D(inkey0, tmp, outkey0,
                                               t = mat + i * nqTot);
                    }
                    else if (stdExp->GetShapeDimension() == 2)
                    {
                        // In keys
                        const LibUtilities::PointsKey &inkey0 =
                            stdExp->GetBasis(0)->GetPointsKey();
                        const LibUtilities::PointsKey &inkey1 =
                            stdExp->GetBasis(1)->GetPointsKey();

                        // Out keys
                        const LibUtilities::PointsKey outkey0(
                            nq[0], stdExp->GetBasis(0)->GetPointsType());
                        const LibUtilities::PointsKey outkey1(
                            nq[1], stdExp->GetBasis(1)->GetPointsType());

                        LibUtilities::Interp2D(inkey0, inkey1, tmp, outkey0,
                                               outkey1, t = mat + i * nqTot);
                    }
                    else if (stdExp->GetShapeDimension() == 3)
                    {
                        // In keys
                        const LibUtilities::PointsKey &inkey0 =
                            stdExp->GetBasis(0)->GetPointsKey();
                        const LibUtilities::PointsKey &inkey1 =
                            stdExp->GetBasis(1)->GetPointsKey();
                        const LibUtilities::PointsKey &inkey2 =
                            stdExp->GetBasis(2)->GetPointsKey();

                        // Out keys
                        const LibUtilities::PointsKey outkey0(
                            nq[0], stdExp->GetBasis(0)->GetPointsType());
                        const LibUtilities::PointsKey outkey1(
                            nq[1], stdExp->GetBasis(1)->GetPointsType());
                        const LibUtilities::PointsKey outkey2(
                            nq[2], stdExp->GetBasis(2)->GetPointsType());

                        LibUtilities::Interp3D(inkey0, inkey1, inkey2, tmp,
                                               outkey0, outkey1, outkey2,
                                               t = mat + i * nqTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case ePhysInterpStdMatTranspose:
            {
                const auto nmTot = stdExp->GetTotPoints();
                const auto nqTot =
                    std::accumulate(nq.begin(), nq.end(), 1, std::multiplies());
                Array<OneD, NekDouble> tmp(nmTot), t(nqTot);
                Array<OneD, NekDouble> mat(nmTot * nqTot);
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;

                    if (stdExp->GetShapeDimension() == 1)
                    {
                        // In keys
                        const LibUtilities::PointsKey &inkey0 =
                            stdExp->GetBasis(0)->GetPointsKey();

                        // Out keys
                        const LibUtilities::PointsKey outkey0(
                            nq[0], stdExp->GetBasis(0)->GetPointsType());

                        LibUtilities::Interp1D(inkey0, tmp, outkey0, t);
                        // Copy to mat with stride nmTot
                        Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                    }
                    else if (stdExp->GetShapeDimension() == 2)
                    {
                        // In keys
                        const LibUtilities::PointsKey &inkey0 =
                            stdExp->GetBasis(0)->GetPointsKey();
                        const LibUtilities::PointsKey &inkey1 =
                            stdExp->GetBasis(1)->GetPointsKey();

                        // Out keys
                        const LibUtilities::PointsKey outkey0(
                            nq[0], stdExp->GetBasis(0)->GetPointsType());
                        const LibUtilities::PointsKey outkey1(
                            nq[1], stdExp->GetBasis(1)->GetPointsType());

                        LibUtilities::Interp2D(inkey0, inkey1, tmp, outkey0,
                                               outkey1, t);
                        // Copy to mat with stride nmTot
                        Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                    }
                    else if (stdExp->GetShapeDimension() == 3)
                    {
                        // In keys
                        const LibUtilities::PointsKey &inkey0 =
                            stdExp->GetBasis(0)->GetPointsKey();
                        const LibUtilities::PointsKey &inkey1 =
                            stdExp->GetBasis(1)->GetPointsKey();
                        const LibUtilities::PointsKey &inkey2 =
                            stdExp->GetBasis(2)->GetPointsKey();

                        // Out keys
                        const LibUtilities::PointsKey outkey0(
                            nq[0], stdExp->GetBasis(0)->GetPointsType());
                        const LibUtilities::PointsKey outkey1(
                            nq[1], stdExp->GetBasis(1)->GetPointsType());
                        const LibUtilities::PointsKey outkey2(
                            nq[2], stdExp->GetBasis(2)->GetPointsType());

                        LibUtilities::Interp3D(inkey0, inkey1, inkey2, tmp,
                                               outkey0, outkey1, outkey2, t);
                        // Copy to mat with stride nmTot
                        Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eMassStdMat:
            {
                Nektar::StdRegions::StdMatrixKey mkey(
                    StdRegions::eMass, stdExp->DetShapeType(), *stdExp);
                Array<OneD, NekDouble> tmp(nmTot), t;
                Array<OneD, NekDouble> mat(nmTot * nmTot);
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->MassMatrixOp(tmp, t = mat + i * nmTot, mkey);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eMassStdMatTranspose:
            {
                Nektar::StdRegions::StdMatrixKey mkey(
                    StdRegions::eMass, stdExp->DetShapeType(), *stdExp);
                Array<OneD, NekDouble> tmp(nmTot), t(nmTot);
                Array<OneD, NekDouble> mat(nmTot * nmTot);
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp, 1);
                    tmp[i] = 1.0;
                    stdExp->MassMatrixOp(tmp, t, mkey);
                    // copy to mat with stride nmTot
                    Vmath::Vcopy(nmTot, &t[0], 1, &mat[i], nmTot);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eInvMassStdMat:
            {
                Nektar::StdRegions::StdMatrixKey mkey(
                    StdRegions::eInvMass, stdExp->DetShapeType(), *stdExp);
                const auto &InvMass = stdExp->GetStdMatrix(mkey);
                Array<OneD, NekDouble> mat(nmTot * nmTot);
                std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, mat.data());

                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            case eInvMassStdMatTranspose:
            {
                Nektar::StdRegions::StdMatrixKey mkey(
                    StdRegions::eInvMass, stdExp->DetShapeType(), *stdExp);
                const auto &InvMass = stdExp->GetStdMatrix(mkey);
                Array<OneD, NekDouble> mat(nmTot * nmTot);

                // copy to mat with stride nmTot
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Vcopy(nmTot, &InvMass->GetRawPtr()[i * nmTot], 1,
                                 &mat[i], nmTot);
                }
                return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid StdMat requested.");
                return MemoryRegion<TData>(0);
                break;
        }
    }

    inline static const std::string m_name = "StdMatDataCreator";
};

} // namespace Nektar::Operators
