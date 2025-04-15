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

#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/Foundations/ManagerAccess.h> // for BasisManager, etc

#include "Operators/Common/NekDataWarehouse.hpp"

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

enum BasisDataType
{
    eNoBasis,
    eBasis,
    eBasisDerivative,
    eDerivative,
    eInterp,
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
                 const BasisDataType basisDataType, const unsigned int npts = 0)
        : m_basisKey(basisKey), m_basisDataType(basisDataType), m_npts(npts)
    {
        hash_combine(m_hash, m_basisKey.GetNumModes(),
                     m_basisKey.GetBasisType(),
                     m_basisKey.GetPointsKey().GetNumPoints(),
                     m_basisKey.GetPointsKey().GetPointsType(),
                     m_basisKey.GetPointsKey().GetFactor(), m_basisDataType,
                     typeid(value_type).name(), "BasisKey");
    }

private:
    LibUtilities::BasisKey m_basisKey;
    BasisDataType m_basisDataType;
    unsigned int m_npts;
};

class BasisDataCreator : public DataCreatorClass
{
public:
    ~BasisDataCreator() override = default;

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const BasisDataKey<TData> &basisDataKey,
                               const unsigned int alignment)
    {
        const auto basis =
            LibUtilities::BasisManager()[basisDataKey.m_basisKey];
        const auto basisDataType = basisDataKey.m_basisDataType;

        switch (basisDataType)
        {
            case eBasis:
            {
                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    basis->GetBdata(), alignment);
                break;
            }
            case eBasisDerivative:
            {
                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    basis->GetDbdata(), alignment);
                break;
            }
            case eWeights:
            {
                auto ndata = basis->GetW().size();
                Array<OneD, NekDouble> wTmp(ndata);

                switch (basis->GetBasisType())
                {
                    case LibUtilities::eModified_B:
                    case LibUtilities::eOrtho_B:
                    {
                        if (basis->GetPointsType() ==
                            LibUtilities::eGaussRadauMAlpha1Beta0)
                        {
                            Vmath::Smul(ndata, 0.5, basis->GetW().data(), 1,
                                        wTmp.data(), 1);
                        }
                        else
                        {
                            const auto z = basis->GetZ();
                            const auto w = basis->GetW();
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
                            Vmath::Smul(ndata, 0.25, basis->GetW().data(), 1,
                                        wTmp.data(), 1);
                        }
                        else if (basis->GetPointsType() ==
                                 LibUtilities::eGaussRadauMAlpha1Beta0)
                        {
                            const auto z = basis->GetZ();
                            const auto w = basis->GetW();
                            for (int i = 0; i < ndata; ++i)
                            {
                                wTmp[i] = 0.25 * (1 - z[i]) * w[i];
                            }
                        }
                        else
                        {
                            const auto z = basis->GetZ();
                            const auto w = basis->GetW();
                            for (int i = 0; i < ndata; ++i)
                            {
                                wTmp[i] = 0.25 * (1 - z[i]) * (1 - z[i]) * w[i];
                            }
                        }
                        break;
                        default:
                        {
                            Vmath::Vcopy(ndata, basis->GetW().data(), 1,
                                         wTmp.data(), 1);
                        }
                    }
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    wTmp, alignment);
            }
            break;
            case eZeros:
            {
                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    basis->GetZ(), alignment);
            }
            break;
            case eDerivative:
            {
                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    basis->GetD()->GetPtr(), alignment);
            }
            break;
            case eInterp:
            {
                const auto np = basisDataKey.m_npts;
                LibUtilities::PointsKey pkey(np, basis->GetPointsType());

                // Need points manager to get correct interpolation matrix
                auto I = LibUtilities::PointsManager()[basis->GetPointsKey()]
                             ->GetI(pkey)
                             ->GetPtr();

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    I, alignment);
            }
            break;
            case eHalfMultOnePlusZero:
            {
                const auto z = basis->GetZ();
                Array<OneD, NekDouble> Tmp(z.size());

                for (int i = 0; i < z.size(); ++i)
                {
                    Tmp[i] = 0.5 * (1.0 + z[i]);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    Tmp, alignment);
            }
            break;
            case eTwoOverOneMinusZero:
            {
                const auto z = basis->GetZ();
                auto n       = z.size();
                Array<OneD, NekDouble> Tmp(n);

                for (int i = 0; i < n; ++i)
                {
                    Tmp[i] = 2 / (1.0 - z[i]);
                }

                return MemoryRegion<TData>::template FromArray<MemSpace>(
                    Tmp, alignment);
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid basis data requested.");
                return MemoryRegion<TData>::Create(0, alignment);
                break;
        }
    }

    inline static const std::string m_name = "BasisDataCreator";
};

enum VandemondeDataType
{
    eNodalToModal,
    eNodalToModalTranspose,
    eModalToNodal
};

class VandemondeDataCreator;

template <typename TData> class VandemondeKey : public BaseKey
{
    friend class VandemondeDataCreator;

public:
    using creator = VandemondeDataCreator;
    typedef TData value_type;

    ~VandemondeKey() override = default;

    VandemondeKey(const VandemondeDataType dataType, const unsigned int exp_idx)
        : m_dataType(dataType), m_exp_idx(exp_idx)
    {
        hash_combine(m_hash, m_dataType, m_exp_idx, typeid(value_type).name(),
                     "VandemondeKey");
    }

private:
    VandemondeDataType m_dataType;
    unsigned int m_exp_idx;
};

class VandemondeDataCreator : public DataCreatorClass
{
public:
    ~VandemondeDataCreator() override = default;
    VandemondeDataCreator(const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const VandemondeKey<TData> &vandemondeKey,
                               const unsigned int alignment)
    {
        auto exp_idx = vandemondeKey.m_exp_idx;
        auto expPtr  = m_expansionList->GetExp(exp_idx);

        auto ncoeffs = expPtr->GetNcoeffs();

        auto vdm    = MemoryRegion<TData>::Create(ncoeffs * ncoeffs, alignment);
        auto vdmptr = vdm.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        DNekMatSharedPtr vdmMat;

        switch (vandemondeKey.m_dataType)
        {
            case eNodalToModal:
            {
                StdRegions::StdMatrixKey Nkey(
                    StdRegions::eInvNBasisTrans, expPtr->DetShapeType(),
                    *expPtr, StdRegions::NullConstFactorMap,
                    StdRegions::NullVarCoeffMap,
                    expPtr->GetNodalPointsKey().GetPointsType());

                vdmMat = expPtr->GetStdMatrix(Nkey);
                goto FwdMat;
                break;
            }
            case eModalToNodal:
            {
                StdRegions::StdMatrixKey Nkey(
                    StdRegions::eNBasisTrans, expPtr->DetShapeType(), *expPtr,
                    StdRegions::NullConstFactorMap, StdRegions::NullVarCoeffMap,
                    expPtr->GetNodalPointsKey().GetPointsType());

                vdmMat = expPtr->GetStdMatrix(Nkey);
                goto FwdMat;
                break;
            }
            case eNodalToModalTranspose:
            {
                StdRegions::StdMatrixKey Nkey(
                    StdRegions::eInvNBasisTrans, expPtr->DetShapeType(),
                    *expPtr, StdRegions::NullConstFactorMap,
                    StdRegions::NullVarCoeffMap,
                    expPtr->GetNodalPointsKey().GetPointsType());

                vdmMat = expPtr->GetStdMatrix(Nkey);
                goto TransMat;
                break;
            }
            FwdMat:
            {
                unsigned int cnt = 0;
                for (unsigned int i = 0; i < ncoeffs; ++i)
                {
                    for (unsigned int j = 0; j < ncoeffs; ++j, ++cnt)
                    {
                        vdmptr[cnt] = vdmMat->GetValue(i, j);
                    }
                }
                break;
            }
            TransMat:
            {
                unsigned int cnt = 0;
                for (unsigned int i = 0; i < ncoeffs; ++i)
                {
                    for (unsigned int j = 0; j < ncoeffs; ++j, ++cnt)
                    {
                        vdmptr[cnt] = vdmMat->GetValue(j, i);
                    }
                }
                break;
            }
        }

        return vdm;
    }

    inline static const std::string m_name = "VandemondeDataCreator";

private:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
