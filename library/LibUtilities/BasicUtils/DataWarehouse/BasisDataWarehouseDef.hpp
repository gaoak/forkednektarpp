///////////////////////////////////////////////////////////////////////////////
//
// File: BasisDataWarehouseDef.hpp
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

#include <LibUtilities/BasicUtils/DataWarehouse/BasisDataWarehouse.hpp>
#include <LibUtilities/Foundations/ManagerAccess.h> // for BasisManager, etc

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::LibUtilities
{

template <typename MemSpace, typename TData>
MemoryRegion<TData> BasisDataCreator::Create(
    const BasisDataKey<TData> &basisDataKey)
{
    const auto basis         = BasisManager()[basisDataKey.m_basisKey];
    const auto basisDataType = basisDataKey.m_basisDataType;

    switch (basisDataType)
    {
        case eBasis:
        {
            return MemoryRegion<TData>::template FromArray<MemSpace>(
                basis->GetBdata());
            break;
        }
        case eBasisDerivative:
        {
            return MemoryRegion<TData>::template FromArray<MemSpace>(
                basis->GetDbdata());
            break;
        }
        case eWeights:
        {
            auto ndata = basis->GetW().size();
            Array<OneD, double> wTmp(ndata);

            switch (basis->GetBasisType())
            {
                case eModified_B:
                case eOrtho_B:
                {
                    if (basis->GetPointsType() == eGaussRadauMAlpha1Beta0)
                    {
                        Vmath::Smul(ndata, 0.5, basis->GetW().data(), 1,
                                    wTmp.data(), 1);
                    }
                    else
                    {
                        const auto z = basis->GetZ();
                        const auto w = basis->GetW();
                        for (unsigned int i = 0; i < ndata; ++i)
                        {
                            wTmp[i] = 0.5 * (1 - z[i]) * w[i];
                        }
                    }
                }
                break;
                case eModified_C:
                case eModifiedPyr_C:
                case eOrtho_C:
                case eOrthoPyr_C:
                {
                    if (basis->GetPointsType() == eGaussRadauMAlpha2Beta0)
                    {
                        Vmath::Smul(ndata, 0.25, basis->GetW().data(), 1,
                                    wTmp.data(), 1);
                    }
                    else if (basis->GetPointsType() == eGaussRadauMAlpha1Beta0)
                    {
                        const auto z = basis->GetZ();
                        const auto w = basis->GetW();
                        for (unsigned int i = 0; i < ndata; ++i)
                        {
                            wTmp[i] = 0.25 * (1 - z[i]) * w[i];
                        }
                    }
                    else
                    {
                        const auto z = basis->GetZ();
                        const auto w = basis->GetW();
                        for (unsigned int i = 0; i < ndata; ++i)
                        {
                            wTmp[i] = 0.25 * (1 - z[i]) * (1 - z[i]) * w[i];
                        }
                    }
                    break;
                }
                default:
                {
                    Vmath::Vcopy(ndata, basis->GetW().data(), 1, wTmp.data(),
                                 1);
                    break;
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(wTmp);
        }
        break;
        case eZeros:
        {
            return MemoryRegion<TData>::template FromArray<MemSpace>(
                basis->GetZ());
        }
        break;
        case eDerivative:
        {
            return MemoryRegion<TData>::template FromArray<MemSpace>(
                basis->GetD()->GetPtr());
        }
        break;
        case eInterpDerivative: // interpolation of nodal derivative matrix
        {
            const auto npTo = basisDataKey.m_npts;
            // Check if we interpolate to
            // a different PointsType specified via m_pointsToType
            // or use the basis' PointsType
            LibUtilities::PointsType ptype =
                basisDataKey.m_toPointsType == LibUtilities::eNoPointsType
                    ? basis->GetPointsType()
                    : basisDataKey.m_toPointsType;
            LibUtilities::PointsKey toPkey(npTo, ptype);

            // use points manager to get correct interpolation matrix
            auto I = LibUtilities::PointsManager()[basis->GetPointsKey()]
                         ->GetI(toPkey)
                         ->GetPtr();

            auto D = basis->GetD()->GetPtr();

            auto npFrom = basis->GetNumPoints();
            Array<OneD, double> ID(npTo * npFrom);

            // calcaulate I x D
            for (unsigned i = 0; i < npTo; ++i)
            {
                for (unsigned j = 0; j < npFrom; ++j)
                {
                    double sum = 0.0;
                    for (unsigned k = 0; k < npFrom; ++k)
                    {
                        sum += I[i + k * npTo] * D[k + j * npFrom];
                    }
                    ID[i + npTo * j] = sum;
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(ID);
        }
        break;
        case eInterp:
        {
            const auto npTo = basisDataKey.m_npts;
            // Check if we interpolate to
            // a different PointsType specified via m_pointsToType
            // or use the basis' PointsType
            PointsType ptype = basisDataKey.m_toPointsType == eNoPointsType
                                   ? basis->GetPointsType()
                                   : basisDataKey.m_toPointsType;
            PointsKey toPkey(npTo, ptype);

            // Need points manager to get correct interpolation matrix
            auto I =
                PointsManager()[basis->GetPointsKey()]->GetI(toPkey)->GetPtr();

            return MemoryRegion<TData>::template FromArray<MemSpace>(I);
        }
        break;
        case eInterpTranspose:
        {
            const auto npTo = basisDataKey.m_npts;
            // Check if we interpolate to
            // a different PointsType specified via m_pointsToType
            // or use the basis' PointsType
            PointsType ptype = basisDataKey.m_toPointsType == eNoPointsType
                                   ? basis->GetPointsType()
                                   : basisDataKey.m_toPointsType;
            PointsKey toPkey(npTo, ptype);

            // Need points manager to get correct interpolation matrix
            auto I =
                PointsManager()[basis->GetPointsKey()]->GetI(toPkey)->GetPtr();

            // Transpose interpolation matrix
            const auto npFrom = basis->GetPointsKey().GetNumPoints();
            Array<OneD, double> tmpI(npFrom * npTo);
            for (int i = 0; i < npFrom; ++i)
            {
                for (int j = 0; j < npTo; ++j)
                {
                    tmpI[i + j * npFrom] = I[j + i * npTo];
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(tmpI);
        }
        break;
        case eHalfMultOnePlusZero:
        {
            const auto z = basis->GetZ();
            Array<OneD, double> Tmp(z.size());

            for (unsigned int i = 0; i < z.size(); ++i)
            {
                Tmp[i] = 0.5 * (1.0 + z[i]);
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(Tmp);
        }
        break;
        case eTwoOverOneMinusZero:
        {
            const auto z = basis->GetZ();
            auto n       = z.size();
            Array<OneD, double> Tmp(n);

            for (unsigned int i = 0; i < n; ++i)
            {
                Tmp[i] = 2 / (1.0 - z[i]);
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(Tmp);
        }
        break;
        default:
            NEKERROR(ErrorUtil::efatal, "invalid basis data requested.");
            return MemoryRegion<TData>(0);
            break;
    }
}

} // namespace Nektar::LibUtilities
