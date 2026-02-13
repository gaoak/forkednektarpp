///////////////////////////////////////////////////////////////////////////////
//
// File: StdMatDataWarehouseDef.hpp
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

#include "Operators/Common/StdMatDataWarehouse.hpp"

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

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

template <typename MemSpace, typename TData>
MemoryRegion<TData> StdMatDataCreator::Create(const StdMatKey<TData> &stdMatKey)
{
    using namespace Nektar::LibUtilities;
    using namespace Nektar::StdRegions;

    const auto shapeType  = stdMatKey.m_shapeType;
    const auto bkey       = stdMatKey.m_basisKeys;
    const auto stdMatType = stdMatKey.m_stdMatType;
    const auto nodaltype  = stdMatKey.m_nodalType;
    const auto &nq        = stdMatKey.m_nq;

    StdExpansionSharedPtr stdExp = nullptr;

    switch (shapeType)
    {
        case ePoint:
        {
            stdExp = MemoryManager<
                Nektar::StdRegions::StdPointExp>::AllocateSharedPtr(bkey[0]);
            break;
        }
        case eSegment:
        {
            stdExp =
                MemoryManager<Nektar::StdRegions::StdSegExp>::AllocateSharedPtr(
                    bkey[0]);
            break;
        }
        case eTriangle:
        {
            stdExp =
                MemoryManager<Nektar::StdRegions::StdTriExp>::AllocateSharedPtr(
                    bkey[0], bkey[1]);
            break;
        }
        case eNodalTri:
        {
            stdExp = MemoryManager<Nektar::StdRegions::StdNodalTriExp>::
                AllocateSharedPtr(bkey[0], bkey[1], nodaltype);
            break;
        }
        case eQuadrilateral:
        {
            stdExp = MemoryManager<
                Nektar::StdRegions::StdQuadExp>::AllocateSharedPtr(bkey[0],
                                                                   bkey[1]);
            break;
        }
        case eTetrahedron:
        {
            stdExp =
                MemoryManager<Nektar::StdRegions::StdTetExp>::AllocateSharedPtr(
                    bkey[0], bkey[1], bkey[2]);
            break;
        }
        case eNodalTet:
        {
            stdExp = MemoryManager<Nektar::StdRegions::StdNodalTetExp>::
                AllocateSharedPtr(bkey[0], bkey[1], bkey[2], nodaltype);
            break;
        }
        case ePyramid:
        {
            stdExp =
                MemoryManager<Nektar::StdRegions::StdPyrExp>::AllocateSharedPtr(
                    bkey[0], bkey[1], bkey[2]);
            break;
        }
        case ePrism:
        {
            stdExp = MemoryManager<
                Nektar::StdRegions::StdPrismExp>::AllocateSharedPtr(bkey[0],
                                                                    bkey[1],
                                                                    bkey[2]);
            break;
        }
        case eNodalPrism:
        {
            stdExp = MemoryManager<Nektar::StdRegions::StdNodalPrismExp>::
                AllocateSharedPtr(bkey[0], bkey[1], bkey[2], nodaltype);
            break;
        }
        case eHexahedron:
        {
            stdExp =
                MemoryManager<Nektar::StdRegions::StdHexExp>::AllocateSharedPtr(
                    bkey[0], bkey[1], bkey[2]);
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
                    stdExp->PhysDeriv(d, tmp,
                                      t = mat + d * nqTot * nqTot + i * nqTot);
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
                    Vmath::Vcopy(nqTot, &t[0], 1, &mat[i + d * nqTot * nqTot],
                                 nqTot);
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eDerivStdMat:
        {
            Array<OneD, NekDouble> tmp0(nmTot), t;
            Array<OneD, NekDouble> tmp1(nqTot);
            Array<OneD, NekDouble> mat(dimension * nqTot * nmTot);
            for (unsigned int d = 0; d < dimension; ++d)
            {
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp0, 1);
                    tmp0[i] = 1.0;
                    stdExp->BwdTrans(tmp0, tmp1);
                    stdExp->PhysDeriv(d, tmp1,
                                      t = mat + d * nqTot * nmTot + i * nqTot);
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eDerivStdMatTranspose:
        {
            Array<OneD, NekDouble> tmp0(nmTot), t(nqTot);
            Array<OneD, NekDouble> tmp1(nqTot);
            Array<OneD, NekDouble> mat(dimension * nqTot * nmTot);
            for (unsigned int d = 0; d < dimension; ++d)
            {
                for (unsigned int i = 0; i < nmTot; ++i)
                {
                    Vmath::Zero(nmTot, tmp0, 1);
                    tmp0[i] = 1.0;
                    stdExp->BwdTrans(tmp0, tmp1);
                    stdExp->PhysDeriv(d, tmp1, t);
                    // copy to mat with stride nqTot
                    Vmath::Vcopy(nqTot, &t[0], 1, &mat[i + d * nqTot * nmTot],
                                 nmTot);
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
                    Vmath::Vcopy(nmTot, &t[0], 1, &mat[i + d * nmTot * nqTot],
                                 nqTot);
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

                    LibUtilities::Interp3D(inkey0, inkey1, inkey2, tmp, outkey0,
                                           outkey1, outkey2,
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

                    LibUtilities::Interp3D(inkey0, inkey1, inkey2, tmp, outkey0,
                                           outkey1, outkey2, t);
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
        case eNodalToModal:
        {
            StdRegions::StdMatrixKey Nkey(
                StdRegions::eInvNBasisTrans, stdExp->DetShapeType(), *stdExp,
                StdRegions::NullConstFactorMap, StdRegions::NullVarCoeffMap,
                StdRegions::NullVarFactorsMap, nodaltype);
            auto vdmMat = stdExp->GetStdMatrix(Nkey);

            auto vdm = MemoryRegion<TData>(nmTot * nmTot);
            auto vdmptr =
                vdm.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            unsigned int cnt = 0;
            for (unsigned int i = 0; i < nmTot; ++i)
            {
                for (unsigned int j = 0; j < nmTot; ++j, ++cnt)
                {
                    vdmptr[cnt] = vdmMat->GetValue(i, j);
                }
            }
            return vdm;
        }
        break;
        case eModalToNodal:
        {
            StdRegions::StdMatrixKey Nkey(
                StdRegions::eNBasisTrans, stdExp->DetShapeType(), *stdExp,
                StdRegions::NullConstFactorMap, StdRegions::NullVarCoeffMap,
                StdRegions::NullVarFactorsMap, nodaltype);
            auto vdmMat = stdExp->GetStdMatrix(Nkey);

            auto vdm = MemoryRegion<TData>(nmTot * nmTot);
            auto vdmptr =
                vdm.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            unsigned int cnt = 0;
            for (unsigned int i = 0; i < nmTot; ++i)
            {
                for (unsigned int j = 0; j < nmTot; ++j, ++cnt)
                {
                    vdmptr[cnt] = vdmMat->GetValue(i, j);
                }
            }
            return vdm;
        }
        break;
        case eNodalToModalTranspose:
        {
            StdRegions::StdMatrixKey Nkey(
                StdRegions::eInvNBasisTrans, stdExp->DetShapeType(), *stdExp,
                StdRegions::NullConstFactorMap, StdRegions::NullVarCoeffMap,
                StdRegions::NullVarFactorsMap, nodaltype);
            auto vdmMat = stdExp->GetStdMatrix(Nkey);

            auto vdm = MemoryRegion<TData>(nmTot * nmTot);
            auto vdmptr =
                vdm.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            unsigned int cnt = 0;
            for (unsigned int i = 0; i < nmTot; ++i)
            {
                for (unsigned int j = 0; j < nmTot; ++j, ++cnt)
                {
                    vdmptr[cnt] = vdmMat->GetValue(j, i);
                }
            }
            return vdm;
        }
        break;
        case eInvMassInteriorStdMat:
        {
            const auto nBoundaryDofs = stdExp->NumBndryCoeffs();
            const auto nInteriorDofs = nmTot - nBoundaryDofs;
            Nektar::StdRegions::StdMatrixKey mkey(
                StdRegions::eMass, stdExp->DetShapeType(), *stdExp);
            const auto &InvMassInterior =
                stdExp->GetStdStaticCondMatrix(mkey)->GetBlock(1, 1);
            Array<OneD, NekDouble> mat(InvMassInterior->GetStorageSize());
            std::copy_n(InvMassInterior->GetRawPtr(),
                        nInteriorDofs * nInteriorDofs, mat.data());

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eInvMassInteriorStdMatTranspose:
        {
            const auto nBoundaryDofs = stdExp->NumBndryCoeffs();
            const auto nInteriorDofs = nmTot - nBoundaryDofs;
            Nektar::StdRegions::StdMatrixKey mkey(
                StdRegions::eMass, stdExp->DetShapeType(), *stdExp);
            const auto &InvMassInterior =
                stdExp->GetStdStaticCondMatrix(mkey)->GetBlock(1, 1);
            InvMassInterior->Transpose();
            Array<OneD, NekDouble> mat(InvMassInterior->GetStorageSize());
            std::copy_n(InvMassInterior->GetRawPtr(),
                        nInteriorDofs * nInteriorDofs, mat.data());

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        default:
            NEKERROR(ErrorUtil::efatal, "invalid StdMat requested.");
            return MemoryRegion<TData>(0);
            break;
    }
}

} // namespace Nektar::Operators
