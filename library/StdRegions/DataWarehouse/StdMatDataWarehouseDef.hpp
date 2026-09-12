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

#include <StdRegions/DataWarehouse/StdMatDataWarehouse.hpp>

#include <LibUtilities/Foundations/Interp.h>
#include <LibUtilities/Foundations/PhysGalerkinProject.h>
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

namespace Nektar::StdRegions
{

template <typename MemSpace, typename TData>
LibUtilities::MemoryRegion<TData> StdMatDataCreator::Create(
    const StdMatKey<TData> &stdMatKey)
{
    using namespace Nektar::LibUtilities;

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
            stdExp = MemoryManager<StdPointExp>::AllocateSharedPtr(bkey[0]);
            break;
        }
        case eSegment:
        {
            stdExp = MemoryManager<StdSegExp>::AllocateSharedPtr(bkey[0]);
            break;
        }
        case eTriangle:
        {
            stdExp =
                MemoryManager<StdTriExp>::AllocateSharedPtr(bkey[0], bkey[1]);
            break;
        }
        case eNodalTri:
        {
            stdExp = MemoryManager<StdNodalTriExp>::AllocateSharedPtr(
                bkey[0], bkey[1], nodaltype);
            break;
        }
        case eQuadrilateral:
        {
            stdExp =
                MemoryManager<StdQuadExp>::AllocateSharedPtr(bkey[0], bkey[1]);
            break;
        }
        case eTetrahedron:
        {
            stdExp = MemoryManager<StdTetExp>::AllocateSharedPtr(
                bkey[0], bkey[1], bkey[2]);
            break;
        }
        case eNodalTet:
        {
            stdExp = MemoryManager<StdNodalTetExp>::AllocateSharedPtr(
                bkey[0], bkey[1], bkey[2], nodaltype);
            break;
        }
        case ePyramid:
        {
            stdExp = MemoryManager<StdPyrExp>::AllocateSharedPtr(
                bkey[0], bkey[1], bkey[2]);
            break;
        }
        case ePrism:
        {
            stdExp = MemoryManager<StdPrismExp>::AllocateSharedPtr(
                bkey[0], bkey[1], bkey[2]);
            break;
        }
        case eNodalPrism:
        {
            stdExp = MemoryManager<StdNodalPrismExp>::AllocateSharedPtr(
                bkey[0], bkey[1], bkey[2], nodaltype);
            break;
        }
        case eHexahedron:
        {
            stdExp = MemoryManager<StdHexExp>::AllocateSharedPtr(
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
            Array<OneD, double> tmp(nmTot), t;
            Array<OneD, double> mat(nmTot * nqTot);
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
            Array<OneD, double> tmp(nmTot), t(nqTot);
            Array<OneD, double> mat(nmTot * nqTot);
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
            Array<OneD, double> tmp(nqTot), t;
            Array<OneD, double> mat(dimension * nqTot * nqTot);
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
            Array<OneD, double> tmp(nqTot), t(nqTot);
            Array<OneD, double> mat(dimension * nqTot * nqTot);
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
            Array<OneD, double> tmp0(nmTot), t;
            Array<OneD, double> tmp1(nqTot);
            Array<OneD, double> mat(dimension * nqTot * nmTot);
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
            Array<OneD, double> tmp0(nmTot), t(nqTot);
            Array<OneD, double> tmp1(nqTot);
            Array<OneD, double> mat(dimension * nqTot * nmTot);
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
            Array<OneD, double> tmp(nqTot), t;
            Array<OneD, double> mat(nmTot * nqTot);
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
            Array<OneD, double> tmp(nqTot), t(nmTot);
            Array<OneD, double> mat(nmTot * nqTot);
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
            Array<OneD, double> tmp(nqTot), t;
            Array<OneD, double> mat(dimension * nmTot * nqTot);
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
            Array<OneD, double> tmp(nqTot), t(nmTot);
            Array<OneD, double> mat(dimension * nmTot * nqTot);
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
            Array<OneD, double> tmp(nmTot), t;
            Array<OneD, double> mat(nmTot * nqTot);
            for (unsigned int i = 0; i < nmTot; ++i)
            {
                Vmath::Zero(nmTot, tmp, 1);
                tmp[i] = 1.0;

                if (stdExp->GetShapeDimension() == 1)
                {
                    // In keys
                    const PointsKey &inkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();

                    // Out keys
                    const PointsKey outkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());

                    Interp1D(inkey0, tmp, outkey0, t = mat + i * nqTot);
                }
                else if (stdExp->GetShapeDimension() == 2)
                {
                    // In keys
                    const PointsKey &inkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const PointsKey &inkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();

                    // Out keys
                    const PointsKey outkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const PointsKey outkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());

                    Interp2D(inkey0, inkey1, tmp, outkey0, outkey1,
                             t = mat + i * nqTot);
                }
                else if (stdExp->GetShapeDimension() == 3)
                {
                    // In keys
                    const PointsKey &inkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const PointsKey &inkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();
                    const PointsKey &inkey2 =
                        stdExp->GetBasis(2)->GetPointsKey();

                    // Out keys
                    const PointsKey outkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const PointsKey outkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());
                    const PointsKey outkey2(
                        nq[2], stdExp->GetBasis(2)->GetPointsType());

                    Interp3D(inkey0, inkey1, inkey2, tmp, outkey0, outkey1,
                             outkey2, t = mat + i * nqTot);
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
            Array<OneD, double> tmp(nmTot), t(nqTot);
            Array<OneD, double> mat(nmTot * nqTot);
            for (unsigned int i = 0; i < nmTot; ++i)
            {
                Vmath::Zero(nmTot, tmp, 1);
                tmp[i] = 1.0;

                if (stdExp->GetShapeDimension() == 1)
                {
                    // In keys
                    const PointsKey &inkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();

                    // Out keys
                    const PointsKey outkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());

                    Interp1D(inkey0, tmp, outkey0, t);
                    // Copy to mat with stride nmTot
                    Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                }
                else if (stdExp->GetShapeDimension() == 2)
                {
                    // In keys
                    const PointsKey &inkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const PointsKey &inkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();

                    // Out keys
                    const PointsKey outkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const PointsKey outkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());

                    Interp2D(inkey0, inkey1, tmp, outkey0, outkey1, t);
                    // Copy to mat with stride nmTot
                    Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                }
                else if (stdExp->GetShapeDimension() == 3)
                {
                    // In keys
                    const PointsKey &inkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const PointsKey &inkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();
                    const PointsKey &inkey2 =
                        stdExp->GetBasis(2)->GetPointsKey();

                    // Out keys
                    const PointsKey outkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const PointsKey outkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());
                    const PointsKey outkey2(
                        nq[2], stdExp->GetBasis(2)->GetPointsType());

                    Interp3D(inkey0, inkey1, inkey2, tmp, outkey0, outkey1,
                             outkey2, t);
                    // Copy to mat with stride nmTot
                    Vmath::Vcopy(nqTot, &t[0], 1, &mat[i], nmTot);
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eGalerkinProjectStdMat:
        {
            // Reverse of ePhysInterpStdMat: input lives on the "fine" (nq)
            // grid, output on the native (basisKeys) grid, via
            // PhysGalerkinProject{1,2,3}D instead of Interp{1,2,3}D.
            const auto nmTot = stdExp->GetTotPoints();
            const auto nqTot =
                std::accumulate(nq.begin(), nq.end(), 1, std::multiplies());
            Array<OneD, double> tmp(nqTot), t;
            Array<OneD, double> mat(nqTot * nmTot);
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                Vmath::Zero(nqTot, tmp, 1);
                tmp[i] = 1.0;

                if (stdExp->GetShapeDimension() == 1)
                {
                    // In (fine) keys
                    const LibUtilities::PointsKey inkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());

                    // Out (native) keys
                    const LibUtilities::PointsKey &outkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();

                    LibUtilities::PhysGalerkinProject1D(inkey0, tmp, outkey0,
                                                        t = mat + i * nmTot);
                }
                else if (stdExp->GetShapeDimension() == 2)
                {
                    // In (fine) keys
                    const LibUtilities::PointsKey inkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const LibUtilities::PointsKey inkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());

                    // Out (native) keys
                    const LibUtilities::PointsKey &outkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const LibUtilities::PointsKey &outkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();

                    LibUtilities::PhysGalerkinProject2D(inkey0, inkey1, tmp,
                                                        outkey0, outkey1,
                                                        t = mat + i * nmTot);
                }
                else if (stdExp->GetShapeDimension() == 3)
                {
                    // In (fine) keys
                    const LibUtilities::PointsKey inkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const LibUtilities::PointsKey inkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());
                    const LibUtilities::PointsKey inkey2(
                        nq[2], stdExp->GetBasis(2)->GetPointsType());

                    // Out (native) keys
                    const LibUtilities::PointsKey &outkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const LibUtilities::PointsKey &outkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();
                    const LibUtilities::PointsKey &outkey2 =
                        stdExp->GetBasis(2)->GetPointsKey();

                    LibUtilities::PhysGalerkinProject3D(
                        inkey0, inkey1, inkey2, tmp, outkey0, outkey1, outkey2,
                        t = mat + i * nmTot);
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eGalerkinProjectStdMatTranspose:
        {
            const auto nmTot = stdExp->GetTotPoints();
            const auto nqTot =
                std::accumulate(nq.begin(), nq.end(), 1, std::multiplies());
            Array<OneD, double> tmp(nqTot), t(nmTot);
            Array<OneD, double> mat(nqTot * nmTot);
            for (unsigned int i = 0; i < nqTot; ++i)
            {
                Vmath::Zero(nqTot, tmp, 1);
                tmp[i] = 1.0;

                if (stdExp->GetShapeDimension() == 1)
                {
                    // In (fine) keys
                    const LibUtilities::PointsKey inkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());

                    // Out (native) keys
                    const LibUtilities::PointsKey &outkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();

                    LibUtilities::PhysGalerkinProject1D(inkey0, tmp, outkey0,
                                                        t);
                    // Copy to mat with stride nqTot
                    Vmath::Vcopy(nmTot, &t[0], 1, &mat[i], nqTot);
                }
                else if (stdExp->GetShapeDimension() == 2)
                {
                    // In (fine) keys
                    const LibUtilities::PointsKey inkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const LibUtilities::PointsKey inkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());

                    // Out (native) keys
                    const LibUtilities::PointsKey &outkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const LibUtilities::PointsKey &outkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();

                    LibUtilities::PhysGalerkinProject2D(inkey0, inkey1, tmp,
                                                        outkey0, outkey1, t);
                    // Copy to mat with stride nqTot
                    Vmath::Vcopy(nmTot, &t[0], 1, &mat[i], nqTot);
                }
                else if (stdExp->GetShapeDimension() == 3)
                {
                    // In (fine) keys
                    const LibUtilities::PointsKey inkey0(
                        nq[0], stdExp->GetBasis(0)->GetPointsType());
                    const LibUtilities::PointsKey inkey1(
                        nq[1], stdExp->GetBasis(1)->GetPointsType());
                    const LibUtilities::PointsKey inkey2(
                        nq[2], stdExp->GetBasis(2)->GetPointsType());

                    // Out (native) keys
                    const LibUtilities::PointsKey &outkey0 =
                        stdExp->GetBasis(0)->GetPointsKey();
                    const LibUtilities::PointsKey &outkey1 =
                        stdExp->GetBasis(1)->GetPointsKey();
                    const LibUtilities::PointsKey &outkey2 =
                        stdExp->GetBasis(2)->GetPointsKey();

                    LibUtilities::PhysGalerkinProject3D(inkey0, inkey1, inkey2,
                                                        tmp, outkey0, outkey1,
                                                        outkey2, t);
                    // Copy to mat with stride nqTot
                    Vmath::Vcopy(nmTot, &t[0], 1, &mat[i], nqTot);
                }
            }

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eMassStdMat:
        {
            StdMatrixKey mkey(eMass, stdExp->DetShapeType(), *stdExp);
            Array<OneD, double> tmp(nmTot), t;
            Array<OneD, double> mat(nmTot * nmTot);
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
            StdMatrixKey mkey(eMass, stdExp->DetShapeType(), *stdExp);
            Array<OneD, double> tmp(nmTot), t(nmTot);
            Array<OneD, double> mat(nmTot * nmTot);
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
            StdMatrixKey mkey(eInvMass, stdExp->DetShapeType(), *stdExp);
            const auto &InvMass = stdExp->GetStdMatrix(mkey);
            Array<OneD, double> mat(nmTot * nmTot);
            std::copy_n(InvMass->GetRawPtr(), nmTot * nmTot, mat.data());

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eInvMassStdMatTranspose:
        {
            StdMatrixKey mkey(eInvMass, stdExp->DetShapeType(), *stdExp);
            const auto &InvMass = stdExp->GetStdMatrix(mkey);
            Array<OneD, double> mat(nmTot * nmTot);

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
            StdMatrixKey Nkey(eInvNBasisTrans, stdExp->DetShapeType(), *stdExp,
                              NullConstFactorMap, NullVarCoeffMap,
                              NullVarFactorsMap, nodaltype);
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
            StdMatrixKey Nkey(eNBasisTrans, stdExp->DetShapeType(), *stdExp,
                              NullConstFactorMap, NullVarCoeffMap,
                              NullVarFactorsMap, nodaltype);
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
            StdMatrixKey Nkey(eInvNBasisTrans, stdExp->DetShapeType(), *stdExp,
                              NullConstFactorMap, NullVarCoeffMap,
                              NullVarFactorsMap, nodaltype);
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
            StdMatrixKey mkey(eMass, stdExp->DetShapeType(), *stdExp);
            const auto &InvMassInterior =
                stdExp->GetStdStaticCondMatrix(mkey)->GetBlock(1, 1);
            Array<OneD, double> mat(InvMassInterior->GetStorageSize());
            std::copy_n(InvMassInterior->GetRawPtr(),
                        nInteriorDofs * nInteriorDofs, mat.data());

            return MemoryRegion<TData>::template FromArray<MemSpace>(mat);
        }
        break;
        case eInvMassInteriorStdMatTranspose:
        {
            const auto nBoundaryDofs = stdExp->NumBndryCoeffs();
            const auto nInteriorDofs = nmTot - nBoundaryDofs;
            StdMatrixKey mkey(eMass, stdExp->DetShapeType(), *stdExp);
            const auto &InvMassInterior =
                stdExp->GetStdStaticCondMatrix(mkey)->GetBlock(1, 1);
            InvMassInterior->Transpose();
            Array<OneD, double> mat(InvMassInterior->GetStorageSize());
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

} // namespace Nektar::StdRegions
