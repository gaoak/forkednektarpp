///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzDeviceSumFacTOPKernels.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/DeviceProperties.hpp"
#include "Operators/Common/Spaces.hpp"

#include "Operators/ElmtOps/BwdTrans/BwdTransDeviceSumFacTOPKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseDeviceSumFacTOPKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivDeviceSumFacTOPKernels.hpp"

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline size_t HelmholtzWorkSpaceSize(
    [[maybe_unused]] const LibUtilities::ShapeType shapeType,
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nm0)
{
    return 0;
}

template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline size_t HelmholtzWorkSpaceSize(
    [[maybe_unused]] const LibUtilities::ShapeType shapeType,
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1)
{
    return 0;
}

template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline size_t HelmholtzWorkSpaceSize(
    [[maybe_unused]] const LibUtilities::ShapeType shapeType,
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const unsigned int nq0,
    [[maybe_unused]] const unsigned int nq1,
    [[maybe_unused]] const unsigned int nq2,
    [[maybe_unused]] const unsigned int nm0,
    [[maybe_unused]] const unsigned int nm1,
    [[maybe_unused]] const unsigned int nm2)
{
    return 0;
}

template <typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int HelmholtzSharedMemorySize(
    const unsigned int nq0, [[maybe_unused]] const unsigned int nm0)
{
    return 4 * nq0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int HelmholtzSharedMemorySize(const unsigned int nq0,
                                              const unsigned int nq1,
                                              const unsigned int nm0,
                                              const unsigned int nm1)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nm0 * nq0 + nm1 * nq1 + 4 * nq0 * nq1 +
               std::max(nq0 * nm1, nm0 * nq1) + 6;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        return nm0 * nq0 + nmTot * nq1 + 4 * nq0 * nq1 + nm0 * nq1 + 6;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot =
            LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
        return nm0 * nq0 + nmTot * nq1 + 4 * nq0 * nq1 + nm0 * nq1 + 6 + nmTot;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename std::enable_if<
              std::is_same_v<Implementation, SumFacTOP>>::type * = nullptr>
inline unsigned int HelmholtzSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 4 * nq0 * nq1 * nq2 +
               std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2) +
               std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2) + 9;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm01 * nq2 + 9;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
        return nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 9;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmTot =
            LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + 4 * nq0 * nq1 * nq2 +
               nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 9;
    }
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const unsigned int inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const TData *NEK_RESTRICT in,
    TData *out, TData *NEK_RESTRICT bwd, const TData lambda,
    const TthreadBlock &threadBlock)
{
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    TData metric[3] = {0.0};
    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 1)
            {
                metric[0] = diffCoeff[0] * df[0];
            }
            else if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[1];
                metric[1] = diffCoeff[1] * df[0] + diffCoeff[2] * df[1];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[1] +
                            diffCoeff[3] * df[2];
                metric[1] = diffCoeff[1] * df[0] + diffCoeff[2] * df[1] +
                            diffCoeff[4] * df[2];
                metric[2] = diffCoeff[3] * df[0] + diffCoeff[4] * df[1] +
                            diffCoeff[5] * df[2];
            }
        }
    }

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int i = idx0; i < nq0; i += stride)
    {
        const unsigned int dfindex = DEFORMED ? i : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 1)
                {
                    metric[0] = diffCoeff[0] * df[i];
                }
                else if (ncoord == 2)
                {
                    metric[0] =
                        diffCoeff[0] * df[i] + diffCoeff[1] * df[dfsize + i];
                    metric[1] =
                        diffCoeff[1] * df[i] + diffCoeff[2] * df[dfsize + i];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[i] +
                                diffCoeff[1] * df[dfsize + i] +
                                diffCoeff[3] * df[2 * dfsize + i];
                    metric[1] = diffCoeff[1] * df[i] +
                                diffCoeff[2] * df[dfsize + i] +
                                diffCoeff[4] * df[2 * dfsize + i];
                    metric[2] = diffCoeff[3] * df[i] +
                                diffCoeff[4] * df[dfsize + i] +
                                diffCoeff[5] * df[2 * dfsize + i];
                }
            }
        }

        TData sum = 0.0;
        for (unsigned int k = 0u; k < ncoord; ++k)
        {
            if (diffCoeff)
            {
                sum += metric[k] * in[k * inoffset + i];
            }
            else
            {
                sum += df[k * dfsize + dfindex] * in[k * inoffset + i];
            }
        }

        if constexpr (DEFORMED)
        {
            bwd[i] *= lambda * jac[i] * w0[i];
            out[i] = sum * jac[i] * w0[i];
        }
        else
        {
            bwd[i] *= lambda * jac[0] * w0[i];
            out[i] = sum * jac[0] * w0[i];
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const unsigned int inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    const TData *NEK_RESTRICT in, TData *out0, TData *out1, TData *metric,
    TData *NEK_RESTRICT bwd, const TData lambda,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            if (ncoord == 2)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[2];
                metric[1] = diffCoeff[0] * df[1] + diffCoeff[1] * df[3];
                metric[2] = diffCoeff[1] * df[0] + diffCoeff[2] * df[2];
                metric[3] = diffCoeff[1] * df[1] + diffCoeff[2] * df[3];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[2] +
                            diffCoeff[3] * df[4];
                metric[1] = diffCoeff[0] * df[1] + diffCoeff[1] * df[3] +
                            diffCoeff[3] * df[5];
                metric[2] = diffCoeff[1] * df[0] + diffCoeff[2] * df[2] +
                            diffCoeff[4] * df[4];
                metric[3] = diffCoeff[1] * df[1] + diffCoeff[2] * df[3] +
                            diffCoeff[4] * df[5];
                metric[4] = diffCoeff[3] * df[0] + diffCoeff[4] * df[2] +
                            diffCoeff[5] * df[4];
                metric[5] = diffCoeff[3] * df[1] + diffCoeff[4] * df[3] +
                            diffCoeff[5] * df[5];
            }

            localBarrier(threadBlock);
        }
    }

    for (unsigned int idx = idx0; idx < nq0 * nq1; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = idx / nq0;
        const unsigned int dfindex = DEFORMED ? idx : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                if (ncoord == 2)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex];
                    metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                                diffCoeff[1] * df[3 * dfsize + dfindex];
                    metric[2] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[2 * dfsize + dfindex];
                    metric[3] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                                diffCoeff[2] * df[3 * dfsize + dfindex];
                }
                else if (ncoord == 3)
                {
                    metric[0] = diffCoeff[0] * df[dfindex] +
                                diffCoeff[1] * df[2 * dfsize + dfindex] +
                                diffCoeff[3] * df[4 * dfsize + dfindex];
                    metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                                diffCoeff[1] * df[3 * dfsize + dfindex] +
                                diffCoeff[3] * df[5 * dfsize + dfindex];
                    metric[2] = diffCoeff[1] * df[dfindex] +
                                diffCoeff[2] * df[2 * dfsize + dfindex] +
                                diffCoeff[4] * df[4 * dfsize + dfindex];
                    metric[3] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                                diffCoeff[2] * df[3 * dfsize + dfindex] +
                                diffCoeff[4] * df[5 * dfsize + dfindex];
                    metric[4] = diffCoeff[3] * df[dfindex] +
                                diffCoeff[4] * df[2 * dfsize + dfindex] +
                                diffCoeff[5] * df[4 * dfsize + dfindex];
                    metric[5] = diffCoeff[3] * df[1 * dfsize + dfindex] +
                                diffCoeff[4] * df[3 * dfsize + dfindex] +
                                diffCoeff[5] * df[5 * dfsize + dfindex];
                }
            }
        }

        TData sum1 = 0.0, sum2 = 0.0;
        if (diffCoeff)
        {
            TData tmp = in[idx];
            sum1 += metric[0u] * tmp;
            sum2 += metric[1u] * tmp;
            tmp = in[inoffset + idx];
            sum1 += metric[2u] * tmp;
            sum2 += metric[3u] * tmp;
            if (ncoord == 3u)
            {
                tmp = in[2u * inoffset + idx];
                sum1 += metric[4u] * tmp;
                sum2 += metric[5u] * tmp;
            }
        }
        else
        {
            TData tmp = in[idx];
            sum1 += df[0u * dfsize + dfindex] * tmp;
            sum2 += df[1u * dfsize + dfindex] * tmp;
            tmp = in[inoffset + idx];
            sum1 += df[2u * dfsize + dfindex] * tmp;
            sum2 += df[3u * dfsize + dfindex] * tmp;
            if (ncoord == 3u)
            {
                tmp = in[2u * inoffset + idx];
                sum1 += df[4u * dfsize + dfindex] * tmp;
                sum2 += df[5u * dfsize + dfindex] * tmp;
            }
        }

        TData tmpQ = w0[i] * w1[j];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        bwd[idx] *= lambda * tmpQ;

        // Moving from standard to collapsed coordinates.
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            out0[idx] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
            out1[idx] = sum2 * tmpQ;
        }
    }

    localBarrier(threadBlock);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const TData *NEK_RESTRICT in,
    TData *out0, TData *out1, TData *out2, TData *metric,
    TData *NEK_RESTRICT bwd, const TData lambda,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot  = nq0 * nq1 * nq2;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    if constexpr (!DEFORMED)
    {
        if (diffCoeff)
        {
            metric[0] = diffCoeff[0] * df[0] + diffCoeff[1] * df[3] +
                        diffCoeff[3] * df[6];
            metric[1] = diffCoeff[0] * df[1] + diffCoeff[1] * df[4] +
                        diffCoeff[3] * df[7];
            metric[2] = diffCoeff[0] * df[2] + diffCoeff[1] * df[5] +
                        diffCoeff[3] * df[8];
            metric[3] = diffCoeff[1] * df[0] + diffCoeff[2] * df[3] +
                        diffCoeff[4] * df[6];
            metric[4] = diffCoeff[1] * df[1] + diffCoeff[2] * df[4] +
                        diffCoeff[4] * df[7];
            metric[5] = diffCoeff[1] * df[2] + diffCoeff[2] * df[5] +
                        diffCoeff[4] * df[8];
            metric[6] = diffCoeff[3] * df[0] + diffCoeff[4] * df[3] +
                        diffCoeff[5] * df[6];
            metric[7] = diffCoeff[3] * df[1] + diffCoeff[4] * df[4] +
                        diffCoeff[5] * df[7];
            metric[8] = diffCoeff[3] * df[2] + diffCoeff[4] * df[5] +
                        diffCoeff[5] * df[8];

            localBarrier(threadBlock);
        }
    }

    for (unsigned int idx = idx0; idx < nq0 * nq1 * nq2; idx += stride)
    {
        const unsigned int i       = idx % nq0;
        const unsigned int j       = (idx / nq0) % nq1;
        const unsigned int k       = idx / (nq0 * nq1);
        const unsigned int dfindex = DEFORMED ? idx : 0;

        if constexpr (DEFORMED)
        {
            if (diffCoeff)
            {
                metric[0] = diffCoeff[0] * df[dfindex] +
                            diffCoeff[1] * df[3 * dfsize + dfindex] +
                            diffCoeff[3] * df[6 * dfsize + dfindex];
                metric[1] = diffCoeff[0] * df[1 * dfsize + dfindex] +
                            diffCoeff[1] * df[4 * dfsize + dfindex] +
                            diffCoeff[3] * df[7 * dfsize + dfindex];
                metric[2] = diffCoeff[0] * df[2 * dfsize + dfindex] +
                            diffCoeff[1] * df[5 * dfsize + dfindex] +
                            diffCoeff[3] * df[8 * dfsize + dfindex];
                metric[3] = diffCoeff[1] * df[dfindex] +
                            diffCoeff[2] * df[3 * dfsize + dfindex] +
                            diffCoeff[4] * df[6 * dfsize + dfindex];
                metric[4] = diffCoeff[1] * df[1 * dfsize + dfindex] +
                            diffCoeff[2] * df[4 * dfsize + dfindex] +
                            diffCoeff[4] * df[7 * dfsize + dfindex];
                metric[5] = diffCoeff[1] * df[2 * dfsize + dfindex] +
                            diffCoeff[2] * df[5 * dfsize + dfindex] +
                            diffCoeff[4] * df[8 * dfsize + dfindex];
                metric[6] = diffCoeff[3] * df[dfindex] +
                            diffCoeff[4] * df[3 * dfsize + dfindex] +
                            diffCoeff[5] * df[6 * dfsize + dfindex];
                metric[7] = diffCoeff[3] * df[1 * dfsize + dfindex] +
                            diffCoeff[4] * df[4 * dfsize + dfindex] +
                            diffCoeff[5] * df[7 * dfsize + dfindex];
                metric[8] = diffCoeff[3] * df[2 * dfsize + dfindex] +
                            diffCoeff[4] * df[5 * dfsize + dfindex] +
                            diffCoeff[5] * df[8 * dfsize + dfindex];
            }
        }

        TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        if (diffCoeff)
        {
            TData tmp = in[idx];
            sum1 += metric[0u] * tmp;
            sum2 += metric[1u] * tmp;
            sum3 += metric[2u] * tmp;
            tmp = in[inoffset + idx];
            sum1 += metric[3u] * tmp;
            sum2 += metric[4u] * tmp;
            sum3 += metric[5u] * tmp;
            tmp = in[2u * inoffset + idx];
            sum1 += metric[6u] * tmp;
            sum2 += metric[7u] * tmp;
            sum3 += metric[8u] * tmp;
        }
        else
        {
            TData tmp = in[idx];
            sum1 += df[0u * dfsize + dfindex] * tmp;
            sum2 += df[1u * dfsize + dfindex] * tmp;
            sum3 += df[2u * dfsize + dfindex] * tmp;
            tmp = in[inoffset + idx];
            sum1 += df[3u * dfsize + dfindex] * tmp;
            sum2 += df[4u * dfsize + dfindex] * tmp;
            sum3 += df[5u * dfsize + dfindex] * tmp;
            tmp = in[2u * inoffset + idx];
            sum1 += df[6u * dfsize + dfindex] * tmp;
            sum2 += df[7u * dfsize + dfindex] * tmp;
            sum3 += df[8u * dfsize + dfindex] * tmp;
        }

        TData tmpQ = w0[i] * w1[j] * w2[k];
        if constexpr (DEFORMED)
        {
            tmpQ *= jac[idx];
        }
        else
        {
            tmpQ *= jac[0];
        }

        bwd[idx] *= lambda * tmpQ;

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            out0[idx] = sum1 * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            out0[idx] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
            out1[idx] = sum2 * tmpQ;
            out2[idx] = sum3 * tmpQ;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData tmp = f2[k] * tmpQ;
            out0[idx] = (sum1 + sum3 * f0[i]) * tmp;
            out1[idx] = (sum2 + sum3 * f1[j]) * tmp;
            out2[idx] = sum3 * tmpQ;
        }
    }

    localBarrier(threadBlock);
}

template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Helmholtz1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nq0,
    const size_t nelmt, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData lambda,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *bwd   = (TData *)shmemptr;
    TData *deriv = bwd + nq0;

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nm0 * e;
        TData *outptr       = out + nm0 * e;

        BwdTransSegSumFacTOPKernel(nm0, nq0, basis0, inptr, bwd, threadBlock);
        PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, D0, dfptr, bwd,
                                             deriv, threadBlock);
        ApplyMetric1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, w0, dfptr,
                                               jacptr, coeff, deriv, deriv, bwd,
                                               lambda, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<true>(nq0, D0, deriv, bwd, threadBlock);
        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Helmholtz2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const bool isModified,
    const unsigned int *NEK_RESTRICT index0, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT nodToMod,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData lambda,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        offset = std::max(nm0 * nq1, nm1 * nq0);
        nmode0 = nm0;
        nmode1 = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        offset = nm0 * nq1;
        nmode0 = nm0;
        nmode1 = nmTot;
    }

    TData *metric   = (TData *)shmemptr;
    TData *bwd      = metric + 6;
    TData *deriv    = bwd + nqTot;
    TData *tmp      = deriv;
    TData *deriv0   = deriv;
    TData *deriv1   = deriv0 + nqTot;
    TData *s_wsp0   = deriv + ncoord * nqTot;
    TData *s_basis0 = s_wsp0 + offset;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    size_t e = getBlockIdx(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }

            localBarrier(threadBlock);
        }

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            BwdTransQuadSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, s_basis0,
                                        s_basis1, tmp, bwd, s_wsp0,
                                        threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel(nm0, nm1, nq0, nq1, nqTot, isModified,
                                       s_basis0, s_basis1, tmp, bwd, s_wsp0,
                                       threadBlock);
        }

        PhysDeriv2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, nqTot, D0, D1, f0, f1, dfptr, bwd, deriv,
            threadBlock);
        if constexpr (DEFORMED)
        {
            TData dmetric[6];
            ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr, coeff,
                deriv, deriv0, deriv1, dmetric, bwd, lambda, threadBlock);
        }
        else
        {
            ApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, nqTot, w0, w1, f0, f1, dfptr, jacptr, coeff,
                deriv, deriv0, deriv1, metric, bwd, lambda, threadBlock);
        }
        SumDerivTensor2DSumFacTOPKernel<true>(nq0, nq1, D0, D1, deriv0, deriv1,
                                              bwd, threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, bwd,
                outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, bwd, tmp, s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void Helmholtz3DSumFacTOPKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t nelmt, const bool isModified,
    const unsigned int *NEK_RESTRICT index0,
    const unsigned int *NEK_RESTRICT index1,
    const unsigned int *NEK_RESTRICT index2,
    const unsigned int *NEK_RESTRICT index3, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out, const TData lambda,
    unsigned char *NEK_RESTRICT shmemptr, const TthreadBlock &threadBlock)
{
    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int offset0 = 0, offset1 = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        offset0 = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
        offset1 = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        offset0 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        offset0 = nm0 * nm1 * nq2;
        offset1 = nm0 * nq1 * nq2;
        nmode0  = nm0;
        nmode1  = nm1;
        nmode2  = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    TData *metric   = (TData *)shmemptr;
    TData *bwd      = metric + 9;
    TData *deriv    = bwd + nqTot;
    TData *tmp      = deriv;
    TData *deriv0   = deriv;
    TData *deriv1   = deriv0 + nqTot;
    TData *deriv2   = deriv1 + nqTot;
    TData *s_wsp0   = deriv2 + nqTot;
    TData *s_wsp1   = s_wsp0 + offset0;
    TData *s_basis0 = s_wsp1 + offset1;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx(threadBlock);
    const unsigned int stride = getLocalRange(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    for (unsigned int idx = idx0; idx < nmode2 * nq2; idx += stride)
    {
        s_basis2[idx] = basis2[idx];
    }

    size_t e = getBlockIdx(threadBlock); // use size_t to prevent overflow
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * e;
        TData *outptr       = out + nmTot * e;

        // Copy to shared memory.
        if constexpr (SHAPE_TYPE == LibUtilities::NodalTet ||
                      SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, tmp, threadBlock);
        }
        else
        {
            // Copy to shared memory.
            for (unsigned int idx = idx0; idx < nmTot; idx += stride)
            {
                tmp[idx] = inptr[idx];
            }
        }

        localBarrier(threadBlock);

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            BwdTransHexSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       s_basis0, s_basis1, s_basis2, tmp, bwd,
                                       s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       isModified, index0, index3, s_basis0,
                                       s_basis1, s_basis2, tmp, bwd, s_wsp0,
                                       s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel(nm0, nm1, nm2, nq0, nq1, nq2, nqTot,
                                       isModified, s_basis0, s_basis1, s_basis2,
                                       tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        PhysDeriv3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
            nq0, nq1, nq2, nqTot, D0, D1, D2, f0, f1, f1m, f2, dfptr, bwd,
            deriv, threadBlock);
        if constexpr (DEFORMED)
        {
            TData dmetric[9];
            ApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nqTot, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, coeff, deriv, deriv0, deriv1, deriv2, dmetric, bwd,
                lambda, threadBlock);
        }
        else
        {
            ApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, nqTot, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, coeff, deriv, deriv0, deriv1, deriv2, metric, bwd,
                lambda, threadBlock);
        }
        SumDerivTensor3DSumFacTOPKernel<true>(nq0, nq1, nq2, D0, D1, D2, deriv0,
                                              deriv1, deriv2, bwd, threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, bwd, outptr, s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, bwd, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<false, false>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, bwd, outptr, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);
        }

        e += getBlockRange(threadBlock);
    }
}

// Non-size based version.
template <typename Implementation, bool DEFORMED, typename TthreadBlock,
          typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Helmholtz1DKernelLauncher(
        const NonTemplated1DSizeParameters sizeParam1D,
        const unsigned int ncoord, const size_t nelmt,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Helmholtz1DSumFacTOPKernel<DEFORMED>(
        ncoord, sizeParam1D.nm0(), sizeParam1D.nq0(), nelmt, basis0, D0, w0, df,
        jac, coeff, in, out, lambda, shmemptr, threadBlock);
}

// Size based template version.
template <
    typename Implementation, bool DEFORMED, unsigned int nm0, unsigned int nq0,
    typename TthreadBlock, typename TData,
    unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(nq0)>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Helmholtz1DKernelLauncher(
        [[maybe_unused]] const Templated1DSizeParameters<nm0, nq0> sizeParam1D,
        const unsigned int ncoord, const size_t nelmt,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Helmholtz1DSumFacTOPKernel<DEFORMED>(ncoord, nm0, nq0, nelmt, basis0, D0,
                                         w0, df, jac, coeff, in, out, lambda,
                                         shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Helmholtz2DKernelLauncher(
        const NonTemplated2DSizeParameters sizeParam2D,
        const unsigned int ncoord, const size_t nelmt, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Helmholtz2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, sizeParam2D.nm0(), sizeParam2D.nm1(), sizeParam2D.nmTot(),
        sizeParam2D.nq0(), sizeParam2D.nq1(), nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, in, out,
        lambda, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nmTot,
          unsigned int nq0, unsigned int nq1, typename TthreadBlock,
          typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Helmholtz2DKernelLauncher(
        [[maybe_unused]] const Templated2DSizeParameters<nm0, nm1, nmTot, nq0,
                                                         nq1>
            sizeParam2D,
        const unsigned int ncoord, const size_t nelmt, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT coeff,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData lambda,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Helmholtz2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nm0, nm1, nmTot, nq0, nq1, nelmt, isModified, index0, basis0,
        basis1, D0, D1, w0, w1, f0, f1, nodToMod, df, jac, coeff, in, out,
        lambda, shmemptr, threadBlock);
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    Helmholtz3DKernelLauncher(
        const NonTemplated3DSizeParameters sizeParam3D, const size_t nelmt,
        const bool isModified, const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Helmholtz3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        sizeParam3D.nm0(), sizeParam3D.nm1(), sizeParam3D.nm2(),
        sizeParam3D.nmTot(), sizeParam3D.nq0(), sizeParam3D.nq1(),
        sizeParam3D.nq2(), nelmt, isModified, index0, index1, index2, index3,
        basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2,
        nodToMod, df, jac, coeff, in, out, lambda, shmemptr, threadBlock);
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, unsigned int nm0, unsigned int nm1, unsigned int nm2,
          unsigned int nmTot, unsigned int nq0, unsigned int nq1,
          unsigned int nq2, typename TthreadBlock, typename TData,
          unsigned int maxThreadPerBlock = GetDeviceBlockSize<Implementation>(
              LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2))>
NEK_DEVICE_KERNEL
    typename std::enable_if<std::is_same_v<Implementation, SumFacTOP>>::type
    __LAUNCH_BOUNDS__(maxThreadPerBlock) Helmholtz3DKernelLauncher(
        [[maybe_unused]] const Templated3DSizeParameters<nm0, nm1, nm2, nmTot,
                                                         nq0, nq1, nq2>
            sizeParam3D,
        const size_t nelmt, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT f0,
        const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
        const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    Helmholtz3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
        nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nelmt, isModified, index0, index1,
        index2, index3, basis0, basis1, basis2, D0, D1, D2, w0, w1, w2, f0, f1,
        f1m, f2, nodToMod, df, jac, coeff, in, out, lambda, shmemptr,
        threadBlock);
}
#endif

} // namespace Nektar::Operators::detail
