///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzBlockOpDeviceSumFacTOPKernels.hpp
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

/**
 * @file HelmholtzBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device thread-only parallel (TOP) sum-factorised kernels and
 * launchers for the weak Helmholtz operator.
 *
 * @details
 * Thread-only parallel kernels for the weak Helmholtz operator: each
 * thread block handles one element with threads collaborating via shared
 * memory.
 */
#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacTOPKernels.hpp>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
/**
 * @brief Global workspace the TOP kernels need: none -- their
 * intermediates live in shared memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t HelmholtzWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/**
 * @brief Global workspace the TOP kernels need: none -- their
 * intermediates live in shared memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t HelmholtzWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/**
 * @brief Global workspace the TOP kernels need: none -- their
 * intermediates live in shared memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t HelmholtzWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int HelmholtzSharedMemorySize(
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0;

    return nDT + 4 * nq0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int HelmholtzSharedMemorySize(
    const TSizeParameter2D sizeParam2D)
{
    // Padding of the inner product's intermediate stage.
    const unsigned int wspPadSize =
        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam2D);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0 + nq1 * nq1;

    // Metric and backward transform, each 16-byte aligned.
    const unsigned int bwdSize = IProductWRTBaseAlignedSize<TData>(6) +
                                 IProductWRTBaseAlignedSize<TData>(nqTot);

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return nDT + bwdSize + nm0 * nq0 + nm1 * nq1 +
               IProductWRTBaseFusedDerivWorkSpaceSize<TData>(
                   sizeParam2D, 3u, std::max(nq0 * nm1, nm0 * nq1) + wspPadSize,
                   0u);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = sizeParam2D.nmTot();

        return nDT + bwdSize + nm0 * nq0 + nmTot * nq1 +
               IProductWRTBaseFusedDerivWorkSpaceSize<TData>(
                   sizeParam2D, 3u, nm0 * nq1 + wspPadSize, 0u);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int HelmholtzSharedMemorySize(
    const TSizeParameter3D sizeParam3D)
{
    // Padding of the inner product's intermediate stages.
    const unsigned int wspPadSize =
        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam3D);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0 + nq1 * nq1 + nq2 * nq2;

    // Metric and backward transform, each 16-byte aligned.
    const unsigned int bwdSize = IProductWRTBaseAlignedSize<TData>(9) +
                                 IProductWRTBaseAlignedSize<TData>(nqTot);

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return nDT + bwdSize + nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
               IProductWRTBaseFusedDerivWorkSpaceSize<TData>(
                   sizeParam3D, 3u,
                   std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2) + wspPadSize,
                   std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2) + wspPadSize);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

        return nDT + bwdSize + nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 +
               IProductWRTBaseFusedDerivWorkSpaceSize<TData>(
                   sizeParam3D, 3u, nm01 * nq2 + wspPadSize,
                   nm0 * nq1 * nq2 + wspPadSize);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;

        return nDT + bwdSize + nm0 * nq0 + nm1 * nq1 + nm02 * nq2 +
               IProductWRTBaseFusedDerivWorkSpaceSize<TData>(
                   sizeParam3D, 3u, nm0 * nm1 * nq2 + wspPadSize,
                   nm0 * nq1 * nq2 + wspPadSize);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();
        const unsigned int nmode2 =
            nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

        return nDT + bwdSize + nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 +
               IProductWRTBaseFusedDerivWorkSpaceSize<TData>(
                   sizeParam3D, 3u, nm0 * nm1 * nq2 + wspPadSize,
                   nm0 * nq1 * nq2 + wspPadSize);
    }
}

/**
 * @brief 1D TOP metric kernel applying diffusion tensor, metric terms and
 * lambda-weighted mass scaling.
 */
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

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

/**
 * @brief Quadrature weight times Jacobian at one point of a
 * two-dimensional element.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData QuadratureFactor2DSumFacTOP(
    const unsigned int i, const unsigned int j,
    [[maybe_unused]] const unsigned int idx, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT jac)
{
    TData tmpQ = w0[i] * w1[j];
    if constexpr (DEFORMED)
    {
        tmpQ *= jac[idx];
    }
    else
    {
        tmpQ *= jac[0];
    }
    return tmpQ;
}

/**
 * @brief Quadrature weight times Jacobian at one point of a
 * three-dimensional element.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData QuadratureFactor3DSumFacTOP(
    const unsigned int i, const unsigned int j, const unsigned int k,
    [[maybe_unused]] const unsigned int idx, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac)
{
    TData tmpQ = w0[i] * w1[j] * w2[k];
    if constexpr (DEFORMED)
    {
        tmpQ *= jac[idx];
    }
    else
    {
        tmpQ *= jac[0];
    }
    return tmpQ;
}

/**
 * @brief Element-wide part of the stiffness metric, two dimensions: with
 * derivative factors uniform over the element and a diffusion tensor,
 * form the metric once in the shared @p metric and synchronise the block;
 * otherwise the metric is formed per point by ApplyMetric2DPointSumFacTOP.
 *
 * @param   ncoord      Cartesian components (coordDim), 2 or 3.
 * @param   df          Derivative factors of this element.
 * @param   diffCoeff   Diffusion tensor, or null for the identity.
 * @param   metric      Metric of the element.
 * @param   threadBlock Thread-block handle supplying the barrier.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric2DSetupSumFacTOP(
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const TData *NEK_RESTRICT df,
    [[maybe_unused]] const TData *NEK_RESTRICT diffCoeff,
    [[maybe_unused]] TData *metric,
    [[maybe_unused]] const TthreadBlock &threadBlock)
{
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
}

/**
 * @brief Element-wide part of the stiffness metric, three dimensions; see
 * ApplyMetric2DSetupSumFacTOP.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric3DSetupSumFacTOP(
    [[maybe_unused]] const TData *NEK_RESTRICT df,
    [[maybe_unused]] const TData *NEK_RESTRICT diffCoeff,
    [[maybe_unused]] TData *metric,
    [[maybe_unused]] const TthreadBlock &threadBlock)
{
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
}

/**
 * @brief Stiffness flux at one point, two dimensions: turn the point's
 * Cartesian derivatives into the fluxes SumDerivTensor2DSumFacTOPKernel
 * contracts, through the diffusion tensor and derivative factors, the
 * quadrature weight and Jacobian, and the collapsed-coordinate factors of
 * a triangle.
 *
 * @tparam SHAPE_TYPE   Quad, Tri or NodalTri.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ncoord      Cartesian components (coordDim), 2 or 3.
 * @param   nqTot       Quadrature points of the element.
 * @param   i,j         The point's index in each direction.
 * @param   idx         The point's flat index.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   f0,f1       Collapsed-coordinate factors (triangles only).
 * @param   df          Derivative factors of this element.
 * @param   jac         Jacobian of this element.
 * @param   diffCoeff   Diffusion tensor, or null for the identity.
 * @param   metric      Metric of the element, formed here per point when
 *                      DEFORMED.
 * @param   deriv       The point's Cartesian derivatives.
 * @param   flux        The point's fluxes.
 *
 * @return The point's quadrature weight times Jacobian.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData ApplyMetric2DPointSumFacTOP(
    const unsigned int ncoord, const unsigned int nqTot, const unsigned int i,
    const unsigned int j, const unsigned int idx, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    TData *metric, const TData (&deriv)[3], TData (&flux)[2])
{
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
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
        TData tmp = deriv[0];
        sum1 += metric[0u] * tmp;
        sum2 += metric[1u] * tmp;
        tmp = deriv[1];
        sum1 += metric[2u] * tmp;
        sum2 += metric[3u] * tmp;
        if (ncoord == 3u)
        {
            tmp = deriv[2];
            sum1 += metric[4u] * tmp;
            sum2 += metric[5u] * tmp;
        }
    }
    else
    {
        TData tmp = deriv[0];
        sum1 += df[0u * dfsize + dfindex] * tmp;
        sum2 += df[1u * dfsize + dfindex] * tmp;
        tmp = deriv[1];
        sum1 += df[2u * dfsize + dfindex] * tmp;
        sum2 += df[3u * dfsize + dfindex] * tmp;
        if (ncoord == 3u)
        {
            tmp = deriv[2];
            sum1 += df[4u * dfsize + dfindex] * tmp;
            sum2 += df[5u * dfsize + dfindex] * tmp;
        }
    }

    const TData tmpQ =
        QuadratureFactor2DSumFacTOP<DEFORMED>(i, j, idx, w0, w1, jac);

    // Moving from standard to collapsed coordinates.
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        flux[0] = sum1 * tmpQ;
        flux[1] = sum2 * tmpQ;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        flux[0] = (sum1 + sum2 * f0[i]) * f1[j] * tmpQ;
        flux[1] = sum2 * tmpQ;
    }

    return tmpQ;
}

/**
 * @brief Stiffness flux at one point, three dimensions; see
 * ApplyMetric2DPointSumFacTOP.
 *
 * @tparam SHAPE_TYPE   Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 *
 * @return The point's quadrature weight times Jacobian.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData ApplyMetric3DPointSumFacTOP(
    const unsigned int nqTot, const unsigned int i, const unsigned int j,
    const unsigned int k, const unsigned int idx, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    TData *metric, const TData (&deriv)[3], TData (&flux)[3])
{
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
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
        TData tmp = deriv[0];
        sum1 += metric[0u] * tmp;
        sum2 += metric[1u] * tmp;
        sum3 += metric[2u] * tmp;
        tmp = deriv[1];
        sum1 += metric[3u] * tmp;
        sum2 += metric[4u] * tmp;
        sum3 += metric[5u] * tmp;
        tmp = deriv[2];
        sum1 += metric[6u] * tmp;
        sum2 += metric[7u] * tmp;
        sum3 += metric[8u] * tmp;
    }
    else
    {
        TData tmp = deriv[0];
        sum1 += df[0u * dfsize + dfindex] * tmp;
        sum2 += df[1u * dfsize + dfindex] * tmp;
        sum3 += df[2u * dfsize + dfindex] * tmp;
        tmp = deriv[1];
        sum1 += df[3u * dfsize + dfindex] * tmp;
        sum2 += df[4u * dfsize + dfindex] * tmp;
        sum3 += df[5u * dfsize + dfindex] * tmp;
        tmp = deriv[2];
        sum1 += df[6u * dfsize + dfindex] * tmp;
        sum2 += df[7u * dfsize + dfindex] * tmp;
        sum3 += df[8u * dfsize + dfindex] * tmp;
    }

    const TData tmpQ =
        QuadratureFactor3DSumFacTOP<DEFORMED>(i, j, k, idx, w0, w1, w2, jac);

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        flux[0] = sum1 * tmpQ;
        flux[1] = sum2 * tmpQ;
        flux[2] = sum3 * tmpQ;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        TData tmp = f2[k] * tmpQ;
        flux[0]   = (sum1 + (sum2 + sum3) * f0[i]) * f1m[j] * tmp;
        flux[1]   = (sum2 + sum3 * f1[j]) * tmp;
        flux[2]   = sum3 * tmpQ;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        flux[0] = (sum1 + sum3 * f0[i]) * f2[k] * tmpQ;
        flux[1] = sum2 * tmpQ;
        flux[2] = sum3 * tmpQ;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        TData tmp = f2[k] * tmpQ;
        flux[0]   = (sum1 + sum3 * f0[i]) * tmp;
        flux[1]   = (sum2 + sum3 * f1[j]) * tmp;
        flux[2]   = sum3 * tmpQ;
    }

    return tmpQ;
}

/**
 * @brief Fused stiffness stage, two dimensions: at each point the
 * Cartesian derivatives of @p in stay in registers and become the fluxes
 * @p out0 and @p out1 that SumDerivTensor2DSumFacTOPKernel contracts.
 *
 * Replaces PhysDeriv2DSumFacTOPKernel followed by a separate metric stage,
 * which staged the derivatives through shared memory and synchronised in
 * between. Ends with a block barrier.
 *
 * @tparam SHAPE_TYPE   Quad, Tri or NodalTri.
 * @tparam DEFORMED     Derivative factors vary point by point.
 * @tparam TthreadBlock Thread-block handle type.
 * @tparam TData        Floating-point type of the field data.
 *
 * @param   ncoord      Cartesian components (coordDim), 2 or 3.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   D0,D1       1D differentiation matrices, column-major.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   f0,f1       Collapsed-coordinate factors (triangles only).
 * @param   df          Derivative factors of this element.
 * @param   jac         Jacobian of this element.
 * @param   diffCoeff   Diffusion tensor, or null for the identity.
 * @param   in          Values of this element in shared memory.
 * @param   out0,out1   Fluxes in shared memory.
 * @param   metric      Metric of the element: shared when uniform, a
 *                      per-thread array when DEFORMED.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivApplyMetric2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1, TData *metric,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    ApplyMetric2DSetupSumFacTOP<DEFORMED>(ncoord, df, diffCoeff, metric,
                                          threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;

        TData deriv[3] = {};
        PhysDeriv2DPointSumFacTOP<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, i, j, idx, D0, D1, f0, f1, df, in, deriv);

        TData flux[2];
        ApplyMetric2DPointSumFacTOP<SHAPE_TYPE, DEFORMED>(
            ncoord, nqTot, i, j, idx, w0, w1, f0, f1, df, jac, diffCoeff,
            metric, deriv, flux);

        out0[idx] = flux[0];
        out1[idx] = flux[1];
    }

    localBarrier(threadBlock);
}

/**
 * @brief Fused stiffness stage, three dimensions: at each point the
 * Cartesian derivatives of @p in stay in registers and become the fluxes
 * @p out0, @p out1 and @p out2 that SumDerivTensor3DSumFacTOPKernel
 * contracts; see PhysDerivApplyMetric2DSumFacTOPKernel. Ends with a block
 * barrier.
 *
 * @tparam SHAPE_TYPE   Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysDerivApplyMetric3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out0, TData *NEK_RESTRICT out1,
    TData *NEK_RESTRICT out2, TData *metric, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    ApplyMetric3DSetupSumFacTOP<DEFORMED>(df, diffCoeff, metric, threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = (idx / nq0) % nq1;
        const unsigned int k = idx / (nq0 * nq1);

        TData deriv[3];
        PhysDeriv3DPointSumFacTOP<SHAPE_TYPE, DEFORMED>(nq0, nq1, nq2, i, j, k,
                                                        idx, D0, D1, D2, f0, f1,
                                                        f1m, f2, df, in, deriv);

        TData flux[3];
        ApplyMetric3DPointSumFacTOP<SHAPE_TYPE, DEFORMED>(
            nqTot, i, j, k, idx, w0, w1, w2, f0, f1, f1m, f2, df, jac,
            diffCoeff, metric, deriv, flux);

        out0[idx] = flux[0];
        out1[idx] = flux[1];
        out2[idx] = flux[2];
    }

    localBarrier(threadBlock);
}

/**
 * @brief Mass term of the fused Helmholtz stage, two dimensions: scale
 * @p bwd at each point by @p lambda times the point's quadrature weight and
 * Jacobian.
 *
 * Runs after PhysDerivApplyMetric2DSumFacTOPKernel, whose closing barrier
 * ends every read of @p bwd for the derivatives. Closes without a barrier:
 * SumDerivTensor2DSumFacTOPKernel, which follows, updates each point of
 * @p bwd from the thread that scaled it.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ScaleMass2DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData lambda, TData *NEK_RESTRICT bwd,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq1; idx += stride)
    {
        bwd[idx] *= lambda * QuadratureFactor2DSumFacTOP<DEFORMED>(
                                 idx % nq0, idx / nq0, idx, w0, w1, jac);
    }
}

/**
 * @brief Mass term of the fused Helmholtz stage, three dimensions; see
 * ScaleMass2DSumFacTOPKernel.
 */
template <bool DEFORMED, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ScaleMass3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT jac,
    const TData lambda, TData *NEK_RESTRICT bwd,
    const TthreadBlock &threadBlock)
{
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nq0 * nq1 * nq2; idx += stride)
    {
        bwd[idx] *= lambda * QuadratureFactor3DSumFacTOP<DEFORMED>(
                                 idx % nq0, (idx / nq0) % nq1,
                                 idx / (nq0 * nq1), idx, w0, w1, w2, jac);
    }
}

/**
 * @brief 1D TOP launcher stringing together the five sum-factorised stages.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    HelmholtzKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT coeff, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData lambda, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int ndf     = ncoord;
    const unsigned int dfsize  = DEFORMED ? nq0 : 1u;
    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    TData *bwd   = (TData *)shmemptr;
    TData *deriv = bwd + nq0;
    TData *s_D0T = bwd + 4 * nq0;

    // Copy to shared memory.
    StageTransposeSumFacTOP(nq0, D0, s_D0T, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nm0 * nelmt * c + nm0 * e;
        TData *outptr       = out + nm0 * nelmt * c + nm0 * e;

        BwdTransSegSumFacTOPKernel<false>(nm0, nq0, basis0, inptr, bwd,
                                          threadBlock);
        PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, D0, dfptr, bwd,
                                             deriv, threadBlock);
        ApplyMetric1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, w0, dfptr,
                                               jacptr, coeff, deriv, deriv, bwd,
                                               lambda, threadBlock);
        SumDerivTensor1DSumFacTOPKernel<false, true>(nq0, s_D0T, deriv, bwd,
                                                     (TData)1.0, threadBlock);
        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, bwd, outptr, (TData)1.0, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief 2D TOP launcher stringing together the five sum-factorised stages.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    HelmholtzKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const bool isModified,
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
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int wsp0Size = 0, nmode0 = 0, nmode1 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wsp0Size = std::max(nm0 * nq1, nm1 * nq0);
        nmode0   = nm0;
        nmode1   = nm1;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        wsp0Size = nm0 * nq1;
        nmode0   = nm0;
        nmode1   = nmTot;
    }

    // Rows of the inner product's input and intermediate stages are padded
    // to IProductWRTBaseRowStride and IProductWRTBaseStageStride, sized by
    // IProductWRTBaseFusedDerivWorkSpaceSize, and every region starts
    // 16-byte aligned.
    // The padded input and the intermediate stages reuse the derivative
    // slots, past the modal values of the nodal shapes.
    const unsigned int nmTotAl = IProductWRTBaseAlignedSize<TData>(nmTot);
    wsp0Size += IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam2D);
    const unsigned int derivWspSize =
        IProductWRTBaseFusedDerivWorkSpaceSize<TData>(sizeParam2D, ncoord,
                                                      wsp0Size, 0u);

    TData *metric = (TData *)shmemptr;
    TData *bwd    = metric + IProductWRTBaseAlignedSize<TData>(6);
    TData *deriv  = bwd + IProductWRTBaseAlignedSize<TData>(nqTot);
    TData *tmp    = deriv;
    TData *deriv0 = deriv;
    TData *deriv1 = deriv0 + nqTot;
    TData *s_ipb  = deriv + nmTotAl;
    TData *s_wsp0 =
        deriv + IProductWRTBaseFusedWorkSpaceOffset<TData>(sizeParam2D);
    TData *s_basis0 = deriv + derivWspSize;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_D0T    = s_basis1 + nmode1 * nq1;
    TData *s_D1T    = s_D0T + nq0 * nq0;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    for (unsigned int idx = idx0; idx < nmode0 * nq0; idx += stride)
    {
        s_basis0[idx] = basis0[idx];
    }

    for (unsigned int idx = idx0; idx < nmode1 * nq1; idx += stride)
    {
        s_basis1[idx] = basis1[idx];
    }

    StageTransposeSumFacTOP(nq0, D0, s_D0T, threadBlock);
    StageTransposeSumFacTOP(nq1, D1, s_D1T, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

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
            BwdTransQuadSumFacTOPKernel<false, true>(nm0, nm1, nq0, nq1, nqTot,
                                                     s_basis0, s_basis1, tmp,
                                                     bwd, s_wsp0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            BwdTransTriSumFacTOPKernel<false>(nm0, nm1, nq0, nq1, nqTot,
                                              isModified, s_basis0, s_basis1,
                                              tmp, bwd, s_wsp0, threadBlock);
        }

        if constexpr (DEFORMED)
        {
            TData dmetric[6] = {0.0};
            PhysDerivApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, D0, D1, w0, w1, f0, f1, dfptr, jacptr, coeff,
                bwd, deriv0, deriv1, dmetric, threadBlock);
        }
        else
        {
            PhysDerivApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                ncoord, nq0, nq1, D0, D1, w0, w1, f0, f1, dfptr, jacptr, coeff,
                bwd, deriv0, deriv1, metric, threadBlock);
        }
        ScaleMass2DSumFacTOPKernel<DEFORMED>(nq0, nq1, w0, w1, jacptr, lambda,
                                             bwd, threadBlock);
        SumDerivTensor2DSumFacTOPKernel<false, true>(nq0, nq1, s_D0T, s_D1T,
                                                     deriv0, deriv1, bwd,
                                                     (TData)1.0, threadBlock);
        const TData *ipb =
            IProductWRTBasePadInput(nq0, nq1, bwd, s_ipb, threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            IProductWRTBaseQuadSumFacTOPKernel<false, false, true>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, s_basis0, s_basis1, ipb,
                outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false, true>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, ipb, outptr, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            IProductWRTBaseTriSumFacTOPKernel<false, false, true>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, ipb, tmp, s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief 3D TOP launcher stringing together the five sum-factorised stages.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    HelmholtzKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        [[maybe_unused]] const unsigned int ncoord, const size_t nelmt,
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
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int wsp0Size = 0, wsp1Size = 0, nmode0 = 0, nmode1 = 0, nmode2 = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wsp0Size = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
        wsp1Size = std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2);
        nmode0   = nm0;
        nmode1   = nm1;
        nmode2   = nm2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        wsp0Size = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
        wsp1Size = nm0 * nq1 * nq2;
        nmode0   = nm0;
        nmode1   = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        nmode2   = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        wsp0Size = nm0 * nm1 * nq2;
        wsp1Size = nm0 * nq1 * nq2;
        nmode0   = nm0;
        nmode1   = nm1;
        nmode2   = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        wsp0Size = nm0 * nm1 * nq2;
        wsp1Size = nm0 * nq1 * nq2;
        nmode0   = nm0;
        nmode1   = nm1;
        nmode2   = nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
    }

    // Rows of the inner product's input and intermediate stages are padded
    // to IProductWRTBaseRowStride and IProductWRTBaseStageStride, sized by
    // IProductWRTBaseFusedDerivWorkSpaceSize, and every region starts
    // 16-byte aligned.
    // The padded input and the intermediate stages reuse the derivative
    // slots, past the modal values of the nodal shapes.
    const unsigned int nmTotAl = IProductWRTBaseAlignedSize<TData>(nmTot);
    const unsigned int wspPadSize =
        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam3D);
    wsp0Size += wspPadSize;
    wsp1Size += wspPadSize;
    const unsigned int derivWspSize =
        IProductWRTBaseFusedDerivWorkSpaceSize<TData>(sizeParam3D, 3u, wsp0Size,
                                                      wsp1Size);

    TData *metric = (TData *)shmemptr;
    TData *bwd    = metric + IProductWRTBaseAlignedSize<TData>(9);
    TData *deriv  = bwd + IProductWRTBaseAlignedSize<TData>(nqTot);
    TData *tmp    = deriv;
    TData *deriv0 = deriv;
    TData *deriv1 = deriv0 + nqTot;
    TData *deriv2 = deriv1 + nqTot;
    TData *s_ipb  = deriv + nmTotAl;
    TData *s_wsp0 =
        deriv + IProductWRTBaseFusedWorkSpaceOffset<TData>(sizeParam3D);
    TData *s_wsp1   = s_wsp0 + IProductWRTBaseAlignedSize<TData>(wsp0Size);
    TData *s_basis0 = deriv + derivWspSize;
    TData *s_basis1 = s_basis0 + nmode0 * nq0;
    TData *s_basis2 = s_basis1 + nmode1 * nq1;
    TData *s_D0T    = s_basis2 + nmode2 * nq2;
    TData *s_D1T    = s_D0T + nq0 * nq0;
    TData *s_D2T    = s_D1T + nq1 * nq1;

    // Copy to shared memory.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

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

    StageTransposeSumFacTOP(nq0, D0, s_D0T, threadBlock);
    StageTransposeSumFacTOP(nq1, D1, s_D1T, threadBlock);
    StageTransposeSumFacTOP(nq2, D2, s_D2T, threadBlock);

    size_t e = getBlockIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const TData *dfptr  = df + ndf * dfsize * e;
        const TData *jacptr = jac + jacsize * e;
        const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
        TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

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
            BwdTransHexSumFacTOPKernel<false, true>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            BwdTransTetSumFacTOPKernel<false>(nm0, nm1, nm2, nq0, nq1, nq2,
                                              nqTot, isModified, index0, index3,
                                              s_basis0, s_basis1, s_basis2, tmp,
                                              bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            BwdTransPrismSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            BwdTransPyrSumFacTOPKernel<false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, s_basis0,
                s_basis1, s_basis2, tmp, bwd, s_wsp0, s_wsp1, threadBlock);
        }
        if constexpr (DEFORMED)
        {
            TData dmetric[9] = {0.0};
            PhysDerivApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, coeff, bwd, deriv0, deriv1, deriv2, dmetric,
                threadBlock);
        }
        else
        {
            PhysDerivApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED>(
                nq0, nq1, nq2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, coeff, bwd, deriv0, deriv1, deriv2, metric,
                threadBlock);
        }
        ScaleMass3DSumFacTOPKernel<DEFORMED>(nq0, nq1, nq2, w0, w1, w2, jacptr,
                                             lambda, bwd, threadBlock);
        SumDerivTensor3DSumFacTOPKernel<false, true>(
            nq0, nq1, nq2, s_D0T, s_D1T, s_D2T, deriv0, deriv1, deriv2, bwd,
            (TData)1.0, threadBlock);
        const TData *ipb =
            IProductWRTBasePadInput(nq0, nq1 * nq2, bwd, s_ipb, threadBlock);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            IProductWRTBaseHexSumFacTOPKernel<false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, s_basis0, s_basis1,
                s_basis2, ipb, outptr, s_wsp0, s_wsp1, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {
            IProductWRTBaseTetSumFacTOPKernel<false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb, outptr,
                s_wsp1, s_wsp0, (TData)1.0, threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            IProductWRTBasePrismSumFacTOPKernel<false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb, tmp, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, tmp, outptr,
                                               threadBlock);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            IProductWRTBasePyrSumFacTOPKernel<false, false, true>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, s_basis0, s_basis1, s_basis2, ipb, outptr, s_wsp1,
                s_wsp0, (TData)1.0, threadBlock);
        }

        e += getBlockRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
