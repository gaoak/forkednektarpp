///////////////////////////////////////////////////////////////////////////////
//
// File: LaplacianBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: Device thread-only parallel (TOP) sum-factorised kernels
// and launchers for the weak Laplacian operator.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file LaplacianBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device thread-only parallel (TOP) sum-factorised kernels and
 * launchers for the weak Laplacian operator.
 *
 * @details
 * Thread-only parallel kernels for the weak Laplacian operator: each
 * thread block handles one element, or a group of them, at a time with
 * threads collaborating via shared memory.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivBlockOpDeviceSumFacTOPKernels.hpp>

// Coefficients are passed as raw pointers. Variable tensor components are
// separated by varDiffCoeffSize scalars. The kernels select the plane and
// element from the base pointer; point offsets follow the same interleaving
// as the input and derivative-factor arrays.

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
/**
 * @brief Values the derivative and workspace region of a 2D or 3D
 * SumFacTOP Laplacian launcher takes in global memory: as
 * LaplacianSharedDerivWorkSpaceSize, packed, with the nmTot modal values
 * and the two unpadded intermediate-stage regions of @p wsp0Size and
 * @p wsp1Size values.
 */
template <typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int LaplacianDerivWorkSpaceSize(
    const TSizeParameter sizeParam, const unsigned int nslots,
    const unsigned int wsp0Size, const unsigned int wsp1Size)
{
    const unsigned int derivSize = nslots * sizeParam.nqTot();
    const unsigned int wspSize   = sizeParam.nmTot() + wsp0Size + wsp1Size;

    return derivSize > wspSize ? derivSize : wspSize;
}

/**
 * @brief Global-memory workspace of the TOP kernels, in TData values: one
 * staging area per element of a thread block's group, the regions following
 * one another.
 *
 * In 1D the nq0 physical values and three derivative slots.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t LaplacianWorkSpaceSize(
    const size_t nelmt, [[maybe_unused]] const unsigned int ncoord,
    const TSizeParameter1D sizeParam1D)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    return 4 * nq0 * nelmt;
}

/**
 * @brief Global-memory workspace of the TOP kernels; see the 1D overload.
 * The metric, the backward transform and the fused derivative and
 * inner-product region, packed and unpadded.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t LaplacianWorkSpaceSize(
    const size_t nelmt, [[maybe_unused]] const unsigned int ncoord,
    const TSizeParameter2D sizeParam2D)
{
    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = sizeParam2D.nqTot();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return (6 + nqTot +
                LaplacianDerivWorkSpaceSize(
                    sizeParam2D, 3u, std::max(nq0 * nm1, nm0 * nq1), 0u)) *
               nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return (6 + nqTot +
                LaplacianDerivWorkSpaceSize(sizeParam2D, 3u, nm0 * nq1, 0u)) *
               nelmt;
    }
}

/**
 * @brief Global-memory workspace of the TOP kernels; see the 2D overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr size_t LaplacianWorkSpaceSize(
    const size_t nelmt, [[maybe_unused]] const unsigned int ncoord,
    const TSizeParameter3D sizeParam3D)
{
    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = sizeParam3D.nqTot();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return (9 + nqTot +
                LaplacianDerivWorkSpaceSize(
                    sizeParam3D, 3u, std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2),
                    std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2))) *
               nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

        return (9 + nqTot +
                LaplacianDerivWorkSpaceSize(sizeParam3D, 3u, nm01 * nq2,
                                            nm0 * nq1 * nq2)) *
               nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism ||
                       SHAPE_TYPE == LibUtilities::Pyr ||
                       SHAPE_TYPE == LibUtilities::NodalPyr)
    {
        return (9 + nqTot +
                LaplacianDerivWorkSpaceSize(sizeParam3D, 3u, nm0 * nm1 * nq2,
                                            nm0 * nq1 * nq2)) *
               nelmt;
    }
}

/**
 * @brief Values the derivative and workspace region of a 2D or 3D
 * SumFacTOP Laplacian launcher takes in shared memory, rounded to
 * SharedMemoryAlignedSize: @p nslots derivative slots of nqTot values, or,
 * if more,
 * IProductWRTBaseFusedWorkSpaceOffset and the two intermediate-stage regions
 * of @p wsp0Size and @p wsp1Size values, padding included, the first 16-byte
 * aligned.
 */
template <typename TData, typename TSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int LaplacianSharedDerivWorkSpaceSize(
    const TSizeParameter sizeParam, const unsigned int nslots,
    const unsigned int wsp0Size, const unsigned int wsp1Size)
{
    const unsigned int derivSize = nslots * sizeParam.nqTot();
    const unsigned int wspSize =
        IProductWRTBaseFusedWorkSpaceOffset<TData>(sizeParam) +
        SharedMemoryAlignedSize<TData>(wsp0Size) + wsp1Size;

    return SharedMemoryAlignedSize<TData>(derivSize > wspSize ? derivSize
                                                              : wspSize);
}

/**
 * @brief One element's workspace of the TOP kernels in shared memory, in
 * TData values: as LaplacianWorkSpaceSize, with each region 16-byte aligned and
 * the inner product's rows and stages padded.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int LaplacianSharedWorkSpaceSize(
    const TSizeParameter2D sizeParam2D)
{
    // Padding of the inner product's intermediate stage.
    const unsigned int wspPadSize =
        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam2D);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = sizeParam2D.nqTot();

    // Metric and backward transform, each 16-byte aligned.
    const unsigned int bwdSize = SharedMemoryAlignedSize<TData>(6) +
                                 SharedMemoryAlignedSize<TData>(nqTot);

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return bwdSize + LaplacianSharedDerivWorkSpaceSize<TData>(
                             sizeParam2D, 3u,
                             std::max(nq0 * nm1, nm0 * nq1) + wspPadSize, 0u);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        return bwdSize + LaplacianSharedDerivWorkSpaceSize<TData>(
                             sizeParam2D, 3u, nm0 * nq1 + wspPadSize, 0u);
    }
}

/// See the 2D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
NEK_HOSTDEVICE_INLINE constexpr unsigned int LaplacianSharedWorkSpaceSize(
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
    const unsigned int nqTot = sizeParam3D.nqTot();

    // Metric and backward transform, each 16-byte aligned.
    const unsigned int bwdSize = SharedMemoryAlignedSize<TData>(9) +
                                 SharedMemoryAlignedSize<TData>(nqTot);

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return bwdSize +
               LaplacianSharedDerivWorkSpaceSize<TData>(
                   sizeParam3D, 3u,
                   std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2) + wspPadSize,
                   std::max(nq0 * nq1 * nm2, nm0 * nm1 * nq2) + wspPadSize);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

        return bwdSize + LaplacianSharedDerivWorkSpaceSize<TData>(
                             sizeParam3D, 3u, nm01 * nq2 + wspPadSize,
                             nm0 * nq1 * nq2 + wspPadSize);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism ||
                       SHAPE_TYPE == LibUtilities::Pyr ||
                       SHAPE_TYPE == LibUtilities::NodalPyr)
    {
        return bwdSize + LaplacianSharedDerivWorkSpaceSize<TData>(
                             sizeParam3D, 3u, nm0 * nm1 * nq2 + wspPadSize,
                             nm0 * nq1 * nq2 + wspPadSize);
    }
}

/**
 * @brief Dynamic shared memory of the TOP kernels, in TData values: the
 * transposed differentiation matrices and basis tables, 16-byte aligned
 * in 2D and 3D, then, with @p useSharedMemoryWsp, the workspace of
 * @p elmtsPerBlock elements: LaplacianWorkSpaceSize in 1D,
 * LaplacianSharedWorkSpaceSize per element in 2D and 3D.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int LaplacianSharedMemorySize(
    const unsigned int ncoord, const TSizeParameter1D sizeParam1D,
    const unsigned int elmtsPerBlock, const bool useSharedMemoryWsp)
{
    const unsigned int nq0 = sizeParam1D.nq0();

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0;

    const unsigned int wspSize =
        useSharedMemoryWsp
            ? static_cast<unsigned int>(
                  LaplacianWorkSpaceSize<SHAPE_TYPE, Implementation>(
                      elmtsPerBlock, ncoord, sizeParam1D))
            : 0u;

    return nDT + wspSize;
}

/// See the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int LaplacianSharedMemorySize(
    [[maybe_unused]] const unsigned int ncoord,
    const TSizeParameter2D sizeParam2D, const unsigned int elmtsPerBlock,
    const bool useSharedMemoryWsp)
{
    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0 + nq1 * nq1;

    const unsigned int wspSize =
        useSharedMemoryWsp
            ? elmtsPerBlock *
                  LaplacianSharedWorkSpaceSize<SHAPE_TYPE, Implementation,
                                               TData>(sizeParam2D)
            : 0u;

    return SharedMemoryAlignedSize<TData>(
               nDT + BasisDataSize2D<SHAPE_TYPE>(sizeParam2D).total()) +
           wspSize;
}

/// See the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TData, typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int LaplacianSharedMemorySize(
    [[maybe_unused]] const unsigned int ncoord,
    const TSizeParameter3D sizeParam3D, const unsigned int elmtsPerBlock,
    const bool useSharedMemoryWsp)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Transposed differentiation matrices for SumDerivTensor.
    const unsigned int nDT = nq0 * nq0 + nq1 * nq1 + nq2 * nq2;

    const unsigned int wspSize =
        useSharedMemoryWsp
            ? elmtsPerBlock *
                  LaplacianSharedWorkSpaceSize<SHAPE_TYPE, Implementation,
                                               TData>(sizeParam3D)
            : 0u;

    return SharedMemoryAlignedSize<TData>(
               nDT + BasisDataSize3D<SHAPE_TYPE>(sizeParam3D).total()) +
           wspSize;
}

/**
 * @brief 1D TOP metric kernel applying diffusion tensor and metric terms.
 */
template <bool DEFORMED, bool VARCOEFF, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void ApplyMetric1DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const unsigned int inoffset, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const size_t varDiffCoeffSize,
    const TData *NEK_RESTRICT in, TData *out, const TthreadBlock &threadBlock)
{
    const size_t diffCoeffStride = VARCOEFF ? varDiffCoeffSize : 1;
    const unsigned int dfsize    = DEFORMED ? nq0 : 1u;

    TData metric[3] = {0.0};
    if constexpr (!DEFORMED && !VARCOEFF)
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

        if constexpr (DEFORMED || VARCOEFF)
        {
            const size_t diffCoeffIndex = VARCOEFF ? i : 0;
            if (diffCoeff)
            {
                if (ncoord == 1)
                {
                    metric[0] = diffCoeff[diffCoeffIndex] * df[dfindex];
                }
                else if (ncoord == 2)
                {
                    metric[0] = diffCoeff[diffCoeffIndex] * df[dfindex] +
                                diffCoeff[diffCoeffStride + diffCoeffIndex] *
                                    df[dfsize + dfindex];
                    metric[1] =
                        diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[dfindex] +
                        diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                            df[dfsize + dfindex];
                }
                else if (ncoord == 3)
                {
                    metric[0] =
                        diffCoeff[diffCoeffIndex] * df[dfindex] +
                        diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[dfsize + dfindex] +
                        diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[2 * dfsize + dfindex];
                    metric[1] =
                        diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[dfindex] +
                        diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                            df[dfsize + dfindex] +
                        diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                            df[2 * dfsize + dfindex];
                    metric[2] =
                        diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[dfindex] +
                        diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                            df[dfsize + dfindex] +
                        diffCoeff[5 * diffCoeffStride + diffCoeffIndex] *
                            df[2 * dfsize + dfindex];
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
            out[i] = sum * jac[i] * w0[i];
        }
        else
        {
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
NEK_DEVICE_INLINE static TData LaplacianQuadratureFactor2DSumFacTOP(
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
NEK_DEVICE_INLINE static TData LaplacianQuadratureFactor3DSumFacTOP(
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
 * otherwise the metric is formed per point by
 * LaplacianApplyMetric2DPointSumFacTOP.
 *
 * @param   ncoord      Cartesian components (coordDim), 2 or 3.
 * @param   df          Derivative factors of this element.
 * @param   diffCoeff   Constant packed tensor, or this element's variable
 *                      tensor (VARCOEFF); null for the identity.
 * @param   metric      Metric of the element.
 * @param   threadBlock Thread-block handle supplying the barrier.
 */
template <bool DEFORMED, bool VARCOEFF, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LaplacianApplyMetric2DSetupSumFacTOP(
    [[maybe_unused]] const unsigned int ncoord,
    [[maybe_unused]] const TData *NEK_RESTRICT df,
    [[maybe_unused]] const TData *NEK_RESTRICT diffCoeff,
    [[maybe_unused]] TData *metric,
    [[maybe_unused]] const TthreadBlock &threadBlock)
{
    if constexpr (!DEFORMED && !VARCOEFF)
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
 * LaplacianApplyMetric2DSetupSumFacTOP.
 */
template <bool DEFORMED, bool VARCOEFF, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LaplacianApplyMetric3DSetupSumFacTOP(
    [[maybe_unused]] const TData *NEK_RESTRICT df,
    [[maybe_unused]] const TData *NEK_RESTRICT diffCoeff,
    [[maybe_unused]] TData *metric,
    [[maybe_unused]] const TthreadBlock &threadBlock)
{
    if constexpr (!DEFORMED && !VARCOEFF)
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
 * @param   diffCoeff   Constant packed tensor, or this element's variable
 *                      tensor (VARCOEFF); null for the identity.
 * @param   metric      Metric of the element, formed here per point when
 *                      DEFORMED.
 * @param   deriv       The point's Cartesian derivatives.
 * @param   flux        The point's fluxes.
 *
 * @return The point's quadrature weight times Jacobian.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool VARCOEFF,
          typename TData>
NEK_DEVICE_INLINE static TData LaplacianApplyMetric2DPointSumFacTOP(
    const unsigned int ncoord, const unsigned int nqTot, const unsigned int i,
    const unsigned int j, const unsigned int idx, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    const size_t varDiffCoeffSize, TData *metric, const TData (&deriv)[3],
    TData (&flux)[2])
{
    const size_t diffCoeffStride = VARCOEFF ? varDiffCoeffSize : 1;
    const unsigned int dfsize    = DEFORMED ? nqTot : 1u;
    const unsigned int dfindex   = DEFORMED ? idx : 0;

    if constexpr (DEFORMED || VARCOEFF)
    {
        const size_t diffCoeffIndex = VARCOEFF ? idx : 0;
        if (diffCoeff)
        {
            if (ncoord == 2)
            {
                metric[0] = diffCoeff[diffCoeffIndex] * df[dfindex] +
                            diffCoeff[diffCoeffStride + diffCoeffIndex] *
                                df[2 * dfsize + dfindex];
                metric[1] =
                    diffCoeff[diffCoeffIndex] * df[1 * dfsize + dfindex] +
                    diffCoeff[diffCoeffStride + diffCoeffIndex] *
                        df[3 * dfsize + dfindex];
                metric[2] =
                    diffCoeff[diffCoeffStride + diffCoeffIndex] * df[dfindex] +
                    diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                        df[2 * dfsize + dfindex];
                metric[3] = diffCoeff[diffCoeffStride + diffCoeffIndex] *
                                df[1 * dfsize + dfindex] +
                            diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                                df[3 * dfsize + dfindex];
            }
            else if (ncoord == 3)
            {
                metric[0] = diffCoeff[diffCoeffIndex] * df[dfindex] +
                            diffCoeff[diffCoeffStride + diffCoeffIndex] *
                                df[2 * dfsize + dfindex] +
                            diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                                df[4 * dfsize + dfindex];
                metric[1] =
                    diffCoeff[diffCoeffIndex] * df[1 * dfsize + dfindex] +
                    diffCoeff[diffCoeffStride + diffCoeffIndex] *
                        df[3 * dfsize + dfindex] +
                    diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                        df[5 * dfsize + dfindex];
                metric[2] =
                    diffCoeff[diffCoeffStride + diffCoeffIndex] * df[dfindex] +
                    diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                        df[2 * dfsize + dfindex] +
                    diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                        df[4 * dfsize + dfindex];
                metric[3] = diffCoeff[diffCoeffStride + diffCoeffIndex] *
                                df[1 * dfsize + dfindex] +
                            diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                                df[3 * dfsize + dfindex] +
                            diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                                df[5 * dfsize + dfindex];
                metric[4] = diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                                df[dfindex] +
                            diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                                df[2 * dfsize + dfindex] +
                            diffCoeff[5 * diffCoeffStride + diffCoeffIndex] *
                                df[4 * dfsize + dfindex];
                metric[5] = diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                                df[1 * dfsize + dfindex] +
                            diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                                df[3 * dfsize + dfindex] +
                            diffCoeff[5 * diffCoeffStride + diffCoeffIndex] *
                                df[5 * dfsize + dfindex];
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
        LaplacianQuadratureFactor2DSumFacTOP<DEFORMED>(i, j, idx, w0, w1, jac);

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
 * LaplacianApplyMetric2DPointSumFacTOP.
 *
 * @tparam SHAPE_TYPE   Hex, Tet, NodalTet, Prism, NodalPrism, Pyr or NodalPyr.
 *
 * @return The point's quadrature weight times Jacobian.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool VARCOEFF,
          typename TData>
NEK_DEVICE_INLINE static TData LaplacianApplyMetric3DPointSumFacTOP(
    const unsigned int nqTot, const unsigned int i, const unsigned int j,
    const unsigned int k, const unsigned int idx, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
    const size_t varDiffCoeffSize, TData *metric, const TData (&deriv)[3],
    TData (&flux)[3])
{
    const size_t diffCoeffStride = VARCOEFF ? varDiffCoeffSize : 1;
    const unsigned int dfsize    = DEFORMED ? nqTot : 1u;
    const unsigned int dfindex   = DEFORMED ? idx : 0;

    if constexpr (DEFORMED || VARCOEFF)
    {
        const size_t diffCoeffIndex = VARCOEFF ? idx : 0;
        if (diffCoeff)
        {
            metric[0] = diffCoeff[diffCoeffIndex] * df[dfindex] +
                        diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[3 * dfsize + dfindex] +
                        diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[6 * dfsize + dfindex];
            metric[1] = diffCoeff[diffCoeffIndex] * df[1 * dfsize + dfindex] +
                        diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[4 * dfsize + dfindex] +
                        diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[7 * dfsize + dfindex];
            metric[2] = diffCoeff[diffCoeffIndex] * df[2 * dfsize + dfindex] +
                        diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[5 * dfsize + dfindex] +
                        diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[8 * dfsize + dfindex];
            metric[3] =
                diffCoeff[diffCoeffStride + diffCoeffIndex] * df[dfindex] +
                diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                    df[3 * dfsize + dfindex] +
                diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                    df[6 * dfsize + dfindex];
            metric[4] = diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[1 * dfsize + dfindex] +
                        diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                            df[4 * dfsize + dfindex] +
                        diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                            df[7 * dfsize + dfindex];
            metric[5] = diffCoeff[diffCoeffStride + diffCoeffIndex] *
                            df[2 * dfsize + dfindex] +
                        diffCoeff[2 * diffCoeffStride + diffCoeffIndex] *
                            df[5 * dfsize + dfindex] +
                        diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                            df[8 * dfsize + dfindex];
            metric[6] =
                diffCoeff[3 * diffCoeffStride + diffCoeffIndex] * df[dfindex] +
                diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                    df[3 * dfsize + dfindex] +
                diffCoeff[5 * diffCoeffStride + diffCoeffIndex] *
                    df[6 * dfsize + dfindex];
            metric[7] = diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[1 * dfsize + dfindex] +
                        diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                            df[4 * dfsize + dfindex] +
                        diffCoeff[5 * diffCoeffStride + diffCoeffIndex] *
                            df[7 * dfsize + dfindex];
            metric[8] = diffCoeff[3 * diffCoeffStride + diffCoeffIndex] *
                            df[2 * dfsize + dfindex] +
                        diffCoeff[4 * diffCoeffStride + diffCoeffIndex] *
                            df[5 * dfsize + dfindex] +
                        diffCoeff[5 * diffCoeffStride + diffCoeffIndex] *
                            df[8 * dfsize + dfindex];
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

    const TData tmpQ = LaplacianQuadratureFactor3DSumFacTOP<DEFORMED>(
        i, j, k, idx, w0, w1, w2, jac);

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
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr ||
                       SHAPE_TYPE == LibUtilities::NodalPyr)
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
 * @param   diffCoeff   Constant packed tensor, or this element's variable
 *                      tensor (VARCOEFF); null for the identity.
 * @param   in          Values of this element in shared memory.
 * @param   out0,out1   Fluxes in shared memory.
 * @param   metric      Metric of the element: shared when uniform, a
 *                      per-thread array when DEFORMED.
 * @param   threadBlock Thread-block handle.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool VARCOEFF,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LaplacianPhysDerivApplyMetric2DSumFacTOPKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const size_t varDiffCoeffSize,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *metric, const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    LaplacianApplyMetric2DSetupSumFacTOP<DEFORMED, VARCOEFF>(
        ncoord, df, diffCoeff, metric, threadBlock);

    for (unsigned int idx = idx0; idx < nqTot; idx += stride)
    {
        const unsigned int i = idx % nq0;
        const unsigned int j = idx / nq0;

        TData deriv[3] = {};
        PhysDeriv2DPointSumFacTOP<SHAPE_TYPE, DEFORMED>(
            ncoord, nq0, nq1, i, j, idx, D0, D1, f0, f1, df, in, deriv);

        TData flux[2];
        LaplacianApplyMetric2DPointSumFacTOP<SHAPE_TYPE, DEFORMED, VARCOEFF>(
            ncoord, nqTot, i, j, idx, w0, w1, f0, f1, df, jac, diffCoeff,
            varDiffCoeffSize, metric, deriv, flux);

        out0[idx] = flux[0];
        out1[idx] = flux[1];
    }

    localBarrier(threadBlock);
}

/**
 * @brief Fused stiffness stage, three dimensions: at each point the
 * Cartesian derivatives of @p in stay in registers and become the fluxes
 * @p out0, @p out1 and @p out2 that SumDerivTensor3DSumFacTOPKernel
 * contracts; see LaplacianPhysDerivApplyMetric2DSumFacTOPKernel. Ends with a
 * block barrier.
 *
 * @tparam SHAPE_TYPE   Hex, Tet, NodalTet, Prism, NodalPrism, Pyr or NodalPyr.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, bool VARCOEFF,
          typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void LaplacianPhysDerivApplyMetric3DSumFacTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT diffCoeff, const size_t varDiffCoeffSize,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out0,
    TData *NEK_RESTRICT out1, TData *NEK_RESTRICT out2, TData *metric,
    const TthreadBlock &threadBlock)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    LaplacianApplyMetric3DSetupSumFacTOP<DEFORMED, VARCOEFF>(
        df, diffCoeff, metric, threadBlock);

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
        LaplacianApplyMetric3DPointSumFacTOP<SHAPE_TYPE, DEFORMED, VARCOEFF>(
            nqTot, i, j, k, idx, w0, w1, w2, f0, f1, f1m, f2, df, jac,
            diffCoeff, varDiffCoeffSize, metric, deriv, flux);

        out0[idx] = flux[0];
        out1[idx] = flux[1];
        out2[idx] = flux[2];
    }

    localBarrier(threadBlock);
}

/**
 * @brief 1D TOP launcher stringing together the five sum-factorised stages,
 * striding the device block over the block's element groups
 * (GetDeviceElmtBlock()).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, bool VARCOEFF, typename TSizeParameter1D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    LaplacianKernelLauncher(
        const TSizeParameter1D sizeParam1D, const unsigned int ncoord,
        const size_t nelmt, const TElmtsPerBlock elmtsPerBlock,
        [[maybe_unused]] const bool isModified,
        const TUseSharedWsp useSharedMemoryWsp,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT D0,
        const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT diffCoeff, const size_t varDiffCoeffSize,
        const unsigned int nhomo, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
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

    const unsigned int basisDataSize = nq0 * nq0;
    TData *s_D0T                     = (TData *)shmemptr;

    // Copy to shared memory.
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);

    const unsigned int c = getBlockIdx<1>(threadBlock);
    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Workspace: the element's share after the basis data, or of the
        // thread block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) + getBlockRange<0>(threadBlock) * c) *
                elmtsPerBlock +
            wspIdx;
        const size_t wspSize =
            LaplacianWorkSpaceSize<SHAPE_TYPE, Implementation>(1u, ncoord,
                                                               sizeParam1D);
        TData *wspbase = isShared ? s_D0T + basisDataSize + wspSize * wspIdx
                                  : wsp + wspSize * globalWspIdx;
        TData *bwd     = wspbase;
        TData *deriv   = bwd + nq0;

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e = elmtBlock.elmtIdx;

            const TData *diffCoeffPtr =
                VARCOEFF ? diffCoeff + (c % nhomo) * nelmt * nq0 + e * nq0
                         : diffCoeff;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nm0 * nelmt * c + nm0 * e;
            TData *outptr       = out + nm0 * nelmt * c + nm0 * e;

            BwdTransSegSumFacTOPKernel<false>(nm0, nq0, basis0, inptr, bwd,
                                              elmtBlock);
            PhysDeriv1DSumFacTOPKernel<DEFORMED>(ncoord, nq0, nq0, D0, dfptr,
                                                 bwd, deriv, elmtBlock);
            ApplyMetric1DSumFacTOPKernel<DEFORMED, VARCOEFF>(
                ncoord, nq0, nq0, w0, dfptr, jacptr, diffCoeffPtr,
                varDiffCoeffSize, deriv, deriv, elmtBlock);
            SumDerivTensor1DSumFacTOPKernel<false, false>(
                nq0, s_D0T, deriv, bwd, (TData)1.0, elmtBlock);
            IProductWRTBaseSegSumFacTOPKernel<false, false>(
                nm0, nq0, basis0, bwd, outptr, (TData)1.0, elmtBlock);
        }
    });
}

/**
 * @brief 2D TOP launcher stringing together the five sum-factorised stages,
 * striding the device block over the block's element groups
 * (GetDeviceElmtBlock()).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, bool VARCOEFF, typename TSizeParameter2D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    LaplacianKernelLauncher(
        const TSizeParameter2D sizeParam2D, const unsigned int ncoord,
        const size_t nelmt, const TElmtsPerBlock elmtsPerBlock,
        const bool isModified, const TUseSharedWsp useSharedMemoryWsp,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT df,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diffCoeff,
        const size_t varDiffCoeffSize, const unsigned int nhomo,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    constexpr bool isNodal = SHAPE_TYPE == LibUtilities::NodalTri;

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = sizeParam2D.nqTot();

    const unsigned int ndf     = 2 * ncoord;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    // The inner product reads rows of length SharedMemoryRowStride. When
    // nq0 is shorter, its input is loaded into the derivative slots, past the
    // modal values of the nodal shapes.
    const unsigned int ldi = SharedMemoryRowStride<TData>(nq0);
    const bool isPadded    = (ldi != nq0);

    // Basis data, then the workspace: 16-byte aligned in shared memory, packed
    // in global memory.
    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);
    TData *s_basis0 = (TData *)shmemptr;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_D0T    = s_basis1 + basisDataSize.basis1();
    TData *s_D1T    = s_D0T + nq0 * nq0;

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);
    LoadTransposeToSharedMemory(nq1, nq1, D1, s_D1T, threadBlock);

    const unsigned int c = getBlockIdx<1>(threadBlock);
    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Inner-product rows are padded in shared memory only.
        constexpr bool padInput = decltype(isShared)::value;

        // Workspace: the element's share after the basis data, or of the
        // thread block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) + getBlockRange<0>(threadBlock) * c) *
                elmtsPerBlock +
            wspIdx;
        const size_t wspSize =
            LaplacianWorkSpaceSize<SHAPE_TYPE, Implementation>(1u, ncoord,
                                                               sizeParam2D);
        TData *wspbase =
            isShared
                ? s_basis0 +
                      SharedMemoryAlignedSize<TData>(basisDataSize.total() +
                                                     nq0 * nq0 + nq1 * nq1) +
                      LaplacianSharedWorkSpaceSize<SHAPE_TYPE, Implementation,
                                                   TData>(sizeParam2D) *
                          wspIdx
                : wsp + wspSize * globalWspIdx;

        TData *metric = wspbase;
        TData *bwd =
            metric + (isShared ? SharedMemoryAlignedSize<TData>(6) : 6u);
        TData *deriv =
            bwd + (isShared ? SharedMemoryAlignedSize<TData>(nqTot) : nqTot);
        TData *modal  = deriv;
        TData *deriv0 = deriv;
        TData *deriv1 = deriv0 + nqTot;
        TData *ipb    = (isShared && isPadded)
                            ? deriv + SharedMemoryAlignedSize<TData>(nmTot)
                            : bwd;
        TData *wsp0 =
            deriv +
            (isShared ? IProductWRTBaseFusedWorkSpaceOffset<TData>(sizeParam2D)
                      : nmTot);

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e = elmtBlock.elmtIdx;

            const TData *diffCoeffPtr =
                VARCOEFF ? diffCoeff + (c % nhomo) * nelmt * nqTot + e * nqTot
                         : diffCoeff;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
            TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

            if constexpr (isNodal)
            {
                // Nodal to Modal in the workspace.
                MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, modal, elmtBlock);
            }
            else
            {
                // Copy to the workspace.
                LoadToSharedMemory(nmTot, inptr, modal, elmtBlock);
                localBarrier(elmtBlock);
            }

            BwdTrans2DSumFacTOPKernel<SHAPE_TYPE, false>(
                nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1,
                modal, bwd, wsp0, elmtBlock);

            TData dmetric[(DEFORMED || VARCOEFF) ? 6 : 1] = {0.0};
            LaplacianPhysDerivApplyMetric2DSumFacTOPKernel<SHAPE_TYPE, DEFORMED,
                                                           VARCOEFF>(
                ncoord, nq0, nq1, D0, D1, w0, w1, f0, f1, dfptr, jacptr,
                diffCoeffPtr, varDiffCoeffSize, bwd, deriv0, deriv1,
                (DEFORMED || VARCOEFF) ? dmetric : metric, elmtBlock);
            SumDerivTensor2DSumFacTOPKernel<false, false>(
                nq0, nq1, s_D0T, s_D1T, deriv0, deriv1, bwd, (TData)1.0,
                elmtBlock);

            if (isShared && isPadded)
            {
                LoadWithPaddingToSharedMemory(nq0, nq1, ldi, bwd, ipb,
                                              elmtBlock);
                localBarrier(elmtBlock);
            }

            IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, false, false,
                                             padInput>(
                nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
                s_basis1, ipb, isNodal ? modal : outptr, wsp0, (TData)1.0,
                elmtBlock);

            if constexpr (isNodal)
            {
                // Multiply by transpose nodToMod to transform coeffs.
                MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, modal,
                                                   outptr, elmtBlock);
            }
        }
    });
}

/**
 * @brief 3D TOP launcher stringing together the five sum-factorised stages,
 * striding the device block over the block's element groups
 * (GetDeviceElmtBlock()).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, bool VARCOEFF, typename TSizeParameter3D,
          typename TElmtsPerBlock, typename TthreadBlock, typename TData,
          typename TUseSharedWsp,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    LaplacianKernelLauncher(
        const TSizeParameter3D sizeParam3D, const unsigned int ncoord,
        const size_t nelmt, const TElmtsPerBlock elmtsPerBlock,
        const bool isModified, const TUseSharedWsp useSharedMemoryWsp,
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
        const TData *NEK_RESTRICT diffCoeff, const size_t varDiffCoeffSize,
        const unsigned int nhomo, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    constexpr bool isNodal = SHAPE_TYPE == LibUtilities::NodalTet ||
                             SHAPE_TYPE == LibUtilities::NodalPrism ||
                             SHAPE_TYPE == LibUtilities::NodalPyr;

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = sizeParam3D.nqTot();

    constexpr unsigned int ndf = 9u;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    unsigned int wsp0Size = 0;
    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wsp0Size = std::max(nq0 * nm1 * nm2, nm0 * nq1 * nq2);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        wsp0Size = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        wsp0Size = nm0 * nm1 * nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr ||
                       SHAPE_TYPE == LibUtilities::NodalPyr)
    {
        wsp0Size = nm0 * nm1 * nq2;
    }

    const unsigned int wspPadSize =
        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(sizeParam3D);
    wsp0Size += wspPadSize;

    // The inner product reads rows of length SharedMemoryRowStride. When
    // nq0 is shorter, its input is loaded into the derivative slots, past the
    // modal values of the nodal shapes.
    const unsigned int ldi = SharedMemoryRowStride<TData>(nq0);
    const bool isPadded    = (ldi != nq0);

    // Basis data, then the workspace: 16-byte aligned in shared memory, packed
    // in global memory.
    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);
    TData *s_basis0 = (TData *)shmemptr;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_basis2 = s_basis1 + basisDataSize.basis1();
    TData *s_D0T    = s_basis2 + basisDataSize.basis2();
    TData *s_D1T    = s_D0T + nq0 * nq0;
    TData *s_D2T    = s_D1T + nq1 * nq1;

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadToSharedMemory(basisDataSize.basis2(), basis2, s_basis2, threadBlock);
    LoadTransposeToSharedMemory(nq0, nq0, D0, s_D0T, threadBlock);
    LoadTransposeToSharedMemory(nq1, nq1, D1, s_D1T, threadBlock);
    LoadTransposeToSharedMemory(nq2, nq2, D2, s_D2T, threadBlock);

    const unsigned int c = getBlockIdx<1>(threadBlock);
    const unsigned int wspIdx =
        GetDeviceSharedWorkSpaceIdx(elmtsPerBlock, threadBlock);
    DispatchWorkSpace(useSharedMemoryWsp, [&](const auto isShared) {
        // Inner-product rows are padded in shared memory only.
        constexpr bool padInput = decltype(isShared)::value;

        // Workspace: the element's share after the basis data, or of the
        // thread block's slice.
        const size_t globalWspIdx =
            (getBlockIdx<0>(threadBlock) + getBlockRange<0>(threadBlock) * c) *
                elmtsPerBlock +
            wspIdx;
        const size_t wspSize =
            LaplacianWorkSpaceSize<SHAPE_TYPE, Implementation>(1u, ncoord,
                                                               sizeParam3D);
        TData *wspbase =
            isShared
                ? s_basis0 +
                      SharedMemoryAlignedSize<TData>(basisDataSize.total() +
                                                     nq0 * nq0 + nq1 * nq1 +
                                                     nq2 * nq2) +
                      LaplacianSharedWorkSpaceSize<SHAPE_TYPE, Implementation,
                                                   TData>(sizeParam3D) *
                          wspIdx
                : wsp + wspSize * globalWspIdx;

        TData *metric = wspbase;
        TData *bwd =
            metric + (isShared ? SharedMemoryAlignedSize<TData>(9) : 9u);
        TData *deriv =
            bwd + (isShared ? SharedMemoryAlignedSize<TData>(nqTot) : nqTot);
        TData *modal  = deriv;
        TData *deriv0 = deriv;
        TData *deriv1 = deriv0 + nqTot;
        TData *deriv2 = deriv1 + nqTot;
        TData *ipb    = (isShared && isPadded)
                            ? deriv + SharedMemoryAlignedSize<TData>(nmTot)
                            : bwd;
        TData *wsp0 =
            deriv +
            (isShared ? IProductWRTBaseFusedWorkSpaceOffset<TData>(sizeParam3D)
                      : nmTot);
        TData *wsp1 =
            wsp0 + (isShared ? SharedMemoryAlignedSize<TData>(wsp0Size)
                             : wsp0Size - wspPadSize);

        for (size_t e0 = static_cast<size_t>(getBlockIdx<0>(threadBlock)) *
                         elmtsPerBlock;
             e0 < nelmt;
             e0 +=
             static_cast<size_t>(getBlockRange<0>(threadBlock)) * elmtsPerBlock)
        {
            const auto elmtBlock =
                GetDeviceElmtBlock(e0, nelmt, elmtsPerBlock, threadBlock);
            const size_t e = elmtBlock.elmtIdx;

            const TData *diffCoeffPtr =
                VARCOEFF ? diffCoeff + (c % nhomo) * nelmt * nqTot + e * nqTot
                         : diffCoeff;
            const TData *dfptr  = df + ndf * dfsize * e;
            const TData *jacptr = jac + jacsize * e;
            const TData *inptr  = in + nmTot * nelmt * c + nmTot * e;
            TData *outptr       = out + nmTot * nelmt * c + nmTot * e;

            if constexpr (isNodal)
            {
                // Nodal to Modal in the workspace.
                MatVecSumFacTOPKernel(nmTot, nodToMod, inptr, modal, elmtBlock);
            }
            else
            {
                // Copy to the workspace.
                LoadToSharedMemory(nmTot, inptr, modal, elmtBlock);
                localBarrier(elmtBlock);
            }

            BwdTrans3DSumFacTOPKernel<SHAPE_TYPE, false>(
                nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
                s_basis0, s_basis1, s_basis2, modal, bwd, wsp0, wsp1,
                elmtBlock);
            TData dmetric[(DEFORMED || VARCOEFF) ? 9 : 1] = {0.0};
            LaplacianPhysDerivApplyMetric3DSumFacTOPKernel<SHAPE_TYPE, DEFORMED,
                                                           VARCOEFF>(
                nq0, nq1, nq2, D0, D1, D2, w0, w1, w2, f0, f1, f1m, f2, dfptr,
                jacptr, diffCoeffPtr, varDiffCoeffSize, bwd, deriv0, deriv1,
                deriv2, (DEFORMED || VARCOEFF) ? dmetric : metric, elmtBlock);
            SumDerivTensor3DSumFacTOPKernel<false, false>(
                nq0, nq1, nq2, s_D0T, s_D1T, s_D2T, deriv0, deriv1, deriv2, bwd,
                (TData)1.0, elmtBlock);

            if (isShared && isPadded)
            {
                LoadWithPaddingToSharedMemory(nq0, nq1 * nq2, ldi, bwd, ipb,
                                              elmtBlock);
                localBarrier(elmtBlock);
            }

            // The collapsed shapes take the two scratches the other way round.
            constexpr bool hex = (SHAPE_TYPE == LibUtilities::Hex);
            IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, false, false,
                                             padInput>(
                nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
                index1, index2, s_basis0, s_basis1, s_basis2, ipb,
                isNodal ? modal : outptr, hex ? wsp0 : wsp1, hex ? wsp1 : wsp0,
                (TData)1.0, elmtBlock);

            if constexpr (isNodal)
            {
                // Multiply by transpose nodToMod to transform coeffs.
                MatVecSumFacTOPKernel<false, true>(nmTot, nodToMod, modal,
                                                   outptr, elmtBlock);
            }
        }
    });
}

#endif

} // namespace Nektar::MultiRegions::detail
