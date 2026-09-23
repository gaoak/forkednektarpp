///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFacKernels.hpp
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
// Description: Device SumFac kernels of the inner product with the
// basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTBaseDeviceSumFacKernels.hpp
 * @brief Device SumFac kernels of the inner product with the basis: one
 * element per thread, with the sum-factorisation intermediates held in a
 * global-memory workspace.
 *
 * @details
 * These are the kernels the Device/SumFac block implementation launches
 * (see IProductWRTBaseDeviceSumFac.hpp); the SumFacTOP counterparts, where
 * a whole thread block cooperates on one element, are in
 * IProductWRTBaseDeviceSumFacTOPKernels.hpp. Everything below is compiled
 * only into device translation units of a device-enabled build
 * (NEKTAR_ENABLE_DEVICE and DEVICE_COMPILE_ONLY).
 *
 * One thread integrates one element. The block storage is reshaped to
 * interleave width warpSize before the launch, so that the elements of a
 * warp are interleaved: element e is handled by lane ilane = e % warpSize
 * of warp iwarp = e / warpSize, and value n of that element sits at offset
 * warpSize * n + ilane within its warp's group. Consecutive lanes therefore
 * address consecutive words. The IProductWRTBaseKernelLauncher overloads --
 * one per dimension, selected by the type of the size parameter, and each
 * in a form with and a form without the quadrature metric -- grid-stride over
 * the elements of the block; the stride, gridDim * blockDim, is a whole
 * number of warps (the launch uses one warp per thread block, see
 * GetDeviceBlockSize), so a thread keeps its lane index for every element
 * it visits. The grid's y dimension selects the component (times
 * homogeneous mode) a thread block works on.
 *
 * Each shape kernel contracts the physical values against one direction's
 * one-dimensional basis table at a time, holding the intermediates in the
 * workspace slices (wsp, or wsp0 and wsp1 in three dimensions) that
 * IProductWRTBaseWorkSpaceSize accounts for. A basis table basisN holds
 * one row of nqN values per mode, so that basisN[m * nqN + i] is mode m
 * evaluated at point i; along the collapsed directions of the triangle,
 * tetrahedron, prism and pyramid the row index m is the shape's combined
 * mode index rather than a single-direction mode number. Physical values
 * are read in point order, with the direction-0 point running fastest, and
 * coefficients are written in the shape's mode ordering: for the
 * tensor-product shapes (quadrilateral, hexahedron) the direction-0 mode
 * index runs fastest, whereas along a collapsed direction it is the
 * innermost index of the combined ordering that does -- q for the
 * triangle, r for the tetrahedron, prism and pyramid.
 *
 * Every kernel comes in two overloads, distinguished by their argument list
 * rather than by a template parameter. The first takes the per-direction
 * quadrature weights w0, w1, w2 and the element Jacobians jac and folds
 * them into the contractions: the Jacobian and the direction-0 weight in
 * the innermost (direction-0) stage, each remaining direction's weight in
 * its own stage. jac is read as one value per element for a regular
 * geometry (DEFORMED false) and as one warp-interleaved value per
 * quadrature point for a deformed one. The second overload omits weights
 * and Jacobians and applies the basis transpose alone, for input that
 * already carries the quadrature metric (see
 * IProductWRTBaseOp::SetIntegration).
 *
 * @p isModified marks a first-direction eModified_A basis, for which the
 * shape kernels add the extra terms the modified expansion carries where
 * its coordinate system collapses: one vertex term for the triangle and
 * the pyramid, two vertex terms plus a singular edge for the tetrahedron,
 * and a singular edge alone for the prism. Each term is identical to the
 * corresponding one in StdRegions/Operators/IProductWRTBaseSumFacStdKernels.hpp
 * and is always accumulated onto the mode it corrects, whatever @p APPEND
 * says. For the nodal shapes the modal kernel of the matching non-nodal
 * shape runs into a workspace slice and MatVecKernel then maps the result
 * to the nodal coefficients with the transpose of the nodal-to-modal
 * matrix -- the reverse of the BwdTrans order.
 *
 * @see IProductWRTBaseSerialAVXSumFacKernels.hpp for the Serial/AVX form of
 * the same decomposition, expressed over SIMD vectors instead of warp
 * lanes.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Workspace a 1D SumFac inner product needs, in TData values: none.
 *
 * The segment kernel contracts its only direction straight into the output
 * and so keeps no intermediates; the 1D IProductWRTBaseKernelLauncher
 * accepts a workspace pointer for signature uniformity and ignores it. The
 * overload completes the family of dimensional overloads the block
 * implementation queries the same way in every dimension (see
 * IProductWRTBaseDeviceSumFac.hpp).
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg).
 * @tparam Implementation    Operators::SumFac; the SumFacTOP overload lives
 *                           in IProductWRTBaseDeviceSumFacTOPKernels.hpp.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam1D  Modal and quadrature sizes of the expansion.
 *
 * @return 0.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTBaseWorkSpaceSize(
    [[maybe_unused]] const size_t nelmt,
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    size_t wspsize = 0;

    if constexpr (SHAPE_TYPE == LibUtilities::Seg)
    {
        wspsize = 0;
    }

    return wspsize;
}

/**
 * @brief Workspace a 2D SumFac inner product needs, in TData values.
 *
 * One flat global-memory region serves the whole launch: a stage needing s
 * values per element occupies s * nelmt consecutive values per component,
 * warp-interleaved exactly like the field data, and the stages follow one
 * another (the 2D IProductWRTBaseKernelLauncher does the slicing, out of a
 * region the caller has scaled by the component count). Per element that
 * is nq1 values -- the direction-0 sums, one per direction-1 quadrature
 * point -- for every 2D shape, plus, for NodalTri, the shape's nmTot modal
 * coefficients ahead of them, which the nodal mapping then reads.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter2D  2D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam2D  Modal and quadrature sizes per direction.
 *
 * @return Workspace size in TData values per component, 0 for unhandled
 * shapes.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter2D sizeParam2D)
{
    size_t wspsize = 0;

    const unsigned int nq1 = sizeParam2D.nq1();

    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        wspsize = nq1 * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
    {
        wspsize = nq1 * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nmTot = sizeParam2D.nmTot();

        wspsize = (nq1 + nmTot) * nelmt;
    }

    return wspsize;
}

/**
 * @brief Workspace a 3D SumFac inner product needs, in TData values.
 *
 * Laid out as in the 2D case: per element nq1 * nq2 values for the
 * direction-0 sums and nq2 values for the direction-1 sums, preceded for
 * the nodal shapes by the shape's nmTot modal coefficients that the nodal
 * mapping reads.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter3D  3D size parameter (see ElmtHelper.hpp).
 *
 * @param   nelmt        Elements in the block, padding included.
 * @param   sizeParam3D  Modal and quadrature sizes per direction.
 *
 * @return Workspace size in TData values per component, 0 for unhandled
 * shapes.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr size_t IProductWRTBaseWorkSpaceSize(
    const size_t nelmt, const TSizeParameter3D sizeParam3D)
{
    size_t wspsize = 0;

    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        wspsize = (nq1 * nq2 + nq2) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
    {
        wspsize = (nq1 * nq2 + nq2) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = (nq1 * nq2 + nq2 + nmTot) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
    {
        wspsize = (nq1 * nq2 + nq2) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        const unsigned int nmTot = sizeParam3D.nmTot();

        wspsize = (nq1 * nq2 + nq2 + nmTot) * nelmt;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        wspsize = (nq1 * nq2 + nq2) * nelmt;
    }

    return wspsize;
}

/**
 * @brief Dynamic shared memory a 1D SumFac launch needs: none.
 *
 * The SumFac kernels keep their intermediates in the global-memory
 * workspace and read the basis tables straight from global memory, so no
 * dynamic shared memory is requested at launch in any dimension. The
 * overloads mirror the SumFacTOP ones in
 * IProductWRTBaseDeviceSumFacTOPKernels.hpp, which do use shared memory, so
 * that the block implementation can query the size the same way for either
 * strategy.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam Implementation    Operators::SumFac.
 * @tparam TSizeParameter1D  1D size parameter (see ElmtHelper.hpp).
 *
 * @param   sizeParam1D  Sizes of the expansion; unused.
 *
 * @return Zero.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter1D_v<TSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter1D sizeParam1D)
{
    return 0;
}

/// @brief Dynamic shared memory a 2D SumFac launch needs: none, for the
/// reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter2D_v<TSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter2D sizeParam2D)
{
    return 0;
}

/// @brief Dynamic shared memory a 3D SumFac launch needs: none, for the
/// reason given in the 1D overload.
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsSizeParameter3D_v<TSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int IProductWRTBaseSharedMemorySize(
    [[maybe_unused]] const TSizeParameter3D sizeParam3D)
{
    return 0;
}

/**
 * @brief Segment SumFac kernel: inner product of one element against each
 * of its nm0 basis functions, quadrature metric included.
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one interleaved value per quadrature
 *                   point rather than a single per-element value.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nm0     Modes of the expansion.
 * @param   nq0     Quadrature points of the expansion.
 * @param   basis0  Basis table (nm0 rows of nq0).
 * @param   w0      Quadrature weights.
 * @param   jac     Warp's Jacobian slice for this element.
 * @param   in      Warp's interleaved physical values, nq0 per element.
 * @param   out     Warp's interleaved coefficients, nm0 per element.
 * @param   scale   Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            const unsigned int index = warpsize * i + ilane;
            if constexpr (DEFORMED)
            {
                sum += in[index] * basis0[p * nq0 + i] * jac[index] * w0[i];
            }
            else
            {
                sum += in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
            }
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[warpsize * p + ilane] += sum;
        }
        else
        {
            out[warpsize * p + ilane] = sum;
        }
    }
}

/**
 * @brief Segment SumFac kernel without the quadrature metric: the basis
 * transpose alone, for input already carrying \f$wJ\f$.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nm0     Modes of the expansion.
 * @param   nq0     Quadrature points of the expansion.
 * @param   basis0  Basis table (nm0 rows of nq0).
 * @param   in      Warp's interleaved physical values.
 * @param   out     Warp's interleaved coefficients.
 * @param   scale   Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseSegSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nq0,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        TData sum = 0.0;
#pragma unroll
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            sum += in[warpsize * i + ilane] * basis0[p * nq0 + i];
        }

        if constexpr (SCALE)
        {
            sum *= scale;
        }

        if constexpr (APPEND)
        {
            out[warpsize * p + ilane] += sum;
        }
        else
        {
            out[warpsize * p + ilane] = sum;
        }
    }
}

/**
 * @brief Quadrilateral SumFac kernel: tensor-product inner product of one
 * element, one direction-0 mode at a time, quadrature metric included.
 *
 * For each direction-0 mode p the direction-0 contraction -- carrying the
 * Jacobian and the direction-0 weights -- reduces the physical values to
 * the nq1 values of @p wsp, one per direction-1 quadrature point; the
 * direction-1 contraction, carrying the direction-1 weights, then turns
 * those into the nm1 coefficients of column p. Coefficients are written as
 * out[q * nm0 + p].
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one value per quadrature point.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nm0,nm1 Modes per direction.
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   basis0  Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1  Direction-1 basis table (nm1 rows of nq1).
 * @param   w0,w1   Quadrature weights per direction.
 * @param   jac     Warp's Jacobian slice for this element.
 * @param   in      Warp's interleaved physical values, nq0 * nq1 per
 *                  element, direction-0 point running fastest.
 * @param   out     Warp's interleaved coefficients, nm0 * nm1 per element.
 * @param   wsp     Warp's workspace slice, nq1 values per element.
 * @param   scale   Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index = warpsize * cnt_ji + ilane;
                if constexpr (DEFORMED)
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[index] * w0[i];
                }
                else
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                }
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[q * nq1 + j] * w1[j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * (nm0 * q + p) + ilane] += sum;
            }
            else
            {
                out[warpsize * (nm0 * q + p) + ilane] = sum;
            }
        }
    }
}

/**
 * @brief Quadrilateral SumFac kernel without the quadrature metric: the two
 * contractions of the metric form with weights and Jacobian dropped.
 *
 * @param   ilane   Lane index of this thread's element within its warp.
 * @param   nm0,nm1 Modes per direction.
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   basis0  Direction-0 basis table.
 * @param   basis1  Direction-1 basis table.
 * @param   in      Warp's interleaved physical values.
 * @param   out     Warp's interleaved coefficients.
 * @param   wsp     Warp's workspace slice, nq1 values per element.
 * @param   scale   Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseQuadSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                sum += in[warpsize * cnt_ji + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[q * nq1 + j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * (nm0 * q + p) + ilane] += sum;
            }
            else
            {
                out[warpsize * (nm0 * q + p) + ilane] = sum;
            }
        }
    }
}

/**
 * @brief Triangular SumFac kernel: inner product of one element in the
 * collapsed coordinate system, quadrature metric included.
 *
 * The structure is that of the quadrilateral kernel, but the direction-1
 * stage runs over the nm1 - p modes (p,q) that follow p in the triangular
 * ordering, indexing @p basis1 by that same combined mode, and the
 * coefficients come out in that flat ordering.
 *
 * With @p isModified set, the collapsed-vertex correction
 * sum_ij in * w0 * w1 * jac * basis0[nq0 + i] * basis1[nq1 + j] is
 * accumulated onto mode 1 -- always accumulated, whatever @p APPEND says --
 * exactly as in the StdRegions IProductTriKernel.
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one value per quadrature point.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1     Modes per direction.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0      Direction-0 basis table (nm0 rows of nq0).
 * @param   basis1      Direction-1 basis table, one row of nq1 per
 *                      combined (p,q) mode.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   jac         Warp's Jacobian slice for this element.
 * @param   in          Warp's interleaved physical values, nq0 * nq1 per
 *                      element.
 * @param   out         Warp's interleaved coefficients in the triangular
 *                      mode ordering.
 * @param   wsp         Warp's workspace slice, nq1 values per element.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index = warpsize * cnt_ji + ilane;
                if constexpr (DEFORMED)
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[index] * w0[i];
                }
                else
                {
                    sum += in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                }
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[mode_pq * nq1 + j] *
                       w1[j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * mode_pq + ilane] += sum;
            }
            else
            {
                out[warpsize * mode_pq + ilane] = sum;
            }
        }
    }

    // Correction for singular vertex in collapsed coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        TData prod = 0.0;
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index = warpsize * (nq0 * j + i) + ilane;

                if constexpr (DEFORMED)
                {
                    prod += in[index] * w0[i] * w1[j] * basis1[nq1 + j] *
                            basis0[nq0 + i] * jac[index];
                }
                else
                {
                    prod += in[index] * w0[i] * w1[j] * basis1[nq1 + j] *
                            basis0[nq0 + i] * jac[0];
                }
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

/**
 * @brief Triangular SumFac kernel without the quadrature metric: as the
 * metric form, including the collapsed-vertex correction onto mode 1, with
 * weights and Jacobian dropped from every term.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1     Modes per direction.
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1   Basis tables per direction.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients.
 * @param   wsp         Warp's workspace slice, nq1 values per element.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTriSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nq0, const unsigned int nq1, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u; p < nm0; ++p)
    {
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                sum += in[warpsize * cnt_ji + ilane] * basis0[p * nq0 + i];
            }
            wsp[warpsize * j + ilane] = sum;
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            TData sum = 0.0;
#pragma unroll
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                sum += wsp[warpsize * j + ilane] * basis1[mode_pq * nq1 + j];
            }

            if constexpr (SCALE)
            {
                sum *= scale;
            }

            if constexpr (APPEND)
            {
                out[warpsize * mode_pq + ilane] += sum;
            }
            else
            {
                out[warpsize * mode_pq + ilane] = sum;
            }
        }
    }

    // Correction for singular vertex in collapsed coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (isModified)
    {
        TData prod = 0.0;
        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
#pragma unroll
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                prod += in[warpsize * cnt_ji + ilane] * basis1[nq1 + j] *
                        basis0[nq0 + i];
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

/**
 * @brief Hexahedral SumFac kernel: three-stage tensor-product inner
 * product of one element, quadrature metric included.
 *
 * For each direction-0 mode p, the direction-0 contraction (carrying the
 * Jacobian and w0) fills @p wsp0 with nq1 * nq2 values; then for each
 * direction-1 mode q the direction-1 contraction (carrying w1) fills
 * @p wsp1 with nq2 values, and the direction-2 contraction (carrying w2)
 * writes the nm2 coefficients out[r * nm0 * nm1 + q * nm0 + p].
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one value per quadrature point.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   jac         Warp's Jacobian slice for this element.
 * @param   in          Warp's interleaved physical values, direction-0
 *                      point running fastest.
 * @param   out         Warp's interleaved coefficients.
 * @param   wsp0        Warp's first workspace slice, nq1 * nq2 values per
 *                      element.
 * @param   wsp1        Warp's second workspace slice, nq2 values per
 *                      element.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
    const TData *NEK_RESTRICT w2, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2; ++r)
            {
                const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;

                TData sum = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum += wsp1[warpsize * k + ilane] * basis2[r * nq2 + k] *
                           w2[k];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_rqp + ilane] += sum;
                }
                else
                {
                    out[warpsize * cnt_rqp + ilane] = sum;
                }
            }
        }
    }
}

/**
 * @brief Hexahedral SumFac kernel without the quadrature metric: the three
 * contractions of the metric form with weights and Jacobian dropped.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseHexSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT basis0,
    const TData *NEK_RESTRICT basis1, const TData *NEK_RESTRICT basis2,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1, const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2; ++r)
            {
                const unsigned int cnt_rqp = nm0 * nm1 * r + nm0 * q + p;

                TData sum = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum += wsp1[warpsize * k + ilane] * basis2[r * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * cnt_rqp + ilane] += sum;
                }
                else
                {
                    out[warpsize * cnt_rqp + ilane] = sum;
                }
            }
        }
    }
}

/**
 * @brief Tetrahedral SumFac kernel: three-stage inner product of one
 * element in the collapsed coordinate system, quadrature metric included.
 *
 * As the hexahedral kernel, but the mode loops follow the tetrahedral
 * ordering -- q runs to nm1 - p and r to nm2 - p - q -- with @p basis1
 * indexed by the combined (p,q) mode and @p basis2 by the combined (p,q,r)
 * mode, whose counter is advanced past the modes an anisotropic order
 * skips.
 *
 * With @p isModified set, the corrections of the modified expansion are
 * accumulated (always accumulated, whatever @p APPEND says) onto the same
 * modes as in the StdRegions IProductTetKernel: the top vertex onto mode
 * 1, the bottom vertex onto mode nm2 and the singular edge onto modes
 * nm2 + r, r = 1 .. nm2 - 2. Only the first NM2_MAX - 2 edge terms are held
 * in registers; higher r are accumulated straight into @p out inside the
 * quadrature loop.
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one value per quadrature point.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction, the latter
 *                      two indexed by combined modes.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   jac         Warp's Jacobian slice for this element.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients in the tetrahedral
 *                      mode ordering.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0;
         ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[mode_pq * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    tmp += wsp1[warpsize * k + ilane] *
                           basis2[mode2 * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    tmp *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += tmp;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = tmp;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
        {
            mode2 += nm2 - p - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM2_MAX = 4;
        TData prod[NM2_MAX - 2]        = {0.0};
        TData topVert                  = 0;
        TData bottomVert               = 0;

        TData jac0;
        if constexpr (!DEFORMED)
        {
            jac0 = jac[0];
        }

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    // Store jac * quadrature weight
                    TData tmp = w0[i] * w1[j] * w2[k];
                    if constexpr (DEFORMED)
                    {
                        tmp *= jac[index];
                    }
                    else
                    {
                        tmp *= jac0;
                    }

                    // top vertex
                    topVert += (basis0[i] * basis1[nq1 + j] +
                                basis0[nq0 + i] * basis1[j] +
                                basis0[nq0 + i] * basis1[nq1 + j]) *
                               basis2[nq2 + k] * in[index] * tmp;

                    // bottom vertex
                    bottomVert += basis0[nq0 + i] * basis1[nq1 + j] *
                                  basis2[k] * in[index] * tmp;

                    // singular edge
                    tmp *= basis1[nq1 + j] * basis0[nq0 + i] * in[index];
                    for (unsigned int r = 1u;
                         r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
                    {
                        prod[r - 1] += basis2[(r + 1) * nq2 + k] * tmp;
                    }
                    for (unsigned int r = NM2_MAX - 1u; r < nm2 - 1u; ++r)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp;
                        }
                    }
                }
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += topVert * scale;
            out[warpsize * nm2 + ilane] += bottomVert * scale;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1] * scale;
            }
        }
        else
        {
            out[warpsize + ilane] += topVert;
            out[warpsize * nm2 + ilane] += bottomVert;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1];
            }
        }
    }
}

/**
 * @brief Tetrahedral SumFac kernel without the quadrature metric: as the
 * metric form, corrections included, with weights and Jacobian dropped
 * from every term.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBaseTetSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pq = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0;
         ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1 - p; ++q, ++mode_pq)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[mode_pq * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p - q; ++r, ++mode2, ++mode_pqr)
            {
                TData tmp = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    tmp += wsp1[warpsize * k + ilane] * basis2[mode2 * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    tmp *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += tmp;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = tmp;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1 - p; q < nm2 - p; ++q)
        {
            mode2 += nm2 - p - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM2_MAX = 4;
        TData prod[NM2_MAX - 2]        = {0.0};
        TData topVert                  = 0;
        TData bottomVert               = 0;
        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    // top vertex
                    topVert += (basis0[i] * basis1[nq1 + j] +
                                basis0[nq0 + i] * basis1[j] +
                                basis0[nq0 + i] * basis1[nq1 + j]) *
                               basis2[nq2 + k] * in[index];

                    // bottom vertex
                    bottomVert += basis0[nq0 + i] * basis1[nq1 + j] *
                                  basis2[k] * in[index];

                    // singular edge
                    TData tmp = basis1[nq1 + j] * basis0[nq0 + i] * in[index];
                    for (unsigned int r = 1u;
                         r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
                    {
                        prod[r - 1] += basis2[(r + 1) * nq2 + k] * tmp;
                    }
                    for (unsigned int r = NM2_MAX - 1; r < nm2 - 1u; ++r)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 + r) + ilane] +=
                                basis2[(r + 1) * nq2 + k] * tmp;
                        }
                    }
                }
            }
        }

        if constexpr (SCALE)
        {
            out[warpsize + ilane] += topVert * scale;
            out[warpsize * nm2 + ilane] += bottomVert * scale;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1] * scale;
            }
        }
        else
        {
            out[warpsize + ilane] += topVert;
            out[warpsize * nm2 + ilane] += bottomVert;
            for (unsigned int r = 1u; r < std::min(NM2_MAX - 1u, nm2 - 1u); ++r)
            {
                out[warpsize * (nm2 + r) + ilane] += prod[r - 1];
            }
        }
    }
}

/**
 * @brief Prismatic SumFac kernel: three-stage inner product of one
 * element, collapsed in the (0,2) plane, quadrature metric included.
 *
 * Direction 1 is a full tensor-product direction (q runs to nm1) while
 * direction 2 is collapsed against direction 0 (r runs to nm2 - p),
 * @p basis2 being indexed by the combined (p,r) mode.
 *
 * With @p isModified set, the collapsed-edge correction is accumulated
 * onto the modes nm2 * q + 1, q = 0 .. nm1 - 1, as in the StdRegions
 * IProductPrismKernel; only the first NM1_MAX terms are held in registers,
 * higher q being accumulated straight into @p out inside the quadrature
 * loop.
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one value per quadrature point.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction, the last
 *                      indexed by the combined (p,r) mode.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   jac         Warp's Jacobian slice for this element.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients in the prismatic
 *                      mode ordering.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
            {
                unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;

                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[(mode_pr + r) * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM1_MAX = 4;
        TData prod[NM1_MAX]            = {0.0};

        TData jac0;
        if constexpr (!DEFORMED)
        {
            jac0 = jac[0];
        }

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    TData tmp = basis2[nq2 + k] * basis0[nq0 + i] * w0[i] *
                                w1[j] * w2[k] * in[index];
                    if constexpr (DEFORMED)
                    {
                        tmp *= jac[index];
                    }
                    else
                    {
                        tmp *= jac0;
                    }

                    for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
                    {
                        prod[q] += tmp * basis1[q * nq1 + j];
                    }
                    for (unsigned int q = NM1_MAX; q < nm1; ++q)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j] * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j];
                        }
                    }
                }
            }
        }

        for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
        {
            if constexpr (SCALE)
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q] * scale;
            }
            else
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q];
            }
        }
    }
}

/**
 * @brief Prismatic SumFac kernel without the quadrature metric: as the
 * metric form, correction included, with weights and Jacobian dropped from
 * every term.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePrismSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode_pqr)
            {
                unsigned int mode_pr = (2u * nm2 - p + 1u) * p / 2u;

                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[(mode_pr + r) * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        constexpr unsigned int NM1_MAX = 4;
        TData prod[NM1_MAX]            = {0.0};

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    TData tmp = basis2[nq2 + k] * basis0[nq0 + i] * in[index];
                    for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
                    {
                        prod[q] += tmp * basis1[q * nq1 + j];
                    }
                    for (unsigned int q = NM1_MAX; q < nm1; ++q)
                    {
                        if constexpr (SCALE)
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j] * scale;
                        }
                        else
                        {
                            out[warpsize * (nm2 * q + 1u) + ilane] +=
                                tmp * basis1[q * nq1 + j];
                        }
                    }
                }
            }
        }

        for (unsigned int q = 0u; q < std::min(NM1_MAX, nm1); ++q)
        {
            if constexpr (SCALE)
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q] * scale;
            }
            else
            {
                out[warpsize * (nm2 * q + 1u) + ilane] += prod[q];
            }
        }
    }
}

/**
 * @brief Pyramidal SumFac kernel: three-stage inner product of one element
 * in the pyramidal collapsed system, quadrature metric included.
 *
 * The mode loops follow the pyramidal ordering, in which direction 2 is
 * collapsed against both other directions (r runs to nm2 - max(p,q)) and
 * @p basis2 is indexed by the combined mode, whose counter skips the
 * entries an anisotropic order leaves out.
 *
 * With @p isModified set, the collapsed top-vertex correction is
 * accumulated onto mode 1, as in the StdRegions IProductPyrKernel.
 *
 * @tparam SCALE     Multiply the result by @p scale.
 * @tparam APPEND    Accumulate onto @p out instead of overwriting it.
 * @tparam DEFORMED  @p jac holds one value per quadrature point.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction, the last
 *                      indexed by the combined mode.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   jac         Warp's Jacobian slice for this element.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients in the pyramidal
 *                      mode ordering.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
    const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
    const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;
                    if constexpr (DEFORMED)
                    {
                        sum_kj += in[index] * basis0[p * nq0 + i] * jac[index] *
                                  w0[i];
                    }
                    else
                    {
                        sum_kj +=
                            in[index] * basis0[p * nq0 + i] * jac[0] * w0[i];
                    }
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < p; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[mode2 * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        for (unsigned int q = p; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k += wsp0[warpsize * cnt_kj + ilane] *
                             basis1[q * nq1 + j] * w1[j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k += wsp1[warpsize * k + ilane] *
                             basis2[mode2 * nq2 + k] * w2[k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1; q < nm2; ++q)
        {
            mode2 += nm2 - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        TData prod = 0.0;

        TData jac0;
        if constexpr (!DEFORMED)
        {
            jac0 = jac[0];
        }

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    const unsigned int index = warpsize * cnt_kji + ilane;

                    // Store jac * quadrature weight
                    TData tmp = w0[i] * w1[j] * w2[k];
                    if constexpr (DEFORMED)
                    {
                        tmp *= jac[index];
                    }
                    else
                    {
                        tmp *= jac0;
                    }

                    // top vertex
                    prod += (basis0[i] * basis1[nq1 + j] +
                             basis0[nq0 + i] * basis1[j] +
                             basis0[nq0 + i] * basis1[nq1 + j]) *
                            basis2[nq2 + k] * in[index] * tmp;
                }
            }
        }

        // Add to existing entry.
        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

/**
 * @brief Pyramidal SumFac kernel without the quadrature metric: as the
 * metric form, correction included, with weights and Jacobian dropped from
 * every term.
 *
 * @param   ilane       Lane index of this thread's element within its
 *                      warp.
 * @param   nm0,nm1,nm2 Modes per direction.
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   in          Warp's interleaved physical values.
 * @param   out         Warp's interleaved coefficients.
 * @param   wsp0,wsp1   Warp's workspace slices of the two intermediate
 *                      stages.
 * @param   scale       Factor applied when @p SCALE is set.
 */
template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void IProductWRTBasePyrSumFacKernel(
    const unsigned int ilane, const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const bool isModified,
    const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
    const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp0, TData *NEK_RESTRICT wsp1,
    const TData scale)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int p = 0u, mode2 = 0u, mode_pqr = 0u; p < nm0; ++p)
    {
        for (unsigned int k = 0u, cnt_kj = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
            {
                TData sum_kj = 0.0;
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    sum_kj +=
                        in[warpsize * cnt_kji + ilane] * basis0[p * nq0 + i];
                }
                wsp0[warpsize * cnt_kj + ilane] = sum_kj;
            }
        }

        for (unsigned int q = 0u; q < p; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - p; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k +=
                        wsp1[warpsize * k + ilane] * basis2[mode2 * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        for (unsigned int q = p; q < nm1; ++q)
        {
            for (unsigned int k = 0u, cnt_kj = 0u; k < nq2; ++k)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int j = 0u; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k +=
                        wsp0[warpsize * cnt_kj + ilane] * basis1[q * nq1 + j];
                }
                wsp1[warpsize * k + ilane] = sum_k;
            }

            for (unsigned int r = 0u; r < nm2 - q; ++r, ++mode2, ++mode_pqr)
            {
                TData sum_k = 0.0;
#pragma unroll
                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    sum_k +=
                        wsp1[warpsize * k + ilane] * basis2[mode2 * nq2 + k];
                }

                if constexpr (SCALE)
                {
                    sum_k *= scale;
                }

                if constexpr (APPEND)
                {
                    out[warpsize * mode_pqr + ilane] += sum_k;
                }
                else
                {
                    out[warpsize * mode_pqr + ilane] = sum_k;
                }
            }
        }

        // increment mode in case order1!=order2
#pragma unroll
        for (unsigned int q = nm1; q < nm2; ++q)
        {
            mode2 += nm2 - q;
        }
    }

    // Add correction for collapsed coordinate.
    if (isModified)
    {
        TData prod = 0.0;
        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; ++k)
        {
            for (unsigned int j = 0u; j < nq1; ++j)
            {
#pragma unroll
                for (unsigned int i = 0u; i < nq0; ++i, ++cnt_kji)
                {
                    // top vertex
                    prod += (basis0[i] * basis1[nq1 + j] +
                             basis0[nq0 + i] * basis1[j] +
                             basis0[nq0 + i] * basis1[nq1 + j]) *
                            basis2[nq2 + k] * in[warpsize * cnt_kji + ilane];
                }
            }
        }

        // Add to existing entry.
        if constexpr (SCALE)
        {
            out[warpsize + ilane] += prod * scale;
        }
        else
        {
            out[warpsize + ilane] += prod;
        }
    }
}

/**
 * @brief Device entry point of the 1D SumFac inner product, quadrature
 * metric included.
 *
 * Unpacks the element sizes from @p sizeParam1D -- runtime values or
 * compile-time constants, see ElmtHelper.hpp -- fetches the shared-memory
 * base, then grid-strides over the block's segments, calling the segment
 * kernel for each. Each iteration derives the thread's lane and warp from
 * the global element index and offsets the input, output and Jacobian
 * pointers to that warp's interleaved group within the component the
 * grid's y index selects; the Jacobian is offset per warp when DEFORMED
 * and per element otherwise, since a regular geometry stores a single
 * value per element. The enable_if on the Implementation tag is what makes
 * the SumFac and SumFacTOP launchers of the same name distinguishable; a
 * size-parameter argument of the wrong dimension is caught by the
 * static_assert in the body. The launch bounds come from
 * GetMaxThreadPerBlock (the warp size for SumFac).
 *
 * @tparam SHAPE_TYPE       Shape of the block's elements (Seg).
 * @tparam Implementation   Operators::SumFac.
 * @tparam SCALE,APPEND,DEFORMED    Passed through to the segment kernel.
 * @tparam TSizeParameter1D Size-parameter type carrying nm0 and nq0.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam1D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      Basis table.
 * @param   w0          Quadrature weights.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused in 1D.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base; unused by SumFac.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const size_t nelmt,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT w0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp,
        const TData scale, [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const unsigned int jacsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nq0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nm0 * (nelmt * c + warpsize * iwarp);
        IProductWRTBaseSegSumFacKernel<SCALE, APPEND, DEFORMED>(
            ilane, nm0, nq0, basis0, w0, jacptr, inptr, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 1D SumFac inner product without the
 * quadrature metric; selected by the absence of the @p w0 and @p jac
 * arguments.
 *
 * The same grid-stride loop as the quadrature overload, calling the
 * unweighted segment kernel.
 *
 * @param   sizeParam1D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A; unused in 1D.
 * @param   basis0      Basis table.
 * @param   nodToMod    Nodal-to-modal matrix; unused in 1D.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Global-memory workspace; unused in 1D.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base; unused by SumFac.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, typename TSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter1D sizeParam1D, const size_t nelmt,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0,
        [[maybe_unused]] const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp, const TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nq0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nm0 * (nelmt * c + warpsize * iwarp);

        IProductWRTBaseSegSumFacKernel<SCALE, APPEND>(ilane, nm0, nq0, basis0,
                                                      inptr, outptr, scale);
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFac inner product, quadrature
 * metric included.
 *
 * Unpacks the element sizes from @p sizeParam2D, fetches the shared-memory
 * base, then grid-strides over the block's elements, calling the shape
 * kernel selected at compile time. Each iteration slices the workspace for
 * the thread's warp and component: nq1 values per element for Quad and
 * Tri; for NodalTri an nmTot slice for the modal result, taken from the
 * front of the workspace, and the nq1 slice behind it, after which
 * MatVecKernel maps the modal coefficients onto @p out with the transpose
 * of @p nodToMod.
 *
 * @tparam SHAPE_TYPE       Quad, Tri or NodalTri.
 * @tparam Implementation   Operators::SumFac.
 * @tparam SCALE,APPEND,DEFORMED    Passed through to the shape kernel,
 *                          except that for NodalTri the shape kernel is
 *                          instantiated with APPEND false and @p APPEND
 *                          governs the nodal mapping instead.
 * @tparam TSizeParameter2D Size-parameter type carrying the mode and point
 *                          counts.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam2D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0      Mode-index table; unused by SumFac, which needs no
 *                      per-thread mode lookup.
 * @param   basis0,basis1   Basis tables per direction.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   nodToMod    Nodal-to-modal matrix, applied transposed; nodal
 *                      shapes only.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Block's workspace, sized by
 *                      IProductWRTBaseWorkSpaceSize and scaled by ncomp.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base; unused by SumFac.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT w0, const TData *NEK_RESTRICT w1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    const unsigned int nqTot   = nq0 * nq1;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wspptr = wsp + nq1 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseQuadSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, w0, w1, jacptr,
                inptr, outptr, wspptr, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wspptr = wsp + nq1 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, inptr, outptr, wspptr, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wspptr  = wsp + nmTot * nelmt * ncomp +
                            nq1 * (nelmt * c + warpsize * iwarp);

            IProductWRTBaseTriSumFacKernel<SCALE, false, DEFORMED>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, w0, w1,
                jacptr, inptr, out1ptr, wspptr, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 2D SumFac inner product without the
 * quadrature metric; selected by the absence of the weight and Jacobian
 * arguments.
 *
 * The same grid-stride loop and workspace slicing as the quadrature
 * overload, calling the unweighted shape kernels.
 *
 * @param   sizeParam2D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0      Mode-index table; unused by SumFac.
 * @param   basis0,basis1   Basis tables per direction.
 * @param   nodToMod    Nodal-to-modal matrix, applied transposed; nodal
 *                      shapes only.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Block's workspace.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base; unused by SumFac.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, typename TSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter2D sizeParam2D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT in,
        TData *NEK_RESTRICT out, TData *NEK_RESTRICT wsp, const TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
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

    const unsigned int nqTot        = nq0 * nq1;
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                 = getGlobalIdx<0>(threadBlock);
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            TData *wspptr = wsp + nq1 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseQuadSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nq0, nq1, basis0, basis1, inptr, outptr,
                wspptr, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            TData *wspptr = wsp + nq1 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTriSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, inptr,
                outptr, wspptr, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTri)
        {
            TData *out1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wspptr  = wsp + nmTot * nelmt * ncomp +
                            nq1 * (nelmt * c + warpsize * iwarp);

            IProductWRTBaseTriSumFacKernel<SCALE, false>(
                ilane, nm0, nm1, nq0, nq1, isModified, basis0, basis1, inptr,
                out1ptr, wspptr, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFac inner product, quadrature
 * metric included.
 *
 * Unpacks the element sizes from @p sizeParam3D, fetches the shared-memory
 * base, then grid-strides over the block's elements, calling the shape
 * kernel selected at compile time. Each iteration slices the workspace for
 * the thread's warp and component: an nq1 * nq2 region followed by an nq2
 * region for the two intermediate stages and, for the nodal shapes, an
 * nmTot region ahead of them for the modal result that MatVecKernel then
 * maps onto @p out with the transpose of @p nodToMod.
 *
 * @note For NodalTet and NodalPrism the shape kernel is instantiated with
 * this launcher's @p APPEND, whereas the 2D launcher instantiates the
 * NodalTri kernel with APPEND false and lets the flag govern the nodal
 * mapping alone. The modal slice is carved from the workspace and is not
 * zeroed, so under APPEND the shape kernel accumulates onto whatever the
 * slice holds before MatVecKernel accumulates the mapped result onto
 * @p out. IProductWRTBase itself launches with APPEND false only.
 *
 * @tparam SHAPE_TYPE       Hex, Tet, NodalTet, Prism, NodalPrism or Pyr.
 * @tparam Implementation   Operators::SumFac.
 * @tparam SCALE,APPEND,DEFORMED    Passed through to the shape kernel.
 * @tparam TSizeParameter3D Size-parameter type carrying the mode and point
 *                          counts.
 * @tparam TthreadBlock     Back-end thread-block handle type.
 * @tparam TData            Floating-point type of the field data.
 *
 * @param   sizeParam3D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1,index2    Mode-index tables; unused by SumFac,
 *                      whose kernels walk the mode ordering with running
 *                      counters instead.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   nodToMod    Nodal-to-modal matrix, applied transposed; nodal
 *                      shapes only.
 * @param   jac         Block's Jacobians.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Block's workspace, sized by
 *                      IProductWRTBaseWorkSpaceSize and scaled by ncomp.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base; unused by SumFac.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, bool DEFORMED, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTBaseKernelLauncher(
        TSizeParameter3D sizeParam3D, const size_t nelmt, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT w0,
        const TData *NEK_RESTRICT w1, const TData *NEK_RESTRICT w2,
        const TData *NEK_RESTRICT nodToMod, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
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

    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int jacsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *jacptr =
            DEFORMED ? jac + jacsize * warpsize * iwarp : jac + e;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseHexSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2, w0,
                w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {

            TData *out1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0    = wsp + nmTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nmTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0    = wsp + nmTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nmTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePyrSumFacKernel<SCALE, APPEND, DEFORMED>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, w0, w1, w2, jacptr, inptr, outptr, wsp0, wsp1, scale);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

/**
 * @brief Device entry point of the 3D SumFac inner product without the
 * quadrature metric; selected by the absence of the weight and Jacobian
 * arguments.
 *
 * The same grid-stride loop and workspace slicing as the quadrature
 * overload, calling the unweighted shape kernels; the note on @p APPEND
 * for the nodal shapes applies here too.
 *
 * @param   sizeParam3D Element sizes.
 * @param   nelmt       Elements in the block, padding included.
 * @param   isModified  First-direction basis is eModified_A.
 * @param   index0,index1,index2    Mode-index tables; unused by SumFac.
 * @param   basis0,basis1,basis2    Basis tables per direction.
 * @param   nodToMod    Nodal-to-modal matrix, applied transposed; nodal
 *                      shapes only.
 * @param   in          Block's physical values, every component
 *                      concatenated.
 * @param   out         Block's coefficients, every component concatenated.
 * @param   wsp         Block's workspace.
 * @param   scale       Factor applied when @p SCALE is set.
 * @param   shmemptr    Dynamic shared-memory base; unused by SumFac.
 * @param   threadBlock Thread-block handle the index helpers read.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool SCALE, bool APPEND, typename TSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    IProductWRTBaseKernelLauncher(
        const TSizeParameter3D sizeParam3D, const size_t nelmt,
        const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nodToMod,
        const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
        TData *NEK_RESTRICT wsp, const TData scale,
        [[maybe_unused]] unsigned char *shmemptr,
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

    const unsigned int nqTot = nq0 * nq1 * nq2;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c     = getBlockIdx<1>(threadBlock);
    const unsigned int ncomp = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nmTot * (nelmt * c + warpsize * iwarp);
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseHexSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, basis0, basis1, basis2,
                inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalTet)
        {

            TData *out1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0    = wsp + nmTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nmTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBaseTetSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, outptr, wsp0, wsp1, scale);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            TData *out1ptr = wsp + nmTot * (nelmt * c + warpsize * iwarp);
            TData *wsp0    = wsp + nmTot * nelmt * ncomp +
                          nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + (nmTot + nq1 * nq2) * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePrismSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, out1ptr, wsp0, wsp1, scale);

            // Multiply by transpose notToMod to transform coeffs.
            MatVecKernel<APPEND, true>(ilane, nmTot, nodToMod, out1ptr, outptr);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            TData *wsp0 = wsp + nq1 * nq2 * (nelmt * c + warpsize * iwarp);
            TData *wsp1 = wsp + nq1 * nq2 * nelmt * ncomp +
                          nq2 * (nelmt * c + warpsize * iwarp);
            IProductWRTBasePyrSumFacKernel<SCALE, APPEND>(
                ilane, nm0, nm1, nm2, nq0, nq1, nq2, isModified, basis0, basis1,
                basis2, inptr, outptr, wsp0, wsp1, scale);
        }
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
