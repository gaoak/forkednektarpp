///////////////////////////////////////////////////////////////////////////////
//
// File: PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp
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
// Description: Device SumFacTOP kernels of the lift-and-project element
// operator.
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file PhysLiftProjectBlockOpDeviceSumFacTOPKernels.hpp
 * @brief Device SumFacTOP kernels of the lift-and-project operator: the
 * direct lift of the Pointwise form, the fused launches of the StdInvMass
 * form and the dense projector's lift, and the Pointwise form and the
 * diagonal inverse mass the separate-operator route takes.
 *
 * @details
 * As the other SumFacTOP Device kernels, the block at width one, each
 * thread block striding over the elements and its threads over the
 * element's points or modes, so that consecutive threads touch consecutive
 * entries. Each kernel has the name and arguments of its SumFac
 * counterpart (PhysLiftProjectBlockOpDeviceSumFacKernels.hpp), restricted to
 * Implementation SumFacTOP, so that the caller launches
 * `Kernel<Implementation, ...>` for either.
 *
 * The fused launches chain the per-element functions of the
 * IProductWRTPhysTrace, IProductWRTBase and BwdTrans SumFacTOP kernels on
 * one shared-memory buffer per thread block, as the Mass kernels chain
 * BwdTrans and IProductWRTBase: the element's inner products, its
 * coefficients, the padded rows the basis transpose reads (conflict-free,
 * as in Mass), two stage scratches shared by the basis transpose, the
 * backward transform and, in two dimensions, the lift, and the basis
 * tables, staged once per
 * thread block; each kernel reserves only the buffers of the stages it runs
 * (PhysLiftProjectSharedLayout()), and its launch is sized from the same
 * sizes. The block size is GetDeviceKernelParameters() of the mode count,
 * as for Mass, the threads striding over points and modes alike.
 *
 * Two kernels take one thread per point of several whole elements per
 * thread block instead, because an element has as few as one point there
 * and one element per block would leave most of each block idle:
 * PhysLiftProjectDirectLiftKernelLauncher, the Pointwise form's direct
 * lift, and PhysLiftProjectLiftJacobianKernelLauncher, the lift and the
 * division by the Jacobian ahead of the dense projector of tetrahedra,
 * prisms and pyramids (PhysLiftProjectDenseTail()). A block takes as many
 * whole elements as fit in 128 points, one at least,
 * and each thread computes its element of the block, then its point. Their
 * launch bounds are the default block size, which is what
 * GetMaxThreadPerBlock() gives for a trace size parameter.
 */

#pragma once

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/BasicUtils/SafeDivide.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsDeviceKernels.hpp>
#include <MultiRegions/ElmtOps/BwdTrans/BwdTransBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceBlockOpDeviceSumFacTOPKernels.hpp>
#include <MultiRegions/ElmtOps/PhysLiftProject/PhysLiftProjectBlockOp.hpp>

#include <type_traits>

namespace Nektar::MultiRegions::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)

/**
 * @brief The integral of one face of an affine element against the hat
 * functions of its in-face point (@p a, @p b): the face values times the
 * face's Jacobian @p jac and the in-face weights, contracted with the
 * eInterp table of each in-face direction that is not collocated. The
 * products and sums are those of IPWRTPhysFaceTOPKernel(), in its order, so
 * that the lift is the face-by-face lift's to the last bit.
 *
 * @param   a,b        The point in the face's first and second in-face
 *                     directions, as a volume point.
 * @param   tnq0,tnq1  Trace points of the two in-face directions.
 * @param   tbasis0,tbasis1  eInterp tables to the trace points,
 *                     `tbasis[p * tnq + i]`.
 * @param   tw0,tw1    In-face trace weights.
 * @param   in         The face's values, `in[j * tnq0 + i]`.
 * @param   isCollocated0,isCollocated1  The in-face direction's trace
 *                     points are the volume points.
 */
template <typename TData>
NEK_DEVICE_INLINE static TData PhysLiftProjectFaceIntegral(
    const unsigned int a, const unsigned int b, const unsigned int tnq0,
    const unsigned int tnq1, const TData *NEK_RESTRICT tbasis0,
    const TData *NEK_RESTRICT tbasis1, const TData *NEK_RESTRICT tw0,
    const TData *NEK_RESTRICT tw1, const TData jac,
    const TData *NEK_RESTRICT in, const bool isCollocated0,
    const bool isCollocated1)
{
    if (isCollocated0 && isCollocated1)
    {
        return in[b * tnq0 + a] * jac * tw0[a] * tw1[b];
    }
    if (isCollocated0)
    {
        TData sum = 0.0;
        for (unsigned int j = 0; j < tnq1; ++j)
        {
            sum += in[j * tnq0 + a] * jac * tbasis1[b * tnq1 + j] * tw1[j];
        }
        return sum * tw0[a];
    }
    if (isCollocated1)
    {
        TData sum = 0.0;
        for (unsigned int i = 0; i < tnq0; ++i)
        {
            sum += in[b * tnq0 + i] * jac * tbasis0[a * tnq0 + i] * tw0[i];
        }
        return sum * tw1[b];
    }
    TData sum = 0.0;
    for (unsigned int j = 0; j < tnq1; ++j)
    {
        TData sum0 = 0.0;
        for (unsigned int i = 0; i < tnq0; ++i)
        {
            sum0 += in[j * tnq0 + i] * jac * tbasis0[a * tnq0 + i] * tw0[i];
        }
        sum += sum0 * tbasis1[b * tnq1 + j] * tw1[j];
    }
    return sum;
}

/**
 * @brief The lift of the @p NFACE faces of one normal direction of an
 * affine element onto its point at @p x along that direction and (@p a,
 * @p b) in the faces' in-face directions, added to @p val face by face:
 * each face's integral (PhysLiftProjectFaceIntegral()) times the
 * end-point interpolant of @p x, or, where the direction's rule has its
 * end points, the integral itself on the face's own plane alone.
 *
 * @param   nx       Volume points along the normal direction.
 * @param   nbasis   End-point interpolant of the direction,
 *                   `nbasis[x * NFACE + f]`.
 * @param   tjac     Jacobian of each of the direction's faces.
 * @param   in       Values of the direction's first face, the others
 *                   following it.
 */
template <unsigned int NFACE, typename TData>
NEK_DEVICE_INLINE static TData PhysLiftProjectAddFaces(
    TData val, const unsigned int x, const unsigned int nx,
    const unsigned int a, const unsigned int b, const unsigned int tnq0,
    const unsigned int tnq1, const TData *NEK_RESTRICT nbasis,
    const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
    const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
    const TData *NEK_RESTRICT tjac, const TData *NEK_RESTRICT in,
    const bool endPtsCollocated, const bool isCollocated0,
    const bool isCollocated1)
{
    for (unsigned int f = 0; f < NFACE; ++f)
    {
        const TData *NEK_RESTRICT face = in + f * tnq0 * tnq1;
        if (endPtsCollocated)
        {
            if (x == f * (nx - 1u))
            {
                val += PhysLiftProjectFaceIntegral(
                    a, b, tnq0, tnq1, tbasis0, tbasis1, tw0, tw1, tjac[f], face,
                    isCollocated0, isCollocated1);
            }
        }
        else
        {
            val += PhysLiftProjectFaceIntegral(a, b, tnq0, tnq1, tbasis0,
                                               tbasis1, tw0, tw1, tjac[f], face,
                                               isCollocated0, isCollocated1) *
                   nbasis[x * NFACE + f];
        }
    }
    return val;
}

/**
 * @brief Size of sum-factorisation stage scratch @p stage (zero or one) of
 * a fused SumFacTOP kernel that runs the lift (@p LIFT), the basis
 * transpose (@p TRANSPOSE) and the backward transform (@p BWD) or some of
 * them: the largest of what the IProductWRTPhysTrace, IProductWRTBase and
 * BwdTrans SumFacTOP kernels it runs ask of it, the last two as the Mass
 * kernels size them, the basis transpose's with its padded rows. The
 * three-dimensional lift (PhysLiftProjectLoadLift3DTOPKernel()) takes no
 * scratch. The
 * collapsed shapes' basis transpose takes the two scratches the other way
 * round (the IProductWRTBase3DSumFacTOPKernel() calls), so that each stage
 * holds the same intermediate for both transforms.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool LIFT, bool TRANSPOSE,
          bool BWD, typename TData, typename TSizeParameter,
          typename TTraceSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr unsigned int PhysLiftProjectStageSize(
    [[maybe_unused]] const TSizeParameter sizeParam,
    [[maybe_unused]] const TTraceSizeParameter traceSizeParam,
    [[maybe_unused]] const unsigned int stage)
{
    unsigned int size = 0u;
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        if (stage == 0)
        {
            const unsigned int nm0 = sizeParam.nm0();
            const unsigned int nq1 = sizeParam.nq1();
            if constexpr (LIFT)
            {
                size =
                    2u * IProductWRTPhysTraceEdgeModeBlockSize(traceSizeParam);
            }
            if constexpr (TRANSPOSE)
            {
                size = std::max(
                    size,
                    nm0 * nq1 +
                        IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                            sizeParam));
            }
            if constexpr (BWD)
            {
                size = std::max(size, (SHAPE_TYPE == LibUtilities::Quad)
                                          ? sizeParam.nm1() * sizeParam.nq0()
                                          : nm0 * nq1);
            }
        }
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const unsigned int nm0  = sizeParam.nm0();
        const unsigned int nm1  = sizeParam.nm1();
        const unsigned int nm2  = sizeParam.nm2();
        const unsigned int nq0  = sizeParam.nq0();
        const unsigned int nq1  = sizeParam.nq1();
        const unsigned int nq2  = sizeParam.nq2();
        const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
        unsigned int transpose = 0u, bwd = 0u;
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            transpose = (stage == 0) ? nm0 * nq1 * nq2 : nm0 * nm1 * nq2;
            bwd       = (stage == 0) ? nq0 * nm1 * nm2 : nq0 * nq1 * nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            transpose = (stage == 0) ? nm01 * nq2 : nm0 * nq1 * nq2;
            bwd       = transpose;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::Pyr ||
                           SHAPE_TYPE == LibUtilities::NodalPyr)
        {
            transpose = (stage == 0) ? nm0 * nm1 * nq2 : nm0 * nq1 * nq2;
            bwd       = transpose;
        }
        if constexpr (TRANSPOSE)
        {
            size = std::max(
                size,
                transpose + IProductWRTBaseWorkSpacePadding<SHAPE_TYPE, TData>(
                                sizeParam));
        }
        if constexpr (BWD)
        {
            size = std::max(size, bwd);
        }
    }
    return size;
}

/**
 * @brief Shared memory one block of a fused SumFacTOP kernel takes, buffer
 * by buffer, in TData values, each rounded up to 16 bytes; zero for a
 * buffer the kernel does not use. The kernels carve their buffers out in
 * this order with no gaps, from the same sizes the launch is sized with.
 */
struct PhysLiftProjectSharedSizes
{
    /// The element's coefficients.
    unsigned int coef = 0u;
    /// Its inner products.
    unsigned int vol = 0u;
    /// Their copy with padded rows, which the basis transpose reads.
    unsigned int ipb = 0u;
    /// The two stage scratches of the lift and the transforms.
    unsigned int wsp0 = 0u;
    unsigned int wsp1 = 0u;
    /// The staged basis tables.
    unsigned int basis = 0u;

    NEK_HOSTDEVICE_INLINE constexpr unsigned int Total() const
    {
        return coef + vol + ipb + wsp0 + wsp1 + basis;
    }
};

/**
 * @brief The shared-memory buffers of fused SumFacTOP kernel @p KERNEL of
 * the StdInvMass form (PhysLiftProjectSharedSizes): the fused launch takes
 * every buffer; the launch before the dense product everything but the
 * coefficients, and the one after it the coefficients, the backward
 * transform's scratch and the basis tables.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, PhysLiftProjectKernel KERNEL,
          typename TData, typename TSizeParameter, typename TTraceSizeParameter>
NEK_HOSTDEVICE_INLINE constexpr PhysLiftProjectSharedSizes
PhysLiftProjectSharedLayout(const TSizeParameter sizeParam,
                            const TTraceSizeParameter traceSizeParam)
{
    // The stages the kernel runs.
    constexpr bool PROJECT = KERNEL == PhysLiftProjectKernel::Fused;
    constexpr bool LIFT    = KERNEL != PhysLiftProjectKernel::ScaleBwdTrans;
    constexpr bool TRANSPOSE =
        PROJECT || KERNEL == PhysLiftProjectKernel::LiftBasisTranspose;
    constexpr bool BWD =
        PROJECT || KERNEL == PhysLiftProjectKernel::ScaleBwdTrans;
    constexpr bool MULTID = !IsSizeParameter1D_v<TSizeParameter>;

    unsigned int nqTot = sizeParam.nq0();
    unsigned int basis = 0u;
    if constexpr (IsSizeParameter2D_v<TSizeParameter>)
    {
        const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam);
        nqTot *= sizeParam.nq1();
        basis = basisDataSize.total();
    }
    else if constexpr (IsSizeParameter3D_v<TSizeParameter>)
    {
        const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam);
        nqTot *= sizeParam.nq1() * sizeParam.nq2();
        basis = basisDataSize.total();
    }

    PhysLiftProjectSharedSizes size;
    if constexpr (BWD)
    {
        size.coef = SharedMemoryAlignedSize<TData>(sizeParam.nmTot());
    }
    if constexpr (LIFT)
    {
        size.vol = SharedMemoryAlignedSize<TData>(nqTot);
    }
    if constexpr (TRANSPOSE && MULTID)
    {
        size.ipb = SharedMemoryAlignedSize<TData>(
            IProductWRTBasePaddedInputSize<TData>(sizeParam));
    }
    size.wsp0 = SharedMemoryAlignedSize<TData>(
        PhysLiftProjectStageSize<SHAPE_TYPE, LIFT, TRANSPOSE, BWD, TData>(
            sizeParam, traceSizeParam, 0));
    size.wsp1 = SharedMemoryAlignedSize<TData>(
        PhysLiftProjectStageSize<SHAPE_TYPE, LIFT, TRANSPOSE, BWD, TData>(
            sizeParam, traceSizeParam, 1));
    if constexpr ((TRANSPOSE || BWD) && MULTID)
    {
        size.basis = SharedMemoryAlignedSize<TData>(basis);
    }
    return size;
}

/**
 * @brief Dynamic shared memory one block of fused SumFacTOP kernel
 * @p KERNEL needs, in TData values: the buffers of each of the
 * @p elmtsPerBlock elements (PhysLiftProjectSharedLayout()) back to back,
 * the basis tables, which the elements share, once.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          PhysLiftProjectKernel KERNEL, typename TData, typename TSizeParameter,
          typename TTraceSizeParameter,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
inline constexpr unsigned int PhysLiftProjectSharedMemorySize(
    const TSizeParameter sizeParam, const TTraceSizeParameter traceSizeParam,
    const unsigned int elmtsPerBlock)
{
    const PhysLiftProjectSharedSizes size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, KERNEL, TData>(sizeParam,
                                                               traceSizeParam);
    return elmtsPerBlock * (size.Total() - size.basis) + size.basis;
}

/// `s_coef *= factor * diag`, or by @p factor alone without @p DIAG, then
/// a barrier.
template <bool DIAG, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectCoefScaleTOPKernel(
    const unsigned int nmTot, const TData factor,
    [[maybe_unused]] const TData *NEK_RESTRICT diag, TData *NEK_RESTRICT s_coef,
    const TthreadBlock &threadBlock)
{
    for (unsigned int m = getLocalIdx<0>(threadBlock); m < nmTot;
         m += getLocalRange<0>(threadBlock))
    {
        if constexpr (DIAG)
        {
            s_coef[m] *= factor * diag[m];
        }
        else
        {
            s_coef[m] *= factor;
        }
    }
    localBarrier(threadBlock);
}

/// The lift of a segment's two end values onto @p s_vol, then a barrier
/// (one dimension has no lift function of its own).
template <typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectSegLiftTOPKernel(
    const unsigned int nq0, const TData *NEK_RESTRICT nbasis0,
    const bool endPtsCollocated0, const TData *NEK_RESTRICT trace,
    TData *NEK_RESTRICT s_vol, const TthreadBlock &threadBlock)
{
    for (unsigned int p = getLocalIdx<0>(threadBlock); p < nq0;
         p += getLocalRange<0>(threadBlock))
    {
        if (endPtsCollocated0)
        {
            if (p == 0u)
            {
                s_vol[p] += trace[0];
            }
            if (p == nq0 - 1u)
            {
                s_vol[p] += trace[1];
            }
        }
        else
        {
            s_vol[p] +=
                nbasis0[2 * p] * trace[0] + nbasis0[2 * p + 1] * trace[1];
        }
    }
    localBarrier(threadBlock);
}

/**
 * @brief The inner products of one affine three-dimensional element with
 * the trace lifted directly onto every point, written to @p s_vol, then a
 * barrier: the load of the inner products and the lift in one pass, each
 * thread taking its own points.
 *
 * Point (p, q, r) takes from each face the integral of the face's values
 * against the hat functions of the point's in-face position, times the
 * end-point interpolant of the point along the face's normal direction, in
 * the packed trace order (the faces normal to direction 0, then 1, then 2;
 * GetLocTraceBlockAttributes()). A face whose trace points are the volume
 * points in both in-face directions reads the one value beside the point; a
 * face with one such direction contracts the other, as face 0 of a
 * tetrahedron on the default rule, whose Gauss-Jacobi(1,0) points in
 * direction 1 are not the volume's Gauss-Legendre points. The products and
 * sums are those of IProductWRTPhysTrace3DFaceTOPKernel() in its order,
 * without its trace-mode scratch and its two barriers per face.
 *
 * @param   in       The element's inner products, in global memory.
 * @param   trace    The element's trace values of the component.
 * @param   tjac     The element's face Jacobians, one per face.
 * @param   s_vol    The inner products with the lift, in shared memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename TthreadBlock,
          typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectLoadLift3DTOPKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const bool lift, const TData *NEK_RESTRICT nbasis0,
    const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
    const unsigned int tnq00, const unsigned int tnq01,
    const unsigned int tnq10, const unsigned int tnq11,
    const unsigned int tnq20, const unsigned int tnq21,
    const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
    const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
    const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
    const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
    const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
    const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
    const TData *NEK_RESTRICT tjac, const bool isCollocated00,
    const bool isCollocated01, const bool isCollocated10,
    const bool isCollocated11, const bool isCollocated20,
    const bool isCollocated21, const bool endPtsCollocated0,
    const bool endPtsCollocated1, const bool endPtsCollocated2,
    const TData *NEK_RESTRICT in, const TData *NEK_RESTRICT trace,
    TData *NEK_RESTRICT s_vol, const TthreadBlock &threadBlock)
{
    constexpr bool tet =
        SHAPE_TYPE == LibUtilities::Tet || SHAPE_TYPE == LibUtilities::NodalTet;
    constexpr unsigned int nface0 = 2u;
    constexpr unsigned int nface1 = tet ? 1u : 2u;
    constexpr unsigned int nface2 = SHAPE_TYPE == LibUtilities::Hex ? 2u : 1u;

    const unsigned int nq01          = nq0 * nq1;
    const unsigned int nqTot         = nq01 * nq2;
    const TData *NEK_RESTRICT trace1 = trace + nface0 * tnq00 * tnq01;
    const TData *NEK_RESTRICT trace2 = trace1 + nface1 * tnq10 * tnq11;

    for (unsigned int pt = getLocalIdx<0>(threadBlock); pt < nqTot;
         pt += getLocalRange<0>(threadBlock))
    {
        TData val = in[pt];
        if (lift)
        {
            const unsigned int p = pt % nq0;
            const unsigned int q = (pt / nq0) % nq1;
            const unsigned int r = pt / nq01;
            val                  = PhysLiftProjectAddFaces<nface0>(
                val, p, nq0, q, r, tnq00, tnq01, nbasis0, tbasis00, tbasis01,
                tw00, tw01, tjac, trace, endPtsCollocated0, isCollocated00,
                isCollocated01);
            val = PhysLiftProjectAddFaces<nface1>(
                val, q, nq1, p, r, tnq10, tnq11, nbasis1, tbasis10, tbasis11,
                tw10, tw11, tjac + nface0, trace1, endPtsCollocated1,
                isCollocated10, isCollocated11);
            val = PhysLiftProjectAddFaces<nface2>(
                val, r, nq2, p, q, tnq20, tnq21, nbasis2, tbasis20, tbasis21,
                tw20, tw21, tjac + nface0 + nface1, trace2, endPtsCollocated2,
                isCollocated20, isCollocated21);
        }
        s_vol[pt] = val;
    }
    localBarrier(threadBlock);
}

/**
 * @brief The inverse mass of an affine element with the scale at width one:
 * `out[m] = scale / J * diag[m] * in[m]`, @p diag left out unless @p DIAG,
 * zero on the padding elements. @p in may be @p out. Grid dimension 1 is
 * the component.
 */
template <typename Implementation, bool DIAG, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectInvMassScaleKernel(
    const size_t nelmt, const unsigned int nmTot, const TData scale,
    const TData *NEK_RESTRICT diag, const TData *NEK_RESTRICT jac,
    const TData *in, TData *out, const TthreadBlock &threadBlock)
{
    const unsigned int c = getBlockIdx<1>(threadBlock);
    for (size_t e = getBlockIdx<0>(threadBlock); e < nelmt;
         e += getBlockRange<0>(threadBlock))
    {
        const size_t offset = (nelmt * c + e) * nmTot;
        const TData factor  = SafeDivide(scale, jac[e]);
        for (unsigned int m = getLocalIdx<0>(threadBlock); m < nmTot;
             m += getLocalRange<0>(threadBlock))
        {
            if constexpr (DIAG)
            {
                out[offset + m] = factor * diag[m] * in[offset + m];
            }
            else
            {
                out[offset + m] = factor * in[offset + m];
            }
        }
    }
}

/**
 * @brief The StdInvMass form with a diagonal standard inverse mass in one
 * dimension at width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        const TTraceSizeParameter1D traceSizeParam1D, const size_t nelmt,
        const TData scale, const bool lift,
        [[maybe_unused]] const bool isModified,
        [[maybe_unused]] const TData *NEK_RESTRICT basis0,
        const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const bool endPtsCollocated0, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT diag, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT coef,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                    TData>(sizeParam1D, traceSizeParam1D);
    TData *s_coef = (TData *)shmemptr;
    TData *s_vol  = s_coef + size.coef;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nq0 * slot;
        TData *outptr      = out + nq0 * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nq0, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            PhysLiftProjectSegLiftTOPKernel(nq0, nbasis0, endPtsCollocated0,
                                            trace + 2 * slot, s_vol,
                                            threadBlock);
        }

        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, s_vol, s_coef, TData(1.0), threadBlock);
        PhysLiftProjectCoefScaleTOPKernel<true>(nm0, SafeDivide(scale, jac[e]),
                                                diag, s_coef, threadBlock);
        BwdTransSegSumFacTOPKernel<APPEND>(nm0, nq0, basis0, s_coef, outptr,
                                           threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a diagonal standard inverse mass in two
 * dimensions at width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const TData scale, const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac, const bool isCollocated0,
        const bool isCollocated1, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const TData *NEK_RESTRICT jac,
        const TData *NEK_RESTRICT diag, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT lift0, unsigned char *shmemptr,
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
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int tnq00 = traceSizeParam2D.nq00();
    const unsigned int tnq10 = traceSizeParam2D.nq10();
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                    TData>(sizeParam2D, traceSizeParam2D);

    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);

    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The inner product reads rows of length SharedMemoryRowStride. When
    // nq0 is shorter, s_vol is loaded into s_ipb.
    const unsigned int ldi = SharedMemoryRowStride<TData>(nq0);
    const bool isPadded    = (ldi != nq0);

    TData *s_coef   = (TData *)shmemptr;
    TData *s_vol    = s_coef + size.coef;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *ipb      = (isPadded) ? s_ipb : s_vol;
    TData *s_basis0 = s_wsp0 + size.wsp0;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *outptr      = out + nqTot * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nqTot, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            IProductWRTPhysTrace2DTOPKernel<SHAPE_TYPE, false, true>(
                nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0, tbasis1,
                tw00, tw10, tjac + numDataJac * e, s_wsp0,
                trace + nqTotTrace * slot, s_vol, isCollocated0, isCollocated1,
                endPtsCollocated0, endPtsCollocated1, threadBlock);
        }

        if (isPadded)
        {
            LoadWithPaddingToSharedMemory(nq0, nq1, ldi, s_vol, ipb,
                                          threadBlock);
            localBarrier(threadBlock);
        }

        IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
            nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
            s_basis1, ipb, s_coef, s_wsp0, (TData)1.0, threadBlock);
        PhysLiftProjectCoefScaleTOPKernel<true>(
            nmTot, SafeDivide(scale, jac[e]), diag, s_coef, threadBlock);
        BwdTrans2DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
            nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1, s_coef,
            outptr, s_wsp0, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a diagonal standard inverse mass of a
 * block of affine three-dimensional elements in one launch, one element
 * per thread block at width one.
 *
 * The block's threads load the element's inner products into shared
 * memory, coalesced, with the trace lifted directly onto each point as it
 * is loaded (PhysLiftProjectLoadLift3DTOPKernel()), apply the basis
 * transpose into the shared coefficients from the padded rows Mass uses,
 * scale them by `scale / J * diag`, and apply the backward transform into
 * @p out, appending when @p APPEND.
 * One shared-memory buffer (PhysLiftProjectSharedLayout()) serves every
 * stage, and the basis tables are staged once per thread block, as Mass
 * does. The arguments are those of the
 * SumFac counterpart; @p vol, @p coef and the global workspaces are not
 * read, the intermediates living in shared memory.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const TData scale, const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
        const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
        const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
        const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
        const TData *NEK_RESTRICT tjac, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const TData *NEK_RESTRICT jac, const TData *NEK_RESTRICT diag,
        const TData *in, const TData *NEK_RESTRICT trace,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *vol,
        [[maybe_unused]] TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT wsp1,
        [[maybe_unused]] TData *NEK_RESTRICT lift0,
        [[maybe_unused]] TData *NEK_RESTRICT lift1, unsigned char *shmemptr,
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

    const unsigned int tnq00 = traceSizeParam3D.nq00();
    const unsigned int tnq01 = traceSizeParam3D.nq01();
    const unsigned int tnq10 = traceSizeParam3D.nq10();
    const unsigned int tnq11 = traceSizeParam3D.nq11();
    const unsigned int tnq20 = traceSizeParam3D.nq20();
    const unsigned int tnq21 = traceSizeParam3D.nq21();
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();

    const auto size =
        PhysLiftProjectSharedLayout<SHAPE_TYPE, PhysLiftProjectKernel::Fused,
                                    TData>(sizeParam3D, traceSizeParam3D);

    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);

    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The inner product reads rows of length SharedMemoryRowStride. When
    // nq0 is shorter, s_vol is loaded into s_ipb.
    const unsigned int ldi = SharedMemoryRowStride<TData>(nq0);
    const bool isPadded    = (ldi != nq0);

    TData *s_coef   = (TData *)shmemptr;
    TData *s_vol    = s_coef + size.coef;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *ipb      = (isPadded) ? s_ipb : s_vol;
    TData *s_wsp1   = s_wsp0 + size.wsp0;
    TData *s_basis0 = s_wsp1 + size.wsp1;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_basis2 = s_basis1 + basisDataSize.basis1();

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadToSharedMemory(basisDataSize.basis2(), basis2, s_basis2, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *outptr      = out + nqTot * slot;

        // The inner products into shared memory with the trace lifted onto
        // them.
        PhysLiftProjectLoadLift3DTOPKernel<SHAPE_TYPE>(
            nq0, nq1, nq2, lift, nbasis0, nbasis1, nbasis2, tnq00, tnq01, tnq10,
            tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
            tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21,
            tjac + numDataJac * e, isCollocated00, isCollocated01,
            isCollocated10, isCollocated11, isCollocated20, isCollocated21,
            endPtsCollocated0, endPtsCollocated1, endPtsCollocated2, inptr,
            trace + nqTotTrace * slot, s_vol, threadBlock);

        if (isPadded)
        {
            LoadWithPaddingToSharedMemory(nq0, nq1 * nq2, ldi, s_vol, ipb,
                                          threadBlock);
            localBarrier(threadBlock);
        }

        constexpr bool hex = (SHAPE_TYPE == LibUtilities::Hex);
        IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
            index1, index2, s_basis0, s_basis1, s_basis2, ipb, s_coef,
            hex ? s_wsp0 : s_wsp1, hex ? s_wsp1 : s_wsp0, (TData)1.0,
            threadBlock);
        PhysLiftProjectCoefScaleTOPKernel<true>(
            nmTot, SafeDivide(scale, jac[e]), diag, s_coef, threadBlock);
        BwdTrans3DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
            nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
            s_basis0, s_basis1, s_basis2, s_coef, outptr, s_wsp0, s_wsp1,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form before the product in one dimension at
 * width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter1D, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        [[maybe_unused]] const TTraceSizeParameter1D traceSizeParam1D,
        const size_t nelmt, const bool lift,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const bool endPtsCollocated0, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *vol, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    TData *s_vol = (TData *)shmemptr;

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nq0 * slot;
        TData *coefptr     = coef + nm0 * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nq0, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            PhysLiftProjectSegLiftTOPKernel(nq0, nbasis0, endPtsCollocated0,
                                            trace + 2 * slot, s_vol,
                                            threadBlock);
        }
        IProductWRTBaseSegSumFacTOPKernel<false, false>(
            nm0, nq0, basis0, s_vol, coefptr, TData(1.0), threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form before the product in two dimensions
 * at width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter2D, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT nbasis0, const TData *NEK_RESTRICT nbasis1,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw10,
        const TData *NEK_RESTRICT tjac, const bool isCollocated0,
        const bool isCollocated1, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const TData *in,
        const TData *NEK_RESTRICT trace, TData *NEK_RESTRICT coef,
        [[maybe_unused]] TData *vol, [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT lift0, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    const unsigned int tnq00 = traceSizeParam2D.nq00();
    const unsigned int tnq10 = traceSizeParam2D.nq10();
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();

    const auto size = PhysLiftProjectSharedLayout<
        SHAPE_TYPE, PhysLiftProjectKernel::LiftBasisTranspose, TData>(
        sizeParam2D, traceSizeParam2D);

    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);

    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The inner product reads rows of length SharedMemoryRowStride. When
    // nq0 is shorter, s_vol is loaded into s_ipb.
    const unsigned int ldi = SharedMemoryRowStride<TData>(nq0);
    const bool isPadded    = (ldi != nq0);

    TData *s_vol    = (TData *)shmemptr;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *ipb      = (isPadded) ? s_ipb : s_vol;
    TData *s_basis0 = s_wsp0 + size.wsp0;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *coefptr     = coef + nmTot * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nqTot, inptr, s_vol, threadBlock);
        localBarrier(threadBlock);

        if (lift)
        {
            IProductWRTPhysTrace2DTOPKernel<SHAPE_TYPE, false, true>(
                nq0, nq1, nbasis0, nbasis1, tnq00, tnq10, tbasis0, tbasis1,
                tw00, tw10, tjac + numDataJac * e, s_wsp0,
                trace + nqTotTrace * slot, s_vol, isCollocated0, isCollocated1,
                endPtsCollocated0, endPtsCollocated1, threadBlock);
        }

        if (isPadded)
        {
            LoadWithPaddingToSharedMemory(nq0, nq1, ldi, s_vol, ipb,
                                          threadBlock);
            localBarrier(threadBlock);
        }

        IProductWRTBase2DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
            nm0, nm1, nmTot, nq0, nq1, nqTot, isModified, index0, s_basis0,
            s_basis1, ipb, coefptr, s_wsp0, (TData)1.0, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, before
 * the product, of a block of three-dimensional affine elements in one
 * launch, one element per thread block at width one: the inner products
 * loaded into shared memory with the trace lifted directly onto each point
 * (PhysLiftProjectLoadLift3DTOPKernel()) and the basis transpose written
 * to @p coef, coalesced. The shared memory holds
 * no coefficients (PhysLiftProjectSharedLayout()).
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TSizeParameter3D, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectLiftBasisTransposeKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const bool lift, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        const unsigned int *NEK_RESTRICT index1,
        const unsigned int *NEK_RESTRICT index2,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
        const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
        const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
        const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
        const TData *NEK_RESTRICT tjac, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const TData *in, const TData *NEK_RESTRICT trace,
        TData *NEK_RESTRICT coef, [[maybe_unused]] TData *vol,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT wsp1,
        [[maybe_unused]] TData *NEK_RESTRICT lift0,
        [[maybe_unused]] TData *NEK_RESTRICT lift1, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const unsigned int tnq00 = traceSizeParam3D.nq00();
    const unsigned int tnq01 = traceSizeParam3D.nq01();
    const unsigned int tnq10 = traceSizeParam3D.nq10();
    const unsigned int tnq11 = traceSizeParam3D.nq11();
    const unsigned int tnq20 = traceSizeParam3D.nq20();
    const unsigned int tnq21 = traceSizeParam3D.nq21();
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();

    const auto size = PhysLiftProjectSharedLayout<
        SHAPE_TYPE, PhysLiftProjectKernel::LiftBasisTranspose, TData>(
        sizeParam3D, traceSizeParam3D);

    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);

    const unsigned int numDataJac =
        LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The inner product reads rows of length SharedMemoryRowStride. When
    // nq0 is shorter, s_vol is loaded into s_ipb.
    const unsigned int ldi = SharedMemoryRowStride<TData>(nq0);
    const bool isPadded    = (ldi != nq0);

    TData *s_vol    = (TData *)shmemptr;
    TData *s_ipb    = s_vol + size.vol;
    TData *s_wsp0   = s_ipb + size.ipb;
    TData *ipb      = (isPadded) ? s_ipb : s_vol;
    TData *s_wsp1   = s_wsp0 + size.wsp0;
    TData *s_basis0 = s_wsp1 + size.wsp1;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_basis2 = s_basis1 + basisDataSize.basis1();

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadToSharedMemory(basisDataSize.basis2(), basis2, s_basis2, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot  = nelmt * c + e;
        const TData *inptr = in + nqTot * slot;
        TData *coefptr     = coef + nmTot * slot;

        // The inner products into shared memory with the trace lifted onto
        // them.
        PhysLiftProjectLoadLift3DTOPKernel<SHAPE_TYPE>(
            nq0, nq1, nq2, lift, nbasis0, nbasis1, nbasis2, tnq00, tnq01, tnq10,
            tnq11, tnq20, tnq21, tbasis00, tbasis01, tbasis10, tbasis11,
            tbasis20, tbasis21, tw00, tw01, tw10, tw11, tw20, tw21,
            tjac + numDataJac * e, isCollocated00, isCollocated01,
            isCollocated10, isCollocated11, isCollocated20, isCollocated21,
            endPtsCollocated0, endPtsCollocated1, endPtsCollocated2, inptr,
            trace + nqTotTrace * slot, s_vol, threadBlock);

        if (isPadded)
        {
            LoadWithPaddingToSharedMemory(nq0, nq1 * nq2, ldi, s_vol, ipb,
                                          threadBlock);
            localBarrier(threadBlock);
        }

        constexpr bool hex = (SHAPE_TYPE == LibUtilities::Hex);
        IProductWRTBase3DSumFacTOPKernel<SHAPE_TYPE, false, false, true>(
            nm0, nm1, nm2, nmTot, nq0, nq1, nq2, nqTot, isModified, index0,
            index1, index2, s_basis0, s_basis1, s_basis2, ipb, coefptr,
            hex ? s_wsp0 : s_wsp1, hex ? s_wsp1 : s_wsp0, (TData)1.0,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form after the product in one dimension at
 * width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter1D,
          typename TTraceSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter1D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter1D sizeParam1D,
        [[maybe_unused]] const TTraceSizeParameter1D traceSizeParam1D,
        const size_t nelmt, const TData scale,
        [[maybe_unused]] const bool isModified,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT jac,
        TData *NEK_RESTRICT coef, TData *NEK_RESTRICT out,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    TData *s_coef = (TData *)shmemptr;

    const unsigned int nm0 = sizeParam1D.nm0();
    const unsigned int nq0 = sizeParam1D.nq0();

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot = nelmt * c + e;
        TData *coefptr    = coef + nm0 * slot;
        TData *outptr     = out + nq0 * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nm0, coefptr, s_coef, threadBlock);
        localBarrier(threadBlock);

        PhysLiftProjectCoefScaleTOPKernel<false>(nm0, SafeDivide(scale, jac[e]),
                                                 (const TData *)nullptr, s_coef,
                                                 threadBlock);
        BwdTransSegSumFacTOPKernel<APPEND>(nm0, nq0, basis0, s_coef, outptr,
                                           threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The dense StdInvMass form after the product in two dimensions at
 * width one; see the three-dimensional overload.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter2D,
          typename TTraceSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter2D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter2D sizeParam2D,
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const TData scale, const bool isModified,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index0,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT jac, TData *NEK_RESTRICT coef,
        TData *NEK_RESTRICT out, [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam2D.nm0();
    const unsigned int nm1   = sizeParam2D.nm1();
    const unsigned int nmTot = sizeParam2D.nmTot();
    const unsigned int nq0   = sizeParam2D.nq0();
    const unsigned int nq1   = sizeParam2D.nq1();
    const unsigned int nqTot = nq0 * nq1;

    const auto size = PhysLiftProjectSharedLayout<
        SHAPE_TYPE, PhysLiftProjectKernel::ScaleBwdTrans, TData>(
        sizeParam2D, traceSizeParam2D);

    const BasisDataSize2D<SHAPE_TYPE> basisDataSize(sizeParam2D);

    TData *s_coef   = (TData *)shmemptr;
    TData *s_wsp0   = s_coef + size.coef;
    TData *s_basis0 = s_wsp0 + size.wsp0;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot = nelmt * c + e;
        TData *coefptr    = coef + nmTot * slot;
        TData *outptr     = out + nqTot * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nmTot, coefptr, s_coef, threadBlock);
        localBarrier(threadBlock);

        PhysLiftProjectCoefScaleTOPKernel<false>(
            nmTot, SafeDivide(scale, jac[e]), (const TData *)nullptr, s_coef,
            threadBlock);
        BwdTrans2DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
            nm0, nm1, nq0, nq1, nqTot, isModified, s_basis0, s_basis1, s_coef,
            outptr, s_wsp0, threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The StdInvMass form with a dense standard inverse mass, after
 * the product, of a block of three-dimensional affine elements in one
 * launch, one element per thread block at width one: the coefficients
 * loaded into shared memory, coalesced, scaled by `scale / J` there, and
 * the backward transform written to @p out, appending when @p APPEND. The
 * shared memory holds neither the inner products nor their padded copy,
 * and its scratch is sized for the backward transform alone
 * (PhysLiftProjectSharedLayout()); the lift's size parameter is passed for
 * the arguments to match the other launches'.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, typename TSizeParameter3D,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TSizeParameter3D>()))
    PhysLiftProjectScaleBwdTransKernelLauncher(
        const TSizeParameter3D sizeParam3D,
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const TData scale, const bool isModified,
        const unsigned int *NEK_RESTRICT index0,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index1,
        [[maybe_unused]] const unsigned int *NEK_RESTRICT index2,
        const unsigned int *NEK_RESTRICT index3,
        const TData *NEK_RESTRICT basis0, const TData *NEK_RESTRICT basis1,
        const TData *NEK_RESTRICT basis2, const TData *NEK_RESTRICT jac,
        TData *NEK_RESTRICT coef, TData *NEK_RESTRICT out,
        [[maybe_unused]] TData *NEK_RESTRICT wsp0,
        [[maybe_unused]] TData *NEK_RESTRICT wsp1, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nm0   = sizeParam3D.nm0();
    const unsigned int nm1   = sizeParam3D.nm1();
    const unsigned int nm2   = sizeParam3D.nm2();
    const unsigned int nmTot = sizeParam3D.nmTot();
    const unsigned int nq0   = sizeParam3D.nq0();
    const unsigned int nq1   = sizeParam3D.nq1();
    const unsigned int nq2   = sizeParam3D.nq2();
    const unsigned int nqTot = nq0 * nq1 * nq2;

    const auto size = PhysLiftProjectSharedLayout<
        SHAPE_TYPE, PhysLiftProjectKernel::ScaleBwdTrans, TData>(
        sizeParam3D, traceSizeParam3D);

    const BasisDataSize3D<SHAPE_TYPE> basisDataSize(sizeParam3D);

    TData *s_coef   = (TData *)shmemptr;
    TData *s_wsp0   = s_coef + size.coef;
    TData *s_wsp1   = s_wsp0 + size.wsp0;
    TData *s_basis0 = s_wsp1 + size.wsp1;
    TData *s_basis1 = s_basis0 + basisDataSize.basis0();
    TData *s_basis2 = s_basis1 + basisDataSize.basis1();

    // Copy to shared memory.
    LoadToSharedMemory(basisDataSize.basis0(), basis0, s_basis0, threadBlock);
    LoadToSharedMemory(basisDataSize.basis1(), basis1, s_basis1, threadBlock);
    LoadToSharedMemory(basisDataSize.basis2(), basis2, s_basis2, threadBlock);

    size_t e             = getBlockIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t slot = nelmt * c + e;
        TData *coefptr    = coef + nmTot * slot;
        TData *outptr     = out + nqTot * slot;

        // Copy to shared memory.
        LoadToSharedMemory(nmTot, coefptr, s_coef, threadBlock);
        localBarrier(threadBlock);

        PhysLiftProjectCoefScaleTOPKernel<false>(
            nmTot, SafeDivide(scale, jac[e]), (const TData *)nullptr, s_coef,
            threadBlock);
        BwdTrans3DSumFacTOPKernel<SHAPE_TYPE, APPEND>(
            nm0, nm1, nm2, nq0, nq1, nq2, nqTot, isModified, index0, index3,
            s_basis0, s_basis1, s_basis2, s_coef, outptr, s_wsp0, s_wsp1,
            threadBlock);

        e += getBlockRange<0>(threadBlock);
    }
}

/**
 * @brief The Pointwise form at width one: `out (+)= scale / (w_q J) * in` at
 * every point, zero on the padding elements. Grid dimension 1 is the
 * component.
 *
 * @param   weights  The quadrature weights at every point (WeightsKey at
 *                   width one).
 * @param   jac      The Jacobians (JacobianKey at width one), one per
 *                   element, or one per point when @p DEFORMED.
 */
template <typename Implementation, bool APPEND, bool DEFORMED,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysLiftProjectPointwiseKernel(
    const size_t nelmt, const unsigned int nqTot, const TData scale,
    const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    const unsigned int c = getBlockIdx<1>(threadBlock);
    for (size_t e = getBlockIdx<0>(threadBlock); e < nelmt;
         e += getBlockRange<0>(threadBlock))
    {
        const size_t offset = (nelmt * c + e) * nqTot;
        for (unsigned int q = getLocalIdx<0>(threadBlock); q < nqTot;
             q += getLocalRange<0>(threadBlock))
        {
            TData jacobian;
            if constexpr (DEFORMED)
            {
                jacobian = jac[e * nqTot + q];
            }
            else
            {
                jacobian = jac[e];
            }
            const TData factor = SafeDivide(scale, weights[q] * jacobian);
            if constexpr (APPEND)
            {
                out[offset + q] += factor * in[offset + q];
            }
            else
            {
                out[offset + q] = factor * in[offset + q];
            }
        }
    }
}

/**
 * @brief The factor `scale / (w J)` of the Pointwise form at point @p pt of
 * element @p e at width one; zero on the padding elements, whose Jacobian is
 * zero.
 */
template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData PhysLiftProjectDirectLiftFactorTOPKernel(
    const size_t e, const unsigned int nqTot, const unsigned int pt,
    const TData scale, const TData *NEK_RESTRICT weights,
    const TData *NEK_RESTRICT jac)
{
    TData jacobian;
    if constexpr (DEFORMED)
    {
        jacobian = jac[nqTot * e + pt];
    }
    else
    {
        jacobian = jac[e];
    }
    return SafeDivide(scale, weights[pt] * jacobian);
}

/// `out[ientry] (+)= factor * val`.
template <bool APPEND, typename TData>
NEK_DEVICE_INLINE static void PhysLiftProjectDirectLiftStoreTOPKernel(
    TData *out, const size_t ientry, const TData factor, const TData val)
{
    if constexpr (APPEND)
    {
        out[ientry] += factor * val;
    }
    else
    {
        out[ientry] = factor * val;
    }
}

/**
 * @brief The lift of the traces of a one-point element onto its point at
 * width one: the sum over its traces, in the packed order, of each trace's
 * integral of the trace values times its Jacobian, times the end-point
 * interpolant of the point along the trace's normal direction. A one-point
 * direction's interpolant onto the trace points is one, so no in-trace
 * interpolant is applied.
 *
 * @tparam SHAPE_TYPE  Tri, Tet, Prism or Pyr.
 * @tparam DEFORMED    The trace Jacobians are per trace point.
 *
 * @param   tnq      Trace points of a trace in each normal direction and
 *                   in-trace direction, `tnq[2 * d + t]` (`tnq[d]` in two
 *                   dimensions, one in-trace direction).
 * @param   nbasis   End-point interpolant of each direction, one row.
 * @param   tw       In-trace weights, in the order of @p tnq.
 * @param   tjacptr  The element's trace Jacobians.
 * @param   traceptr The element's trace values of the component.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static TData PhysLiftProjectOnePointLiftTOPKernel(
    const unsigned int *tnq, const TData *const *nbasis, const TData *const *tw,
    const TData *NEK_RESTRICT tjacptr, const TData *NEK_RESTRICT traceptr)
{
    constexpr unsigned int dim = LibUtilities::ShapeTypeDimMap[SHAPE_TYPE];
    // The traces of each direction, taken at compile time: the host table
    // cannot be indexed at run time in device code.
    constexpr unsigned int nface0 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
    constexpr unsigned int nface1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    constexpr unsigned int nface2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];
    const unsigned int nfaces[3] = {nface0, nface1, nface2};
    TData lift                   = 0.0;
    unsigned int k               = 0;
    unsigned int iface           = 0;
    for (unsigned int d = 0; d < dim; ++d)
    {
        const unsigned int tnq0 = (dim == 3) ? tnq[2 * d] : tnq[d];
        const unsigned int tnq1 = (dim == 3) ? tnq[2 * d + 1] : 1u;
        const TData *tw0        = (dim == 3) ? tw[2 * d] : tw[d];
        const TData *tw1        = (dim == 3) ? tw[2 * d + 1] : nullptr;
        for (unsigned int f = 0; f < nfaces[d]; ++f, ++iface)
        {
            TData sum = 0.0;
            for (unsigned int j = 0; j < tnq1; ++j)
            {
                for (unsigned int i = 0; i < tnq0; ++i, ++k)
                {
                    TData faceJac;
                    if constexpr (DEFORMED)
                    {
                        faceJac = tjacptr[k];
                    }
                    else
                    {
                        faceJac = tjacptr[iface];
                    }
                    TData val = traceptr[k] * faceJac * tw0[i];
                    if (dim == 3)
                    {
                        val *= tw1[j];
                    }
                    sum += val;
                }
            }
            lift += sum * nbasis[d][f];
        }
    }
    return lift;
}

/**
 * @brief The Pointwise form of a block of segments on SumFacTOP, the trace
 * lifted directly onto every point: `out (+)= scale / (w J) * (in + h_0(p)
 * g_0 + h_1(p) g_1)`. See the three-dimensional launcher; a segment's
 * traces are points, with no Jacobian or weight, and @p tjac is not read.
 *
 * One thread per volume point of a thread block's elements, over every
 * component, the block at width one. A thread block takes @p elmtsPerBlock
 * whole elements at a time, as many as fit in 128
 * points (one at least): an element has as few as one point here, so one
 * element per block would leave most of each block idle. Each thread computes
 * its element of the block first, then its point (GetDevicePackedThread());
 * consecutive threads take consecutive points of an element and then of the
 * next, so the reads and writes of the volume are coalesced. An element of
 * more points than the block has threads is taken alone, its points strided
 * over the block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TTraceSizeParameter1D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter1D>()))
    PhysLiftProjectDirectLiftKernelLauncher(
        const TTraceSizeParameter1D traceSizeParam1D, const size_t nelmt,
        const unsigned int ncomp, const unsigned int elmtsPerBlock,
        const TData scale, const bool lift, const TData *NEK_RESTRICT nbasis0,
        [[maybe_unused]] const TData *NEK_RESTRICT tjac,
        const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
        const TData *in, const TData *NEK_RESTRICT trace, TData *out,
        const TthreadBlock &threadBlock)
{
    const unsigned int nqTot   = traceSizeParam1D.nm0();
    const size_t compSize      = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize = 2u * nelmt;

    // This thread's element of the block, then its points; a block may hold
    // several elements, and a lone element strides over the whole block.
    const DevicePackedThread thread =
        GetDevicePackedThread(nqTot, elmtsPerBlock, threadBlock);
    for (size_t e0 =
             static_cast<size_t>(getBlockIdx(threadBlock)) * elmtsPerBlock;
         e0 < nelmt;
         e0 += static_cast<size_t>(getBlockRange(threadBlock)) * elmtsPerBlock)
    {
        const size_t e = e0 + thread.elmt;
        if (thread.elmt < elmtsPerBlock && e < nelmt)
        {
            for (unsigned int pt = thread.firstPoint; pt < nqTot;
                 pt += thread.pointStride)
            {
                const TData factor =
                    PhysLiftProjectDirectLiftFactorTOPKernel<DEFORMED>(
                        e, nqTot, pt, scale, weights, jac);
                const TData h0 = nbasis0[2u * pt];
                const TData h1 = nbasis0[2u * pt + 1u];

                for (unsigned int c = 0; c < ncomp; ++c)
                {
                    const size_t ientry = compSize * c + nqTot * e + pt;
                    TData val           = in[ientry];
                    if (lift)
                    {
                        const TData *traceptr =
                            trace + compTraceSize * c + 2u * e;
                        val += h0 * traceptr[0] + h1 * traceptr[1];
                    }
                    PhysLiftProjectDirectLiftStoreTOPKernel<APPEND>(
                        out, ientry, factor, val);
                }
            }
        }
    }
}

/**
 * @brief The Pointwise form of a block of two-dimensional elements on
 * SumFacTOP, the trace lifted directly onto every point. See the
 * three-dimensional launcher: quadrilaterals whose edges have the volume
 * points, each point adding its four edges' contributions (the two
 * direction-0 edges, then the two direction-1 edges), and triangles of one
 * point.
 *
 * One thread per volume point of a thread block's elements, over every
 * component, the block at width one. A thread block takes @p elmtsPerBlock
 * whole elements at a time, as many as fit in 128
 * points (one at least): an element has as few as one point here, so one
 * element per block would leave most of each block idle. Each thread computes
 * its element of the block first, then its point (GetDevicePackedThread());
 * consecutive threads take consecutive points of an element and then of the
 * next, so the reads and writes of the volume are coalesced. An element of
 * more points than the block has threads is taken alone, its points strided
 * over the block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TTraceSizeParameter2D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter2D>()))
    PhysLiftProjectDirectLiftKernelLauncher(
        const TTraceSizeParameter2D traceSizeParam2D, const size_t nelmt,
        const unsigned int ncomp, const unsigned int elmtsPerBlock,
        const TData scale, const bool lift, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT tw00,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tjac,
        const TData *NEK_RESTRICT weights, const TData *NEK_RESTRICT jac,
        const TData *in, const TData *NEK_RESTRICT trace, TData *out,
        const TthreadBlock &threadBlock)
{
    static_assert(SHAPE_TYPE == LibUtilities::Quad ||
                      SHAPE_TYPE == LibUtilities::Tri,
                  "The direct lift takes quadrilaterals and triangles.");

    constexpr bool isCollapsed   = SHAPE_TYPE != LibUtilities::Quad;
    constexpr unsigned int nedge = LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The trace size parameter's nm0() and nm1() are the volume quadrature
    // point counts here (#m_nm), as in the trace operators.
    const unsigned int nq0   = traceSizeParam2D.nm0();
    const unsigned int nq1   = traceSizeParam2D.nm1();
    const unsigned int nqTot = nq0 * nq1;
    const unsigned int nqTotTrace =
        traceSizeParam2D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac = DEFORMED ? nqTotTrace : nedge;

    const size_t compSize      = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize = static_cast<size_t>(nqTotTrace) * nelmt;

    // This thread's element of the block, then its points; a block may hold
    // several elements, and a lone element strides over the whole block.
    const DevicePackedThread thread =
        GetDevicePackedThread(nqTot, elmtsPerBlock, threadBlock);
    for (size_t e0 =
             static_cast<size_t>(getBlockIdx(threadBlock)) * elmtsPerBlock;
         e0 < nelmt;
         e0 += static_cast<size_t>(getBlockRange(threadBlock)) * elmtsPerBlock)
    {
        const size_t e = e0 + thread.elmt;
        if (thread.elmt < elmtsPerBlock && e < nelmt)
        {
            for (unsigned int pt = thread.firstPoint; pt < nqTot;
                 pt += thread.pointStride)
            {
                const TData factor =
                    PhysLiftProjectDirectLiftFactorTOPKernel<DEFORMED>(
                        e, nqTot, pt, scale, weights, jac);
                const TData *tjacptr     = tjac + numDataJac * e;
                const size_t traceOffset = static_cast<size_t>(nqTotTrace) * e;
                const size_t volOffset   = static_cast<size_t>(nqTot) * e + pt;

                if constexpr (!isCollapsed)
                {
                    // The point of each edge beside this point, the edge's
                    // weight there and the end-point interpolant along its
                    // normal.
                    const unsigned int p        = pt % nq0;
                    const unsigned int q        = pt / nq0;
                    const unsigned int k[nedge] = {q, nq1 + q, 2u * nq1 + p,
                                                   2u * nq1 + nq0 + p};
                    const TData w[nedge] = {tw00[q], tw00[q], tw10[p], tw10[p]};
                    const TData h[nedge] = {
                        nbasis0[2u * p], nbasis0[2u * p + 1u], nbasis1[2u * q],
                        nbasis1[2u * q + 1u]};
                    // Each edge's factor, formed once for every component.
                    TData coef[nedge];
                    for (unsigned int iedge = 0; iedge < nedge; ++iedge)
                    {
                        TData edgeJac;
                        if constexpr (DEFORMED)
                        {
                            edgeJac = tjacptr[k[iedge]];
                        }
                        else
                        {
                            edgeJac = tjacptr[iedge];
                        }
                        coef[iedge] = edgeJac * w[iedge] * h[iedge];
                    }
                    for (unsigned int c = 0; c < ncomp; ++c)
                    {
                        const size_t ientry = compSize * c + volOffset;
                        TData val           = in[ientry];
                        if (lift)
                        {
                            const TData *traceptr =
                                trace + compTraceSize * c + traceOffset;
                            for (unsigned int iedge = 0; iedge < nedge; ++iedge)
                            {
                                val += traceptr[k[iedge]] * coef[iedge];
                            }
                        }
                        PhysLiftProjectDirectLiftStoreTOPKernel<APPEND>(
                            out, ientry, factor, val);
                    }
                }
                else
                {
                    const unsigned int tnq[2] = {traceSizeParam2D.nq00(),
                                                 traceSizeParam2D.nq10()};
                    const TData *nbasis[2]    = {nbasis0, nbasis1};
                    const TData *tw[2]        = {tw00, tw10};
                    for (unsigned int c = 0; c < ncomp; ++c)
                    {
                        const size_t ientry = compSize * c + volOffset;
                        TData val           = in[ientry];
                        if (lift)
                        {
                            val +=
                                PhysLiftProjectOnePointLiftTOPKernel<SHAPE_TYPE,
                                                                     DEFORMED>(
                                    tnq, nbasis, tw, tjacptr,
                                    trace + compTraceSize * c + traceOffset);
                        }
                        PhysLiftProjectDirectLiftStoreTOPKernel<APPEND>(
                            out, ientry, factor, val);
                    }
                }
            }
        }
    }
}

/**
 * @brief The Pointwise form of a block of three-dimensional elements on
 * SumFacTOP, the trace lifted directly onto every point, over every
 * component: `out (+)= scale / (w J) * (in + L g)`, zero on the padding
 * elements, whose Jacobian is zero.
 *
 * One thread per volume point of a thread block's elements, over every
 * component, the block at width one. A thread block takes @p elmtsPerBlock
 * whole elements at a time, as many as fit in 128
 * points (one at least): an element has as few as one point here, so one
 * element per block would leave most of each block idle. Each thread computes
 * its element of the block first, then its point (GetDevicePackedThread());
 * consecutive threads take consecutive points of an element and then of the
 * next, so the reads and writes of the volume are coalesced. An element of
 * more points than the block has threads is taken alone, its points strided
 * over the block.
 *
 * @details
 * Hexahedra whose faces have the volume points in both in-face directions:
 * the lift of face @em f onto volume point (p, q, r) is then the face's
 * value at the point beside it, times the face's Jacobian there, its two
 * in-face weights there and the end-point interpolant of the point along
 * the face's normal direction, \f$h(\pm 1)\f$, a delta where the rule has
 * its end points. Each point adds its six faces' contributions directly, in
 * the packed trace order (direction 0's low and high faces, then direction
 * 1's and direction 2's: the hexahedron's faces 4, 2, 1, 3, 0 and 5), each
 * formed as the face-by-face lift of IProductWRTPhysTrace forms it.
 * Consecutive points of an element read consecutive values of the faces
 * normal to directions 1 and 2, and the same value of those normal to
 * direction 0, a broadcast; each face's Jacobian is a broadcast across the
 * element's points.
 *
 * Tetrahedra, prisms and pyramids of one point: every face contributes its
 * integral of the trace values times its Jacobian, times the one-point
 * end-point interpolant (PhysLiftProjectOnePointLiftTOPKernel()).
 *
 * @tparam SHAPE_TYPE  Hex, Tet, Prism or Pyr.
 * @tparam APPEND      Accumulate onto @p out.
 * @tparam DEFORMED    The Jacobians are per point.
 *
 * @param   traceSizeParam3D  Volume point counts per direction (nm0(),
 *                            nm1(), nm2()) and trace point counts per face
 *                            pair.
 * @param   nelmt     Elements in the block, including padding.
 * @param   ncomp     Components (times homogeneous modes) of the block.
 * @param   elmtsPerBlock  Elements per thread block
 *                    (GetElmtsPerBlock()).
 * @param   scale     Factor the result is scaled by.
 * @param   lift      Lift @p trace; without it only the scale is applied.
 * @param   nbasis0,nbasis1,nbasis2  End-point interpolant of each
 *                    direction, `nbasis[p * ntrace + side]`.
 * @param   tw00,tw01,tw10,tw11,tw20,tw21  Weights of each face pair's
 *                    in-face directions (FetchTraceTables()).
 * @param   tjac      Trace Jacobians (JacobianLocTraceKey at width one),
 *                    one per face, or one per trace point when @p DEFORMED.
 * @param   weights   Quadrature weights at every point (WeightsKey at width
 *                    one).
 * @param   jac       Jacobians (JacobianKey at width one), one per element,
 *                    or one per point when @p DEFORMED.
 * @param   in        Inner products; may be @p out.
 * @param   trace     Element-local trace values.
 * @param   out       Result.
 * @param   threadBlock  The thread block.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, typename TTraceSizeParameter3D,
          typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    PhysLiftProjectDirectLiftKernelLauncher(
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const unsigned int ncomp, const unsigned int elmtsPerBlock,
        const TData scale, const bool lift, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
        const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
        const TData *NEK_RESTRICT tjac, const TData *NEK_RESTRICT weights,
        const TData *NEK_RESTRICT jac, const TData *in,
        const TData *NEK_RESTRICT trace, TData *out,
        const TthreadBlock &threadBlock)
{
    static_assert(SHAPE_TYPE == LibUtilities::Hex ||
                      SHAPE_TYPE == LibUtilities::Tet ||
                      SHAPE_TYPE == LibUtilities::Prism ||
                      SHAPE_TYPE == LibUtilities::Pyr,
                  "The direct lift takes hexahedra, tetrahedra, prisms and "
                  "pyramids.");

    constexpr bool isCollapsed   = SHAPE_TYPE != LibUtilities::Hex;
    constexpr unsigned int nface = LibUtilities::ShapeTypeNumTraces[SHAPE_TYPE];

    // The trace size parameter's nm0()..nm2() are the volume quadrature
    // point counts here (#m_nm), as in the trace operators.
    const unsigned int nq0   = traceSizeParam3D.nm0();
    const unsigned int nq1   = traceSizeParam3D.nm1();
    const unsigned int nq2   = traceSizeParam3D.nm2();
    const unsigned int nq01  = nq0 * nq1;
    const unsigned int nqTot = nq01 * nq2;
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int numDataJac = DEFORMED ? nqTotTrace : nface;

    const size_t compSize      = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize = static_cast<size_t>(nqTotTrace) * nelmt;

    // This thread's element of the block, then its points; a block may hold
    // several elements, and a lone element strides over the whole block.
    const DevicePackedThread thread =
        GetDevicePackedThread(nqTot, elmtsPerBlock, threadBlock);
    for (size_t e0 =
             static_cast<size_t>(getBlockIdx(threadBlock)) * elmtsPerBlock;
         e0 < nelmt;
         e0 += static_cast<size_t>(getBlockRange(threadBlock)) * elmtsPerBlock)
    {
        const size_t e = e0 + thread.elmt;
        if (thread.elmt < elmtsPerBlock && e < nelmt)
        {
            for (unsigned int pt = thread.firstPoint; pt < nqTot;
                 pt += thread.pointStride)
            {
                const TData factor =
                    PhysLiftProjectDirectLiftFactorTOPKernel<DEFORMED>(
                        e, nqTot, pt, scale, weights, jac);
                const TData *tjacptr     = tjac + numDataJac * e;
                const size_t traceOffset = static_cast<size_t>(nqTotTrace) * e;
                const size_t volOffset   = static_cast<size_t>(nqTot) * e + pt;

                if constexpr (!isCollapsed)
                {
                    const unsigned int p = pt % nq0;
                    const unsigned int q = (pt / nq0) % nq1;
                    const unsigned int r = pt / nq01;

                    // The point of each face beside this point, the face's
                    // in-face weights there and the end-point interpolant along
                    // its normal.
                    const unsigned int fsz0     = nq1 * nq2;
                    const unsigned int fsz1     = nq0 * nq2;
                    const unsigned int off1     = 2u * fsz0;
                    const unsigned int off2     = off1 + 2u * fsz1;
                    const unsigned int k[nface] = {
                        r * nq1 + q,        fsz0 + r * nq1 + q,
                        off1 + r * nq0 + p, off1 + fsz1 + r * nq0 + p,
                        off2 + q * nq0 + p, off2 + nq01 + q * nq0 + p};
                    const TData w0[nface] = {tw00[q], tw00[q], tw10[p],
                                             tw10[p], tw20[p], tw20[p]};
                    const TData w1[nface] = {tw01[r], tw01[r], tw11[r],
                                             tw11[r], tw21[q], tw21[q]};
                    const TData h[nface]  = {
                        nbasis0[2u * p], nbasis0[2u * p + 1u],
                        nbasis1[2u * q], nbasis1[2u * q + 1u],
                        nbasis2[2u * r], nbasis2[2u * r + 1u]};
                    // Each face's factor, formed once for every component.
                    TData coef[nface];
                    for (unsigned int iface = 0; iface < nface; ++iface)
                    {
                        TData faceJac;
                        if constexpr (DEFORMED)
                        {
                            faceJac = tjacptr[k[iface]];
                        }
                        else
                        {
                            faceJac = tjacptr[iface];
                        }
                        coef[iface] =
                            faceJac * w0[iface] * w1[iface] * h[iface];
                    }
                    for (unsigned int c = 0; c < ncomp; ++c)
                    {
                        const size_t ientry = compSize * c + volOffset;
                        TData val           = in[ientry];
                        if (lift)
                        {
                            const TData *traceptr =
                                trace + compTraceSize * c + traceOffset;
                            for (unsigned int iface = 0; iface < nface; ++iface)
                            {
                                val += traceptr[k[iface]] * coef[iface];
                            }
                        }
                        PhysLiftProjectDirectLiftStoreTOPKernel<APPEND>(
                            out, ientry, factor, val);
                    }
                }
                else
                {
                    const unsigned int tnq[6] = {
                        traceSizeParam3D.nq00(), traceSizeParam3D.nq01(),
                        traceSizeParam3D.nq10(), traceSizeParam3D.nq11(),
                        traceSizeParam3D.nq20(), traceSizeParam3D.nq21()};
                    const TData *nbasis[3] = {nbasis0, nbasis1, nbasis2};
                    const TData *tw[6] = {tw00, tw01, tw10, tw11, tw20, tw21};
                    for (unsigned int c = 0; c < ncomp; ++c)
                    {
                        const size_t ientry = compSize * c + volOffset;
                        TData val           = in[ientry];
                        if (lift)
                        {
                            val +=
                                PhysLiftProjectOnePointLiftTOPKernel<SHAPE_TYPE,
                                                                     DEFORMED>(
                                    tnq, nbasis, tw, tjacptr,
                                    trace + compTraceSize * c + traceOffset);
                        }
                        PhysLiftProjectDirectLiftStoreTOPKernel<APPEND>(
                            out, ientry, factor, val);
                    }
                }
            }
        }
    }
}

/// Components each thread of the dense form's lift takes at once
/// (PhysLiftProjectLiftJacobianKernelLauncher), the compressible flow
/// solver's five, where one component per thread would already fill the
/// device; one otherwise.
static constexpr unsigned int physLiftProjectCompChunk = 5u;

/// Whether the faces of a direction are contractions
/// (PhysLiftProjectLiftDir): never, always, or as the collocation of their
/// in-face directions says at run time.
enum class PhysLiftProjectStaged
{
    Never,
    Always,
    RunTime
};

/// PhysLiftProjectLiftJacobianKernelLauncher's STAGE_MASK when the contracted
/// directions are taken at run time; otherwise bit @em d says direction
/// @em d's faces are contractions.
static constexpr unsigned int physLiftProjectStageRunTime = 8u;

/// The PhysLiftProjectStaged of direction @p DIR under @p STAGE_MASK.
template <unsigned int STAGE_MASK, unsigned int DIR>
inline constexpr PhysLiftProjectStaged physLiftProjectStagedDir =
    STAGE_MASK == physLiftProjectStageRunTime ? PhysLiftProjectStaged::RunTime
    : ((STAGE_MASK >> DIR) & 1u)              ? PhysLiftProjectStaged::Always
                                              : PhysLiftProjectStaged::Never;

/**
 * @brief The @p NFACE faces of one normal direction of an affine element in
 * the dense projector's lift (PhysLiftProjectLiftJacobianKernelLauncher): their
 * trace points in the two in-face directions, where the faces start in the
 * element's packed trace, and the tables of their integrals against the hat
 * functions of the volume points. A face whose trace points are the volume
 * points in both in-face directions is read at one point; otherwise
 * (#Staged()) at most one in-face direction has other points, and the
 * integral is a contraction over that direction (FaceIntegral()), which the
 * kernel forms once per element in shared memory.
 */
template <unsigned int NFACE, PhysLiftProjectStaged STAGED, typename TData>
struct PhysLiftProjectLiftDir
{
    unsigned int tnq0, tnq1, na, nb, offset;
    const TData *NEK_RESTRICT tbasis0;
    const TData *NEK_RESTRICT tbasis1;
    const TData *NEK_RESTRICT tw0;
    const TData *NEK_RESTRICT tw1;
    bool isCollocated0, isCollocated1;

    /**
     * @param   tnq0,tnq1  Trace points of the two in-face directions.
     * @param   na,nb      Volume points of the two in-face directions.
     * @param   offset     The first face's start in the packed trace.
     * @param   tbasis0,tbasis1  eInterp tables onto the trace points.
     * @param   tw0,tw1    In-face trace weights.
     * @param   isCollocated0,isCollocated1  The in-face direction's trace
     *                     points are the volume points.
     */
    NEK_DEVICE_INLINE PhysLiftProjectLiftDir(
        const unsigned int tnq0, const unsigned int tnq1, const unsigned int na,
        const unsigned int nb, const unsigned int offset,
        const TData *NEK_RESTRICT tbasis0, const TData *NEK_RESTRICT tbasis1,
        const TData *NEK_RESTRICT tw0, const TData *NEK_RESTRICT tw1,
        const bool isCollocated0, const bool isCollocated1)
        : tnq0(tnq0), tnq1(tnq1), na(na), nb(nb), offset(offset),
          tbasis0(tbasis0), tbasis1(tbasis1), tw0(tw0), tw1(tw1),
          isCollocated0(isCollocated0), isCollocated1(isCollocated1)
    {
    }

    /// The faces' integrals are contractions, formed in shared memory;
    /// known at compile time unless @p STAGED is RunTime.
    NEK_DEVICE_INLINE bool Staged() const
    {
        if constexpr (STAGED == PhysLiftProjectStaged::RunTime)
        {
            return !(isCollocated0 && isCollocated1);
        }
        else
        {
            return STAGED == PhysLiftProjectStaged::Always;
        }
    }

    /// Shared-memory entries per component: one per face and volume point
    /// of the face's in-face directions, `na * nb` per face.
    NEK_DEVICE_INLINE unsigned int StageSize() const
    {
        return Staged() ? NFACE * na * nb : 0u;
    }

    /// The integral of the face at @p face against the hat functions of
    /// volume point (@p a, @p b) of its in-face directions, contracted over
    /// the direction whose trace points are not the volume's.
    NEK_DEVICE_INLINE TData FaceIntegral(const TData *NEK_RESTRICT face,
                                         const unsigned int a,
                                         const unsigned int b) const
    {
        TData sum = 0.0;
        if (isCollocated0)
        {
            for (unsigned int j = 0; j < tnq1; ++j)
            {
                sum += tbasis1[b * tnq1 + j] * tw1[j] * tw0[a] *
                       face[j * tnq0 + a];
            }
        }
        else
        {
            for (unsigned int j = 0; j < tnq0; ++j)
            {
                sum += tbasis0[a * tnq0 + j] * tw0[j] * tw1[b] *
                       face[b * tnq0 + j];
            }
        }
        return sum;
    }

    /// Shared-memory entry @p s of the direction, of the element's trace
    /// values at @p trace (FaceIntegral()).
    NEK_DEVICE_INLINE TData Stage(const TData *NEK_RESTRICT trace,
                                  const unsigned int s) const
    {
        const unsigned int nab = na * nb;
        const unsigned int f   = s / nab;
        const unsigned int ab  = s % nab;
        return FaceIntegral(trace + offset + f * tnq0 * tnq1, ab % na, ab / na);
    }
};

/**
 * @brief What one point (@p x along the normal, @p a, @p b in the faces)
 * reads of one direction's faces (PhysLiftProjectLiftDir), formed once per
 * thread: the face entry and its weight where the face is read at one
 * point, otherwise the shared-memory entry, and per face the end-point
 * interpolant along the normal.
 */
template <unsigned int NFACE, PhysLiftProjectStaged STAGED, typename TData>
struct PhysLiftProjectLiftPoint
{
    unsigned int index;
    TData weight;
    TData h[NFACE];

    NEK_DEVICE_INLINE PhysLiftProjectLiftPoint(
        const PhysLiftProjectLiftDir<NFACE, STAGED, TData> &dir,
        const unsigned int stageOffset, const unsigned int x,
        const unsigned int nx, const unsigned int a, const unsigned int b,
        const TData *NEK_RESTRICT nbasis, const bool endPtsCollocated)
    {
        if (dir.Staged())
        {
            index  = stageOffset + b * dir.na + a;
            weight = 0.0;
        }
        else
        {
            index  = dir.offset + b * dir.tnq0 + a;
            weight = dir.tw0[a] * dir.tw1[b];
        }
#pragma unroll
        for (unsigned int f = 0; f < NFACE; ++f)
        {
            h[f] = endPtsCollocated ? TData(x == f * (nx - 1u) ? 1.0 : 0.0)
                                    : nbasis[x * NFACE + f];
        }
    }

    /// Add the lift of the direction's faces onto this point to @p val, the
    /// point's values of the chunk's components: each face's integral (read
    /// from the trace values at @p trace, @p compTraceSize apart, or from
    /// the shared-memory entries at @p staged, @p stageSize apart) times its
    /// Jacobian (@p tjac) and the end-point interpolant of the point.
    template <unsigned int CHUNK>
    NEK_DEVICE_INLINE void AddFaceLifts(
        const PhysLiftProjectLiftDir<NFACE, STAGED, TData> &dir,
        const TData *NEK_RESTRICT trace, const size_t compTraceSize,
        const TData *staged, const unsigned int stageSize,
        const TData *NEK_RESTRICT tjac, const unsigned int ncompLeft,
        TData (&val)[CHUNK]) const
    {
        const bool isStaged = dir.Staged();
        const unsigned int faceStep =
            isStaged ? dir.na * dir.nb : dir.tnq0 * dir.tnq1;
#pragma unroll
        for (unsigned int f = 0; f < NFACE; ++f)
        {
            const TData scale = tjac[f] * h[f];
#pragma unroll
            for (unsigned int cc = 0; cc < CHUNK; ++cc)
            {
                if (cc < ncompLeft)
                {
                    const TData sum =
                        isStaged ? staged[stageSize * cc + index + f * faceStep]
                                 : weight * trace[compTraceSize * cc + index +
                                                  f * faceStep];
                    val[cc] += sum * scale;
                }
            }
        }
    }
};

/**
 * @brief The StdInvMass form of a block of affine three-dimensional
 * elements on SumFacTOP, at width one, before the dense projector
 * (PhysLiftProjectDenseTail()): `out = (in + L g) / J` at every point, zero
 * on the padding elements, whose Jacobian is zero.
 *
 * @details
 * Each thread block takes @p elmtsPerBlock elements, one thread per point of
 * each, and a chunk of @p CHUNK components, the grid's second dimension. Point
 * (p, q, r) takes from each face, in the packed trace order, the face's
 * integral against the hat functions of the point's in-face position
 * times the end-point interpolant of the point along the face's normal:
 * one value where the face's trace points are the volume points in both
 * in-face directions, otherwise a contraction over the other direction, as
 * on face 0 of a tetrahedron on the default rule, whose Gauss-Jacobi(1,0)
 * points in direction 1 are not the volume's Gauss-Legendre points. The
 * block's threads first form those contractions, once per element and
 * component, in shared memory (PhysLiftProjectLiftDir), so that each point
 * then reads one value per face, where every point formed its own before;
 * which directions are contractions is a template argument, so that a
 * direction read at one point carries no contraction code. What a point
 * reads of each direction is formed once per thread
 * (PhysLiftProjectLiftPoint) and serves every element and component.
 *
 * @tparam SHAPE_TYPE  Tet, Prism or Pyr.
 * @tparam CHUNK       Components per thread: physLiftProjectCompChunk or
 *                     one.
 * @tparam STAGE_MASK  Bit @em d set where direction @em d's faces are
 *                     contractions (one in-face direction's trace points are
 *                     not the volume's), or physLiftProjectStageRunTime to
 *                     take that from the collocation flags at run time.
 *
 * @param   traceSizeParam3D  Volume point counts per direction (nm0(),
 *                    nm1(), nm2()) and trace point counts per face pair.
 * @param   nelmt     Elements in the block, including padding.
 * @param   ncomp     Components (times homogeneous modes) of the block.
 * @param   elmtsPerBlock       Elements per thread block, of at least
 * `elmtsPerBlock * nqTot` threads.
 * @param   lift      Lift @p trace; without it only the division is applied.
 * @param   nbasis0,nbasis1,nbasis2  End-point interpolant of each
 *                    direction, `nbasis[p * ntrace + side]`.
 * @param   tbasis00,tbasis01,tbasis10,tbasis11,tbasis20,tbasis21  eInterp
 *                    tables of each face pair's in-face directions onto
 *                    their trace points (FetchTraceTables()).
 * @param   tw00,tw01,tw10,tw11,tw20,tw21  Weights of each face pair's
 *                    in-face directions.
 * @param   tjac      Trace Jacobians, one per face.
 * @param   isCollocated00,isCollocated01,isCollocated10,isCollocated11,
 *          isCollocated20,isCollocated21  The in-face direction's trace
 *                    points are the volume points; one of each pair at
 *                    least.
 * @param   endPtsCollocated0,endPtsCollocated1,endPtsCollocated2  The
 *                    direction's rule has its end points.
 * @param   jac       Jacobians, one per element.
 * @param   in        Inner products; may be @p out.
 * @param   trace     Element-local trace values.
 * @param   out       Result.
 * @param   shmemptr  `elmtsPerBlock * CHUNK` times the directions' StageSize()
 * values.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          unsigned int CHUNK, unsigned int STAGE_MASK,
          typename TTraceSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFacTOP>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void __LAUNCH_BOUNDS__(
    (GetMaxThreadPerBlock<Implementation, TTraceSizeParameter3D>()))
    PhysLiftProjectLiftJacobianKernelLauncher(
        const TTraceSizeParameter3D traceSizeParam3D, const size_t nelmt,
        const unsigned int ncomp, const unsigned int elmtsPerBlock,
        const bool lift, const TData *NEK_RESTRICT nbasis0,
        const TData *NEK_RESTRICT nbasis1, const TData *NEK_RESTRICT nbasis2,
        const TData *NEK_RESTRICT tbasis00, const TData *NEK_RESTRICT tbasis01,
        const TData *NEK_RESTRICT tbasis10, const TData *NEK_RESTRICT tbasis11,
        const TData *NEK_RESTRICT tbasis20, const TData *NEK_RESTRICT tbasis21,
        const TData *NEK_RESTRICT tw00, const TData *NEK_RESTRICT tw01,
        const TData *NEK_RESTRICT tw10, const TData *NEK_RESTRICT tw11,
        const TData *NEK_RESTRICT tw20, const TData *NEK_RESTRICT tw21,
        const TData *NEK_RESTRICT tjac, const bool isCollocated00,
        const bool isCollocated01, const bool isCollocated10,
        const bool isCollocated11, const bool isCollocated20,
        const bool isCollocated21, const bool endPtsCollocated0,
        const bool endPtsCollocated1, const bool endPtsCollocated2,
        const TData *NEK_RESTRICT jac, const TData *in,
        const TData *NEK_RESTRICT trace, TData *out, unsigned char *shmemptr,
        const TthreadBlock &threadBlock)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");
    static_assert(SHAPE_TYPE == LibUtilities::Tet ||
                      SHAPE_TYPE == LibUtilities::Prism ||
                      SHAPE_TYPE == LibUtilities::Pyr,
                  "The dense projector takes tetrahedra, prisms and "
                  "pyramids.");
    FETCH_SHARED_MEMORY(shmemptr);
    TData *shared = reinterpret_cast<TData *>(shmemptr);

    constexpr unsigned int nface0 = 2u;
    constexpr unsigned int nface1 = SHAPE_TYPE == LibUtilities::Tet ? 1u : 2u;
    constexpr unsigned int nface2 = 1u;
    constexpr unsigned int nface  = nface0 + nface1 + nface2;
    constexpr PhysLiftProjectStaged staged0 =
        physLiftProjectStagedDir<STAGE_MASK, 0>;
    constexpr PhysLiftProjectStaged staged1 =
        physLiftProjectStagedDir<STAGE_MASK, 1>;
    constexpr PhysLiftProjectStaged staged2 =
        physLiftProjectStagedDir<STAGE_MASK, 2>;

    // The trace size parameter's nm0()..nm2() are the volume quadrature
    // point counts here (#m_nm), as in the trace operators.
    const unsigned int nq0   = traceSizeParam3D.nm0();
    const unsigned int nq1   = traceSizeParam3D.nm1();
    const unsigned int nq2   = traceSizeParam3D.nm2();
    const unsigned int nq01  = nq0 * nq1;
    const unsigned int nqTot = nq01 * nq2;
    const unsigned int tnq00 = traceSizeParam3D.nq00();
    const unsigned int tnq01 = traceSizeParam3D.nq01();
    const unsigned int tnq10 = traceSizeParam3D.nq10();
    const unsigned int tnq11 = traceSizeParam3D.nq11();
    const unsigned int tnq20 = traceSizeParam3D.nq20();
    const unsigned int tnq21 = traceSizeParam3D.nq21();
    const unsigned int nqTotTrace =
        traceSizeParam3D.template nqTotTrace<SHAPE_TYPE>();
    const unsigned int off1 = nface0 * tnq00 * tnq01;
    const unsigned int off2 = off1 + nface1 * tnq10 * tnq11;

    const PhysLiftProjectLiftDir<nface0, staged0, TData> dir0(
        tnq00, tnq01, nq1, nq2, 0u, tbasis00, tbasis01, tw00, tw01,
        isCollocated00, isCollocated01);
    const PhysLiftProjectLiftDir<nface1, staged1, TData> dir1(
        tnq10, tnq11, nq0, nq2, off1, tbasis10, tbasis11, tw10, tw11,
        isCollocated10, isCollocated11);
    const PhysLiftProjectLiftDir<nface2, staged2, TData> dir2(
        tnq20, tnq21, nq0, nq1, off2, tbasis20, tbasis21, tw20, tw21,
        isCollocated20, isCollocated21);
    const unsigned int stage0    = dir0.StageSize();
    const unsigned int stage1    = dir1.StageSize();
    const unsigned int stage2    = dir2.StageSize();
    const unsigned int stageSize = stage0 + stage1 + stage2;
    const bool staged            = lift && stageSize > 0u;

    const size_t compSize      = static_cast<size_t>(nqTot) * nelmt;
    const size_t compTraceSize = static_cast<size_t>(nqTotTrace) * nelmt;

    // This thread's element of the thread block, then its point
    // (GetDevicePackedThread()), and what the point reads of each
    // direction's faces, for every element: a block may hold several
    // elements. A thread takes one point, the dense projector's elements
    // being no larger than a block.
    const DevicePackedThread thread =
        GetDevicePackedThread(nqTot, elmtsPerBlock, threadBlock);
    const unsigned int tid    = getLocalIdx(threadBlock);
    const unsigned int el     = thread.elmt;
    const bool active         = el < elmtsPerBlock && thread.firstPoint < nqTot;
    const unsigned int pt     = active ? thread.firstPoint : 0u;
    const unsigned int p      = pt % nq0;
    const unsigned int q      = (pt / nq0) % nq1;
    const unsigned int r      = pt / nq01;
    const unsigned int c0     = CHUNK * getBlockIdx<1>(threadBlock);
    const unsigned int nleft  = ncomp - c0;
    const unsigned int nthr   = getLocalRange(threadBlock);
    const unsigned int nstage = elmtsPerBlock * CHUNK * stageSize;
    const PhysLiftProjectLiftPoint<nface0, staged0, TData> pt0(
        dir0, 0u, p, nq0, q, r, nbasis0, endPtsCollocated0);
    const PhysLiftProjectLiftPoint<nface1, staged1, TData> pt1(
        dir1, stage0, q, nq1, p, r, nbasis1, endPtsCollocated1);
    const PhysLiftProjectLiftPoint<nface2, staged2, TData> pt2(
        dir2, stage0 + stage1, r, nq2, p, q, nbasis2, endPtsCollocated2);

    for (size_t e0 =
             static_cast<size_t>(getBlockIdx(threadBlock)) * elmtsPerBlock;
         e0 < nelmt;
         e0 += static_cast<size_t>(getBlockRange(threadBlock)) * elmtsPerBlock)
    {
        // The contractions of the chunk's components of the thread block's
        // elements, one entry per thread at a time.
        if (staged)
        {
            for (unsigned int w = tid; w < nstage; w += nthr)
            {
                const unsigned int s  = w % stageSize;
                const unsigned int cc = (w / stageSize) % CHUNK;
                const size_t e        = e0 + w / (stageSize * CHUNK);
                if (e < nelmt && cc < nleft)
                {
                    const TData *traceptr = trace + compTraceSize * (c0 + cc) +
                                            static_cast<size_t>(nqTotTrace) * e;
                    shared[w] = s < stage0 ? dir0.Stage(traceptr, s)
                                : s < stage0 + stage1
                                    ? dir1.Stage(traceptr, s - stage0)
                                    : dir2.Stage(traceptr, s - stage0 - stage1);
                }
            }
            localBarrier(threadBlock);
        }

        const size_t e = e0 + el;
        if (active && e < nelmt)
        {
            const TData factor   = SafeDivide(TData(1.0), jac[e]);
            const TData *tjacptr = tjac + nface * e;
            const size_t ientry  = compSize * c0 + nqTot * e + pt;
            // Zero past the last component, so every entry is defined.
            TData val[CHUNK];
#pragma unroll
            for (unsigned int cc = 0; cc < CHUNK; ++cc)
            {
                val[cc] = cc < nleft ? in[ientry + compSize * cc] : TData(0.0);
            }
            if (lift)
            {
                const TData *traceptr = trace + compTraceSize * c0 +
                                        static_cast<size_t>(nqTotTrace) * e;
                const TData *stagedptr = shared + stageSize * CHUNK * el;
                pt0.AddFaceLifts(dir0, traceptr, compTraceSize, stagedptr,
                                 stageSize, tjacptr, nleft, val);
                pt1.AddFaceLifts(dir1, traceptr, compTraceSize, stagedptr,
                                 stageSize, tjacptr + nface0, nleft, val);
                pt2.AddFaceLifts(dir2, traceptr, compTraceSize, stagedptr,
                                 stageSize, tjacptr + nface0 + nface1, nleft,
                                 val);
            }
#pragma unroll
            for (unsigned int cc = 0; cc < CHUNK; ++cc)
            {
                if (cc < nleft)
                {
                    out[ientry + compSize * cc] = factor * val[cc];
                }
            }
        }
        if (staged)
        {
            localBarrier(threadBlock);
        }
    }
}

#endif

} // namespace Nektar::MultiRegions::detail
