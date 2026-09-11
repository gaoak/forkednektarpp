///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTPhysNormalDerivTraceSerialAVXGenericKernels.hpp
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
// Description: Serial/AVX kernels of the lift against the normal derivative
// of the test function
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTPhysNormalDerivTraceSerialAVXGenericKernels.hpp
 * @brief Serial and AVX kernels of the surface inner product against the
 * normal derivative of the volume cardinal basis.
 *
 * @details
 * The operator is
 * \f[ \langle \partial_n \phi, g \rangle_{\partial E}
 *     = \sum_d \langle H_d\, \partial_{\eta_d} \phi, g \rangle_{\partial E} ,
 * \f]
 * one trace inner product per element direction, each the ordinary
 * IProductWRTPhysTrace with the basis in direction @em d replaced by its
 * derivative. \f$H_d\f$ comes from the element local factors the data
 * warehouse supplies through the Duffy change of coordinates
 * \f$\xi \to \eta\f$: its regular part is already in the factor array,
 * component @em d, and its singular \f$2/(1 - \eta)\f$ part is applied on
 * the volume by ScaleAddCollapsedFacKernel().
 *
 * For term @em d the derivative table appears wherever a slot's element
 * direction is @em d: as the normal table of direction @em d, and as the
 * tangential table of the trace slots that run along @em d. A derivative
 * slot is never collocated, and a direction never has its end points
 * collocated for its own term.
 *
 * The edge and face cores are those of the plain trace lift in
 * IProductWRTPhysTraceSerialAVXGenericKernels.hpp; this file only decides
 * which tables each core sees and where the collapsed factors go. The
 * three IProductWRTPhysNormalDerivTraceKernelLauncher() overloads are the
 * entry points the block operator's OperatorND() reaches; overload
 * resolution picks the dimension from the number of arguments.
 *
 * Instantiation. The size parameter reaches these kernels templated
 * where the shape switch matched a compiled point count, so the cores'
 * loop bounds are constants as they are for the other trace operators.
 * To keep the number of instantiations in hand, the direction term and
 * the accumulate flag are run-time arguments rather than template
 * parameters: one body per size serves all terms and both modes, and only
 * the end-point variants of the shared cores are instantiated twice.
 *
 * @see IProductWRTPhysNormalDerivTraceSerialAVXGeneric.hpp for the block
 * operator that calls these kernels.
 * @see IProductWRTPhysNormalDerivTraceDeviceGenericKernels.hpp for the
 * same decomposition packed for warp lanes instead of SIMD vectors.
 */

#pragma once

#include <cstring>

#include "Operators/ElmtOps/ElmtHelper.hpp"
#include <LibUtilities/BasicUtils/ShapeType.hpp>
#include <LibUtilities/SimdLib/tinysimd.hpp>

#include "Operators/ElmtOps/IProductWRTPhysTrace/IProductWRTPhysTraceSerialAVXGenericKernels.hpp"

namespace Nektar::Operators::detail
{

/**
 * @brief The one-dimensional lift: \f$\langle \partial_n \phi, g \rangle\f$
 * over a segment's two vertices.
 *
 * A segment's traces are single points, so there is no trace quadrature to
 * sum over, no tangential table and no trace Jacobian. What is left is
 * the point evaluation
 * \f[ out_p \mathrel{+}= \sum_{v=0,1} \partial h_p(\xi_v)\, F_v\, g_v , \f]
 * with \f$F_v\f$ the normal derivative factor at that vertex, which the
 * data warehouse supplies as the single component of the factor array.
 * There is no end-point collocated shortcut: the interpolation to a
 * collocated end point is a selection matrix, which its derivative never
 * is.
 *
 * @param   append  Accumulate onto @p out instead of zeroing it first.
 */
template <typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysNormalDerivTrace1DKernel(
    const unsigned nm0, const simd_type *dnbasis0, const simd_type *jac,
    const simd_type *in, simd_type *out, const bool append)
{
    if (!append)
    {
        std::memset((void *)out, 0, nm0 * sizeof(simd_type));
    }

    for (unsigned p = 0; p < nm0; ++p)
    {
        simd_type sum = static_cast<typename simd_type::scalarType>(0.0);
        sum.fma(dnbasis0[2 * p] * jac[0], in[0]);
        sum.fma(dnbasis0[2 * p + 1] * jac[1], in[1]);

        out[p] = out[p] + sum;
    }
}

/**
 * @brief Scale a volume workspace by the collapsed factors a trace group
 * still needs, then accumulate it into the output.
 *
 * @param   useD1,useD2     Apply \f$2/(1 - \eta_1)\f$, \f$2/(1 - \eta_2)\f$.
 * @param   twoOver1,twoOver2   Those factors at the element's own points
 *                          of directions 1 and 2; unread when the matching
 *                          flag is false. In two dimensions @p nm2 is one
 *                          and @p twoOver2 is unread.
 */
template <typename simd_type>
NEK_FORCE_INLINE static void ScaleAddCollapsedFacKernel(
    const unsigned nm0, const unsigned nm1, const unsigned nm2,
    const bool useD1, const bool useD2, const simd_type *twoOver1,
    const simd_type *twoOver2, const simd_type *volwsp, simd_type *out)
{
    for (unsigned k = 0; k < nm2; ++k)
    {
        for (unsigned j = 0; j < nm1; ++j)
        {
            simd_type f = static_cast<typename simd_type::scalarType>(1.0);
            if (useD1)
            {
                f = f * twoOver1[j];
            }
            if (useD2)
            {
                f = f * twoOver2[k];
            }

            const unsigned o = (k * nm1 + j) * nm0;
            for (unsigned i = 0; i < nm0; ++i)
            {
                out[o + i].fma(volwsp[o + i], f);
            }
        }
    }
}

/**
 * @brief One edge group of a term of the two-dimensional operator,
 * accumulated into @p out.
 *
 * @c EDGE_NORMAL_DIR names the group: the element coordinate direction its
 * edges are normal to. The tables handed in are already those of the term
 * being evaluated, so this only routes the group through the volume
 * workspace when it carries the collapsed factor.
 *
 * @param   useD1   The group carries \f$2/(1 - \eta_1)\f$ and is
 *                  accumulated through @p volwsp. A run-time flag so that
 *                  one group body per size serves every direction term.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          unsigned EDGE_NORMAL_DIR, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysNormalDerivTraceEdgeGroupKernel(
    const bool useD1, const unsigned nm0, const unsigned nm1,
    const unsigned tnq00, [[maybe_unused]] const unsigned tnq10,
    const simd_type *nbasis, const simd_type *tbasis, const simd_type *tw,
    const simd_type *twoOver1, const simd_type *jac, const simd_type *in,
    simd_type *wsp, simd_type *volwsp, simd_type *out, const bool isCollocated,
    const bool endPtsCollocated)
{
    const unsigned numDataOut = nm0 * nm1;

    // Edge group offsets into the input and the factors, matching what
    // the plain trace lift does internally. Only the group being
    // instantiated reads its own offsets, so the others go unused.
    [[maybe_unused]] constexpr unsigned nedgeN1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];

    [[maybe_unused]] const unsigned off1  = 2 * tnq00;
    [[maybe_unused]] const unsigned joff1 = (DEFORMED) ? off1 : 2;

    simd_type *dst = useD1 ? volwsp : out;
    if (useD1)
    {
        std::memset((void *)dst, 0, numDataOut * sizeof(simd_type));
    }

    if constexpr (EDGE_NORMAL_DIR == 0)
    {
        if (endPtsCollocated)
        {
            IPWRTPhysTraceEdgeN0Kernel<true, DEFORMED>(
                0, 2, nm0, nm1, nbasis, tnq00, tbasis, tw, jac, wsp, in, dst,
                isCollocated);
        }
        else
        {
            IPWRTPhysTraceEdgeN0Kernel<false, DEFORMED>(
                0, 2, nm0, nm1, nbasis, tnq00, tbasis, tw, jac, wsp, in, dst,
                isCollocated);
        }
    }
    else
    {
        if (endPtsCollocated)
        {
            IPWRTPhysTraceEdgeN1Kernel<true, DEFORMED>(
                0, nedgeN1, nm0, nm1, nbasis, tnq10, tbasis, tw, jac + joff1,
                wsp, in + off1, dst, isCollocated);
        }
        else
        {
            IPWRTPhysTraceEdgeN1Kernel<false, DEFORMED>(
                0, nedgeN1, nm0, nm1, nbasis, tnq10, tbasis, tw, jac + joff1,
                wsp, in + off1, dst, isCollocated);
        }
    }

    if (useD1)
    {
        // nm2 is one: the volume is a single eta_2 layer in 2D.
        ScaleAddCollapsedFacKernel(nm0, nm1, 1u, true, false, twoOver1,
                                   twoOver1, volwsp, out);
    }
}

/**
 * @brief Term @p derivDir of the two-dimensional operator: the trace
 * inner product taken with the direction @p derivDir table replaced by
 * its derivative.
 *
 * The edge groups are evaluated apart so the group normal to the collapsed
 * direction can drop its factor. A collapsed direction has exactly one
 * trace and it always sits at \f$\eta_1 = -1\f$, where
 * \f$2/(1 - \eta_1)\f$ is one, so group 1 never scales. Splitting rather
 * than scaling the combined result means this does not rely on that edge
 * depositing on the \f$\eta_1 = -1\f$ row, so the element is free to carry
 * a direction 1 quadrature with no end point, and the trace any quadrature
 * at all.
 *
 * @param   derivDir    Element direction whose table is the derivative.
 * @param   append      Accumulate onto @p out instead of zeroing it first.
 * @param   jac         This term's component of the factor array for the
 *                      element group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysNormalDerivTracePerDir2DKernel(
    const unsigned derivDir, const unsigned nm0, const unsigned nm1,
    const unsigned tnq00, const unsigned tnq10, const simd_type *nbasis0,
    const simd_type *nbasis1, const simd_type *dnbasis0,
    const simd_type *dnbasis1, const simd_type *tbasis0,
    const simd_type *tbasis1, const simd_type *dtbasis0,
    const simd_type *dtbasis1, const simd_type *tw0, const simd_type *tw1,
    const simd_type *twoOver1, const simd_type *jac, const simd_type *in,
    simd_type *wsp, simd_type *volwsp, simd_type *out, const unsigned traceDir0,
    const unsigned traceDir1, const bool append, const bool isColl0,
    const bool isColl1, const bool endPtsColl0, const bool endPtsColl1)
{
    const bool c1 = CompNeedsCollapsedFac(SHAPE_TYPE, derivDir, 1);

    if (!append)
    {
        std::memset((void *)out, 0, nm0 * nm1 * sizeof(simd_type));
    }

    // Term derivDir picks the derivative table wherever a slot's element
    // direction is derivDir. A derivative slot is never collocated, and a
    // direction never has its own end points collocated for its own term.
    const simd_type *nb0 = (0 == derivDir) ? dnbasis0 : nbasis0;
    const simd_type *nb1 = (1 == derivDir) ? dnbasis1 : nbasis1;
    const simd_type *tb0 = (traceDir0 == derivDir) ? dtbasis0 : tbasis0;
    const simd_type *tb1 = (traceDir1 == derivDir) ? dtbasis1 : tbasis1;
    const bool cd0       = (traceDir0 == derivDir) ? false : isColl0;
    const bool cd1       = (traceDir1 == derivDir) ? false : isColl1;
    const bool epd0      = (0 == derivDir) ? false : endPtsColl0;
    const bool epd1      = (1 == derivDir) ? false : endPtsColl1;

    // The groups are written out so that each keeps its constant normal
    // direction; the term is a run-time index, so this body is
    // instantiated once per size and serves every term.
    IPWRTPhysNormalDerivTraceEdgeGroupKernel<SHAPE_TYPE, DEFORMED, 0>(
        c1, nm0, nm1, tnq00, tnq10, nb0, tb0, tw0, twoOver1, jac, in, wsp,
        volwsp, out, cd0, epd0);
    IPWRTPhysNormalDerivTraceEdgeGroupKernel<SHAPE_TYPE, DEFORMED, 1>(
        false, nm0, nm1, tnq00, tnq10, nb1, tb1, tw1, twoOver1, jac, in, wsp,
        volwsp, out, cd1, epd1);
}

/**
 * @brief One face group of a term of the three-dimensional operator,
 * accumulated into @p out.
 *
 * @c FACE_NORMAL_DIR names the group: the element coordinate direction its
 * faces are normal to. The groups are evaluated apart so each can drop the
 * collapsed factors it does not need. Where a factor is wanted the group
 * goes into @p volwsp first and is scaled there, on the element's own
 * points, because the factor is singular at the collapsed apex, a
 * quadrature point of the trace but never of the element.
 *
 * @param   useD1,useD2     The group carries \f$2/(1 - \eta_1)\f$,
 *                          \f$2/(1 - \eta_2)\f$. Run-time flags so that one
 *                          group body per size serves every direction term.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          unsigned FACE_NORMAL_DIR, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysNormalDerivTraceFaceGroupKernel(
    const bool useD1, const bool useD2, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const unsigned tnq00, const unsigned tnq01,
    const unsigned tnq10, const unsigned tnq11,
    [[maybe_unused]] const unsigned tnq20,
    [[maybe_unused]] const unsigned tnq21, const simd_type *nbasis,
    const simd_type *tbasisA, const simd_type *tbasisB, const simd_type *twA,
    const simd_type *twB, const simd_type *twoOver1, const simd_type *twoOver2,
    const simd_type *jac, const simd_type *in, simd_type *wsp0, simd_type *wsp1,
    simd_type *volwsp, simd_type *out, const bool isCollocatedA,
    const bool isCollocatedB, const bool endPtsCollocated)
{
    const unsigned numDataOut = nm0 * nm1 * nm2;

    // Face group offsets into the input and the factors, matching what
    // the plain trace lift does internally. Only the group being
    // instantiated reads its own offsets, so the others go unused.
    [[maybe_unused]] constexpr unsigned nfaceN1 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
    [[maybe_unused]] constexpr unsigned nfaceN2 =
        LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];

    [[maybe_unused]] const unsigned off1  = 2 * tnq00 * tnq01;
    [[maybe_unused]] const unsigned joff1 = (DEFORMED) ? off1 : 2;
    [[maybe_unused]] const unsigned dd2   = nfaceN1 * tnq10 * tnq11;
    [[maybe_unused]] const unsigned off2  = off1 + dd2;
    [[maybe_unused]] const unsigned joff2 =
        joff1 + ((DEFORMED) ? dd2 : nfaceN1);

    const bool scale = useD1 || useD2;
    simd_type *dst   = scale ? volwsp : out;
    if (scale)
    {
        std::memset((void *)dst, 0, numDataOut * sizeof(simd_type));
    }

    if constexpr (FACE_NORMAL_DIR == 0)
    {
        if (endPtsCollocated)
        {
            IPWRTPhysTraceFaceN0Kernel<true, DEFORMED>(
                0, 2, nm0, nm1, nm2, nbasis, tnq00, tnq01, tbasisA, tbasisB,
                twA, twB, jac, wsp0, wsp1, in, dst, isCollocatedA,
                isCollocatedB);
        }
        else
        {
            IPWRTPhysTraceFaceN0Kernel<false, DEFORMED>(
                0, 2, nm0, nm1, nm2, nbasis, tnq00, tnq01, tbasisA, tbasisB,
                twA, twB, jac, wsp0, wsp1, in, dst, isCollocatedA,
                isCollocatedB);
        }
    }
    else if constexpr (FACE_NORMAL_DIR == 1)
    {
        if (endPtsCollocated)
        {
            IPWRTPhysTraceFaceN1Kernel<true, DEFORMED>(
                0, nfaceN1, nm0, nm1, nm2, nbasis, tnq10, tnq11, tbasisA,
                tbasisB, twA, twB, jac + joff1, wsp0, wsp1, in + off1, dst,
                isCollocatedA, isCollocatedB);
        }
        else
        {
            IPWRTPhysTraceFaceN1Kernel<false, DEFORMED>(
                0, nfaceN1, nm0, nm1, nm2, nbasis, tnq10, tnq11, tbasisA,
                tbasisB, twA, twB, jac + joff1, wsp0, wsp1, in + off1, dst,
                isCollocatedA, isCollocatedB);
        }
    }
    else
    {
        if (endPtsCollocated)
        {
            IPWRTPhysTraceFaceN2Kernel<true, DEFORMED>(
                0, nfaceN2, nm0, nm1, nm2, nbasis, tnq20, tnq21, tbasisA,
                tbasisB, twA, twB, jac + joff2, wsp0, wsp1, in + off2, dst,
                isCollocatedA, isCollocatedB);
        }
        else
        {
            IPWRTPhysTraceFaceN2Kernel<false, DEFORMED>(
                0, nfaceN2, nm0, nm1, nm2, nbasis, tnq20, tnq21, tbasisA,
                tbasisB, twA, twB, jac + joff2, wsp0, wsp1, in + off2, dst,
                isCollocatedA, isCollocatedB);
        }
    }

    if (scale)
    {
        ScaleAddCollapsedFacKernel(nm0, nm1, nm2, useD1, useD2, twoOver1,
                                   twoOver2, volwsp, out);
    }
}

/**
 * @brief Term @p derivDir of the three-dimensional operator: the trace
 * inner product taken with the direction @p derivDir table replaced by
 * its derivative.
 *
 * A face group normal to a collapsed direction sits at \f$\eta = -1\f$,
 * where that direction's factor is one, so group @em k drops the direction
 * @em k factor. That is a property of the shape, not of the quadrature, so
 * it does not lean on the end points being collocated. With no collapsed
 * direction nothing is scaled and this is the plain trace lift written out
 * one group at a time.
 *
 * @param   derivDir    Element direction whose table is the derivative.
 * @param   append      Accumulate onto @p out instead of zeroing it first.
 * @param   jac         This term's component of the factor array for the
 *                      element group.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IPWRTPhysNormalDerivTracePerDir3DKernel(
    const unsigned derivDir, const unsigned nm0, const unsigned nm1,
    const unsigned nm2, const unsigned tnq00, const unsigned tnq01,
    const unsigned tnq10, const unsigned tnq11, const unsigned tnq20,
    const unsigned tnq21, const simd_type *nbasis0, const simd_type *nbasis1,
    const simd_type *nbasis2, const simd_type *dnbasis0,
    const simd_type *dnbasis1, const simd_type *dnbasis2,
    const simd_type *tbasis00, const simd_type *tbasis01,
    const simd_type *tbasis10, const simd_type *tbasis11,
    const simd_type *tbasis20, const simd_type *tbasis21,
    const simd_type *dtbasis00, const simd_type *dtbasis01,
    const simd_type *dtbasis10, const simd_type *dtbasis11,
    const simd_type *dtbasis20, const simd_type *dtbasis21,
    const simd_type *tw00, const simd_type *tw01, const simd_type *tw10,
    const simd_type *tw11, const simd_type *tw20, const simd_type *tw21,
    const simd_type *twoOver1, const simd_type *twoOver2, const simd_type *jac,
    const simd_type *in, simd_type *wsp0, simd_type *wsp1, simd_type *volwsp,
    simd_type *out, const unsigned traceDir00, const unsigned traceDir01,
    const unsigned traceDir10, const unsigned traceDir11,
    const unsigned traceDir20, const unsigned traceDir21, const bool append,
    const bool isColl00, const bool isColl01, const bool isColl10,
    const bool isColl11, const bool isColl20, const bool isColl21,
    const bool endPtsColl0, const bool endPtsColl1, const bool endPtsColl2)
{
    const bool c1 = CompNeedsCollapsedFac(SHAPE_TYPE, derivDir, 1);
    const bool c2 = CompNeedsCollapsedFac(SHAPE_TYPE, derivDir, 2);

    if (!append)
    {
        std::memset((void *)out, 0, nm0 * nm1 * nm2 * sizeof(simd_type));
    }

    // Term derivDir picks the derivative table wherever a slot's element
    // direction is derivDir. A derivative slot is never collocated, and a
    // direction never has its own end points collocated for its own term.
    const simd_type *nb0  = (0 == derivDir) ? dnbasis0 : nbasis0;
    const simd_type *nb1  = (1 == derivDir) ? dnbasis1 : nbasis1;
    const simd_type *nb2  = (2 == derivDir) ? dnbasis2 : nbasis2;
    const simd_type *tb00 = (traceDir00 == derivDir) ? dtbasis00 : tbasis00;
    const simd_type *tb01 = (traceDir01 == derivDir) ? dtbasis01 : tbasis01;
    const simd_type *tb10 = (traceDir10 == derivDir) ? dtbasis10 : tbasis10;
    const simd_type *tb11 = (traceDir11 == derivDir) ? dtbasis11 : tbasis11;
    const simd_type *tb20 = (traceDir20 == derivDir) ? dtbasis20 : tbasis20;
    const simd_type *tb21 = (traceDir21 == derivDir) ? dtbasis21 : tbasis21;
    const bool cd00       = (traceDir00 == derivDir) ? false : isColl00;
    const bool cd01       = (traceDir01 == derivDir) ? false : isColl01;
    const bool cd10       = (traceDir10 == derivDir) ? false : isColl10;
    const bool cd11       = (traceDir11 == derivDir) ? false : isColl11;
    const bool cd20       = (traceDir20 == derivDir) ? false : isColl20;
    const bool cd21       = (traceDir21 == derivDir) ? false : isColl21;
    const bool epd0       = (0 == derivDir) ? false : endPtsColl0;
    const bool epd1       = (1 == derivDir) ? false : endPtsColl1;
    const bool epd2       = (2 == derivDir) ? false : endPtsColl2;

    // The groups are written out so that each keeps its constant normal
    // direction; the term is a run-time index, so this body is
    // instantiated once per size and serves every term.
    IPWRTPhysNormalDerivTraceFaceGroupKernel<SHAPE_TYPE, DEFORMED, 0>(
        c1, c2, nm0, nm1, nm2, tnq00, tnq01, tnq10, tnq11, tnq20, tnq21, nb0,
        tb00, tb01, tw00, tw01, twoOver1, twoOver2, jac, in, wsp0, wsp1, volwsp,
        out, cd00, cd01, epd0);
    IPWRTPhysNormalDerivTraceFaceGroupKernel<SHAPE_TYPE, DEFORMED, 1>(
        false, c2, nm0, nm1, nm2, tnq00, tnq01, tnq10, tnq11, tnq20, tnq21, nb1,
        tb10, tb11, tw10, tw11, twoOver1, twoOver2, jac, in, wsp0, wsp1, volwsp,
        out, cd10, cd11, epd1);
    IPWRTPhysNormalDerivTraceFaceGroupKernel<SHAPE_TYPE, DEFORMED, 2>(
        c1, false, nm0, nm1, nm2, tnq00, tnq01, tnq10, tnq11, tnq20, tnq21, nb2,
        tb20, tb21, tw20, tw21, twoOver1, twoOver2, jac, in, wsp0, wsp1, volwsp,
        out, cd20, cd21, epd2);
}

/**
 * @brief Launcher for one element group of segments.
 *
 * The one-dimensional arm of the overload set OperatorND() calls; the
 * higher-dimensional arms follow below, and overload resolution picks
 * between them by the number of arguments the index sequences expand to.
 * A segment's traces are points, so it takes no tangential table, no
 * weight and no workspace, and only the derivative table and the vertex
 * factors are read.
 *
 * @tparam SHAPE_TYPE   Seg; unread, the path being the same for every
 *                      one-dimensional expansion.
 * @tparam DEFORMED     Unread: a segment carries one factor per vertex
 *                      whatever its geometry.
 *
 * @param   append      Accumulate onto the volume field instead of
 *                      zeroing it first.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysNormalDerivTraceKernelLauncher(
    const TTraceSizeParameter1D sizeParam1D,
    [[maybe_unused]] const simd_type *nbasis0, const simd_type *dnbasis0,
    [[maybe_unused]] const simd_type *twoOver0, const simd_type *jac,
    [[maybe_unused]] const size_t jacCompStride,
    [[maybe_unused]] simd_type *volwsp, const simd_type *in, simd_type *out,
    const bool append, [[maybe_unused]] const bool endPtsCollocated0)
{
    static_assert(IsTraceSizeParameter1D_v<TTraceSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter1D or "
                  "TemplatedTraceSizeParameter1D.");

    IProductWRTPhysNormalDerivTrace1DKernel(sizeParam1D.nm0(), dnbasis0, jac,
                                            in, out, append);
}

/**
 * @brief All edges of one two-dimensional element group: the two terms of
 * the operator, one per element direction, accumulated in turn.
 *
 * @tparam SHAPE_TYPE  Quad, Tri or NodalTri.
 * @tparam DEFORMED    Factors vary point by point along the traces.
 *
 * @param   append              Accumulate onto @p out instead of zeroing
 *                              it first.
 * @param   sizeParam2D         Volume and trace point counts of the shape.
 * @param   nbasis0,nbasis1     eInterp tables \f$h_p(\pm 1)\f$ per
 *                              direction.
 * @param   dnbasis0,dnbasis1   Their derivatives.
 * @param   tbasis0,tbasis1     eInterp tables to the trace points, per
 *                              direction, and @p dtbasis0, @p dtbasis1
 *                              their derivatives.
 * @param   tw0,tw1             Trace quadrature weights per direction.
 * @param   twoOver0,twoOver1   \f$2/(1 - \eta)\f$ at the element points per
 *                              direction; direction 0 is never collapsed
 *                              and its slot is null and unread.
 * @param   jac                 Factor array of this element group, term 0;
 *                              term 1 sits @p jacCompStride SIMD vectors
 *                              further on.
 * @param   wsp                 Trace-mode scratch, `2 * max(nm0,nm1)`.
 * @param   volwsp              Volume-sized scratch for the collapsed
 *                              factor scaling; unread on a quadrilateral.
 * @param   traceDir0,traceDir1 Element direction each trace slot runs
 *                              along.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysNormalDerivTraceKernelLauncher(
    const TTraceSizeParameter2D sizeParam2D, const simd_type *nbasis0,
    const simd_type *nbasis1, const simd_type *dnbasis0,
    const simd_type *dnbasis1, const simd_type *tbasis0,
    const simd_type *tbasis1, const simd_type *dtbasis0,
    const simd_type *dtbasis1, const simd_type *tw0, const simd_type *tw1,
    [[maybe_unused]] const simd_type *twoOver0, const simd_type *twoOver1,
    const simd_type *jac, const size_t jacCompStride, simd_type *wsp,
    simd_type *volwsp, const simd_type *in, simd_type *out,
    const unsigned traceDir0, const unsigned traceDir1, const bool append,
    const bool isColl0, const bool isColl1, const bool endPtsColl0,
    const bool endPtsColl1)
{
    static_assert(IsTraceSizeParameter2D_v<TTraceSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter2D or "
                  "TemplatedTraceSizeParameter2D.");

    // Shape size.
    const unsigned nm0   = sizeParam2D.nm0();
    const unsigned nm1   = sizeParam2D.nm1();
    const unsigned tnq00 = sizeParam2D.nq00();
    const unsigned tnq10 = sizeParam2D.nq10();

    // One term per element direction; the later terms append onto the
    // first. The term is a run-time index so that the size-specialised
    // body is instantiated once per size rather than once per term.
    for (unsigned d = 0; d < 2; ++d)
    {
        IPWRTPhysNormalDerivTracePerDir2DKernel<SHAPE_TYPE, DEFORMED>(
            d, nm0, nm1, tnq00, tnq10, nbasis0, nbasis1, dnbasis0, dnbasis1,
            tbasis0, tbasis1, dtbasis0, dtbasis1, tw0, tw1, twoOver1,
            jac + d * jacCompStride, in, wsp, volwsp, out, traceDir0, traceDir1,
            append || d > 0, isColl0, isColl1, endPtsColl0, endPtsColl1);
    }
}

/**
 * @brief All faces of one three-dimensional element group: the three
 * terms of the operator, one per element direction, accumulated in turn.
 *
 * @tparam SHAPE_TYPE  Hex, Prism, NodalPrism, Pyr, Tet or NodalTet.
 * @tparam DEFORMED    Factors vary point by point along the traces.
 *
 * @param   append          Accumulate onto @p out instead of zeroing it
 *                          first.
 * @param   sizeParam3D     Volume and trace point counts of the shape.
 * @param   nbasis0,nbasis1,nbasis2     eInterp tables \f$h_p(\pm 1)\f$ per
 *                          direction, and @p dnbasis0 to @p dnbasis2 their
 *                          derivatives.
 * @param   tbasis00,...,tbasis21   eInterp tables to the trace points, per
 *                          normal direction and tangential direction, and
 *                          @p dtbasis00 to @p dtbasis21 their derivatives.
 * @param   tw00,...,tw21   Trace quadrature weights in the same order.
 * @param   twoOver0,twoOver1,twoOver2  \f$2/(1 - \eta)\f$ at the element
 *                          points per direction; direction 0 is never
 *                          collapsed and its slot is null and unread, as is
 *                          any direction the shape does not collapse.
 * @param   jac             Factor array of this element group, term 0;
 *                          term @em d sits `d * jacCompStride` SIMD vectors
 *                          further on.
 * @param   wsp0            Trace-mode workspace, two faces' worth.
 * @param   wsp1            Scratch for the fully general face contraction.
 * @param   volwsp          Volume-sized scratch for the collapsed factor
 *                          scaling; unread on a hexahedron.
 * @param   traceDir00,...,traceDir21   Element direction each trace slot
 *                          runs along.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TTraceSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTPhysNormalDerivTraceKernelLauncher(
    const TTraceSizeParameter3D sizeParam3D, const simd_type *nbasis0,
    const simd_type *nbasis1, const simd_type *nbasis2,
    const simd_type *dnbasis0, const simd_type *dnbasis1,
    const simd_type *dnbasis2, const simd_type *tbasis00,
    const simd_type *tbasis01, const simd_type *tbasis10,
    const simd_type *tbasis11, const simd_type *tbasis20,
    const simd_type *tbasis21, const simd_type *dtbasis00,
    const simd_type *dtbasis01, const simd_type *dtbasis10,
    const simd_type *dtbasis11, const simd_type *dtbasis20,
    const simd_type *dtbasis21, const simd_type *tw00, const simd_type *tw01,
    const simd_type *tw10, const simd_type *tw11, const simd_type *tw20,
    const simd_type *tw21, [[maybe_unused]] const simd_type *twoOver0,
    const simd_type *twoOver1, const simd_type *twoOver2, const simd_type *jac,
    const size_t jacCompStride, simd_type *wsp0, simd_type *wsp1,
    simd_type *volwsp, const simd_type *in, simd_type *out,
    const unsigned traceDir00, const unsigned traceDir01,
    const unsigned traceDir10, const unsigned traceDir11,
    const unsigned traceDir20, const unsigned traceDir21, const bool append,
    const bool isColl00, const bool isColl01, const bool isColl10,
    const bool isColl11, const bool isColl20, const bool isColl21,
    const bool endPtsColl0, const bool endPtsColl1, const bool endPtsColl2)
{
    static_assert(IsTraceSizeParameter3D_v<TTraceSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedTraceSizeParameter3D or "
                  "TemplatedTraceSizeParameter3D.");

    // Shape size.
    const unsigned nm0   = sizeParam3D.nm0();
    const unsigned nm1   = sizeParam3D.nm1();
    const unsigned nm2   = sizeParam3D.nm2();
    const unsigned tnq00 = sizeParam3D.nq00();
    const unsigned tnq01 = sizeParam3D.nq01();
    const unsigned tnq10 = sizeParam3D.nq10();
    const unsigned tnq11 = sizeParam3D.nq11();
    const unsigned tnq20 = sizeParam3D.nq20();
    const unsigned tnq21 = sizeParam3D.nq21();

    // One term per element direction; the later terms append onto the
    // first. The term is a run-time index so that the size-specialised
    // body is instantiated once per size rather than once per term.
    for (unsigned d = 0; d < 3; ++d)
    {
        IPWRTPhysNormalDerivTracePerDir3DKernel<SHAPE_TYPE, DEFORMED>(
            d, nm0, nm1, nm2, tnq00, tnq01, tnq10, tnq11, tnq20, tnq21, nbasis0,
            nbasis1, nbasis2, dnbasis0, dnbasis1, dnbasis2, tbasis00, tbasis01,
            tbasis10, tbasis11, tbasis20, tbasis21, dtbasis00, dtbasis01,
            dtbasis10, dtbasis11, dtbasis20, dtbasis21, tw00, tw01, tw10, tw11,
            tw20, tw21, twoOver1, twoOver2, jac + d * jacCompStride, in, wsp0,
            wsp1, volwsp, out, traceDir00, traceDir01, traceDir10, traceDir11,
            traceDir20, traceDir21, append || d > 0, isColl00, isColl01,
            isColl10, isColl11, isColl20, isColl21, endPtsColl0, endPtsColl1,
            endPtsColl2);
    }
}

} // namespace Nektar::Operators::detail
