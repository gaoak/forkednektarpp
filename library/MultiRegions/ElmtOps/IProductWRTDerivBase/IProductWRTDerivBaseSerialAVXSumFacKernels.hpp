///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXSumFacKernels.hpp
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
// Description: Kernels of the Serial/AVX sum-factorised inner product
// with the gradient of the basis.
///////////////////////////////////////////////////////////////////////////////

/**
 * @file IProductWRTDerivBaseSerialAVXSumFacKernels.hpp
 * @brief Kernels of the Serial/AVX sum-factorised inner product with the
 * gradient of the basis: the geometric stage that turns the input vector
 * into one weighted field per reference direction, and the launchers that
 * follow it with the transposed tensor derivative, one element per SIMD
 * lane.
 *
 * @details
 * IProductWRTDerivBaseKernelLauncher is the entry point the block operator
 * (IProductWRTDerivBaseSerialAVXSumFac.hpp) calls for one element group;
 * its one-, two- and three-dimensional overloads are selected by the type
 * of the size parameter. Each runs the StdAlignDerivBase kernel of its
 * dimension, which per quadrature point contracts the input's coordinate
 * components against the derivative factors,
 * \f[ g_d = \sum_k \frac{\partial \xi_d}{\partial x_k} f_k , \f]
 * applies the collapsed-coordinate corrections of the simplex shapes and
 * multiplies by the Jacobian and the tensor-product quadrature weights;
 * then SumDerivTensorNDKernel
 * (StdRegions/Operators/PhysDerivSumFacStdKernels.hpp, reached through the
 * PhysDerivSerialAVXSumFacKernels.hpp include) applies the transposed
 * one-dimensional derivative matrix of each direction and sums the
 * directions into a single field. The basis contraction of the Coeff path
 * is not part of these launchers: the block operator calls
 * IProductWRTBaseKernelLauncher (IProductWRTBaseSerialAVXSumFacKernels.hpp,
 * included here for that purpose) on the summed field itself.
 *
 * @p simd_type is the SIMD vector type of one interleaved element group,
 * so one index steps over whole groups and each lane carries its own
 * element; on the Serial execution space the type is scalar. The
 * derivative factors are read untransposed: the indim * dimension factors
 * of one point are contiguous, indexed k * dimension + d. When
 * @p DEFORMED is false the factors and the Jacobian are read once per
 * element group rather than once per point.
 *
 * On the simplex shapes the reference derivative is taken in the collapsed
 * coordinates eta, so the chain-rule factors relating xi to eta enter.
 * Because the operator is the transpose of a derivative, those factors are
 * applied here -- to the direction fields, before the transposed
 * derivative -- rather than after it. The in-code comments give each
 * factor's expression; the tables themselves come from BasisDataKey
 * (eHalfMultOnePlusZero, eTwoOverOneMinusZero).
 */

#pragma once

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp>
#include <MultiRegions/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Contract the input vector with the derivative factors of a
 * segment and apply the quadrature metric.
 *
 * Writes \f$g_0\f$ for every point of the element group, multiplied by the
 * Jacobian and the quadrature weight of each point, so the result is ready
 * for the transposed derivative and, on the Coeff path, for the
 * metric-free IProduct kernel.
 *
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam simd_type   SIMD vector type of one interleaved element group.
 *
 * @param   nq0       Quadrature points of the element.
 * @param   indim     Coordinate dimension: number of input components
 *                    read.
 * @param   df_ptr    Derivative factors of the element group.
 * @param   df_tmp    Caller-owned scratch for the factors of the current
 *                    point.
 * @param   inoffset  Stride, in vectors, between the input's coordinate
 *                    components.
 * @param   in        Input vector \f${\bf f}\f$.
 * @param   out       Receives the weighted \f$g_0\f$, @p nq0 vectors.
 * @param   jac_ptr   Jacobians of the element group.
 * @param   w         Quadrature weights of the element.
 */
template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void StdAlignDerivBase1D(
    const unsigned int nq0, const unsigned int indim, const simd_type *df_ptr,
    simd_type *df_tmp, const size_t inoffset, const simd_type *in,
    simd_type *out, const simd_type *jac_ptr,
    const typename simd_type::scalarType *w)
{
    simd_type jac = 1.0;
    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if constexpr (!DEFORMED)
    {
        // unroll very small loops
        jac       = jac_ptr[0];
        df_tmp[0] = df_ptr[0];
        if (indim >= 2)
        {
            df_tmp[1] = df_ptr[1];
        }
        if (indim == 3)
        {
            df_tmp[2] = df_ptr[2];
        }
    }

    for (unsigned i = 0; i < nq0; ++i)
    {
        if constexpr (DEFORMED)
        {
            jac       = jac_ptr[i];
            df_tmp[0] = df_ptr[i * indim];
            if (indim >= 2)
            {
                df_tmp[1] = df_ptr[i * indim + 1];
            }
            if (indim == 3)
            {
                df_tmp[2] = df_ptr[i * indim + 2];
            }
        }

        simd_type sum = 0.0;
        for (unsigned int d = 0; d < indim; ++d)
        {
            simd_type inval = in[d * inoffset + i];
            sum.fma(inval, df_tmp[d]);
        }
        out[i] = sum * jac * w[i];
    }
}

/**
 * @brief Contract the input vector with the derivative factors of a
 * quadrilateral or triangle and apply the quadrature metric.
 *
 * Writes the two weighted direction fields \f$g_0\f$, \f$g_1\f$ in point
 * order (direction 0 fastest). For the triangles the collapsed correction
 * is applied to \f$g_0\f$ before the weighting, using @p Fac0 and
 * @p Fac1, which are unused for quadrilaterals.
 *
 * @tparam SHAPE_TYPE  Shape of the block's elements.
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam simd_type   SIMD vector type of one interleaved element group.
 *
 * @param   nq0,nq1     Quadrature points per direction.
 * @param   indim       Coordinate dimension: number of input components
 *                      read.
 * @param   df_ptr      Derivative factors of the element group,
 *                      2 * indim per point.
 * @param   df_tmp      Caller-owned scratch for the factors of the current
 *                      point.
 * @param   inoffset    Stride, in vectors, between the input's coordinate
 *                      components.
 * @param   inptr       Input vector \f${\bf f}\f$.
 * @param   out         Two output fields, one per reference direction.
 * @param   Fac0        Table of (1 + eta_0)/2 (triangles).
 * @param   Fac1        Table of 2/(1 - eta_1) (triangles).
 * @param   jac_ptr     Jacobians of the element group.
 * @param   w0,w1       Quadrature weights per direction.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void StdAlignDerivBase2D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int indim,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    const simd_type *inptr, simd_type *out[2],
    [[maybe_unused]] const simd_type *Fac0,
    [[maybe_unused]] const simd_type *Fac1, const simd_type *jac_ptr,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1)
{
    const auto ndf = 2 * indim;
    simd_type jac  = 1.0;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if constexpr (!DEFORMED)
    {
        jac       = jac_ptr[0];
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];

        if (indim == 3)
        {
            df_tmp[4] = df_ptr[4];
            df_tmp[5] = df_ptr[5];
        }
    }

    simd_type f1;
    unsigned int cnt_ji = 0;
    for (unsigned int j = 0; j < nq1; ++j)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            f1 = simd_type(Fac1[j]);
        }

        for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            if constexpr (DEFORMED)
            {
                jac       = jac_ptr[cnt_ji];
                df_tmp[0] = df_ptr[cnt_ji * ndf];
                df_tmp[1] = df_ptr[cnt_ji * ndf + 1];
                df_tmp[2] = df_ptr[cnt_ji * ndf + 2];
                df_tmp[3] = df_ptr[cnt_ji * ndf + 3];

                if (indim == 3)
                {
                    df_tmp[4] = df_ptr[cnt_ji * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_ji * ndf + 5];
                }
            }

            simd_type in0 = inptr[cnt_ji];
            simd_type in1 = inptr[inoffset + cnt_ji];

            simd_type out0 = df_tmp[0] * in0;
            out0.fma(df_tmp[2], in1);

            simd_type out1 = df_tmp[1] * in0;
            out1.fma(df_tmp[3], in1);

            if (indim == 3)
            {
                simd_type in2 = inptr[2 * inoffset + cnt_ji];
                out0.fma(df_tmp[4], in2);
                out1.fma(df_tmp[5], in2);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                // Multiply by geometric factors
                simd_type f0 = Fac0[i];

                // Scale by geometric factor 2/(1-z1)
                out0 *= f1;
                // Scale by geometric factor (1+z0)/(1-z1)
                simd_type c1 = f0 * out1;
                out0.fma(c1, f1);
            }

            // store outputs
            simd_type WJ   = jac * w0[i] * w1[j];
            out[0][cnt_ji] = out0 * WJ;
            out[1][cnt_ji] = out1 * WJ;
        }
    }
}

/**
 * @brief Contract the input vector with the derivative factors of a
 * three-dimensional element and apply the quadrature metric.
 *
 * Writes the three weighted direction fields in point order (direction 0
 * fastest). A 3D element's coordinate dimension is always three, so three
 * input components are read and the derivative factors have nine entries
 * per point. The number of additive collapsed correction terms differs per
 * shape -- none for the hexahedron, one for the prisms, two for the
 * pyramid and the tetrahedra -- and they are applied before the
 * weighting. That count is not the number of factor tables a shape reads,
 * which is larger; the tables a shape does not read are ignored.
 *
 * @tparam SHAPE_TYPE  Shape of the block's elements.
 * @tparam DEFORMED    True when the geometric factors vary within an
 *                     element.
 * @tparam simd_type   SIMD vector type of one interleaved element group.
 *
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   df_ptr      Derivative factors of the element group, nine per
 *                      point.
 * @param   df_tmp      Caller-owned scratch for the factors of the current
 *                      point.
 * @param   inoffset    Stride, in vectors, between the input's coordinate
 *                      components.
 * @param   Fac0        Table of (1 + eta_0)/2.
 * @param   Fac1        Table of (1 + eta_1)/2.
 * @param   Fac1m       Table of 2/(1 - eta_1).
 * @param   Fac2        Table of 2/(1 - eta_2).
 * @param   jac_ptr     Jacobians of the element group.
 * @param   w0,w1,w2    Quadrature weights per direction.
 * @param   inptr       Input vector \f${\bf f}\f$.
 * @param   out         Three output fields, one per reference direction.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void StdAlignDerivBase3D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    [[maybe_unused]] const simd_type *Fac0,
    [[maybe_unused]] const simd_type *Fac1,
    [[maybe_unused]] const simd_type *Fac1m,
    [[maybe_unused]] const simd_type *Fac2, const simd_type *jac_ptr,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2, const simd_type *inptr,
    simd_type *out[3])
{
    const auto ndf = 9;
    simd_type jac  = 1.0;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if constexpr (!DEFORMED)
    {
        jac       = jac_ptr[0];
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
        df_tmp[4] = df_ptr[4];
        df_tmp[5] = df_ptr[5];
        df_tmp[6] = df_ptr[6];
        df_tmp[7] = df_ptr[7];
        df_tmp[8] = df_ptr[8];
    }

    unsigned int cnt_kji = 0;
    simd_type f0, f1, f1m, f2;

    for (unsigned int k = 0; k < nq2; ++k)
    {
        if constexpr (SHAPE_TYPE != LibUtilities::Hex)
        {
            f2 = simd_type(Fac2[k]);
        }

        for (unsigned int j = 0; j < nq1; ++j)
        {
            if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                          (SHAPE_TYPE == LibUtilities::NodalTet))
            {
                f1  = simd_type(Fac1[j]);
                f1m = simd_type(Fac1m[j]);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                f1 = simd_type(Fac1[j]);
            }

            simd_type w12 = w1[j] * w2[k];

            for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
            {
                if constexpr (DEFORMED)
                {
                    jac       = jac_ptr[cnt_kji];
                    df_tmp[0] = df_ptr[cnt_kji * ndf];
                    df_tmp[1] = df_ptr[cnt_kji * ndf + 1];
                    df_tmp[2] = df_ptr[cnt_kji * ndf + 2];
                    df_tmp[3] = df_ptr[cnt_kji * ndf + 3];
                    df_tmp[4] = df_ptr[cnt_kji * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_kji * ndf + 5];
                    df_tmp[6] = df_ptr[cnt_kji * ndf + 6];
                    df_tmp[7] = df_ptr[cnt_kji * ndf + 7];
                    df_tmp[8] = df_ptr[cnt_kji * ndf + 8];
                }

                simd_type in0 = inptr[cnt_kji];
                simd_type in1 = inptr[inoffset + cnt_kji];
                simd_type in2 = inptr[2 * inoffset + cnt_kji];

                simd_type out0 = df_tmp[0] * in0;
                out0.fma(df_tmp[3], in1);
                out0.fma(df_tmp[6], in2);

                simd_type out1 = df_tmp[1] * in0;
                out1.fma(df_tmp[4], in1);
                out1.fma(df_tmp[7], in2);

                simd_type out2 = df_tmp[2] * in0;
                out2.fma(df_tmp[5], in1);
                out2.fma(df_tmp[8], in2);

                if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                              (SHAPE_TYPE == LibUtilities::NodalTet))
                {
                    // (out0 + (out1 + out2)*(1+z0)/2) * 2/(1 - z1) * 2/(1 - z2)
                    f0 = Fac0[i];
                    out0.fma(out1 + out2, f0);
                    out0 *= f1m * f2;

                    // (out1 + out2 * (1+z1)/2 ) * 2/(1 - z2)
                    out1.fma(out2, f1);
                    out1 *= f2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    // (out0 +  out2 * (1 + z0)/2 ) * 2/(1 - z2)
                    f0 = Fac0[i];
                    out0.fma(out2, f0);
                    out0 *= f2;

                    // (out1 + out2 * (1+z1)/2 ) * 2/(1 - z2)
                    out1.fma(out2, f1);
                    out1 *= f2;
                }
                else if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                                   (SHAPE_TYPE == LibUtilities::NodalPrism))
                {
                    // (out0 +  out2 * (1 + z0)/2 ) * 2/(1 - z2)
                    f0 = Fac0[i];
                    out0.fma(out2, f0);
                    out0 *= f2;
                }

                // store outputs
                simd_type WJ    = jac * w0[i] * w12;
                out[0][cnt_kji] = out0 * WJ;
                out[1][cnt_kji] = out1 * WJ;
                out[2][cnt_kji] = out2 * WJ;
            }
        }
    }
}

/**
 * @brief Segment launcher: form the weighted direction field of one
 * element group and apply the transposed derivative.
 *
 * StdAlignDerivBase1D writes \f$g_0\f$ into
 * @p tmp0, and SumDerivTensor1DKernel then writes
 * \f$D_0^{\mathsf T} g_0\f$ to @p out, overwriting it.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements (Seg); unused.
 * @tparam DEFORMED          True when the geometric factors vary within an
 *                           element.
 * @tparam TSizeParameter1D  1D size parameter, modal or physical-space.
 * @tparam simd_type         SIMD vector type of one interleaved element
 *                           group.
 *
 * @param   sizeParam1D Sizes of the expansion; only nq0 is read.
 * @param   indim       Coordinate dimension: number of input components
 *                      read.
 * @param   df_ptr      Derivative factors of the element group.
 * @param   df_tmp      Caller-owned scratch for the factors of the current
 *                      point.
 * @param   inoffset    Stride, in vectors, between the input's coordinate
 *                      components.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   jac         Jacobians of the element group.
 * @param   w0          Quadrature weights of the expansion.
 * @param   D0          One-dimensional derivative matrix.
 * @param   tmp0        Plane of nq0 vectors receiving \f$g_0\f$.
 * @param   out         Plane of nq0 vectors receiving the transposed
 *                      derivative.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTDerivBaseKernelLauncher(
    const TSizeParameter1D sizeParam1D, const unsigned int indim,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    const simd_type *in, const simd_type *jac,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *D0, simd_type *tmp0, simd_type *out)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D> ||
                      IsPhysSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be a one dimensional size "
                  "parameter.");

    const unsigned int nq0 = sizeParam1D.nq0();

    // Align the derivative with the standard element.
    StdAlignDerivBase1D<DEFORMED>(nq0, indim, df_ptr, df_tmp, inoffset, in,
                                  tmp0, jac, w0);

    // Apply the transposed derivative and sum up.
    SumDerivTensor1DKernel<false>(nq0, tmp0, D0, out);
}

/**
 * @brief Quadrilateral or triangle launcher: form the two weighted
 * direction fields of one element group and apply the transposed
 * derivatives.
 *
 * StdAlignDerivBase2D writes \f$g_0\f$ and \f$g_1\f$ into @p tmp0 and
 * @p tmp1 -- including, for the triangles, the collapsed corrections from
 * @p f0 and @p f1 -- and SumDerivTensor2DKernel sums their transposed
 * derivatives into @p out, overwriting it.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam DEFORMED          True when the geometric factors vary within an
 *                           element.
 * @tparam TSizeParameter2D  2D size parameter, modal or physical-space.
 * @tparam simd_type         SIMD vector type of one interleaved element
 *                           group.
 *
 * @param   sizeParam2D Sizes per direction; only nq0 and nq1 are read.
 * @param   indim       Coordinate dimension: number of input components
 *                      read.
 * @param   df_ptr      Derivative factors of the element group.
 * @param   df_tmp      Caller-owned scratch for the factors of the current
 *                      point.
 * @param   inoffset    Stride, in vectors, between the input's coordinate
 *                      components.
 * @param   in          Input vector \f${\bf f}\f$.
 * @param   f0          Table of (1 + eta_0)/2 (triangles).
 * @param   f1          Table of 2/(1 - eta_1) (triangles).
 * @param   jac         Jacobians of the element group.
 * @param   w0,w1       Quadrature weights per direction.
 * @param   D0,D1       One-dimensional derivative matrices.
 * @param   tmp0,tmp1   Planes of nq0 * nq1 vectors receiving the direction
 *                      fields.
 * @param   out         Plane of nq0 * nq1 vectors receiving the summed
 *                      transposed derivative.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTDerivBaseKernelLauncher(
    const TSizeParameter2D sizeParam2D, const unsigned int indim,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    const simd_type *in, const simd_type *f0, const simd_type *f1,
    const simd_type *jac, const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, simd_type *tmp0, simd_type *tmp1,
    simd_type *out)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D> ||
                      IsPhysSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be a two dimensional size "
                  "parameter.");

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    simd_type *tmpPtr[2] = {tmp0, tmp1};

    // Align the derivative with the standard element.
    StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(nq0, nq1, indim, df_ptr, df_tmp,
                                              inoffset, in, tmpPtr, f0, f1, jac,
                                              w0, w1);

    // Apply the transposed derivative and sum up.
    SumDerivTensor2DKernel<false>(nq0, nq1, tmpPtr[0], tmpPtr[1], D0, D1, out);
}

/**
 * @brief Three-dimensional launcher: form the three weighted direction
 * fields of one element group and apply the transposed derivatives.
 *
 * StdAlignDerivBase3D writes the direction fields into @p tmp0 ... @p tmp2
 * -- with the shape's collapsed corrections from the four factor tables --
 * and SumDerivTensor3DKernel sums their transposed derivatives into
 * @p out, overwriting it. The coordinate dimension of a 3D element is
 * always three, so @p indim is unused.
 *
 * @tparam SHAPE_TYPE        Shape of the block's elements.
 * @tparam DEFORMED          True when the geometric factors vary within an
 *                           element.
 * @tparam TSizeParameter3D  3D size parameter, modal or physical-space.
 * @tparam simd_type         SIMD vector type of one interleaved element
 *                           group.
 *
 * @param   sizeParam3D     Sizes per direction; only the point counts are
 *                          read.
 * @param   indim           Coordinate dimension; unused.
 * @param   df_ptr          Derivative factors of the element group, nine
 *                          per point.
 * @param   df_tmp          Caller-owned scratch for the factors of the
 *                          current point.
 * @param   inoffset        Stride, in vectors, between the input's
 *                          coordinate components.
 * @param   in              Input vector \f${\bf f}\f$.
 * @param   f0              Table of (1 + eta_0)/2.
 * @param   f1              Table of (1 + eta_1)/2.
 * @param   f1m             Table of 2/(1 - eta_1).
 * @param   f2              Table of 2/(1 - eta_2).
 * @param   jac             Jacobians of the element group.
 * @param   w0,w1,w2        Quadrature weights per direction.
 * @param   D0,D1,D2        One-dimensional derivative matrices.
 * @param   tmp0,tmp1,tmp2  Planes of nqTot vectors receiving the direction
 *                          fields.
 * @param   out             Plane of nqTot vectors receiving the summed
 *                          transposed derivative.
 */
template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTDerivBaseKernelLauncher(
    const TSizeParameter3D sizeParam3D,
    [[maybe_unused]] const unsigned int indim, const simd_type *df_ptr,
    simd_type *df_tmp, const size_t inoffset, const simd_type *in,
    const simd_type *f0, const simd_type *f1, const simd_type *f1m,
    const simd_type *f2, const simd_type *jac,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, simd_type *tmp0, simd_type *tmp1,
    simd_type *tmp2, simd_type *out)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D> ||
                      IsPhysSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be a three dimensional size "
                  "parameter.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    simd_type *tmpPtr[3] = {tmp0, tmp1, tmp2};

    // Align the derivative with the standard element.
    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(nq0, nq1, nq2, df_ptr, df_tmp,
                                              inoffset, f0, f1, f1m, f2, jac,
                                              w0, w1, w2, in, tmpPtr);

    // Apply the transposed derivative and sum up.
    SumDerivTensor3DKernel<false>(nq0, nq1, nq2, tmpPtr[0], tmpPtr[1],
                                  tmpPtr[2], D0, D1, D2, out);
}

} // namespace Nektar::MultiRegions::detail
