///////////////////////////////////////////////////////////////////////////////
//
// File: CurlCurlBlockOpSerialAVXStdMatKernels.hpp
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
 * @file CurlCurlBlockOpSerialAVXStdMatKernels.hpp
 * @brief Pointwise curl kernels of the Serial/AVX StdMat curl-curl block
 * operator (see CurlCurlBlockOpSerialAVXStdMat.hpp).
 *
 * @details
 * Each kernel takes the reference-space derivatives of one element group,
 * as produced by the standard-matrix multiplies of the block operator,
 * chain-rules them into Cartesian derivatives through the geometric
 * derivative factors and combines those into a curl in the same sweep, one
 * element per SIMD lane. The block operator uses them twice per group: to
 * form the vorticity omega = curl u and then the output curl omega. On a
 * multi-plane 3DH1 block it calls each 2D kernel once per group and plane,
 * for the plane part of a single curl.
 */

#pragma once

#include <LibUtilities/BasicUtils/NekInline.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Scalar curl of a two-dimensional vector field,
 * omega = d(u1)/dx0 - d(u0)/dx1, per quadrature point with one element per
 * SIMD lane.
 *
 * The derivative-factor entry k * 2 + d maps the reference-direction-d
 * derivative into the Cartesian-k gradient component. For regular geometry
 * one set of four factors is stored per element group; for deformed
 * geometry one per quadrature point. The chain rule and the curl are
 * applied in a single sweep.
 *
 * @tparam ExecSpace Execution space; unused.
 * @tparam DEFORMED  Read per-point rather than per-element derivative
 *                   factors.
 * @tparam TData     SIMD vector type of the data (a scalar type on the
 *                   Serial execution space).
 *
 * @param   nqTot   Quadrature points per element.
 * @param   df_ptr  Derivative factors: four vectors per element group
 *                  (regular) or per point (deformed).
 * @param   deriv   Reference-space derivatives of the two components,
 *                  ordered as deriv[(c * 2 + d) * nqTot + i] for component
 *                  c and direction d.
 * @param   omega   Output plane of nqTot vectors.
 */
template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Curl2DScalarStdMatKernel(const unsigned int nqTot,
                                                      const TData *df_ptr,
                                                      const TData *deriv,
                                                      TData *omega)
{
    constexpr unsigned int ndf = 4u;

    TData df_tmp[4];
    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
    }

    for (unsigned int i = 0; i < nqTot; ++i)
    {
        const TData d0u = deriv[i];
        const TData d1u = deriv[nqTot + i];
        const TData d0v = deriv[2u * nqTot + i];
        const TData d1v = deriv[3u * nqTot + i];

        if constexpr (DEFORMED)
        {
            df_tmp[0] = df_ptr[ndf * i];
            df_tmp[1] = df_ptr[ndf * i + 1];
            df_tmp[2] = df_ptr[ndf * i + 2];
            df_tmp[3] = df_ptr[ndf * i + 3];
        }

        TData dvdx = d0v * df_tmp[0];
        dvdx.fma(d1v, df_tmp[1]);
        TData dudy = d0u * df_tmp[2];
        dudy.fma(d1u, df_tmp[3]);

        omega[i] = dvdx - dudy;
    }
}

/**
 * @brief Vector curl of a two-dimensional scalar field,
 * out = ( d(omega)/dx1, -d(omega)/dx0 ), per quadrature point with one
 * element per SIMD lane.
 *
 * The derivative factors are read as in Curl2DScalarStdMatKernel; both
 * output components come out of the same sweep, so the sign flip needs no
 * pass of its own.
 *
 * @tparam ExecSpace Execution space; unused.
 * @tparam DEFORMED  Read per-point rather than per-element derivative
 *                   factors.
 * @tparam TData     SIMD vector type of the data (a scalar type on the
 *                   Serial execution space).
 *
 * @param   nqTot   Quadrature points per element.
 * @param   df_ptr  Derivative factors: four vectors per element group
 *                  (regular) or per point (deformed).
 * @param   deriv   Reference-space derivatives of omega, ordered as
 *                  deriv[d * nqTot + i] for direction d.
 * @param   out0    First output component, nqTot vectors.
 * @param   out1    Second output component, nqTot vectors.
 */
template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Curl2DVectorStdMatKernel(const unsigned int nqTot,
                                                      const TData *df_ptr,
                                                      const TData *deriv,
                                                      TData *out0, TData *out1)
{
    constexpr unsigned int ndf = 4u;

    TData df_tmp[4];
    if constexpr (!DEFORMED)
    {
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
    }

    for (unsigned int i = 0; i < nqTot; ++i)
    {
        const TData d0 = deriv[i];
        const TData d1 = deriv[nqTot + i];

        if constexpr (DEFORMED)
        {
            df_tmp[0] = df_ptr[ndf * i];
            df_tmp[1] = df_ptr[ndf * i + 1];
            df_tmp[2] = df_ptr[ndf * i + 2];
            df_tmp[3] = df_ptr[ndf * i + 3];
        }

        TData dwdx = d0 * df_tmp[0];
        dwdx.fma(d1, df_tmp[1]);
        TData dwdy = d0 * df_tmp[2];
        dwdy.fma(d1, df_tmp[3]);

        out0[i] = dwdy;
        out1[i] = -dwdx;
    }
}

/**
 * @brief Curl of a three-dimensional vector field, per quadrature point
 * with one element per SIMD lane; used for both passes of the curl-curl
 * operator, omega = curl u and out = curl omega.
 *
 * The derivative-factor entry k * 3 + d maps the reference-direction-d
 * derivative into the Cartesian-k gradient component; nine factors are
 * stored per element group (regular) or per point (deformed). Only the six
 * off-diagonal Cartesian derivatives are formed, the diagonal ones
 * cancelling out of the curl:
 *     out0 = d(u2)/dx1 - d(u1)/dx2,
 *     out1 = d(u0)/dx2 - d(u2)/dx0,
 *     out2 = d(u1)/dx0 - d(u0)/dx1.
 *
 * @tparam ExecSpace Execution space; unused.
 * @tparam DEFORMED  Read per-point rather than per-element derivative
 *                   factors.
 * @tparam TData     SIMD vector type of the data (a scalar type on the
 *                   Serial execution space).
 *
 * @param   nqTot   Quadrature points per element.
 * @param   df_ptr  Derivative factors: nine vectors per element group
 *                  (regular) or per point (deformed).
 * @param   deriv   Reference-space derivatives of the three components,
 *                  ordered as deriv[(c * 3 + d) * nqTot + i] for component
 *                  c and direction d.
 * @param   out0    First output component, nqTot vectors.
 * @param   out1    Second output component, nqTot vectors.
 * @param   out2    Third output component, nqTot vectors.
 */
template <typename ExecSpace, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void Curl3DStdMatKernel(const unsigned int nqTot,
                                                const TData *df_ptr,
                                                const TData *deriv, TData *out0,
                                                TData *out1, TData *out2)
{
    constexpr unsigned int ndf = 9u;

    TData df_tmp[9];
    if constexpr (!DEFORMED)
    {
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

    for (unsigned int i = 0; i < nqTot; ++i)
    {
        TData d0[3], d1[3], d2[3];
        d0[0] = deriv[i];
        d1[0] = deriv[nqTot + i];
        d2[0] = deriv[2u * nqTot + i];
        d0[1] = deriv[3u * nqTot + i];
        d1[1] = deriv[4u * nqTot + i];
        d2[1] = deriv[5u * nqTot + i];
        d0[2] = deriv[6u * nqTot + i];
        d1[2] = deriv[7u * nqTot + i];
        d2[2] = deriv[8u * nqTot + i];

        if constexpr (DEFORMED)
        {
            df_tmp[0] = df_ptr[ndf * i + 0];
            df_tmp[1] = df_ptr[ndf * i + 1];
            df_tmp[2] = df_ptr[ndf * i + 2];
            df_tmp[3] = df_ptr[ndf * i + 3];
            df_tmp[4] = df_ptr[ndf * i + 4];
            df_tmp[5] = df_ptr[ndf * i + 5];
            df_tmp[6] = df_ptr[ndf * i + 6];
            df_tmp[7] = df_ptr[ndf * i + 7];
            df_tmp[8] = df_ptr[ndf * i + 8];
        }

        TData dudy = d0[0] * df_tmp[3];
        dudy.fma(d1[0], df_tmp[4]);
        dudy.fma(d2[0], df_tmp[5]);

        TData dudz = d0[0] * df_tmp[6];
        dudz.fma(d1[0], df_tmp[7]);
        dudz.fma(d2[0], df_tmp[8]);

        TData dvdx = d0[1] * df_tmp[0];
        dvdx.fma(d1[1], df_tmp[1]);
        dvdx.fma(d2[1], df_tmp[2]);

        TData dvdz = d0[1] * df_tmp[6];
        dvdz.fma(d1[1], df_tmp[7]);
        dvdz.fma(d2[1], df_tmp[8]);

        TData dwdx = d0[2] * df_tmp[0];
        dwdx.fma(d1[2], df_tmp[1]);
        dwdx.fma(d2[2], df_tmp[2]);

        TData dwdy = d0[2] * df_tmp[3];
        dwdy.fma(d1[2], df_tmp[4]);
        dwdy.fma(d2[2], df_tmp[5]);

        out0[i] = dwdy - dvdz; // dw/dy - dv/dz
        out1[i] = dudz - dwdx; // du/dz - dw/dx
        out2[i] = dvdx - dudy; // dv/dx - du/dy
    }
}

} // namespace Nektar::MultiRegions::detail
