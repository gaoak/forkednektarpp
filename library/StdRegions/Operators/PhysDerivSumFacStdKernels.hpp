///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivSumFacStdKernels.hpp
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
// Description: The tensorial derivative kernels of the sum-factorised
// physical derivative, for one field or for several at once, used by the
// SIMD element operators and in a scalar manner within StdRegions
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

template <typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor1DKernel(
    const unsigned int nq0, const simd_type *in,
    const typename simd_type::scalarType *D0, simd_type *out_d0)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    for (unsigned int i = 0; i < nq0; ++i)
    { // Row index of D0 matrix

        simd_type prod_sum = 0.0;
        for (unsigned int k = 0; k < nq0; ++k)
        {                                    // Col index of D0, row index of IN
            simd_type v1 = D0[k * nq0 + i];  // Load 1x
            simd_type v2 = simd_type(in[k]); // Load 1x

            prod_sum.fma(v1, v2);
        }

        out_d0[i] = prod_sum; // Store 1x
    }
}

template <bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor1DKernel(
    const unsigned int nq0, const simd_type *in0,
    const typename simd_type::scalarType *D0, simd_type *out)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0^T * in
    for (unsigned int i = 0; i < nq0; ++i)
    { // Row index of D0 matrix
        simd_type prod_sum = 0.0;
        for (unsigned int k = 0; k < nq0; ++k)
        { // Col index of D0, row index of IN
            simd_type v1 = D0[i * nq0 + k];
            simd_type v2 = in0[k];

            prod_sum.fma(v1, v2);
        }

        if constexpr (APPEND)
        {
            out[i] += prod_sum; // Store 1x
        }
        else
        {
            out[i] = prod_sum; // Store 1x
        }
    }
}

template <typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor2DKernel(
    const unsigned int nq0, const unsigned int nq1, const simd_type *in,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, simd_type *out_d0,
    simd_type *out_d1, bool Deriv0 = true, bool Deriv1 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    if (Deriv0) // backwards compatibility with StdRegsion PhyTensorDeriv
    {
        for (unsigned int i = 0; i < nq0; ++i)
        { // Row index of D0 matrix
            for (unsigned int j = 0; j < nq1; ++j)
            { // Col index of IN matrix

                simd_type prod_sum = 0.0;
                for (unsigned int k = 0; k < nq0; ++k)
                { // Col index of D0, row index of IN
                    simd_type v1 = D0[k * nq0 + i]; // Load 1x
                    simd_type v2 = in[j * nq0 + k]; // Load 1x

                    prod_sum.fma(v1, v2);
                }

                out_d0[j * nq0 + i] = prod_sum; // Store 1x
            }
        }
    }

    // D1 * in
    if (Deriv1)
    {
        // in * D1^T
        for (unsigned int i = 0; i < nq0; ++i)
        { // row index for grid
            for (unsigned int j = 0; j < nq1; ++j)
            { // Column index for D1^T (row idx for D1)

                simd_type prod_sum = 0.0;
                for (unsigned int k = 0; k < nq1; ++k)
                {
                    simd_type v1 = in[k * nq0 + i]; // Load 1x
                    simd_type v2 = D1[k * nq1 + j]; // Load 1x

                    prod_sum.fma(v1, v2);
                }

                out_d1[j * nq0 + i] = prod_sum; // Store 1x
            }
        }
    }
}

template <bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor2DKernel(
    const unsigned int nq0, const unsigned int nq1, const simd_type *in0,
    const simd_type *in1, const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, simd_type *out,
    bool Deriv0 = true, bool Deriv1 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0^T * in
    if (Deriv0) // backwards compatibility with StdRegsion PhyTensorDeriv
    {
        for (unsigned int i = 0; i < nq0; ++i)
        { // Row index of D0 matrix
            for (unsigned int j = 0; j < nq1; ++j)
            { // Col index of IN matrix

                simd_type prod_sum = 0.0;
                for (unsigned int k = 0; k < nq0; ++k)
                { // Col index of D0, row index of IN
                    simd_type v1 = D0[i * nq0 + k];  // Load 1x
                    simd_type v2 = in0[j * nq0 + k]; // Load 1x

                    prod_sum.fma(v1, v2);
                }

                if constexpr (APPEND)
                {
                    out[j * nq0 + i] += prod_sum; // Store 1x
                }
                else
                {
                    out[j * nq0 + i] = prod_sum; // Store 1x
                }
            }
        }
    }

    // D1^T * in
    if (Deriv1)
    {
        // in * D1^T
        for (unsigned int j = 0; j < nq1; ++j)
        { // Column index for D1^T (row idx for D1)
            for (unsigned int i = 0; i < nq0; ++i)
            { // row index for grid

                simd_type prod_sum = 0.0;
                for (unsigned int k = 0; k < nq1; ++k)
                {
                    simd_type v1 = in1[k * nq0 + i]; // Load 1x
                    simd_type v2 = D1[j * nq1 + k];  // Load 1x

                    prod_sum.fma(v1, v2);
                }

                out[j * nq0 + i] += prod_sum; // Store 1x
            }
        }
    }
}

template <typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *in, const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, simd_type *out_d0,
    simd_type *out_d1, simd_type *out_d2, bool Deriv0 = true,
    bool Deriv1 = true, bool Deriv2 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // Direction 0
    if (Deriv0)
    {
        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int j = 0; j < nq1 * nq2; ++j)
            {
                simd_type prod_sum = 0.0;
                for (unsigned int k = 0; k < nq0; ++k)
                {
                    simd_type v1 = D0[k * nq0 + i]; // Load 1x
                    simd_type v2 = in[j * nq0 + k]; // Load 1x

                    prod_sum.fma(v1, v2);
                }

                out_d0[j * nq0 + i] = prod_sum; // Store 1x
            }
        }
    }

    // Direction 1
    if (Deriv1)
    {
        for (unsigned int block = 0; block < nq2; ++block)
        {
            unsigned int start = block * nq0 * nq1;

            for (unsigned int i = 0; i < nq0; ++i)
            {
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    simd_type prod_sum = 0.0;
                    for (unsigned int k = 0; k < nq1; ++k)
                    {
                        simd_type v1 = in[start + k * nq0 + i]; // Load 1x
                        simd_type v2 = D1[k * nq1 + j];         // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    out_d1[start + j * nq0 + i] = prod_sum; // Store 1x
                }
            }
        }
    }

    // Direction 2
    if (Deriv2)
    {
        for (unsigned int i = 0; i < nq0 * nq1; ++i)
        {
            for (unsigned int j = 0; j < nq2; ++j)
            {
                simd_type prod_sum = 0.0;
                for (unsigned int k = 0; k < nq2; ++k)
                {
                    simd_type v1 = simd_type(in[k * nq0 * nq1 + i]); // Load 1x
                    simd_type v2 = D2[k * nq2 + j];                  // Load 1x

                    prod_sum.fma(v1, v2);
                }

                out_d2[j * nq0 * nq1 + i] = prod_sum; // Store 1x
            }
        }
    }
}

template <bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *in0, const simd_type *in1, const simd_type *in2,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, simd_type *out,
    bool Deriv0 = true, bool Deriv1 = true, bool Deriv2 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // Compared with PhysDerivTensor3DKernel, here we must multiply by the
    // transpose of D matrix

    // Direction 0
    if (Deriv0)
    {
        for (unsigned int p = 0; p < nq0; ++p)
        {
            unsigned int cnt_kji = 0, cnt_kj = 0;
            for (unsigned int k = 0; k < nq2; ++k)
            {
                for (unsigned int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type prod_sum = 0.0;
                    for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        simd_type v1 = D0[p * nq0 + i]; // Load 1x
                        simd_type v2 = in0[cnt_kji];    // Load 1x

                        prod_sum.fma(v1, v2);
                    }
                    if constexpr (APPEND)
                    {
                        out[cnt_kj * nq0 + p] += prod_sum; // Store 1x
                    }
                    else
                    {
                        out[cnt_kj * nq0 + p] = prod_sum; // Store 1x
                    }
                }
            }
        }
    }

    // Direction 1
    if (Deriv1)
    {
        for (unsigned int block = 0; block < nq2; ++block)
        {
            unsigned int start = block * nq0 * nq1;

            for (unsigned int i = 0; i < nq0; ++i)
            {
                for (unsigned int j = 0; j < nq1; ++j)
                {
                    simd_type prod_sum = 0.0;
                    for (unsigned int k = 0; k < nq1; ++k)
                    {
                        simd_type v1 = in1[start + k * nq0 + i]; // Load 1x
                        simd_type v2 = D1[j * nq1 + k];          // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    out[start + j * nq0 + i] += prod_sum; // Store 1x
                }
            }
        }
    }

    // Direction 2
    if (Deriv2)
    {
        unsigned int cnt_hi = 0;
        for (unsigned int h = 0; h < nq1; ++h)
        {
            for (unsigned int i = 0; i < nq0; ++i, ++cnt_hi)
            {
                for (unsigned int j = 0; j < nq2; ++j)
                {
                    simd_type prod_sum = 0.0;
                    for (unsigned int k = 0; k < nq2; ++k)
                    {
                        simd_type v1 = in2[k * nq0 * nq1 + cnt_hi]; // Load 1x
                        simd_type v2 = D2[j * nq2 + k];             // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    out[j * nq0 * nq1 + cnt_hi] += prod_sum; // Store 1x
                }
            }
        }
    }
}

/**
 * @brief The collapsed-coordinate derivative of @p NV fields at once: the
 * line's values of every field are held in registers and each derivative
 * entry is loaded once for all of them. `out[v * varStride + i]` holds
 * field v.
 * More than @p MAXQ points exceed the register block: the derivatives are
 * then formed one field at a time by PhysDerivTensor1DKernel().
 *
 * @tparam NV     Fields differentiated together.
 * @tparam MAXQ   Points per direction the register block holds.
 *
 * @param   nq0     Quadrature points.
 * @param   in      The NV fields, `nq0` each.
 * @param   D0      Differentiation matrix, column major.
 * @param   out     The NV derivatives, `nq0` each.
 * @param   varStride Distance between consecutive fields of @p out, at
 *                    least `nq0`.
 */
template <unsigned int NV, unsigned int MAXQ, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor1DLinesKernel(
    const unsigned int nq0, const simd_type *const in[NV],
    const typename simd_type::scalarType *D0, simd_type *out,
    const unsigned int varStride)
{
    if (nq0 > MAXQ)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            PhysDerivTensor1DKernel(nq0, in[v], D0, out + v * varStride);
        }
        return;
    }
    simd_type line[NV][MAXQ], sum[NV];

    for (unsigned int v = 0; v < NV; ++v)
    {
        for (unsigned int k = 0; k < nq0; ++k)
        {
            line[v][k] = in[v][k];
        }
    }
    for (unsigned int i = 0; i < nq0; ++i)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            sum[v] = simd_type(0.0);
        }
        for (unsigned int k = 0; k < nq0; ++k)
        {
            const simd_type entry = D0[k * nq0 + i];
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v].fma(entry, line[v][k]);
            }
        }
        for (unsigned int v = 0; v < NV; ++v)
        {
            out[v * varStride + i] = sum[v];
        }
    }
}

/**
 * @brief The transposed derivative contraction of @p NV fields at once into
 * the NV outputs, `out[v] = (-) D0^T in[v]`, negated with @p NEGATE, as
 * SumDerivTensor1DKernel() forms it one field at a time, with the line's
 * inputs for every field held in registers. `in[v * varStride + i]`
 * holds field v. More than @p MAXQ points exceed the register block: that
 * kernel then forms the contraction one field at a time, negated after.
 *
 * @tparam NEGATE The contraction is negated.
 * @tparam NV     Fields contracted together.
 * @tparam MAXQ   Points per direction the register block holds.
 *
 * @param   nq0     Quadrature points.
 * @param   in      The NV inputs, `nq0` each.
 * @param   varStride Distance between consecutive fields of @p in, at
 *                    least `nq0`.
 * @param   D0      Differentiation matrix, column major.
 * @param   out     The NV outputs, `nq0` each; overwritten.
 */
template <bool NEGATE, unsigned int NV, unsigned int MAXQ, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor1DLinesKernel(
    const unsigned int nq0, const simd_type *in, const unsigned int varStride,
    const typename simd_type::scalarType *D0, simd_type *const out[NV])
{
    if (nq0 > MAXQ)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            SumDerivTensor1DKernel<false>(nq0, in + v * varStride, D0, out[v]);
            if constexpr (NEGATE)
            {
                for (unsigned int q = 0; q < nq0; ++q)
                {
                    out[v][q] = -out[v][q];
                }
            }
        }
        return;
    }
    simd_type line[NV][MAXQ], sum[NV];

    for (unsigned int v = 0; v < NV; ++v)
    {
        for (unsigned int i = 0; i < nq0; ++i)
        {
            line[v][i] = in[v * varStride + i];
        }
    }
    for (unsigned int p = 0; p < nq0; ++p)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            sum[v] = simd_type(0.0);
        }
        for (unsigned int i = 0; i < nq0; ++i)
        {
            const simd_type entry = D0[p * nq0 + i];
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v].fma(entry, line[v][i]);
            }
        }
        for (unsigned int v = 0; v < NV; ++v)
        {
            if constexpr (NEGATE)
            {
                out[v][p] = -sum[v];
            }
            else
            {
                out[v][p] = sum[v];
            }
        }
    }
}

/**
 * @brief The two collapsed-coordinate derivatives of @p NV fields at once,
 * line by line: a line's values of every field are held in registers and
 * every output of the line is formed from them, so each derivative entry
 * is loaded once for all the fields. `out[v * varStride + d * nqTot + q]`
 * holds
 * direction d of field v, as PhysDerivTensor2DKernel() lays its two
 * outputs out. More than @p MAXQ points in a direction exceed the register
 * block: that kernel then forms the derivatives one field at a time.
 *
 * @tparam NV     Fields differentiated together.
 * @tparam MAXQ   Points per direction the register block holds.
 *
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   in      The NV fields, `nq0 * nq1` each.
 * @param   D0,D1   Differentiation matrices, column major.
 * @param   out     The 2 NV derivatives, `nq0 * nq1` each.
 * @param   varStride Distance between consecutive fields of @p out, at
 *                    least `2 * nq0 * nq1`.
 */
template <unsigned int NV, unsigned int MAXQ, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor2DLinesKernel(
    const unsigned int nq0, const unsigned int nq1,
    const simd_type *const in[NV], const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, simd_type *out,
    const unsigned int varStride)
{
    const unsigned int nqTot = nq0 * nq1;
    if (nq0 > MAXQ || nq1 > MAXQ)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            PhysDerivTensor2DKernel(nq0, nq1, in[v], D0, D1,
                                    out + v * varStride,
                                    out + v * varStride + nqTot);
        }
        return;
    }
    simd_type line[NV][MAXQ], sum[NV];

    // Direction 0: lines along i, contiguous.
    for (unsigned int j = 0; j < nq1; ++j)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int k = 0; k < nq0; ++k)
            {
                line[v][k] = in[v][j * nq0 + k];
            }
        }
        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int k = 0; k < nq0; ++k)
            {
                const simd_type entry = D0[k * nq0 + i];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][k]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                out[v * varStride + j * nq0 + i] = sum[v];
            }
        }
    }
    // Direction 1: lines along j, stride nq0.
    for (unsigned int i = 0; i < nq0; ++i)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int k = 0; k < nq1; ++k)
            {
                line[v][k] = in[v][k * nq0 + i];
            }
        }
        for (unsigned int j = 0; j < nq1; ++j)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int k = 0; k < nq1; ++k)
            {
                const simd_type entry = D1[k * nq1 + j];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][k]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                out[v * varStride + nqTot + j * nq0 + i] = sum[v];
            }
        }
    }
}

/**
 * @brief The sum of the two transposed derivative contractions of @p NV
 * fields at once, line by line, into the NV outputs: `out[v] = (-)(D0^T
 * in[v][0] + D1^T in[v][1])`, negated with @p NEGATE, as
 * SumDerivTensor2DKernel() forms it one field at a time, with a line's
 * inputs for every field held in registers. `in[v * varStride + d * nqTot
 * + q]`
 * holds direction d of field v. More than @p MAXQ points in a direction
 * exceed the register block: that kernel then forms the sum one field at a
 * time, negated after.
 *
 * @tparam NEGATE The sum is negated.
 * @tparam NV     Fields contracted together.
 * @tparam MAXQ   Points per direction the register block holds.
 *
 * @param   nq0,nq1 Quadrature points per direction.
 * @param   in      The 2 NV inputs, `nq0 * nq1` each.
 * @param   varStride Distance between consecutive fields of @p in, at
 *                    least `2 * nq0 * nq1`.
 * @param   D0,D1   Differentiation matrices, column major.
 * @param   out     The NV outputs, `nq0 * nq1` each; overwritten.
 */
template <bool NEGATE, unsigned int NV, unsigned int MAXQ, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor2DLinesKernel(
    const unsigned int nq0, const unsigned int nq1, const simd_type *in,
    const unsigned int varStride, const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, simd_type *const out[NV])
{
    const unsigned int nqTot = nq0 * nq1;
    if (nq0 > MAXQ || nq1 > MAXQ)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            SumDerivTensor2DKernel<false>(nq0, nq1, in + v * varStride,
                                          in + v * varStride + nqTot, D0, D1,
                                          out[v]);
            if constexpr (NEGATE)
            {
                for (unsigned int q = 0; q < nqTot; ++q)
                {
                    out[v][q] = -out[v][q];
                }
            }
        }
        return;
    }
    simd_type line[NV][MAXQ], sum[NV];

    // Direction 0: D0^T along i; writes the output.
    for (unsigned int j = 0; j < nq1; ++j)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                line[v][i] = in[v * varStride + j * nq0 + i];
            }
        }
        for (unsigned int p = 0; p < nq0; ++p)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int i = 0; i < nq0; ++i)
            {
                const simd_type entry = D0[p * nq0 + i];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][i]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                if constexpr (NEGATE)
                {
                    out[v][j * nq0 + p] = -sum[v];
                }
                else
                {
                    out[v][j * nq0 + p] = sum[v];
                }
            }
        }
    }
    // Direction 1: D1^T along j; accumulates.
    for (unsigned int i = 0; i < nq0; ++i)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int k = 0; k < nq1; ++k)
            {
                line[v][k] = in[v * varStride + nqTot + k * nq0 + i];
            }
        }
        for (unsigned int j = 0; j < nq1; ++j)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int k = 0; k < nq1; ++k)
            {
                const simd_type entry = D1[j * nq1 + k];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][k]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                if constexpr (NEGATE)
                {
                    out[v][j * nq0 + i] -= sum[v];
                }
                else
                {
                    out[v][j * nq0 + i] += sum[v];
                }
            }
        }
    }
}

/**
 * @brief The three collapsed-coordinate derivatives of @p NV fields at
 * once, line by line, as PhysDerivTensor2DLinesKernel() forms the two:
 * `out[v * varStride + d * nqTot + q]` holds direction d of field v, as
 * PhysDerivTensor3DKernel() lays its three outputs out, and that kernel
 * forms the derivatives one field at a time beyond @p MAXQ points in a
 * direction.
 *
 * @tparam NV     Fields differentiated together.
 * @tparam MAXQ   Points per direction the register block holds.
 *
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   in          The NV fields, `nq0 * nq1 * nq2` each.
 * @param   D0,D1,D2    Differentiation matrices, column major.
 * @param   out         The 3 NV derivatives, `nq0 * nq1 * nq2` each.
 * @param   varStride   Distance between consecutive fields of @p out, at
 *                      least `3 * nq0 * nq1 * nq2`.
 */
template <unsigned int NV, unsigned int MAXQ, typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor3DLinesKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *const in[NV], const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, simd_type *out,
    const unsigned int varStride)
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    if (nq0 > MAXQ || nq1 > MAXQ || nq2 > MAXQ)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            PhysDerivTensor3DKernel(
                nq0, nq1, nq2, in[v], D0, D1, D2, out + v * varStride,
                out + v * varStride + nqTot, out + v * varStride + 2 * nqTot);
        }
        return;
    }
    simd_type line[NV][MAXQ], sum[NV];

    // Direction 0: lines along i, contiguous.
    for (unsigned int j = 0; j < nq1 * nq2; ++j)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int k = 0; k < nq0; ++k)
            {
                line[v][k] = in[v][j * nq0 + k];
            }
        }
        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int k = 0; k < nq0; ++k)
            {
                const simd_type entry = D0[k * nq0 + i];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][k]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                out[v * varStride + j * nq0 + i] = sum[v];
            }
        }
    }
    // Direction 1: lines along j within each plane of constant k, stride
    // nq0.
    for (unsigned int plane = 0; plane < nq2; ++plane)
    {
        const unsigned int start = plane * nq0 * nq1;
        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                for (unsigned int k = 0; k < nq1; ++k)
                {
                    line[v][k] = in[v][start + k * nq0 + i];
                }
            }
            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v] = simd_type(0.0);
                }
                for (unsigned int k = 0; k < nq1; ++k)
                {
                    const simd_type entry = D1[k * nq1 + j];
                    for (unsigned int v = 0; v < NV; ++v)
                    {
                        sum[v].fma(entry, line[v][k]);
                    }
                }
                for (unsigned int v = 0; v < NV; ++v)
                {
                    out[v * varStride + nqTot + start + j * nq0 + i] = sum[v];
                }
            }
        }
    }
    // Direction 2: lines across the planes, stride nq0 * nq1.
    for (unsigned int i = 0; i < nq0 * nq1; ++i)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int k = 0; k < nq2; ++k)
            {
                line[v][k] = in[v][k * nq0 * nq1 + i];
            }
        }
        for (unsigned int j = 0; j < nq2; ++j)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int k = 0; k < nq2; ++k)
            {
                const simd_type entry = D2[k * nq2 + j];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][k]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                out[v * varStride + 2 * nqTot + j * nq0 * nq1 + i] = sum[v];
            }
        }
    }
}

/**
 * @brief The sum of the three transposed derivative contractions of @p NV
 * fields at once, line by line, into the NV outputs: `out[v] = (-)(D0^T
 * in[v][0] + D1^T in[v][1] + D2^T in[v][2])`, negated with @p NEGATE, as
 * SumDerivTensor3DKernel() forms it one field at a time, with a line's
 * inputs for every field held in registers. `in[v * varStride + d * nqTot
 * + q]`
 * holds direction d of field v. More than @p MAXQ points in a direction
 * exceed the register block: that kernel then forms the sum one field at a
 * time, negated after.
 *
 * @tparam NEGATE The sum is negated.
 * @tparam NV     Fields contracted together.
 * @tparam MAXQ   Points per direction the register block holds.
 *
 * @param   nq0,nq1,nq2 Quadrature points per direction.
 * @param   in          The 3 NV inputs, `nq0 * nq1 * nq2` each.
 * @param   varStride   Distance between consecutive fields of @p in, at
 *                      least `3 * nq0 * nq1 * nq2`.
 * @param   D0,D1,D2    Differentiation matrices, column major.
 * @param   out         The NV outputs, `nq0 * nq1 * nq2` each;
 *                      overwritten.
 */
template <bool NEGATE, unsigned int NV, unsigned int MAXQ, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor3DLinesKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *in, const unsigned int varStride,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, simd_type *const out[NV])
{
    const unsigned int nqTot = nq0 * nq1 * nq2;
    if (nq0 > MAXQ || nq1 > MAXQ || nq2 > MAXQ)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            SumDerivTensor3DKernel<false>(
                nq0, nq1, nq2, in + v * varStride, in + v * varStride + nqTot,
                in + v * varStride + 2 * nqTot, D0, D1, D2, out[v]);
            if constexpr (NEGATE)
            {
                for (unsigned int q = 0; q < nqTot; ++q)
                {
                    out[v][q] = -out[v][q];
                }
            }
        }
        return;
    }
    simd_type line[NV][MAXQ], sum[NV];

    // Direction 0: D0^T along i; writes the output.
    for (unsigned int j = 0; j < nq1 * nq2; ++j)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int i = 0; i < nq0; ++i)
            {
                line[v][i] = in[v * varStride + j * nq0 + i];
            }
        }
        for (unsigned int p = 0; p < nq0; ++p)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int i = 0; i < nq0; ++i)
            {
                const simd_type entry = D0[p * nq0 + i];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][i]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                if constexpr (NEGATE)
                {
                    out[v][j * nq0 + p] = -sum[v];
                }
                else
                {
                    out[v][j * nq0 + p] = sum[v];
                }
            }
        }
    }
    // Direction 1: D1^T along j within each plane of constant k;
    // accumulates.
    for (unsigned int plane = 0; plane < nq2; ++plane)
    {
        const unsigned int start = plane * nq0 * nq1;
        for (unsigned int i = 0; i < nq0; ++i)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                for (unsigned int k = 0; k < nq1; ++k)
                {
                    line[v][k] =
                        in[v * varStride + nqTot + start + k * nq0 + i];
                }
            }
            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v] = simd_type(0.0);
                }
                for (unsigned int k = 0; k < nq1; ++k)
                {
                    const simd_type entry = D1[j * nq1 + k];
                    for (unsigned int v = 0; v < NV; ++v)
                    {
                        sum[v].fma(entry, line[v][k]);
                    }
                }
                for (unsigned int v = 0; v < NV; ++v)
                {
                    if constexpr (NEGATE)
                    {
                        out[v][start + j * nq0 + i] -= sum[v];
                    }
                    else
                    {
                        out[v][start + j * nq0 + i] += sum[v];
                    }
                }
            }
        }
    }
    // Direction 2: D2^T across the planes; accumulates.
    for (unsigned int i = 0; i < nq0 * nq1; ++i)
    {
        for (unsigned int v = 0; v < NV; ++v)
        {
            for (unsigned int k = 0; k < nq2; ++k)
            {
                line[v][k] = in[v * varStride + 2 * nqTot + k * nq0 * nq1 + i];
            }
        }
        for (unsigned int j = 0; j < nq2; ++j)
        {
            for (unsigned int v = 0; v < NV; ++v)
            {
                sum[v] = simd_type(0.0);
            }
            for (unsigned int k = 0; k < nq2; ++k)
            {
                const simd_type entry = D2[j * nq2 + k];
                for (unsigned int v = 0; v < NV; ++v)
                {
                    sum[v].fma(entry, line[v][k]);
                }
            }
            for (unsigned int v = 0; v < NV; ++v)
            {
                if constexpr (NEGATE)
                {
                    out[v][j * nq0 * nq1 + i] -= sum[v];
                }
                else
                {
                    out[v][j * nq0 * nq1 + i] += sum[v];
                }
            }
        }
    }
}

/**
 * @brief PhysDerivTensor1DLinesKernel(), PhysDerivTensor2DLinesKernel() or
 * PhysDerivTensor3DLinesKernel() by dimension, for a composite operator
 * templated on it.
 *
 * @tparam NDIM        Dimension, one to three.
 * @param  sizeParam   Any size parameter with nq0() to nq<NDIM-1>().
 * @param  derivMatrix The NDIM differentiation matrices, column major.
 * @param  varStride   Distance between consecutive fields of @p out.
 */
template <unsigned int NDIM, unsigned int NV, unsigned int MAXQ, typename TSize,
          typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensorLinesKernelND(
    const TSize &sizeParam, const simd_type *const in[NV],
    const typename simd_type::scalarType *const *derivMatrix, simd_type *out,
    const unsigned int varStride)
{
    static_assert(NDIM >= 1 && NDIM <= 3, "One to three dimensions.");
    if constexpr (NDIM == 1)
    {
        PhysDerivTensor1DLinesKernel<NV, MAXQ>(sizeParam.nq0(), in,
                                               derivMatrix[0], out, varStride);
    }
    else if constexpr (NDIM == 2)
    {
        PhysDerivTensor2DLinesKernel<NV, MAXQ>(sizeParam.nq0(), sizeParam.nq1(),
                                               in, derivMatrix[0],
                                               derivMatrix[1], out, varStride);
    }
    else
    {
        PhysDerivTensor3DLinesKernel<NV, MAXQ>(
            sizeParam.nq0(), sizeParam.nq1(), sizeParam.nq2(), in,
            derivMatrix[0], derivMatrix[1], derivMatrix[2], out, varStride);
    }
}

/**
 * @brief SumDerivTensor1DLinesKernel(), SumDerivTensor2DLinesKernel() or
 * SumDerivTensor3DLinesKernel() by dimension, for a composite operator
 * templated on it.
 *
 * @tparam NDIM        Dimension, one to three.
 * @param  sizeParam   Any size parameter with nq0() to nq<NDIM-1>().
 * @param  derivMatrix The NDIM differentiation matrices, column major.
 * @param  varStride   Distance between consecutive fields of @p in.
 */
template <bool NEGATE, unsigned int NDIM, unsigned int NV, unsigned int MAXQ,
          typename TSize, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensorLinesKernelND(
    const TSize &sizeParam, const simd_type *in, const unsigned int varStride,
    const typename simd_type::scalarType *const *derivMatrix,
    simd_type *const out[NV])
{
    static_assert(NDIM >= 1 && NDIM <= 3, "One to three dimensions.");
    if constexpr (NDIM == 1)
    {
        SumDerivTensor1DLinesKernel<NEGATE, NV, MAXQ>(
            sizeParam.nq0(), in, varStride, derivMatrix[0], out);
    }
    else if constexpr (NDIM == 2)
    {
        SumDerivTensor2DLinesKernel<NEGATE, NV, MAXQ>(
            sizeParam.nq0(), sizeParam.nq1(), in, varStride, derivMatrix[0],
            derivMatrix[1], out);
    }
    else
    {
        SumDerivTensor3DLinesKernel<NEGATE, NV, MAXQ>(
            sizeParam.nq0(), sizeParam.nq1(), sizeParam.nq2(), in, varStride,
            derivMatrix[0], derivMatrix[1], derivMatrix[2], out);
    }
}
