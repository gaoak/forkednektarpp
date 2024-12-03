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
// Description: Inner kernel functions leveraging the  AVX impelmentaiton of
// PHysDeriv. They are then used in a scalar manner within  StdRegions
//
///////////////////////////////////////////////////////////////////////////////

template <typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor1DKernel(
    const size_t nq0, const typename simd_type::vectorType *in,
    const simd_type *D0, typename simd_type::scalarType *out_d0)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    for (int i = 0; i < nq0; ++i)
    { // Row index of D0 matrix

        simd_type prod_sum = 0.0;
        for (int k = 0; k < nq0; ++k)
        {                                    // Col index of D0, row index of IN
            simd_type v1 = D0[k * nq0 + i];  // Load 1x
            simd_type v2 = simd_type(in[k]); // Load 1x

            prod_sum.fma(v1, v2);
        }

        prod_sum.store(out_d0 + i * simd_type::width);
    }
}

template <typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor2DKernel(
    const size_t nq0, const size_t nq1,
    const typename simd_type::vectorType *in, const simd_type *D0,
    const simd_type *D1, typename simd_type::scalarType *out_d0,
    typename simd_type::scalarType *out_d1, bool Deriv0 = true,
    bool Deriv1 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    if (Deriv0) // backwards compatibility with StdRegsion PhyTensorDeriv
    {
        for (int i = 0; i < nq0; ++i)
        { // Row index of D0 matrix
            for (int j = 0; j < nq1; ++j)
            { // Col index of IN matrix

                simd_type prod_sum = 0.0;
                for (int k = 0; k < nq0; ++k)
                { // Col index of D0, row index of IN
                    simd_type v1 = D0[k * nq0 + i];            // Load 1x
                    simd_type v2 = simd_type(in[j * nq0 + k]); // Load 1x

                    prod_sum.fma(v1, v2);
                }

                prod_sum.store(out_d0 +
                               (j * nq0 + i) * simd_type::width); // Store 1x
            }
        }
    }

    // D1 * in
    if (Deriv1)
    {
        // in * D1^T
        for (int i = 0; i < nq0; ++i)
        { // row index for grid
            for (int j = 0; j < nq1; ++j)
            { // Column index for D1^T (row idx for D1)

                simd_type prod_sum = 0.0;
                for (int k = 0; k < nq1; ++k)
                {
                    simd_type v1 = simd_type(in[k * nq0 + i]); // Load 1x
                    simd_type v2 = D1[k * nq1 + j];            // Load 1x

                    prod_sum.fma(v1, v2);
                }

                prod_sum.store(out_d1 +
                               (j * nq0 + i) * simd_type::width); // Store 1x
            }
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor2DKernel(
    const size_t nq0, const size_t nq1,
    const typename simd_type::vectorType *in0,
    const typename simd_type::vectorType *in1, const simd_type *w0,
    const simd_type *w1, const simd_type *jac, const simd_type *D0,
    const simd_type *D1, typename simd_type::scalarType *out,
    bool Deriv0 = true, bool Deriv1 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    if (Deriv0) // backwards compatibility with StdRegsion PhyTensorDeriv
    {
        for (int i = 0; i < nq0; ++i)
        { // Row index of D0 matrix
            for (int j = 0; j < nq1; ++j)
            { // Col index of IN matrix

                simd_type prod_sum = 0.0;
                for (int k = 0; k < nq0; ++k)
                { // Col index of D0, row index of IN
                    simd_type v1 = D0[i * nq0 + k];             // Load 1x
                    simd_type v2 = simd_type(in0[j * nq0 + k]); // Load 1x

                    simd_type jac_val;
                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[j * nq0 + k];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }
                    v2 *= w0[k] * jac_val; // Load 1x

                    prod_sum.fma(v1, v2);
                }

                prod_sum *= w1[j]; // Load 1x

                prod_sum.store(out +
                               (j * nq0 + i) * simd_type::width); // Store 1x
            }
        }
    }

    // D1 * in
    if (Deriv1)
    {
        // in * D1^T
        for (int j = 0; j < nq1; ++j)
        { // Column index for D1^T (row idx for D1)
            for (int i = 0; i < nq0; ++i)
            { // row index for grid

                simd_type prod_sum = 0.0;
                for (int k = 0; k < nq1; ++k)
                {
                    simd_type v1 = simd_type(in1[k * nq0 + i]); // Load 1x
                    simd_type v2 = D1[j * nq1 + k];             // Load 1x

                    simd_type jac_val;
                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[k * nq0 + i];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }
                    v2 *= w1[k] * jac_val; // Load 4x

                    prod_sum.fma(v1, v2);
                }

                prod_sum *= w0[i]; // Load 1x

                simd_type temp;
                temp.load(out + (j * nq0 + i) * simd_type::width);
                temp += prod_sum;
                temp.store(out + (j * nq0 + i) * simd_type::width); // Store 1x
            }
        }
    }
}

template <typename simd_type>
NEK_FORCE_INLINE static void PhysDerivTensor3DKernel(
    const size_t nq0, const size_t nq1, const size_t nq2,
    const typename simd_type::vectorType *in, const simd_type *D0,
    const simd_type *D1, const simd_type *D2,
    typename simd_type::scalarType *out_d0,
    typename simd_type::scalarType *out_d1,
    typename simd_type::scalarType *out_d2, bool Deriv0 = true,
    bool Deriv1 = true, bool Deriv2 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // Direction 0
    if (Deriv0)
    {
        for (int i = 0; i < nq0; ++i)
        {
            for (int j = 0; j < nq1 * nq2; ++j)
            {
                simd_type prod_sum = 0.0;
                for (int k = 0; k < nq0; ++k)
                {
                    simd_type v1 = D0[k * nq0 + i];            // Load 1x
                    simd_type v2 = simd_type(in[j * nq0 + k]); // Load 1x

                    prod_sum.fma(v1, v2);
                }

                // out_d0[j * nq0 + i] = prod_sum; // Store 1x
                prod_sum.store(out_d0 + (j * nq0 + i) * simd_type::width);
            }
        }
    }

    // Direction 1
    if (Deriv1)
    {
        for (int block = 0; block < nq2; ++block)
        {
            int start = block * nq0 * nq1;

            for (int i = 0; i < nq0; ++i)
            {
                for (int j = 0; j < nq1; ++j)
                {
                    simd_type prod_sum = 0.0;
                    for (int k = 0; k < nq1; ++k)
                    {
                        simd_type v1 =
                            simd_type(in[start + k * nq0 + i]); // Load 1x
                        simd_type v2 = D1[k * nq1 + j];         // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    // out_d1[start + j * nq0 + i] = prod_sum; // Store 1x
                    prod_sum.store(out_d1 +
                                   (start + j * nq0 + i) * simd_type::width);
                }
            }
        }
    }

    // Direction 2
    if (Deriv2)
    {
        for (int i = 0; i < nq0 * nq1; ++i)
        {
            for (int j = 0; j < nq2; ++j)
            {
                simd_type prod_sum = 0.0;
                for (int k = 0; k < nq2; ++k)
                {
                    simd_type v1 = simd_type(in[k * nq0 * nq1 + i]); // Load 1x
                    simd_type v2 = D2[k * nq2 + j];                  // Load 1x

                    prod_sum.fma(v1, v2);
                }

                // out_d2[j * nq0 * nq1 + i] = prod_sum; // Store 1x
                prod_sum.store(out_d2 + (j * nq0 * nq1 + i) * simd_type::width);
            }
        }
    }
}

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void SumDerivTensor3DKernel(
    const size_t nq0, const size_t nq1, const size_t nq2,
    const typename simd_type::vectorType *in0,
    const typename simd_type::vectorType *in1,
    const typename simd_type::vectorType *in2, const simd_type *w0,
    const simd_type *w1, const simd_type *w2, const simd_type *jac,
    const simd_type *D0, const simd_type *D1, const simd_type *D2,
    typename simd_type::scalarType *out, bool Deriv0 = true, bool Deriv1 = true,
    bool Deriv2 = true)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // Compared with PhysDerivTensor3DKernel, here we must multiply by the
    // transpose of D matrix

    // Direction 0
    if (Deriv0)
    {
        for (int p = 0; p < nq0; ++p)
        {
            int cnt_kji = 0, cnt_kj = 0;
            for (int k = 0; k < nq2; ++k)
            {
                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type prod_sum = 0.0;
                    for (int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        simd_type v1 = D0[p * nq0 + i];         // Load 1x
                        simd_type v2 = simd_type(in0[cnt_kji]); // Load 1x

                        simd_type jac_val;
                        if constexpr (DEFORMED)
                        {
                            jac_val = jac[cnt_kji];
                        }
                        else
                        {
                            jac_val = jac[0];
                        }
                        v2 *= w0[i] * jac_val; // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    prod_sum *= w1[j] * w2[k]; // Load 2x

                    // out[cnt_hj * nq0 + i] = prod_sum; // Store 1x
                    prod_sum.store(out + (cnt_kj * nq0 + p) * simd_type::width);
                }
            }
        }
    }

    // Direction 1
    if (Deriv1)
    {
        for (int block = 0; block < nq2; ++block)
        {
            int start = block * nq0 * nq1;

            for (int i = 0; i < nq0; ++i)
            {
                for (int j = 0; j < nq1; ++j)
                {
                    simd_type prod_sum = 0.0;
                    for (int k = 0; k < nq1; ++k)
                    {
                        simd_type v1 =
                            simd_type(in1[start + k * nq0 + i]); // Load 1x
                        simd_type v2 = D1[j * nq1 + k];          // Load 1x

                        simd_type jac_val;
                        if constexpr (DEFORMED)
                        {
                            jac_val = jac[start + k * nq0 + i];
                        }
                        else
                        {
                            jac_val = jac[0];
                        }
                        v1 *= w1[k] * jac_val; // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    prod_sum *= w0[i] * w2[block]; // Load 1x

                    // out[start + j * nq0 + i] += prod_sum; // Store 1x
                    simd_type temp;
                    temp.load(out + (start + j * nq0 + i) * simd_type::width);
                    temp += prod_sum;
                    temp.store(out + (start + j * nq0 + i) * simd_type::width);
                }
            }
        }
    }

    // Direction 2
    if (Deriv2)
    {
        int cnt_ih = 0;
        for (int h = 0; h < nq1; ++h)
        {
            for (int i = 0; i < nq0; ++i, ++cnt_ih)
            {
                for (int j = 0; j < nq2; ++j)
                {
                    simd_type prod_sum = 0.0;
                    for (int k = 0; k < nq2; ++k)
                    {
                        simd_type v1 =
                            simd_type(in2[k * nq0 * nq1 + cnt_ih]); // Load 1x
                        simd_type v2 = D2[j * nq2 + k];             // Load 1x

                        simd_type jac_val;
                        if constexpr (DEFORMED)
                        {
                            jac_val = jac[k * nq0 * nq1 + cnt_ih];
                        }
                        else
                        {
                            jac_val = jac[0];
                        }
                        v1 *= w2[k] * jac_val; // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    prod_sum *= w0[i] * w1[h]; // Load 2x

                    // out[j * nq0 * nq1 + i] += prod_sum; // Store 1x
                    simd_type temp;
                    temp.load(out +
                              (j * nq0 * nq1 + cnt_ih) * simd_type::width);
                    temp += prod_sum;
                    temp.store(out +
                               (j * nq0 * nq1 + cnt_ih) * simd_type::width);
                }
            }
        }
    }
}
