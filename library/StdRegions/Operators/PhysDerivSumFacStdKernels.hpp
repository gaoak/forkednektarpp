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

NEK_FORCE_INLINE static void PhysDerivTensor1DKernel(
    const size_t nq0, const vec_t::vectorType *in, const vec_t *D0,
    vec_t::scalarType *out_d0)
{
    // All matricies are column major ordered since operators used to
    // be computed via BLAS.

    // D0 * in
    for (int i = 0; i < nq0; ++i)
    { // Row index of D0 matrix

        vec_t prod_sum = 0.0;
        for (int k = 0; k < nq0; ++k)
        {                               // Col index of D0, row index of IN
            vec_t v1 = D0[k * nq0 + i]; // Load 1x
            vec_t v2 = vec_t(in[k]);    // Load 1x

            prod_sum.fma(v1, v2);
        }

        prod_sum.store(out_d0 + i * vec_t::width);
    }
}

NEK_FORCE_INLINE static void PhysDerivTensor2DKernel(
    const size_t nq0, const size_t nq1, const vec_t::vectorType *in,
    const vec_t *D0, const vec_t *D1, vec_t::scalarType *out_d0,
    vec_t::scalarType *out_d1, bool Deriv0 = true, bool Deriv1 = true)
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

                vec_t prod_sum = 0.0;
                for (int k = 0; k < nq0; ++k)
                { // Col index of D0, row index of IN
                    vec_t v1 = D0[k * nq0 + i];        // Load 1x
                    vec_t v2 = vec_t(in[j * nq0 + k]); // Load 1x

                    prod_sum.fma(v1, v2);
                }

                prod_sum.store(out_d0 +
                               (j * nq0 + i) * vec_t::width); // Store 1x
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

                vec_t prod_sum = 0.0;
                for (int k = 0; k < nq1; ++k)
                {
                    vec_t v1 = vec_t(in[k * nq0 + i]); // Load 1x
                    vec_t v2 = D1[k * nq1 + j];        // Load 1x

                    prod_sum.fma(v1, v2);
                }

                prod_sum.store(out_d1 +
                               (j * nq0 + i) * vec_t::width); // Store 1x
            }
        }
    }
}

NEK_FORCE_INLINE static void PhysDerivTensor3DKernel(
    const size_t nq0, const size_t nq1, const size_t nq2,
    const vec_t::vectorType *in, const vec_t *D0, const vec_t *D1,
    const vec_t *D2, vec_t::scalarType *out_d0, vec_t::scalarType *out_d1,
    vec_t::scalarType *out_d2, bool Deriv0 = true, bool Deriv1 = true,
    bool Deriv2 = true)
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
                vec_t prod_sum = 0.0;
                for (int k = 0; k < nq0; ++k)
                {
                    vec_t v1 = D0[k * nq0 + i];        // Load 1x
                    vec_t v2 = vec_t(in[j * nq0 + k]); // Load 1x

                    prod_sum.fma(v1, v2);
                }

                // out_d0[j * nq0 + i] = prod_sum; // Store 1x
                prod_sum.store(out_d0 + (j * nq0 + i) * vec_t::width);
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
                    vec_t prod_sum = 0.0;
                    for (int k = 0; k < nq1; ++k)
                    {
                        vec_t v1 = vec_t(in[start + k * nq0 + i]); // Load 1x
                        vec_t v2 = D1[k * nq1 + j];                // Load 1x

                        prod_sum.fma(v1, v2);
                    }

                    // out_d1[start + j * nq0 + i] = prod_sum; // Store 1x
                    prod_sum.store(out_d1 +
                                   (start + j * nq0 + i) * vec_t::width);
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
                vec_t prod_sum = 0.0;
                for (int k = 0; k < nq2; ++k)
                {
                    vec_t v1 = vec_t(in[k * nq0 * nq1 + i]); // Load 1x
                    vec_t v2 = D2[k * nq2 + j];              // Load 1x

                    prod_sum.fma(v1, v2);
                }

                // out_d2[j * nq0 * nq1 + i] = prod_sum; // Store 1x
                prod_sum.store(out_d2 + (j * nq0 * nq1 + i) * vec_t::width);
            }
        }
    }
}
