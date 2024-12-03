///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseSumFacStdKernels.hpp
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
// Description: Inner kernel functions for AVX impelmentaiton of
// IProductWRTBase. There are then used in a scalar manner within
// StdRegions
//
///////////////////////////////////////////////////////////////////////////////

template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void ScaleAppend(simd_type &store, simd_type &pos,
                                         [[maybe_unused]] double scale)
{
    if constexpr (SCALE && APPEND)
    {
        store.fma(pos, simd_type(scale));
    }
    else if constexpr (APPEND)
    {
        store = store + pos;
    }
    else if constexpr (SCALE)
    {
        store = pos * scale;
    }
    else
    {
        store = pos;
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductSegKernel(
    const size_t nm0, const size_t nq0,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *w0, const simd_type *jac,
    typename simd_type::scalarType *out, double scale = 1.0)
{
    for (int p = 0; p < nm0; ++p)
    {
        simd_type sum = 0.0;

        for (int i = 0; i < nq0; ++i)
        {
            simd_type jac_val;

            if constexpr (DEFORMED)
            {
                jac_val = jac[i]; // J for each quadrature point.
            }
            else
            {
                jac_val = jac[0];
            }

            simd_type prod =
                simd_type(in[i]) * basis0[p * nq0 + i] * jac_val; // Load 2x
            sum.fma(prod, w0[i]);                                 // Load 1x
        }

        // Modes are reversed from what they normally are for tris, tets etc.
        simd_type tmp;
        tmp.load(out);
        ScaleAppend<SCALE, APPEND>(tmp, sum, scale); // Store x1
        tmp.store(out);
        out += simd_type::width;
    }
}

template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductSegKernel(
    const size_t nm0, const size_t nq0,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    typename simd_type::scalarType *out, double scale = 1.0)
{
    for (int p = 0; p < nm0; ++p)
    {
        simd_type sum = 0.0;

        for (int i = 0; i < nq0; ++i)
        {
            sum.fma(simd_type(in[i]), basis0[p * nq0 + i]); // Load 2x
        }

        // Modes are reversed from what they normally are for tris, tets etc.
        simd_type tmp;
        tmp.load(out);
        ScaleAppend<SCALE, APPEND>(tmp, sum, scale); // Store x1
        tmp.store(out);
        out += simd_type::width;
    }
}

#if 0 
template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductQuadKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    const typename simd_type::vectorType *in, const simd_type *basis0, const simd_type *basis1,
    const simd_type *w0, const simd_type *w1, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_j, // nq1
    typename simd_type::scalarType *out, double scale = 1.0)
{
    for (int p = 0; p < nm0; ++p)
    {
        int cnt_ji = 0;
        for (int j = 0; j < nq1; ++j)
        {
            simd_type sum_j = 0.0;

            for (int i = 0; i < nq0; ++i, ++cnt_ji)
            {
                simd_type jac_val;

                if constexpr (DEFORMED)
                {
                    jac_val = jac[j * nq0 + i]; // J for each quadrature point.
                }
                else
                {
                    jac_val = jac[0];
                }

                simd_type prod = simd_type(in[cnt_ji]) * basis0[p * nq0 + i] *
                             jac_val;   // Load 2x
                sum_j.fma(prod, w0[i]); // Load 1x
            }

            sums_j[j] = sum_j; // Store 1
        }

        for (int q = 0; q < nm1; ++q)
        {
            simd_type sum = 0.0;

            for (int j = 0; j < nq1; ++j)
            {
                simd_type prod = sums_j[j] * basis1[q * nq1 + j]; // Load 2x
                sum.fma(prod, w1[j]);                         // Load 1x
            }

            // Modes are reversed from what they normally are for tris, tets
            // etc.
            simd_type tmp;
            tmp.load(out + (q * nm0 + p) * simd_type::width);
            ScaleAppend<SCALE, APPEND>(tmp, sum, scale); // Store x1
            tmp.store(out + (q * nm0 + p) * simd_type::width);
        }
    }
}
#else
template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductQuadKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *w0, const simd_type *w1,
    const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_j, // nq1
    typename simd_type::scalarType *out, double scale = 1.0,
    const bool CollDir0 = false, const bool CollDir1 = false)
{
    for (int p = 0; p < nm0; ++p)
    {
        if (CollDir0)
        {
            for (int j = 0; j < nq1; ++j)
            {
                simd_type jac_val;

                if constexpr (DEFORMED)
                {
                    jac_val = jac[j * nq0 + p]; // J for each quadrature point.
                }
                else
                {
                    jac_val = jac[0];
                }

                sums_j[j] = simd_type(in[j * nq0 + p]) * jac_val * w0[p];
            }
        }
        else
        {
            int cnt_ji = 0;
            for (int j = 0; j < nq1; ++j)
            {
                simd_type sum_j = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    simd_type jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val =
                            jac[j * nq0 + i]; // J for each quadrature point.
                    }
                    else
                    {
                        jac_val = jac[0];
                    }

                    simd_type prod = simd_type(in[cnt_ji]) *
                                     basis0[p * nq0 + i] * jac_val; // Load 2x
                    sum_j.fma(prod, w0[i]);                         // Load 1x
                }

                sums_j[j] = sum_j; // Store 1
            }
        }

        if (CollDir1)
        {
            for (int q = 0; q < nm1; ++q)
            {
                simd_type sum = sums_j[q] * w1[q];

                // Modes are reversed from what they normally are for tris, tets
                // etc.
                simd_type tmp;
                tmp.load(out + (q * nm0 + p) * simd_type::width);
                ScaleAppend<SCALE, APPEND>(tmp, sum, scale); // Store x1
                tmp.store(out + (q * nm0 + p) * simd_type::width);
            }
        }
        else
        {
            for (int q = 0; q < nm1; ++q)
            {
                simd_type sum = 0.0;

                for (int j = 0; j < nq1; ++j)
                {
                    simd_type prod = sums_j[j] * basis1[q * nq1 + j]; // Load 2x
                    sum.fma(prod, w1[j]);                             // Load 1x
                }

                // Modes are reversed from what they normally are for tris, tets
                // etc.
                simd_type tmp;
                tmp.load(out + (q * nm0 + p) * simd_type::width);
                ScaleAppend<SCALE, APPEND>(tmp, sum, scale); // Store x1
                tmp.store(out + (q * nm0 + p) * simd_type::width);
            }
        }
    }
}

// inner product without quadrature metric wJ
template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductQuadKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_j, // nq1
    typename simd_type::scalarType *out, double scale = 1.0,
    const bool CollDir0 = false, const bool CollDir1 = false)
{
    for (int p = 0; p < nm0; ++p)
    {
        if (CollDir0)
        {
            for (int j = 0; j < nq1; ++j)
            {
                sums_j[j] = simd_type(in[j * nq0 + p]);
            }
        }
        else
        {
            int cnt_ji = 0;
            for (int j = 0; j < nq1; ++j)
            {
                simd_type sum_j = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_ji)
                {
                    // Load 2x
                    sum_j.fma(simd_type(in[cnt_ji]), basis0[p * nq0 + i]);
                }

                sums_j[j] = sum_j; // Store 1
            }
        }

        if (CollDir1)
        {
            for (int q = 0; q < nm1; ++q)
            {
                // Modes are reversed from what they normally are for tris, tets
                // etc.
                simd_type tmp;
                tmp.load(out + (q * nm0 + p) * simd_type::width);
                ScaleAppend<SCALE, APPEND>(tmp, sums_j[q], scale); // Store x1
                tmp.store(out + (q * nm0 + p) * simd_type::width);
            }
        }
        else
        {
            for (int q = 0; q < nm1; ++q)
            {
                simd_type sum = 0.0;

                for (int j = 0; j < nq1; ++j)
                {
                    sum.fma(sums_j[j], basis1[q * nq1 + j]); // Load 2x
                }

                // Modes are reversed from what they normally are for tris, tets
                simd_type tmp;
                tmp.load(out + (q * nm0 + p) * simd_type::width);
                ScaleAppend<SCALE, APPEND>(tmp, sum, scale); // Store x1
                tmp.store(out + (q * nm0 + p) * simd_type::width);
            }
        }
    }
}
#endif

template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductTriKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    const bool modBasis, const typename simd_type::vectorType *in,
    const simd_type *basis0, const simd_type *basis1, const simd_type *w0,
    const simd_type *w1, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &eta0_sums, // nq1
    typename simd_type::scalarType *out, double scale = 1.0)
{
    int mode = 0;

    for (int p = 0; p < nm0; ++p)
    {
        int eta_idx = 0;

        // Our inner loop is phi_p not phi_pq since we want to put as
        // much work as we can in the p-only loop instead of the full
        // pq loop.
        for (int eta1 = 0; eta1 < nq1; ++eta1)
        {
            simd_type eta0_sum = 0.0;

            for (int eta0 = 0; eta0 < nq0; ++eta0, ++eta_idx)
            {
                // eta0_sum += phi_p(eta0) * fn(eta0, eta1) * J * w0(eta0)
                simd_type jac_val;
                if constexpr (DEFORMED)
                {
                    jac_val = jac[eta1 * nq0 + eta0];
                }
                else
                {
                    jac_val = jac[0];
                }

                simd_type prod = simd_type(in[eta_idx]) *
                                 basis0[p * nq0 + eta0] * jac_val; // Load 2x
                eta0_sum.fma(prod, w0[eta0]);                      // Load 1x
            }

            eta0_sums[eta1] = eta0_sum;
        }

        for (int q = 0; q < nm1 - p; ++q, ++mode)
        {
            simd_type sum_eta1 = 0.0;
            for (int eta1 = 0; eta1 < nq1; ++eta1)
            {
                simd_type prod =
                    eta0_sums[eta1] * basis1[mode * nq1 + eta1]; // Load 2x
                sum_eta1.fma(prod, w1[eta1]);                    // Load 1x
            }
            simd_type tmp;
            tmp.load(out + mode * simd_type::width);
            ScaleAppend<SCALE, APPEND>(tmp, sum_eta1, scale); // Store x1
            tmp.store(out + mode * simd_type::width);
        }
    }

    // Correction for singular vertex in collpased coordinates in
    // modified basis.  Basically we add phi_1 * phi_01 * (weighting,
    // etc) to mode 00 With contributions from every quadrature point
    if (modBasis)
    {
        int eta_idx        = 0;
        simd_type iprod_01 = 0.0; // T(outptr + VW); //Load 1x

        for (int eta1 = 0; eta1 < nq1; ++eta1)
        {
            simd_type preweight_eta1;

            if constexpr (DEFORMED)
            {
                preweight_eta1 = w1[eta1] * basis1[nq1 + eta1];
            }
            else
            {
                preweight_eta1 = w1[eta1] * jac[0] * basis1[nq1 + eta1];
            }

            for (int eta0 = 0; eta0 < nq0; ++eta0, ++eta_idx)
            {
                simd_type prod =
                    simd_type(in[eta_idx]) * preweight_eta1 * w0[eta0];

                if constexpr (DEFORMED)
                {
                    prod = prod * jac[eta1 * nq0 + eta0];
                }

                simd_type basis_val1 = basis0[nq0 + eta0];
                iprod_01.fma(prod, basis_val1);
            }
        }
        simd_type tmp;
        tmp.load(out + 1 * simd_type::width);
        ScaleAppend<SCALE, true>(tmp, iprod_01, scale);
        tmp.store(out + 1 * simd_type::width);
    }
}

// inner product without quadrature metric wJ
template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductTriKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    const bool modBasis, const typename simd_type::vectorType *in,
    const simd_type *basis0, const simd_type *basis1,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &eta0_sums, // nq1
    typename simd_type::scalarType *out, double scale = 1.0)
{
    int mode = 0;

    for (int p = 0; p < nm0; ++p)
    {
        int eta_idx = 0;

        // Our inner loop is phi_p not phi_pq since we want to put as
        // much work as we can in the p-only loop instead of the full
        // pq loop.
        for (int eta1 = 0; eta1 < nq1; ++eta1)
        {
            simd_type eta0_sum = 0.0;

            for (int eta0 = 0; eta0 < nq0; ++eta0, ++eta_idx)
            {
                // Load 2x
                eta0_sum.fma(simd_type(in[eta_idx]), basis0[p * nq0 + eta0]);
            }

            eta0_sums[eta1] = eta0_sum;
        }

        for (int q = 0; q < nm1 - p; ++q, ++mode)
        {
            simd_type sum_eta1 = 0.0;
            for (int eta1 = 0; eta1 < nq1; ++eta1)
            {
                sum_eta1.fma(eta0_sums[eta1],
                             basis1[mode * nq1 + eta1]); // Load 2x
            }
            simd_type tmp;
            tmp.load(out + mode * simd_type::width);
            ScaleAppend<SCALE, APPEND>(tmp, sum_eta1, scale); // Store x1
            tmp.store(out + mode * simd_type::width);
        }
    }

    // Correction for singular vertex in collpased coordinates in
    // modified basis.  Basically we add phi_1 * phi_01 * (weighting,
    // etc) to mode 00 With contributions from every quadrature point
    if (modBasis)
    {
        int eta_idx        = 0;
        simd_type iprod_01 = 0.0; // T(outptr + VW); //Load 1x

        for (int eta1 = 0; eta1 < nq1; ++eta1)
        {
            for (int eta0 = 0; eta0 < nq0; ++eta0, ++eta_idx)
            {
                simd_type prod = simd_type(in[eta_idx]) * basis1[nq1 + eta1];

                iprod_01.fma(prod, basis0[nq0 + eta0]);
            }
        }
        simd_type tmp;
        tmp.load(out + 1 * simd_type::width);
        ScaleAppend<SCALE, true>(tmp, iprod_01, scale);
        tmp.store(out + 1 * simd_type::width);
    }
}

#if 0 
template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductHexKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const typename simd_type::vectorType *in,
    const simd_type *basis0, const simd_type *basis1, const simd_type *basis2,
    const simd_type *w0, const simd_type *w1, const simd_type *w2, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_kj, // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k,  // nq2
    typename simd_type::scalarType *out, double scale = 1.0)
{
    for (int p = 0; p < nm0; ++p)
    {
        int cnt_kji = 0, cnt_kj = 0;

        for (int k = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                simd_type sum_kj = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {

                    simd_type jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[nq0 * nq1 * k + nq0 * j + i];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }

                    simd_type prod = simd_type(in[cnt_kji]) * basis0[i + nq0 * p] *
                                 jac_val;    // load 2x
                    sum_kj.fma(prod, w0[i]); // Load 1x
                }

                sums_kj[cnt_kj] = sum_kj;
            }
        }

        for (int q = 0; q < nm1; ++q)
        {
            cnt_kj = 0;
            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type prod =
                        sums_kj[cnt_kj] * basis1[q * nq1 + j]; // Load 2x
                    sum_k.fma(prod, w1[j]);                    // Load 1x
                }

                sums_k[k] = sum_k;
            }

            for (int r = 0; r < nm2; ++r)
            {
                simd_type sum = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    simd_type prod = sums_k[k] * basis2[r * nq2 + k]; // Load 2x
                    sum.fma(prod, w2[k]);                         // Load 1x
                }
                simd_type temp;
                temp.load(out + (r * nm0 * nm1 + q * nm0 + p) * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum, scale); // Store x1
                temp.store(out + (r * nm0 * nm1 + q * nm0 + p) * simd_type::width);
            }
        }
    }
}
#else
template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductHexKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2, const simd_type *w0,
    const simd_type *w1, const simd_type *w2, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>>
        &sums_kj,                                                   // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k, // nq2
    typename simd_type::scalarType *out, double scale = 1.0,
    const bool CollDir0 = false, const bool CollDir1 = false,
    const bool CollDir2 = false)
{
    for (int p = 0; p < nm0; ++p)
    {
        if (CollDir0)
        {
            int cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type jac_val;
                    int cnt_kjp = nq0 * nq1 * k + nq0 * j + p;

                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[cnt_kjp];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }

                    sums_kj[cnt_kj] = simd_type(in[cnt_kjp]) * jac_val * w0[p];
                }
            }
        }
        else
        {
            int cnt_kji = 0, cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type sum_kj = 0.0;

                    for (int i = 0; i < nq0; ++i, ++cnt_kji)
                    {

                        simd_type jac_val;

                        if constexpr (DEFORMED)
                        {
                            jac_val = jac[nq0 * nq1 * k + nq0 * j + i];
                        }
                        else
                        {
                            jac_val = jac[0];
                        }

                        simd_type prod = simd_type(in[cnt_kji]) *
                                         basis0[i + nq0 * p] *
                                         jac_val; // load 2x
                        sum_kj.fma(prod, w0[i]);  // Load 1x
                    }

                    sums_kj[cnt_kj] = sum_kj;
                }
            }
        }

        for (int q = 0; q < nm1; ++q)
        {

            if (CollDir1)
            {
                for (int k = 0; k < nq2; ++k)
                {
                    sums_k[k] = sums_kj[k * nq1 + q] * w1[q];
                }
            }
            else
            {
                int cnt_kj = 0;
                for (int k = 0; k < nq2; ++k)
                {
                    simd_type sum_k = 0.0;

                    for (int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        simd_type prod = sums_kj[cnt_kj] * basis1[q * nq1 + j];
                        sum_k.fma(prod, w1[j]);
                    }

                    sums_k[k] = sum_k;
                }
            }

            if (CollDir2)
            {
                for (int r = 0; r < nm2; ++r)
                {
                    simd_type sum = sums_k[r] * w2[r];

                    simd_type temp;
                    temp.load(out +
                              (r * nm0 * nm1 + q * nm0 + p) * simd_type::width);
                    ScaleAppend<SCALE, APPEND>(temp, sum, scale);
                    temp.store(out + (r * nm0 * nm1 + q * nm0 + p) *
                                         simd_type::width);
                }
            }
            else
            {
                for (int r = 0; r < nm2; ++r)
                {
                    simd_type sum = 0.0;

                    for (int k = 0; k < nq2; ++k)
                    {
                        simd_type prod = sums_k[k] * basis2[r * nq2 + k];
                        sum.fma(prod, w2[k]);
                    }
                    simd_type temp;
                    temp.load(out +
                              (r * nm0 * nm1 + q * nm0 + p) * simd_type::width);
                    ScaleAppend<SCALE, APPEND>(temp, sum, scale);
                    temp.store(out + (r * nm0 * nm1 + q * nm0 + p) *
                                         simd_type::width);
                }
            }
        }
    }
}

// inner product without quadrature metric wJ
template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductHexKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2,
    std::vector<simd_type, tinysimd::allocator<simd_type>>
        &sums_kj,                                                   // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k, // nq2
    typename simd_type::scalarType *out, double scale = 1.0,
    const bool CollDir0 = false, const bool CollDir1 = false,
    const bool CollDir2 = false)
{
    for (int p = 0; p < nm0; ++p)
    {
        if (CollDir0)
        {
            int cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    int cnt_kjp     = nq0 * nq1 * k + nq0 * j + p;
                    sums_kj[cnt_kj] = simd_type(in[cnt_kjp]);
                }
            }
        }
        else
        {
            int cnt_kji = 0, cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type sum_kj = 0.0;

                    for (int i = 0; i < nq0; ++i, ++cnt_kji)
                    {
                        // Load 2x
                        sum_kj.fma(simd_type(in[cnt_kji]), basis0[i + nq0 * p]);
                    }

                    sums_kj[cnt_kj] = sum_kj;
                }
            }
        }

        for (int q = 0; q < nm1; ++q)
        {

            if (CollDir1)
            {
                for (int k = 0; k < nq2; ++k)
                {
                    sums_k[k] = sums_kj[k * nq1 + q];
                }
            }
            else
            {
                int cnt_kj = 0;
                for (int k = 0; k < nq2; ++k)
                {
                    simd_type sum_k = 0.0;

                    for (int j = 0; j < nq1; ++j, ++cnt_kj)
                    {
                        sum_k.fma(sums_kj[cnt_kj], basis1[q * nq1 + j]);
                    }

                    sums_k[k] = sum_k;
                }
            }

            if (CollDir2)
            {
                for (int r = 0; r < nm2; ++r)
                {
                    simd_type temp;
                    temp.load(out +
                              (r * nm0 * nm1 + q * nm0 + p) * simd_type::width);
                    ScaleAppend<SCALE, APPEND>(temp, sums_k[r], scale);
                    temp.store(out + (r * nm0 * nm1 + q * nm0 + p) *
                                         simd_type::width);
                }
            }
            else
            {
                for (int r = 0; r < nm2; ++r)
                {
                    simd_type sum = 0.0;

                    for (int k = 0; k < nq2; ++k)
                    {
                        sum.fma(sums_k[k], basis2[r * nq2 + k]);
                    }
                    simd_type temp;
                    temp.load(out +
                              (r * nm0 * nm1 + q * nm0 + p) * simd_type::width);
                    ScaleAppend<SCALE, APPEND>(temp, sum, scale);
                    temp.store(out + (r * nm0 * nm1 + q * nm0 + p) *
                                         simd_type::width);
                }
            }
        }
    }
}
#endif

template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductTetKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const bool isModified,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2, const simd_type *w0,
    const simd_type *w1, const simd_type *w2, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>>
        &sums_kj,                                                   // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k, // nq2
    typename simd_type::scalarType *out, double scale = 1.0)
{
    for (int p = 0, mode = 0, mode2 = 0, cnt_pqr = 0; p < nm0; ++p)
    {
        int cnt_kji = 0;

        for (int k = 0, cnt_kj = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                // Unroll first entry of for loop below and multiply by
                // quadrature weights in dir0 & jacobian.
                simd_type jac_val;

                if constexpr (DEFORMED)
                {
                    jac_val = jac[nq0 * nq1 * k + nq0 * j];
                }
                else
                {
                    jac_val = jac[0];
                }

                simd_type sum_kj = simd_type(in[cnt_kji]) * basis0[nq0 * p] *
                                   jac_val * w0[0]; // Load 3x
                ++cnt_kji;

                for (int i = 1; i < nq0; ++i, ++cnt_kji)
                {
                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[nq0 * nq1 * k + nq0 * j + i];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }

                    simd_type inxmm = simd_type(in[cnt_kji]) *
                                      basis0[i + nq0 * p] * jac_val; // Load 2x
                    sum_kj.fma(inxmm, w0[i]);                        // Load 1x
                }

                sums_kj[cnt_kj] = sum_kj; // Store 1x
            }
        }

        for (int q = 0; q < nm1 - p; ++q, ++mode)
        {
            int cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k =
                    basis1[mode * nq1] * sums_kj[cnt_kj] * w1[0]; // Load 3x
                ++cnt_kj;

                for (int j = 1; j < nq1; ++j, ++cnt_kj)
                {
                    simd_type tmp2 =
                        basis1[mode * nq1 + j] * sums_kj[cnt_kj]; // Load 2x
                    sum_k.fma(tmp2, w1[j]);                       // Load 1x
                }

                sums_k[k] = sum_k; // Store 1x
            }

            for (int r = 0; r < nm2 - p - q; ++r, ++mode2, ++cnt_pqr)
            {
                simd_type tmp =
                    sums_k[0] * basis2[mode2 * nq2] * w2[0]; // Load 3x

                for (int k = 1; k < nq2; ++k)
                {
                    simd_type tmp2 =
                        sums_k[k] * basis2[mode2 * nq2 + k]; // Load 2x
                    tmp.fma(tmp2, w2[k]);                    // Load 1x
                }
                simd_type temp;
                temp.load(out + cnt_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, tmp, scale); // Store 1x
                temp.store(out + cnt_pqr * simd_type::width);
            }
        }

        // increment mode in case order1!=order2
        for (int q = nm1 - p; q < nm2 - p; ++q)
        {
            mode2 += nm2 - p - q;
        }
    }

    if (isModified)
    {
        for (int k = 0, cnt = 0; k < nq2; ++k)
        {
            simd_type tmpQ2 = w2[k]; // Load 1x
            if constexpr (!DEFORMED)
            {
                tmpQ2 = tmpQ2 * jac[0];
            }

            for (int j = 0; j < nq1; ++j)
            {
                simd_type tmpQ1 = tmpQ2 * w1[j]; // Load 1x

                for (int i = 0; i < nq0; ++i, ++cnt)
                {
                    // Store jac * quadrature weight
                    simd_type tmpQ  = tmpQ1 * w0[i];      // Load 1x
                    simd_type tmpIn = simd_type(in[cnt]); // Load 1x

                    if constexpr (DEFORMED)
                    {
                        tmpQ = tmpQ * jac[k * nq0 * nq1 + j * nq0 + i];
                    }

                    // top vertex
                    //
                    simd_type tmp = basis0[i] * basis1[nq1 + j]; // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[j]);         // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[nq1 + j]);   // Load 2x
                    tmp = tmp * basis2[nq2 + k];                 // Load 1x
                    tmp = tmp * tmpIn;

                    // add to existing entry
                    simd_type tmpOut = tmp * tmpQ;
                    simd_type temp;
                    temp.load(out + 1 * simd_type::width);
                    ScaleAppend<SCALE, true>(temp, tmpOut, scale); // Store 1x
                    temp.store(out + 1 * simd_type::width);

                    // bottom vertex
                    //
                    tmp = basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] *
                          tmpIn; // Load 3x
                    tmpOut = tmp * tmpQ;
                    temp.load(out + nm2 * simd_type::width);
                    ScaleAppend<SCALE, true>(temp, tmpOut, scale); // Store 1x
                    temp.store(out + nm2 * simd_type::width);

                    // singular edge
                    for (int r = 1; r < nm2 - 1; ++r)
                    {
                        tmp = basis2[(r + 1) * nq2 + k] * basis1[nq1 + j] *
                              basis0[nq0 + i] * tmpIn; // Load 3x
                        tmpOut = tmp * tmpQ;
                        temp.load(out + (nm2 + r) * simd_type::width);
                        ScaleAppend<SCALE, true>(temp, tmpOut,
                                                 scale); // Store 1x
                        temp.store(out + (nm2 + r) * simd_type::width);
                    }
                }
            }
        }
    }
}

// inner product without quadrature metric wJ
template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductTetKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const bool isModified,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2,
    std::vector<simd_type, tinysimd::allocator<simd_type>>
        &sums_kj,                                                   // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k, // nq2
    typename simd_type::scalarType *out, double scale = 1.0)
{
    for (int p = 0, mode = 0, mode2 = 0, cnt_pqr = 0; p < nm0; ++p)
    {
        int cnt_kji = 0;

        for (int k = 0, cnt_kj = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                simd_type sum_kj(0.0);

                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    // Load 1x
                    sum_kj.fma(simd_type(in[cnt_kji]), basis0[i + nq0 * p]);
                }

                sums_kj[cnt_kj] = sum_kj; // Store 1x
            }
        }

        for (int q = 0; q < nm1 - p; ++q, ++mode)
        {
            int cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k(0.0);

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k.fma(basis1[mode * nq1 + j],
                              sums_kj[cnt_kj]); // Load 1x
                }

                sums_k[k] = sum_k; // Store 1x
            }

            for (int r = 0; r < nm2 - p - q; ++r, ++mode2, ++cnt_pqr)
            {
                simd_type sum(0.0);

                for (int k = 0; k < nq2; ++k)
                {
                    sum.fma(sums_k[k], basis2[mode2 * nq2 + k]); // Load 2x
                }
                simd_type temp;
                temp.load(out + cnt_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum, scale); // Store 1x
                temp.store(out + cnt_pqr * simd_type::width);
            }
        }

        // increment mode in case order1!=order2
        for (int q = nm1 - p; q < nm2 - p; ++q)
        {
            mode2 += nm2 - p - q;
        }
    }

    if (isModified)
    {
        for (int k = 0, cnt = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j)
            {
                for (int i = 0; i < nq0; ++i, ++cnt)
                {
                    simd_type tmpIn = simd_type(in[cnt]); // Load 1x
                    // top vertex
                    //
                    simd_type tmp = basis0[i] * basis1[nq1 + j]; // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[j]);         // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[nq1 + j]);   // Load 2x
                    tmp = tmp * basis2[nq2 + k];                 // Load 1x
                    tmp = tmp * tmpIn;

                    // add to existing entry
                    simd_type temp;
                    temp.load(out + 1 * simd_type::width);
                    ScaleAppend<SCALE, true>(temp, tmp, scale); // Store 1x
                    temp.store(out + 1 * simd_type::width);

                    // bottom vertex
                    //
                    tmp = basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] *
                          tmpIn; // Load 3x
                    temp.load(out + nm2 * simd_type::width);
                    ScaleAppend<SCALE, true>(temp, tmp, scale); // Store 1x
                    temp.store(out + nm2 * simd_type::width);

                    // singular edge
                    for (int r = 1; r < nm2 - 1; ++r)
                    {
                        tmp = basis2[(r + 1) * nq2 + k] * basis1[nq1 + j] *
                              basis0[nq0 + i] * tmpIn; // Load 3x
                        temp.load(out + (nm2 + r) * simd_type::width);
                        ScaleAppend<SCALE, true>(temp, tmp,
                                                 scale); // Store 1x
                        temp.store(out + (nm2 + r) * simd_type::width);
                    }
                }
            }
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductPrismKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const bool isModified,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2, const simd_type *w0,
    const simd_type *w1, const simd_type *w2, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>>
        &sums_kj,                                                   // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k, // nq2
    std::vector<simd_type, tinysimd::allocator<simd_type>> &corr_q, // nm1
    typename simd_type::scalarType *out, double scale = 1.0)
{
    int mode_pr = 0, mode_pqr = 0;

    for (int p = 0; p < nm0; ++p)
    {
        int cnt_kji = 0, cnt_kj = 0;

        for (int k = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                simd_type sum_kj = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    simd_type jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[nq0 * nq1 * k + nq0 * j + i];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }

                    simd_type prod =
                        basis0[nq0 * p + i] * jac_val * w0[i]; // load 2x
                    simd_type fn = simd_type(in[cnt_kji]);     // load 1x
                    sum_kj.fma(prod, fn);
                }

                sums_kj[cnt_kj] = sum_kj; // store 1x
            }
        }

        for (int q = 0; q < nm1; ++q)
        {
            cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k.fma(basis1[q * nq1 + j] * w1[j],
                              sums_kj[cnt_kj]); // Load 3x
                }

                sums_k[k] = sum_k; // Store 1x
            }

            // Start with nesting. Should be able to move out of q
            // loop and sotre identical copies...
            for (int r = 0; r < nm2 - p; ++r, ++mode_pqr)
            {
                simd_type sum_k = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    sum_k.fma(basis2[(mode_pr + r) * nq2 + k] * w2[k],
                              sums_k[k]); // Load 3x
                }
                simd_type temp;
                temp.load(out + mode_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum_k, scale); // Store 1x
                temp.store(out + mode_pqr * simd_type::width);
            }
        }

        mode_pr += nm2 - p;
    }

    if (isModified)
    {
        // Corrections for singular edge
        for (int q = 0; q < nm1; ++q)
        {
            corr_q[q] = 0.0; // T(outptr + (nm2*q + 1)*VW);
        }

        int cnt_kji = 0;
        for (int k = 0; k < nq2; ++k)
        {
            simd_type k_weight = w2[k];
            if constexpr (!DEFORMED)
            {
                k_weight = k_weight * jac[0];
            }

            for (int j = 0; j < nq1; ++j)
            {
                simd_type kj_weight = k_weight * w1[j];
                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {

                    simd_type kji_weight = kj_weight * w0[i];
                    simd_type prod       = kji_weight * simd_type(in[cnt_kji]);

                    if constexpr (DEFORMED)
                    {
                        prod *= jac[k * nq1 * nq0 + j * nq0 + i];
                    }

                    simd_type basis_2 = basis2[nq2 + k];
                    simd_type basis_0 = basis0[nq0 + i];
                    // Add phi_1q1 to phi_0q1
                    for (int q = 0; q < nm1; ++q)
                    {
                        simd_type basis_1 = basis1[q * nq1 + j];

                        corr_q[q].fma(basis_2 * basis_1, basis_0 * prod);
                    }
                }
            }
        }

        simd_type temp;
        for (int q = 0; q < nm1; ++q)
        {
            temp.load(out + (nm2 * q + 1) * simd_type::width);
            ScaleAppend<SCALE, true>(temp, corr_q[q], scale);
            temp.store(out + (nm2 * q + 1) * simd_type::width);
        }
    }
}

// inner product without quadrature metric wJ
template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductPrismKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const bool isModified,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2,
    std::vector<simd_type, tinysimd::allocator<simd_type>>
        &sums_kj,                                                   // nq2 * nq1
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k, // nq2
    std::vector<simd_type, tinysimd::allocator<simd_type>> &corr_q, // nm1
    typename simd_type::scalarType *out, double scale = 1.0)
{
    int mode_pr = 0, mode_pqr = 0;

    for (int p = 0; p < nm0; ++p)
    {
        int cnt_kji = 0, cnt_kj = 0;

        for (int k = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                simd_type sum_kj = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    // load 2x
                    sum_kj.fma(simd_type(in[cnt_kji]), basis0[nq0 * p + i]);
                }

                sums_kj[cnt_kj] = sum_kj; // store 1x
            }
        }

        for (int q = 0; q < nm1; ++q)
        {
            cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    // Load 2x
                    sum_k.fma(basis1[q * nq1 + j], sums_kj[cnt_kj]);
                }

                sums_k[k] = sum_k; // Store 1x
            }

            // Start with nesting. Should be able to move out of q
            // loop and sotre identical copies...
            for (int r = 0; r < nm2 - p; ++r, ++mode_pqr)
            {
                simd_type sum_k = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    // Load 2x
                    sum_k.fma(basis2[(mode_pr + r) * nq2 + k], sums_k[k]);
                }
                simd_type temp;
                temp.load(out + mode_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum_k, scale); // Store 1x
                temp.store(out + mode_pqr * simd_type::width);
            }
        }

        mode_pr += nm2 - p;
    }

    if (isModified)
    {
        // Corrections for singular edge
        for (int q = 0; q < nm1; ++q)
        {
            corr_q[q] = 0.0; // T(outptr + (nm2*q + 1)*VW);
        }

        int cnt_kji = 0;
        for (int k = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j)
            {
                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    simd_type basis_2 = basis2[nq2 + k];
                    simd_type prod = basis0[nq0 + i] * simd_type(in[cnt_kji]);
                    // Add phi_1q1 to phi_0q1
                    for (int q = 0; q < nm1; ++q)
                    {
                        corr_q[q].fma(basis_2 * basis1[q * nq1 + j], prod);
                    }
                }
            }
        }

        simd_type temp;
        for (int q = 0; q < nm1; ++q)
        {
            temp.load(out + (nm2 * q + 1) * simd_type::width);
            ScaleAppend<SCALE, true>(temp, corr_q[q], scale);
            temp.store(out + (nm2 * q + 1) * simd_type::width);
        }
    }
}

template <bool SCALE, bool APPEND, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void IProductPyrKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const bool isModified,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2, const simd_type *w0,
    const simd_type *w1, const simd_type *w2, const simd_type *jac,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_kj,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k,
    typename simd_type::scalarType *out, double scale = 1.0)
{
    int mode_pqr = 0, cnt_pqr = 0;

    for (int p = 0; p < nm0; ++p)
    {
        int cnt_kji = 0, cnt_kj = 0;

        for (int k = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                simd_type sum_kj = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    simd_type jac_val;

                    if constexpr (DEFORMED)
                    {
                        jac_val = jac[nq0 * nq1 * k + nq0 * j + i];
                    }
                    else
                    {
                        jac_val = jac[0];
                    }

                    simd_type prod =
                        basis0[nq0 * p + i] * jac_val * w0[i]; // load 2x
                    simd_type fn = simd_type(in[cnt_kji]);     // load 1x
                    sum_kj.fma(prod, fn);
                }

                sums_kj[cnt_kj] = sum_kj; // store 1x
            }
        }

        for (int q = 0; q < p; ++q)
        {
            cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k.fma(basis1[q * nq1 + j] * w1[j],
                              sums_kj[cnt_kj]); // Load 3x
                }

                sums_k[k] = sum_k; // Store 1x
            }

            for (int r = 0; r < nm2 - p; ++r, ++mode_pqr, ++cnt_pqr)
            {
                simd_type sum_k = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    sum_k.fma(basis2[mode_pqr * nq2 + k] * w2[k],
                              sums_k[k]); // Load 3x
                }
                simd_type temp;
                temp.load(out + cnt_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum_k, scale); // Store 1x
                temp.store(out + cnt_pqr * simd_type::width);
            }
        }

        for (int q = p; q < nm1; ++q)
        {
            cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    sum_k.fma(basis1[q * nq1 + j] * w1[j],
                              sums_kj[cnt_kj]); // Load 3x
                }
                sums_k[k] = sum_k; // Store 1x
            }

            for (int r = 0; r < nm2 - q; ++r, ++mode_pqr, ++cnt_pqr)
            {
                simd_type sum_k = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    sum_k.fma(basis2[mode_pqr * nq2 + k] * w2[k],
                              sums_k[k]); // Load 3x
                }
                simd_type temp;
                temp.load(out + cnt_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum_k, scale); // Store 1x
                temp.store(out + cnt_pqr * simd_type::width);
            }
        }

        // increment mode in case order1!=order2
        for (int q = nm1; q < nm2; ++q)
        {
            mode_pqr += nm2 - q;
        }
    }

    if (isModified)
    {
        for (int k = 0, cnt = 0; k < nq2; ++k)
        {
            simd_type tmpQ2 = w2[k]; // Load 1x

            if constexpr (!DEFORMED)
            {
                tmpQ2 = tmpQ2 * jac[0];
            }

            for (int j = 0; j < nq1; ++j)
            {
                simd_type tmpQ1 = tmpQ2 * w1[j]; // Load 1x

                for (int i = 0; i < nq0; ++i, ++cnt)
                {
                    // Store jac * quadrature weight
                    simd_type tmpQ  = tmpQ1 * w0[i];      // Load 1x
                    simd_type tmpIn = simd_type(in[cnt]); // Load 1x

                    if constexpr (DEFORMED)
                    {
                        tmpQ = tmpQ * jac[k * nq0 * nq1 + j * nq0 + i];
                    }

                    // top vertex
                    //
                    simd_type tmp = basis0[i] * basis1[nq1 + j]; // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[j]);         // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[nq1 + j]);   // Load 2x
                    tmp = tmp * basis2[nq2 + k];                 // Load 1x
                    tmp = tmp * tmpIn;

                    // add to existing entry
                    simd_type tmpOut = tmp * tmpQ;
                    simd_type temp;
                    temp.load(out + 1 * simd_type::width);
                    ScaleAppend<SCALE, true>(temp, tmpOut, scale); // Store 1x
                    temp.store(out + 1 * simd_type::width);
                }
            }
        }
    }
}

// inner product without quadrature metric wJ
template <bool SCALE, bool APPEND, typename simd_type>
NEK_FORCE_INLINE static void IProductPyrKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const bool isModified,
    const typename simd_type::vectorType *in, const simd_type *basis0,
    const simd_type *basis1, const simd_type *basis2,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_kj,
    std::vector<simd_type, tinysimd::allocator<simd_type>> &sums_k,
    typename simd_type::scalarType *out, double scale = 1.0)
{
    int mode_pqr = 0, cnt_pqr = 0;

    for (int p = 0; p < nm0; ++p)
    {
        int cnt_kji = 0, cnt_kj = 0;

        for (int k = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j, ++cnt_kj)
            {
                simd_type sum_kj = 0.0;

                for (int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    // load 2x
                    sum_kj.fma(simd_type(in[cnt_kji]), basis0[nq0 * p + i]);
                }

                sums_kj[cnt_kj] = sum_kj; // store 1x
            }
        }

        for (int q = 0; q < p; ++q)
        {
            cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    // Load 2x
                    sum_k.fma(basis1[q * nq1 + j], sums_kj[cnt_kj]);
                }

                sums_k[k] = sum_k; // Store 1x
            }

            for (int r = 0; r < nm2 - p; ++r, ++mode_pqr, ++cnt_pqr)
            {
                simd_type sum_k = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    // Load 2x
                    sum_k.fma(basis2[mode_pqr * nq2 + k], sums_k[k]);
                }
                simd_type temp;
                temp.load(out + cnt_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum_k, scale); // Store 1x
                temp.store(out + cnt_pqr * simd_type::width);
            }
        }

        for (int q = p; q < nm1; ++q)
        {
            cnt_kj = 0;

            for (int k = 0; k < nq2; ++k)
            {
                simd_type sum_k = 0.0;

                for (int j = 0; j < nq1; ++j, ++cnt_kj)
                {
                    // Load 2x
                    sum_k.fma(basis1[q * nq1 + j], sums_kj[cnt_kj]);
                }
                sums_k[k] = sum_k; // Store 1x
            }

            for (int r = 0; r < nm2 - q; ++r, ++mode_pqr, ++cnt_pqr)
            {
                simd_type sum_k = 0.0;

                for (int k = 0; k < nq2; ++k)
                {
                    // Load 3x
                    sum_k.fma(basis2[mode_pqr * nq2 + k], sums_k[k]);
                }
                simd_type temp;
                temp.load(out + cnt_pqr * simd_type::width);
                ScaleAppend<SCALE, APPEND>(temp, sum_k, scale); // Store 1x
                temp.store(out + cnt_pqr * simd_type::width);
            }
        }

        // increment mode in case order1!=order2
        for (int q = nm1; q < nm2; ++q)
        {
            mode_pqr += nm2 - q;
        }
    }

    if (isModified)
    {
        for (int k = 0, cnt = 0; k < nq2; ++k)
        {
            for (int j = 0; j < nq1; ++j)
            {
                for (int i = 0; i < nq0; ++i, ++cnt)
                {
                    // Store jac * quadrature weight
                    simd_type tmpIn = simd_type(in[cnt]); // Load 1x

                    // top vertex
                    //
                    simd_type tmp = basis0[i] * basis1[nq1 + j]; // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[j]);         // Load 2x
                    tmp.fma(basis0[nq0 + i], basis1[nq1 + j]);   // Load 2x
                    tmp = tmp * basis2[nq2 + k];                 // Load 1x
                    tmp = tmp * tmpIn;

                    // add to existing entry
                    simd_type temp;
                    temp.load(out + 1 * simd_type::width);
                    ScaleAppend<SCALE, true>(temp, tmp, scale); // Store 1x
                    temp.store(out + 1 * simd_type::width);
                }
            }
        }
    }
}