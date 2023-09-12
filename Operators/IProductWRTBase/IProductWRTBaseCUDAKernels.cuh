namespace Nektar::Operators::detail
{
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseSegKernel(const size_t nm0, const size_t nq0,
                                         const size_t nelmt,
                                         const TData *basis0, const TData *w0,
                                         const TData *jac, const TData *in,
                                         TData *out, TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Assign pointers.
    const TData *inptr  = in + (nq0 * e);
    TData *outptr       = out + (nm0 * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * e) : jac + e;

    // Compute inner product.
    for (size_t p = 0; p < nm0; ++p)
    {
        TData sum = 0.0;
        for (size_t i = 0; i < nq0; ++i)
        {
            TData jac_val = DEFORMED ? jacptr[i] : jacptr[0];
            sum += inptr[i] * basis0[p * nq0 + i] * jac_val * w0[i];
        }
        if (SCALE)
        {
            sum *= scale;
        }
        outptr[p] = APPEND ? outptr[p] + sum : sum;
    }
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseQuadKernel(
    const size_t nm0, const size_t nm1, const size_t nq0, const size_t nq1,
    const size_t nelmt, const TData *basis0, const TData *basis1,
    const TData *w0, const TData *w1, const TData *jac, const TData *in,
    TData *out, TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Allocate workspace memory.
    TData *wsp = new TData[nq1];

    // Assign pointers.
    const TData *inptr  = in + (nq0 * nq1 * e);
    TData *outptr       = out + (nm0 * nm1 * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * nq1 * e) : jac + e;

    // Compute inner product.
    for (size_t p = 0; p < nm0; ++p)
    {
        size_t cnt_ji = 0;
        for (size_t j = 0; j < nq1; ++j)
        {
            TData sum = 0.0;
            for (size_t i = 0; i < nq0; ++i)
            {
                TData jac_val = DEFORMED ? jacptr[j * nq0 + i] : jacptr[0];
                sum += inptr[cnt_ji++] * basis0[p * nq0 + i] * jac_val * w0[i];
            }
            wsp[j] = sum;
        }

        for (size_t q = 0; q < nm1; ++q)
        {
            TData sum = 0.0;
            for (size_t j = 0; j < nq1; ++j)
            {
                sum += wsp[j] * basis1[q * nq1 + j] * w1[j];
            }
            if (SCALE)
            {
                sum *= scale;
            }
            outptr[q * nm0 + p] = APPEND ? outptr[q * nm0 + p] + sum : sum;
        }
    }
    delete wsp;
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseTriKernel(
    const size_t nm0, const size_t nm1, const size_t nmTot, const size_t nq0,
    const size_t nq1, const size_t nelmt, const bool correct,
    const TData *basis0, const TData *basis1, const TData *w0, const TData *w1,
    const TData *jac, const TData *in, TData *out, TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Allocate workspace memory.
    TData *wsp = new TData[nq1];

    // Assign pointers.
    const TData *inptr  = in + (nq0 * nq1 * e);
    TData *outptr       = out + (nmTot * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * nq1 * e) : jac + e;

    // Compute inner product.
    size_t mode = 0;
    for (size_t p = 0; p < nm0; ++p)
    {
        size_t eta_idx = 0;
        for (size_t eta1 = 0; eta1 < nq1; ++eta1)
        {
            TData sum = 0.0;
            for (size_t eta0 = 0; eta0 < nq0; ++eta0)
            {
                TData jac_val =
                    DEFORMED ? jacptr[eta1 * nq0 + eta0] : jacptr[0];
                sum += inptr[eta_idx++] * basis0[p * nq0 + eta0] * jac_val *
                       w0[eta0];
            }
            wsp[eta1] = sum;
        }

        for (size_t q = 0; q < nm1 - p; ++q)
        {
            TData sum = 0.0;
            for (size_t eta1 = 0; eta1 < nq1; ++eta1)
            {
                sum += wsp[eta1] * basis1[mode * nq1 + eta1] * w1[eta1];
            }
            if (SCALE)
            {
                sum *= scale;
            }
            outptr[mode++] = APPEND ? outptr[mode] + sum : sum;
        }
    }

    // Correction for singular vertex in collpased coordinates.
    // Basically we add phi_1 * phi_01 * (weighting, etc) to mode 00
    // With contributions from every quadrature point
    if (correct)
    {
        size_t eta_idx = 0;
        TData iprod_01 = 0.0;
        for (size_t eta1 = 0; eta1 < nq1; ++eta1)
        {
            TData tmp = w1[eta1] * basis1[nq1 + eta1];

            if (!DEFORMED)
            {
                tmp *= jacptr[0];
            }

            for (size_t eta0 = 0; eta0 < nq0; ++eta0)
            {
                TData prod = inptr[eta_idx++] * tmp * w0[eta0];
                if (DEFORMED)
                {
                    prod *= jacptr[eta1 * nq0 + eta0];
                }
                iprod_01 += prod * basis0[nq0 + eta0];
            }
        }
        outptr[1] += SCALE ? iprod_01 * scale : iprod_01;
    }
    delete wsp;
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseHexKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nq0,
    const size_t nq1, const size_t nq2, const size_t nelmt, const TData *basis0,
    const TData *basis1, const TData *basis2, const TData *w0, const TData *w1,
    const TData *w2, const TData *jac, const TData *in, TData *out,
    TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Allocate workspace memory.
    TData *wsp0 = new TData[nq2 * nq1];
    TData *wsp1 = new TData[nq2];

    // Assign pointers.
    const TData *inptr  = in + (nq0 * nq1 * nq2 * e);
    TData *outptr       = out + (nm0 * nm1 * nm2 * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * nq1 * nq2 * e) : jac + e;

    // Compute inner product.
    for (size_t p = 0; p < nm0; ++p)
    {
        size_t cnt_kji = 0, cnt_kj = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            for (size_t j = 0; j < nq1; ++j)
            {
                TData sum_kj = 0.0;
                for (size_t i = 0; i < nq0; ++i)
                {
                    TData jac_val = DEFORMED
                                        ? jacptr[nq0 * nq1 * k + nq0 * j + i]
                                        : jacptr[0];
                    sum_kj += inptr[cnt_kji++] * basis0[i + nq0 * p] * jac_val *
                              w0[i];
                }
                wsp0[cnt_kj++] = sum_kj;
            }
        }

        for (size_t q = 0; q < nm1; ++q)
        {
            cnt_kj = 0;
            for (size_t k = 0; k < nq2; ++k)
            {
                TData sum_k = 0.0;
                for (size_t j = 0; j < nq1; ++j)
                {
                    sum_k += wsp0[cnt_kj++] * basis1[q * nq1 + j] * w1[j];
                }
                wsp1[k] = sum_k;
            }

            for (size_t r = 0; r < nm2; ++r)
            {
                TData sum = 0.0;
                for (size_t k = 0; k < nq2; ++k)
                {
                    sum += wsp1[k] * basis2[r * nq2 + k] * w2[k];
                }
                if (SCALE)
                {
                    sum *= scale;
                }
                outptr[r * nm0 * nm1 + q * nm0 + p] =
                    APPEND ? outptr[r * nm0 * nm1 + q * nm0 + p] + sum : sum;
            }
        }
    }
    delete wsp0;
    delete wsp1;
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBaseTetKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nmTot,
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t nelmt,
    const bool correct, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, const TData *in, TData *out, TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Allocate workspace memory.
    TData *wsp0 = new TData[nq2 * nq1];
    TData *wsp1 = new TData[nq2];

    // Assign pointers.
    const TData *inptr  = in + (nq0 * nq1 * nq2 * e);
    TData *outptr       = out + (nmTot * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * nq1 * nq2 * e) : jac + e;

    // Compute inner product.
    size_t cnt_pqr = 0;
    for (size_t p = 0, mode = 0, mode2 = 0; p < nm0; ++p)
    {
        size_t cnt_kji = 0, cnt_kj = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            for (size_t j = 0; j < nq1; ++j)
            {
                TData jac_val =
                    DEFORMED ? jacptr[nq0 * nq1 * k + nq0 * j] : jacptr[0];
                TData sum_kj =
                    inptr[cnt_kji++] * basis0[nq0 * p] * jac_val * w0[0];
                for (size_t i = 1; i < nq0; ++i)
                {
                    jac_val = DEFORMED ? jacptr[nq0 * nq1 * k + nq0 * j + i]
                                       : jacptr[0];
                    sum_kj += inptr[cnt_kji++] * basis0[i + nq0 * p] * jac_val *
                              w0[i];
                }
                wsp0[cnt_kj++] = sum_kj;
            }
        }

        for (size_t q = 0; q < nm1 - p; ++q, ++mode)
        {
            size_t cnt_kj = 0;
            for (size_t k = 0; k < nq2; ++k)
            {
                TData sum_k = basis1[mode * nq1] * wsp0[cnt_kj++] * w1[0];
                for (size_t j = 1; j < nq1; ++j)
                {
                    sum_k += basis1[mode * nq1 + j] * wsp0[cnt_kj++] * w1[j];
                }
                wsp1[k] = sum_k;
            }

            for (size_t r = 0; r < nm2 - p - q; ++r, ++mode2)
            {
                TData tmp = wsp1[0] * basis2[mode2 * nq2] * w2[0];
                for (size_t k = 1; k < nq2; ++k)
                {
                    tmp += wsp1[k] * basis2[mode2 * nq2 + k] * w2[k];
                }
                if (SCALE)
                {
                    tmp *= scale;
                }
                outptr[cnt_pqr++] = APPEND ? outptr[cnt_pqr] + tmp : tmp;
            }
        }
    }

    // Add correction for collapsed coordinate.
    if (correct)
    {
        size_t cnt = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            TData tmpQ2 = w2[k];
            if (!DEFORMED)
            {
                tmpQ2 *= jacptr[0];
            }

            for (size_t j = 0; j < nq1; ++j)
            {
                TData tmpQ1 = tmpQ2 * w1[j];
                for (size_t i = 0; i < nq0; ++i)
                {
                    // Store jac * quadrature weight
                    TData tmpQ = tmpQ1 * w0[i];

                    if (DEFORMED)
                    {
                        tmpQ *= jacptr[k * nq0 * nq1 + j * nq0 + i];
                    }

                    // top vertex
                    //
                    TData tmp = basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k];
                    tmp *= inptr[cnt] * tmpQ;

                    // add to existing entry
                    outptr[1] += SCALE ? tmp * scale : tmp;

                    // bottom vertex
                    //
                    tmp = basis0[nq0 + i] * basis1[nq1 + j] * basis2[k] *
                          inptr[cnt] * tmpQ;
                    outptr[nm2] += SCALE ? tmp * scale : tmp;

                    // singular edge
                    for (size_t r = 1; r < nm2 - 1; ++r)
                    {
                        tmp = basis2[(r + 1) * nq2 + k] * basis1[nq1 + j] *
                              basis0[nq0 + i] * inptr[cnt] * tmpQ;
                        outptr[nm2 + r] += SCALE ? tmp * scale : tmp;
                    }
                    cnt++;
                }
            }
        }
    }
    delete wsp0;
    delete wsp1;
}

template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBasePrismKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nmTot,
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t nelmt,
    const bool correct, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, const TData *in, TData *out, TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Allocate workspace memory.
    TData *wsp0 = new TData[nq2 * nq1];
    TData *wsp1 = new TData[nq2];
    TData *wsp2 = new TData[nm1];

    // Assign pointers.
    const TData *inptr  = in + (nq0 * nq1 * nq2 * e);
    TData *outptr       = out + (nmTot * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * nq1 * nq2 * e) : jac + e;

    // Compute inner product.
    size_t mode_pr = 0, mode_pqr = 0;
    for (size_t p = 0; p < nm0; ++p)
    {
        size_t cnt_kji = 0, cnt_kj = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            for (size_t j = 0; j < nq1; ++j)
            {
                TData sum_kj = 0.0;
                for (size_t i = 0; i < nq0; ++i)
                {
                    TData jac_val = DEFORMED
                                        ? jacptr[nq0 * nq1 * k + nq0 * j + i]
                                        : jacptr[0];
                    sum_kj += inptr[cnt_kji++] * basis0[nq0 * p + i] * jac_val *
                              w0[i];
                }
                wsp0[cnt_kj++] = sum_kj;
            }
        }

        for (size_t q = 0; q < nm1; ++q)
        {
            cnt_kj = 0;
            for (size_t k = 0; k < nq2; ++k)
            {
                TData sum_k = basis1[q * nq1] * w1[0] * wsp0[cnt_kj++];
                for (size_t j = 1; j < nq1; ++j)
                {
                    sum_k += basis1[q * nq1 + j] * w1[j] * wsp0[cnt_kj++];
                }
                wsp1[k] = sum_k;
            }

            for (int r = 0; r < nm2 - p; ++r)
            {
                TData sum_k = basis2[(mode_pr + r) * nq2] * w2[0] * wsp1[0];
                for (size_t k = 1; k < nq2; ++k)
                {
                    sum_k += basis2[(mode_pr + r) * nq2 + k] * w2[k] * wsp1[k];
                }
                if (SCALE)
                {
                    sum_k *= scale;
                }
                outptr[mode_pqr++] = APPEND ? outptr[mode_pqr] + sum_k : sum_k;
            }
        }
        mode_pr += nm2 - p;
    }

    // Add correction for collapsed coordinate.
    if (correct)
    {
        for (size_t q = 0; q < nm1; ++q)
        {
            wsp2[q] = 0.0;
        }

        size_t cnt_kji = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            TData k_weight = w2[k];
            if (!DEFORMED)
            {
                k_weight *= jacptr[0];
            }

            for (size_t j = 0; j < nq1; ++j)
            {
                TData kj_weight = k_weight * w1[j];
                for (size_t i = 0; i < nq0; ++i)
                {
                    TData prod = kj_weight * w0[i] * inptr[cnt_kji++];
                    if (DEFORMED)
                    {
                        prod *= jacptr[k * nq1 * nq0 + j * nq0 + i];
                    }

                    for (size_t q = 0; q < nm1; ++q)
                    {
                        wsp2[q] += prod * basis2[nq2 + k] *
                                   basis1[q * nq1 + j] * basis0[nq0 + i];
                    }
                }
            }
        }

        for (size_t q = 0; q < nm1; ++q)
        {
            outptr[nm2 * q + 1] += SCALE ? wsp2[q] * scale : wsp2[q];
        }
    }
    delete wsp0;
    delete wsp1;
    delete wsp2;
}

// NOTE: Not workign when nm2 > nm1
template <typename TData, bool SCALE, bool APPEND, bool DEFORMED>
__global__ void IProductWRTBasePyrKernel(
    const size_t nm0, const size_t nm1, const size_t nm2, const size_t nmTot,
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t nelmt,
    const bool correct, const TData *basis0, const TData *basis1,
    const TData *basis2, const TData *w0, const TData *w1, const TData *w2,
    const TData *jac, const TData *in, TData *out, TData scale = 1.0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Allocate workspace memory.
    TData *wsp0 = new TData[nq2 * nq1];
    TData *wsp1 = new TData[nq2];

    // Assign pointers.
    const TData *inptr  = in + (nq0 * nq1 * nq2 * e);
    TData *outptr       = out + (nmTot * e);
    const TData *jacptr = DEFORMED ? jac + (nq0 * nq1 * nq2 * e) : jac + e;

    // Compute inner product.
    size_t mode_pqr = 0;
    for (size_t p = 0; p < nm0; ++p)
    {
        size_t cnt_kji = 0, cnt_kj = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            for (size_t j = 0; j < nq1; ++j)
            {
                TData sum_kj = 0.0;
                for (size_t i = 0; i < nq0; ++i)
                {
                    TData jac_val = DEFORMED
                                        ? jacptr[nq0 * nq1 * k + nq0 * j + i]
                                        : jacptr[0];
                    sum_kj += inptr[cnt_kji++] * basis0[nq0 * p + i] * jac_val *
                              w0[i];
                }
                wsp0[cnt_kj++] = sum_kj;
            }
        }

        for (size_t q = 0; q < p; ++q)
        {
            cnt_kj = 0;
            for (size_t k = 0; k < nq2; ++k)
            {
                TData sum_k = basis1[q * nq1] * w1[0] * wsp0[cnt_kj++];
                for (size_t j = 1; j < nq1; ++j)
                {
                    sum_k += basis1[q * nq1 + j] * w1[j] * wsp0[cnt_kj++];
                }
                wsp1[k] = sum_k;
            }

            for (size_t r = 0; r < nm2 - p; ++r)
            {
                TData sum_k = basis2[mode_pqr * nq2] * w2[0] * wsp1[0];
                for (size_t k = 1; k < nq2; ++k)
                {
                    sum_k += basis2[mode_pqr * nq2 + k] * w2[k] * wsp1[k];
                }
                if (SCALE)
                {
                    sum_k *= scale;
                }
                outptr[mode_pqr++] = APPEND ? outptr[mode_pqr] + sum_k : sum_k;
            }
        }

        for (size_t q = p; q < nm1; ++q)
        {
            cnt_kj = 0;
            for (size_t k = 0; k < nq2; ++k)
            {
                TData sum_k = basis1[q * nq1] * w1[0] * wsp0[cnt_kj++];
                for (size_t j = 1; j < nq1; ++j)
                {
                    sum_k += basis1[q * nq1 + j] * w1[j] * wsp0[cnt_kj++];
                }
                wsp1[k] = sum_k;
            }

            for (size_t r = 0; r < nm2 - q; ++r)
            {
                TData sum_k = basis2[mode_pqr * nq2] * w2[0] * wsp1[0];
                for (size_t k = 1; k < nq2; ++k)
                {
                    sum_k += basis2[mode_pqr * nq2 + k] * w2[k] * wsp1[k];
                }
                if (SCALE)
                {
                    sum_k *= scale;
                }
                outptr[mode_pqr++] = APPEND ? outptr[mode_pqr] + sum_k : sum_k;
            }
        }
    }

    // Add correction for collapsed coordinate.
    if (correct)
    {
        size_t cnt = 0;
        for (size_t k = 0; k < nq2; ++k)
        {
            TData tmpQ2 = w2[k];
            if (!DEFORMED)
            {
                tmpQ2 *= jacptr[0];
            }

            for (size_t j = 0; j < nq1; ++j)
            {
                TData tmpQ1 = tmpQ2 * w1[j];
                for (size_t i = 0; i < nq0; ++i)
                {
                    // Store jac * quadrature weight
                    TData tmpQ = tmpQ1 * w0[i];
                    if (DEFORMED)
                    {
                        tmpQ *= jacptr[k * nq0 * nq1 + j * nq0 + i];
                    }

                    // top vertex
                    TData tmp = basis0[i] * basis1[nq1 + j];
                    tmp += basis0[nq0 + i] * basis1[j];
                    tmp += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp *= basis2[nq2 + k];
                    tmp *= inptr[cnt++] * tmpQ;

                    // add to existing entry
                    outptr[1] += SCALE ? tmp * scale : tmp;
                }
            }
        }
    }
    delete wsp0;
    delete wsp1;
}

} // namespace Nektar::Operators::detail
