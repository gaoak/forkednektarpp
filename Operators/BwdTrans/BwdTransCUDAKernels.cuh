namespace Nektar::Operators::detail
{
template <typename TData>
__global__ void BwdTransSegKernel(const size_t nm0, const size_t nq0,
                                  const size_t nelmt, const TData *basis0,
                                  const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const TData *inptr = in + (nm0 * e);
    TData *outptr      = out + (nq0 * e);

    for (size_t i = 0; i < nq0; ++i)
    {
        TData tmp = inptr[0] * basis0[i];
        for (size_t p = 1; p < nm0; ++p)
        {
            tmp += inptr[p] * basis0[p * nq0 + i];
        }
        outptr[i] = tmp;
    }
}

template <typename TData>
__global__ void BwdTransQuadKernel(const size_t nm0, const size_t nm1,
                                   const size_t nq0, const size_t nq1,
                                   const size_t nelmt, const TData *basis0,
                                   const TData *basis1, const TData *in,
                                   TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    TData *wsp = new TData[nq0 * nm1];

    const TData *inptr = in + (nm0 * nm1 * e);
    TData *outptr      = out + (nq0 * nq1 * e);

    size_t cnt_iq = 0;
    for (size_t i = 0; i < nq0; ++i)
    {
        size_t cnt_pq = 0;
        for (size_t q = 0; q < nm1; ++q)
        {
            TData tmp = inptr[cnt_pq++] * basis0[i];
            for (size_t p = 1; p < nm0; ++p)
            {
                tmp += inptr[cnt_pq++] * basis0[p * nq0 + i];
            }
            wsp[cnt_iq++] = tmp;
        }
    }

    size_t cnt_ij = 0;
    for (size_t j = 0; j < nq1; ++j)
    {
        size_t cnt_iq = 0;
        for (size_t i = 0; i < nq0; ++i)
        {
            TData tmp = wsp[cnt_iq++] * basis1[j];
            for (size_t q = 1; q < nm1; ++q)
            {
                tmp += wsp[cnt_iq++] * basis1[q * nq1 + j];
            }
            outptr[cnt_ij++] = tmp;
        }
    }

    delete wsp;
}

template <typename TData>
__global__ void BwdTransTriKernel(const size_t nm0, const size_t nm1,
                                  const size_t nmTot, const size_t nq0,
                                  const size_t nq1, const size_t nelmt,
                                  const bool correct, const TData *basis0,
                                  const TData *basis1, const TData *in,
                                  TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    TData *wsp = new TData[nm0];

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * e);

    size_t cnt_ij = 0;
    for (size_t j = 0; j < nq1; ++j)
    {
        size_t mode = 0;
        for (size_t p = 0; p < nm0; ++p)
        {
            TData tmp = 0.0;
            for (size_t q = 0; q < (nm1 - p); ++q)
            {
                tmp += basis1[mode * nq1 + j] * inptr[mode];
                mode++;
            }
            wsp[p] = tmp;
        }

        for (size_t i = 0; i < nq0; ++i)
        {
            TData tmp = wsp[0] * basis0[i];
            for (size_t p = 1; p < nm0; ++p)
            {
                tmp += wsp[p] * basis0[p * nq0 + i];
            }

            if (correct)
            {
                tmp += inptr[1] * basis0[nq0 + i] * basis1[nq1 + j];
            }
            outptr[cnt_ij++] = tmp;
        }
    }

    delete wsp;
}

template <typename TData>
__global__ void BwdTransHexKernel(const size_t nm0, const size_t nm1,
                                  const size_t nm2, const size_t nq0,
                                  const size_t nq1, const size_t nq2,
                                  const size_t nelmt, const TData *basis0,
                                  const TData *basis1, const TData *basis2,
                                  const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    TData *wsp0 = new TData[nq0 * nm1 * nm2];
    TData *wsp1 = new TData[nq0 * nq1 * nm2];

    const TData *inptr = in + (nm0 * nm1 * nm2 * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t cnt_irq = 0;
    for (size_t i = 0; i < nq0; ++i)
    {
        size_t cnt_rqp = 0;
        for (size_t r = 0; r < nm2; ++r)
        {
            for (size_t q = 0; q < nm1; ++q)
            {
                TData tmp = inptr[cnt_rqp++] * basis0[i];
                for (size_t p = 1; p < nm0; ++p)
                {
                    tmp += inptr[cnt_rqp++] * basis0[p * nq0 + i];
                }
                wsp0[cnt_irq++] = tmp;
            }
        }
    }

    size_t cnt_jir = 0;
    for (size_t j = 0; j < nq1; ++j)
    {
        size_t cnt_irq = 0;
        for (size_t i = 0; i < nq0; ++i)
        {
            for (size_t r = 0; r < nm2; ++r)
            {
                TData tmp = wsp0[cnt_irq++] * basis1[j];
                for (size_t q = 1; q < nm1; ++q)
                {
                    tmp += wsp0[cnt_irq++] * basis1[q * nq1 + j];
                }
                wsp1[cnt_jir++] = tmp;
            }
        }
    }

    size_t cnt_kji = 0;
    for (size_t k = 0; k < nq2; ++k)
    {
        size_t cnt_jir = 0;
        for (size_t j = 0; j < nq1; ++j)
        {
            for (size_t i = 0; i < nq0; ++i)
            {
                TData tmp = wsp1[cnt_jir++] * basis2[k];
                for (size_t r = 1; r < nm2; ++r)
                {
                    tmp += wsp1[cnt_jir++] * basis2[r * nq2 + k];
                }
                outptr[cnt_kji++] = tmp;
            }
        }
    }

    delete wsp0;
    delete wsp1;
}

template <typename TData> // not working for nm2 > nm1
__global__ void BwdTransTetKernel(const size_t nm0, const size_t nm1,
                                  const size_t nm2, const size_t nmTot,
                                  const size_t nq0, const size_t nq1,
                                  const size_t nq2, const size_t nelmt,
                                  const bool correct, const TData *basis0,
                                  const TData *basis1, const TData *basis2,
                                  const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t nm01 = (2 * nm1 - nm0 + 1) * nm0 / 2;
    TData *fpq  = new TData[nm01];
    TData *fp   = new TData[nm0];

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t cnt_kji = 0;
    for (size_t k = 0; k < nq2; ++k)
    {
        size_t cnt_pq = 0;
        size_t mode   = 0;
        for (size_t p = 0; p < nm0; ++p)
        {
            for (size_t q = 0; q < nm1 - p; ++q)
            {
                TData tmp = basis2[k + nq2 * mode] * inptr[mode++];
                for (size_t r = 1; r < nm2 - p - q; ++r)
                {
                    tmp += basis2[k + nq2 * mode] * inptr[mode++];
                }
                fpq[cnt_pq++] = tmp;
            }

            // increment mode in case order1!=order2
            for (size_t q = nm1 - p; q < nm2 - p; ++q)
            {
                mode += nm2 - p - q;
            }
        }

        for (size_t j = 0; j < nq1; ++j)
        {
            mode   = 0;
            cnt_pq = 0;
            for (size_t p = 0; p < nm0; ++p)
            {
                TData tmp = fpq[cnt_pq++] * basis1[mode * nq1 + j];
                for (size_t q = 1; q < nm1 - p; ++q)
                {
                    tmp += fpq[cnt_pq++] * basis1[(mode + q) * nq1 + j];
                }

                fp[p] = tmp;
                mode += nm1 - p;
            }

            for (size_t i = 0; i < nq0; ++i)
            {
                TData tmp = basis0[i] * fp[0];
                for (size_t p = 1; p < nm0; ++p)
                {
                    tmp += basis0[p * nq0 + i] * fp[p];
                }

                if (correct)
                {
                    // top vertex
                    TData tmp1 = basis0[i] * basis1[nq1 + j];
                    tmp1 += basis0[nq0 + i] * basis1[j];
                    tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp1 *= basis2[nq2 + k];
                    tmp += tmp1 * inptr[1];

                    // bottom vertex
                    tmp1 = basis0[nq0 + i] * basis1[nq1 + j];
                    tmp1 *= basis2[k];
                    tmp += tmp1 * inptr[nm2];

                    // singular edge
                    for (size_t r = 1; r < nm2 - 1; ++r)
                    {
                        tmp1 = basis1[nq1 + j] * basis0[nq0 + i];
                        tmp1 *= basis2[(r + 1) * nq2 + k];
                        tmp += tmp1 * inptr[nm2 + r];
                    }
                }
                outptr[cnt_kji++] = tmp;
            }
        }
    }

    delete fpq;
    delete fp;
}

template <typename TData>
__global__ void BwdTransPrismKernel(const size_t nm0, const size_t nm1,
                                    const size_t nm2, const size_t nmTot,
                                    const size_t nq0, const size_t nq1,
                                    const size_t nq2, const size_t nelmt,
                                    const bool correct, const TData *basis0,
                                    const TData *basis1, const TData *basis2,
                                    const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    TData *fpq = new TData[nm0 * nm1];
    TData *fp  = new TData[nm0];

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t cnt_kji = 0;
    for (size_t k = 0; k < nq2; ++k)
    {
        size_t mode_pqr = 0;
        size_t mode_pr  = 0;
        size_t mode_pq  = 0;
        for (size_t p = 0; p < nm0; ++p)
        {
            for (size_t q = 0; q < nm1; ++q)
            {
                TData tmp = 0.0;
                for (size_t r = 0; r < nm2 - p; ++r)
                {
                    tmp += inptr[mode_pqr++] * basis2[(mode_pr + r) * nq2 + k];
                }
                fpq[mode_pq++] = tmp;
            }
            mode_pr += nm2 - p;
        }

        for (size_t j = 0; j < nq1; ++j)
        {
            size_t mode_pq = 0;
            for (size_t p = 0; p < nm0; ++p)
            {
                TData tmp = fpq[mode_pq++] * basis1[j];
                for (size_t q = 1; q < nm1; ++q)
                {
                    tmp += fpq[mode_pq++] * basis1[q * nq1 + j];
                }
                fp[p] = tmp;
            }

            for (size_t i = 0; i < nq0; ++i)
            {
                TData tmp = fp[0] * basis0[i];
                for (size_t p = 1; p < nm0; ++p)
                {
                    tmp += fp[p] * basis0[p * nq0 + i];
                }

                if (correct)
                {
                    for (size_t q = 0; q < nm1; ++q)
                    {
                        tmp += basis2[nq2 + k] * basis1[q * nq1 + j] *
                               basis0[nq0 + i] * inptr[q * nm2 + 1];
                    }
                }
                outptr[cnt_kji++] = tmp;
            }
        }
    }

    delete fpq;
    delete fp;
}

template <typename TData> // not working for nm2 > nm1
__global__ void BwdTransPyrKernel(const size_t nm0, const size_t nm1,
                                  const size_t nm2, const size_t nmTot,
                                  const size_t nq0, const size_t nq1,
                                  const size_t nq2, const size_t nelmt,
                                  const bool correct, const TData *basis0,
                                  const TData *basis1, const TData *basis2,
                                  const TData *in, TData *out)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    TData *fpq = new TData[nm0 * nm1];
    TData *fp  = new TData[nm0];

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t cnt_kji = 0;
    for (int k = 0; k < nq2; ++k)
    {
        size_t mode_pqr = 0;
        size_t mode_pq  = 0;
        for (size_t p = 0; p < nm0; ++p)
        {
            for (size_t q = 0; q < p; ++q)
            {
                TData tmp = 0.0;
                for (size_t r = 0; r < nm2 - p; ++r)
                {
                    tmp += basis2[mode_pqr * nq2 + k] * inptr[mode_pqr++];
                }
                fpq[mode_pq++] = tmp;
            }

            for (size_t q = p; q < nm1; ++q)
            {
                TData tmp = 0.0;
                for (size_t r = 0; r < nm2 - q; ++r)
                {
                    tmp += basis2[mode_pqr * nq2 + k] * inptr[mode_pqr++];
                }
                fpq[mode_pq++] = tmp;
            }

            // increment mode in case nm2>nm1
            for (size_t q = nm1; q < nm2 - p; ++q)
            {
                mode_pqr += nm2 - q;
            }
        }

        for (size_t j = 0; j < nq1; ++j)
        {
            size_t mode_pq = 0;
            for (size_t p = 0; p < nm0; ++p)
            {
                TData tmp = fpq[mode_pq++] * basis1[j];
                for (int q = 1; q < nm1; ++q)
                {
                    tmp += fpq[mode_pq++] * basis1[q * nq1 + j];
                }
                fp[p] = tmp;
            }

            for (size_t i = 0; i < nq0; ++i)
            {
                TData tmp = fp[0] * basis0[i];
                for (size_t p = 1; p < nm0; ++p)
                {
                    tmp += fp[p] * basis0[p * nq0 + i];
                }

                if (correct)
                {
                    // top vertex
                    TData tmp1 = basis0[i] * basis1[nq1 + j];
                    tmp1 += basis0[nq0 + i] * basis1[j];
                    tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
                    tmp1 *= basis2[nq2 + k];
                    tmp += tmp1 * inptr[1];
                }
                outptr[cnt_kji++] = tmp;
            }
        }
    }

    delete fpq;
    delete fp;
}

template <typename TData>
__global__ void BwdTransSegKernel_QP(const size_t nm0, const size_t nq0,
                                     const TData *basis0, const TData *in,
                                     TData *out)
{
    size_t e = blockIdx.x;

    const TData *inptr = in + (nm0 * e);
    TData *outptr      = out + (nq0 * e);

    size_t i = threadIdx.x;
    if (i < nq0)
    {
        TData tmp = inptr[0] * basis0[i];
        for (size_t p = 1; p < nm0; p++)
        {
            tmp += inptr[p] * basis0[p * nq0 + i];
        }
        outptr[i] = tmp;
    }
}

template <typename TData>
__global__ void BwdTransQuadKernel_QP(const size_t nm0, const size_t nm1,
                                      const size_t nq0, const size_t nq1,
                                      const TData *basis0, const TData *basis1,
                                      const TData *in, TData *wsp, TData *out)
{
    size_t e = blockIdx.x;

    const TData *inptr = in + (nm0 * nm1 * e);
    TData *outptr      = out + (nq0 * nq1 * e);

    size_t i, j, q;
    i = threadIdx.x;
    q = threadIdx.y;
    if (i < nq0 && q < nm1)
    {
        size_t cnt_iq = nm1 * i + q;
        size_t cnt_pq = nm0 * q;
        TData tmp     = inptr[cnt_pq++] * basis0[i];
        for (size_t p = 1; p < nm0; ++p)
        {
            tmp += inptr[cnt_pq++] * basis0[p * nq0 + i];
        }
        wsp[cnt_iq] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    if (i < nq0 && j < nq1)
    {
        size_t cnt_iq = nm1 * i;
        size_t cnt_ij = nq0 * j + i;
        TData tmp     = wsp[cnt_iq++] * basis1[j];
        for (size_t q = 1; q < nm1; ++q)
        {
            tmp += wsp[cnt_iq++] * basis1[q * nq1 + j];
        }
        outptr[cnt_ij] = tmp;
    }
}

template <typename TData>
__global__ void BwdTransTriKernel_QP(const size_t nm0, const size_t nm1,
                                     const size_t nmTot, const size_t nq0,
                                     const size_t nq1, const bool correct,
                                     const TData *basis0, const TData *basis1,
                                     const TData *in, TData *wsp, TData *out)
{
    size_t e = blockIdx.x;

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * e);

    size_t i, j, p;
    p = threadIdx.x;
    j = threadIdx.y;
    if (p < nm0 && j < nq1)
    {
        size_t mode = (2 * nm1 - p + 1) * p / 2;
        TData tmp   = 0.0;
        for (size_t q = 0; q < (nm1 - p); ++q)
        {
            tmp += basis1[mode * nq1 + j] * inptr[mode];
            mode++;
        }
        wsp[nm0 * j + p] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    if (i < nq0 && j < nq1)
    {
        size_t cnt_ij = nq0 * j + i;
        TData tmp     = wsp[nm0 * j] * basis0[i];
        for (size_t p = 1; p < nm0; ++p)
        {
            tmp += wsp[nm0 * j + p] * basis0[p * nq0 + i];
        }

        if (correct)
        {
            tmp += inptr[1] * basis0[nq0 + i] * basis1[nq1 + j];
        }

        outptr[cnt_ij] = tmp;
    }
}

template <typename TData>
__global__ void BwdTransHexKernel_QP(const size_t nm0, const size_t nm1,
                                     const size_t nm2, const size_t nq0,
                                     const size_t nq1, const size_t nq2,
                                     const TData *basis0, const TData *basis1,
                                     const TData *basis2, const TData *in,
                                     TData *wsp0, TData *wsp1, TData *out)
{
    size_t e = blockIdx.x;

    const TData *inptr = in + (nm0 * nm1 * nm2 * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t i, j, k, q, r;
    i = threadIdx.x;
    q = threadIdx.y;
    r = threadIdx.z;
    if (i < nq0 && q < nm1 && r < nm2)
    {
        size_t cnt_rqp = nm1 * nm0 * r + nm0 * q;
        size_t cnt_irq = nm1 * nm2 * i + nm1 * r + q;
        TData tmp      = inptr[cnt_rqp++] * basis0[i];
        for (size_t p = 1; p < nm0; ++p)
        {
            tmp += inptr[cnt_rqp++] * basis0[p * nq0 + i];
        }
        wsp0[cnt_irq] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    r = threadIdx.z;
    if (i < nq0 && j < nq1 && r < nm2)
    {
        size_t cnt_irq = nm1 * nm2 * i + nm1 * r;
        size_t cnt_jir = nq0 * nm2 * j + nm2 * i + r;
        TData tmp      = wsp0[cnt_irq++] * basis1[j];
        for (size_t q = 1; q < nm1; ++q)
        {
            tmp += wsp0[cnt_irq++] * basis1[q * nq1 + j];
        }
        wsp1[cnt_jir] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (i < nq0 && j < nq1 && k < nq2)
    {
        size_t cnt_jir = nq0 * nm2 * j + nm2 * i;
        size_t cnt_kji = nq0 * nq1 * k + nq0 * j + i;
        TData tmp      = wsp1[cnt_jir++] * basis2[k];
        for (size_t r = 1; r < nm2; ++r)
        {
            tmp += wsp1[cnt_jir++] * basis2[r * nq2 + k];
        }
        outptr[cnt_kji] = tmp;
    }
}

template <typename TData> // not working for nm2 > nm1
__global__ void BwdTransTetKernel_QP(const size_t nm0, const size_t nm1,
                                     const size_t nm2, const size_t nmTot,
                                     const size_t nq0, const size_t nq1,
                                     const size_t nq2, const bool correct,
                                     const TData *basis0, const TData *basis1,
                                     const TData *basis2, const TData *in,
                                     TData *fpq, TData *fp, TData *out)
{
    size_t nm01 = (2 * nm1 - nm0 + 1) * nm0 / 2;
    size_t e    = blockIdx.x;

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t i, j, k, p, q;
    p = threadIdx.x;
    q = threadIdx.y;
    k = threadIdx.z;
    if (p < nm0 && q < nm1 && k < nq2)
    {
        size_t cnt_pq = nm01 * k + (2 * nm1 - p + 1) * p / 2 + q;
        size_t mode   = (2 * (nm2 - p) - q + 1) * q / 2;
        for (size_t n = 0; n < p; ++n)
        {
            mode += (nm2 - n + 1) * (nm2 - n) / 2;
        }

        if (q < nm1 - p)
        {
            TData tmp = basis2[k + nq2 * mode] * inptr[mode++];
            for (size_t r = 1; r < nm2 - p - q; ++r)
            {
                tmp += basis2[k + nq2 * mode] * inptr[mode++];
            }
            fpq[cnt_pq] = tmp;
        }
    }

    __syncthreads();

    p = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (p < nm0 && j < nq1 && k < nq2)
    {
        size_t cnt_pq = nm01 * k + (2 * nm1 - p + 1) * p / 2;
        size_t mode   = (2 * nm1 - p + 1) * p / 2;
        size_t mode_p = nm0 * nq1 * k + nm0 * j + p;
        TData tmp     = fpq[cnt_pq++] * basis1[mode * nq1 + j];
        for (size_t q = 1; q < nm1 - p; ++q)
        {
            tmp += fpq[cnt_pq++] * basis1[(mode + q) * nq1 + j];
        }
        fp[mode_p] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (i < nq0 && j < nq1 && k < nq2)
    {
        size_t cnt_kji = nq0 * nq1 * k + nq0 * j + i;
        size_t mode_p  = nm0 * nq1 * k + nm0 * j;
        TData tmp      = basis0[i] * fp[mode_p++];
        for (size_t p = 1; p < nm0; ++p)
        {
            tmp += basis0[p * nq0 + i] * fp[mode_p++];
        }

        if (correct)
        {
            // top vertex
            TData tmp1 = basis0[i] * basis1[nq1 + j];
            tmp1 += basis0[nq0 + i] * basis1[j];
            tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
            tmp1 *= basis2[nq2 + k];
            tmp += tmp1 * inptr[1];

            // bottom vertex
            tmp1 = basis0[nq0 + i] * basis1[nq1 + j];
            tmp1 *= basis2[k];
            tmp += tmp1 * inptr[nm2];

            // singular edge
            for (size_t r = 1; r < nm2 - 1; ++r)
            {
                tmp1 = basis1[nq1 + j] * basis0[nq0 + i];
                tmp1 *= basis2[(r + 1) * nq2 + k];
                tmp += tmp1 * inptr[nm2 + r];
            }
        }

        outptr[cnt_kji] = tmp;
    }
}

template <typename TData>
__global__ void BwdTransPrismKernel_QP(const size_t nm0, const size_t nm1,
                                       const size_t nm2, const size_t nmTot,
                                       const size_t nq0, const size_t nq1,
                                       const size_t nq2, const bool correct,
                                       const TData *basis0, const TData *basis1,
                                       const TData *basis2, const TData *in,
                                       TData *fpq, TData *fp, TData *out)
{
    size_t e = blockIdx.x;

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t i, j, k, p, q;
    p = threadIdx.x;
    q = threadIdx.y;
    k = threadIdx.z;
    if (p < nm0 && q < nm1 && k < nq2)
    {
        size_t mode_pr  = (2 * nm2 - p + 1) * p / 2;
        size_t mode_pqr = mode_pr * nm1 + (nm2 - p) * q;
        size_t mode_pq  = nm0 * nm1 * k + nm1 * p + q;
        TData tmp       = 0.0;
        for (size_t r = 0; r < nm2 - p; ++r)
        {
            tmp += inptr[mode_pqr++] * basis2[(mode_pr + r) * nq2 + k];
        }
        fpq[mode_pq] = tmp;
    }

    __syncthreads();

    p = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (p < nm0 && j < nq1 && k < nq2)
    {
        size_t mode_pq = nm0 * nm1 * k + nm1 * p;
        size_t mode_p  = nm0 * nq1 * k + nm0 * j + p;
        TData tmp      = fpq[mode_pq++] * basis1[j];
        for (int q = 1; q < nm1; ++q)
        {
            tmp += fpq[mode_pq++] * basis1[q * nq1 + j];
        }
        fp[mode_p] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (i < nq0 && j < nq1 && k < nq2)
    {
        size_t cnt_kji = nq0 * nq1 * k + nq0 * j + i;
        size_t mode_p  = nm0 * nq1 * k + nm0 * j;
        TData tmp      = fp[mode_p++] * basis0[i];
        for (int p = 1; p < nm0; ++p)
        {
            tmp += fp[mode_p++] * basis0[p * nq0 + i];
        }

        if (correct)
        {
            for (int q = 0; q < nm1; ++q)
            {
                tmp += basis2[nq2 + k] * basis1[q * nq1 + j] * basis0[nq0 + i] *
                       inptr[q * nm2 + 1];
            }
        }
        outptr[cnt_kji] = tmp;
    }
}

template <typename TData> // not working for nm2 > nm1
__global__ void BwdTransPyrKernel_QP(const size_t nm0, const size_t nm1,
                                     const size_t nm2, const size_t nmTot,
                                     const size_t nq0, const size_t nq1,
                                     const size_t nq2, const bool correct,
                                     const TData *basis0, const TData *basis1,
                                     const TData *basis2, const TData *in,
                                     TData *fpq, TData *fp, TData *out)
{
    size_t e = blockIdx.x;

    const TData *inptr = in + (nmTot * e);
    TData *outptr      = out + (nq0 * nq1 * nq2 * e);

    size_t i, j, k, p, q;
    p = threadIdx.x;
    q = threadIdx.y;
    k = threadIdx.z;
    if (p < nm0 && q < nm1 && k < nq2)
    {
        size_t mode_pq  = nm0 * nm1 * k + nm1 * p + q;
        size_t mode_tmp = 0;
        for (size_t n = 0; n < p; ++n)
        {
            mode_tmp += n * (nm2 - n);
            mode_tmp += ((2 * nm2 - nm1 - n + 1) * (nm1 - n)) / 2;
            if (nm2 > nm1 && nm2 - nm1 > n)
            {
                mode_tmp += (((nm2 - nm1) + (n + 1)) * (nm2 - nm1 - n)) / 2;
            }
        }

        if (q < p)
        {
            size_t mode_pqr = mode_tmp + q * (nm2 - p);
            TData tmp       = 0.0;
            for (size_t r = 0; r < nm2 - p; ++r)
            {
                tmp += basis2[mode_pqr * nq2 + k] * inptr[mode_pqr++];
            }
            fpq[mode_pq] = tmp;
        }
        else if (q < nm1)
        {
            size_t mode_pqr = mode_tmp + p * (nm2 - p);
            mode_pqr += ((2 * (nm2 - p) - (q - p) + 1) * (q - p)) / 2;
            TData tmp = 0.0;
            for (size_t r = 0; r < nm2 - q; ++r)
            {
                tmp += basis2[mode_pqr * nq2 + k] * inptr[mode_pqr++];
            }
            fpq[mode_pq] = tmp;
        }
    }

    __syncthreads();

    p = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (p < nm0 && j < nq1 && k < nq2)
    {
        size_t mode_p  = nm0 * nq1 * k + nm0 * j + p;
        size_t mode_pq = nm0 * nm1 * k + nm1 * p;
        TData tmp      = fpq[mode_pq++] * basis1[j];
        for (size_t q = 1; q < nm1; ++q)
        {
            tmp += fpq[mode_pq++] * basis1[q * nq1 + j];
        }
        fp[mode_p] = tmp;
    }

    __syncthreads();

    i = threadIdx.x;
    j = threadIdx.y;
    k = threadIdx.z;
    if (i < nq0 && j < nq1 && k < nq2)
    {
        size_t cnt_kji = nq0 * nq1 * k + nq0 * j + i;
        size_t mode_p  = nm0 * nq1 * k + nm0 * j;
        TData tmp      = fp[mode_p++] * basis0[i];
        for (size_t p = 1; p < nm0; ++p)
        {
            tmp += fp[mode_p++] * basis0[p * nq0 + i];
        }

        if (correct)
        {
            // top vertex
            TData tmp1 = basis0[i] * basis1[nq1 + j];
            tmp1 += basis0[nq0 + i] * basis1[j];
            tmp1 += basis0[nq0 + i] * basis1[nq1 + j];
            tmp1 *= basis2[nq2 + k];
            tmp += tmp1 * inptr[1];
        }
        outptr[cnt_kji] = tmp;
    }
}

} // namespace Nektar::Operators::detail
