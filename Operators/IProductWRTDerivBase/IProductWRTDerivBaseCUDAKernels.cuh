namespace Nektar::Operators::detail
{
template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBaseSegKernel(const size_t nq0,
                                              const size_t ncoord,
                                              const size_t nelmt,
                                              const size_t dfSize, TData *df,
                                              TData *in0, TData *in1,
                                              TData *in2, TData *out0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t ndf = ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * e);
    if (ncoord > 1)
    {
        inptr[1] = in1 + (nq0 * e);
    }
    if (ncoord > 2)
    {
        inptr[2] = in2 + (nq0 * e);
    }

    TData *outptr0 = out0 + (nq0 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t i = 0; i < nq0; ++i)
    {
        size_t dfindex = DEFORMED ? i : 0;
        outptr0[i]     = dfptr[0][dfindex] * inptr[0][i];
        for (size_t d = 1; d < ncoord; ++d)
        {
            outptr0[i] += (dfptr[d][dfindex] * inptr[d][i]);
        }
    }
    delete[] inptr;
    delete[] dfptr;
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBaseQuadKernel(
    const size_t nq0, const size_t nq1, const size_t ncoord, const size_t nelmt,
    const size_t dfSize, TData *df, TData *in0, TData *in1, TData *in2,
    TData *out0, TData *out1)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const auto ndf = 2 * ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * nq1 * e);
    inptr[1]      = in1 + (nq0 * nq1 * e);
    if (ncoord > 2)
    {
        inptr[2] = in2 + (nq0 * nq1 * e);
    }

    TData *outptr0 = out0 + (nq0 * nq1 * e);
    TData *outptr1 = out1 + (nq0 * nq1 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t i = 0; i < nq0 * nq1; ++i)
    {
        size_t dfindex = DEFORMED ? i : 0;
        outptr0[i]     = dfptr[0][dfindex] * inptr[0][i];
        outptr1[i]     = dfptr[1][dfindex] * inptr[0][i];
        for (size_t d = 1; d < ncoord; ++d)
        {
            outptr0[i] += (dfptr[2 * d][dfindex] * inptr[d][i]);
            outptr1[i] += (dfptr[2 * d + 1][dfindex] * inptr[d][i]);
        }
    }
    delete[] inptr;
    delete[] dfptr;
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBaseTriKernel(
    const size_t nq0, const size_t nq1, const size_t ncoord, const size_t nelmt,
    const TData *Z0, const TData *Z1, const size_t dfSize, TData *df,
    TData *in0, TData *in1, TData *in2, TData *out0, TData *out1)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const auto ndf = 2 * ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * nq1 * e);
    inptr[1]      = in1 + (nq0 * nq1 * e);
    if (ncoord > 2)
    {
        inptr[2] = in2 + (nq0 * nq1 * e);
    }

    TData *outptr0 = out0 + (nq0 * nq1 * e);
    TData *outptr1 = out1 + (nq0 * nq1 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t j = 0, cnt_ji = 0; j < nq1; ++j)
    {
        TData f0 = 2.0 / (1.0 - Z1[j]);
        for (size_t i = 0; i < nq0; ++i, ++cnt_ji)
        {
            size_t dfindex  = DEFORMED ? cnt_ji : 0;
            outptr0[cnt_ji] = dfptr[0][dfindex] * inptr[0][cnt_ji];
            outptr1[cnt_ji] = dfptr[1][dfindex] * inptr[0][cnt_ji];
            for (size_t d = 1; d < ncoord; ++d)
            {
                outptr0[cnt_ji] += (dfptr[2 * d][dfindex] * inptr[d][cnt_ji]);
                outptr1[cnt_ji] +=
                    (dfptr[2 * d + 1][dfindex] * inptr[d][cnt_ji]);
            }

            // Multiply by geometric factors
            TData f1 = 0.5 * (1.0 + Z0[i]);
            outptr0[cnt_ji] += outptr1[cnt_ji] * f1;
            outptr0[cnt_ji] *= f0;
        }
    }
    delete[] inptr;
    delete[] dfptr;
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBaseHexKernel(
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t ncoord,
    const size_t nelmt, const size_t dfSize, TData *df, TData *in0, TData *in1,
    TData *in2, TData *out0, TData *out1, TData *out2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const auto ndf = 3 * ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * nq1 * nq2 * e);
    inptr[1]      = in1 + (nq0 * nq1 * nq2 * e);
    inptr[2]      = in2 + (nq0 * nq1 * nq2 * e);

    TData *outptr0 = out0 + (nq0 * nq1 * nq2 * e);
    TData *outptr1 = out1 + (nq0 * nq1 * nq2 * e);
    TData *outptr2 = out2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t i = 0; i < nq0 * nq1 * nq2; ++i)
    {
        size_t dfindex = DEFORMED ? i : 0;
        outptr0[i]     = dfptr[0][dfindex] * inptr[0][i];
        outptr1[i]     = dfptr[1][dfindex] * inptr[0][i];
        outptr2[i]     = dfptr[2][dfindex] * inptr[0][i];
        for (size_t d = 1; d < ncoord; ++d)
        {
            outptr0[i] += (dfptr[3 * d][dfindex] * inptr[d][i]);
            outptr1[i] += (dfptr[3 * d + 1][dfindex] * inptr[d][i]);
            outptr2[i] += (dfptr[3 * d + 2][dfindex] * inptr[d][i]);
        }
    }
    delete[] inptr;
    delete[] dfptr;
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBaseTetKernel(
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t ncoord,
    const size_t nelmt, const TData *Z0, const TData *Z1, const TData *Z2,
    const size_t dfSize, TData *df, TData *in0, TData *in1, TData *in2,
    TData *out0, TData *out1, TData *out2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const auto ndf = 3 * ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * nq1 * nq2 * e);
    inptr[1]      = in1 + (nq0 * nq1 * nq2 * e);
    inptr[2]      = in2 + (nq0 * nq1 * nq2 * e);

    TData *outptr0 = out0 + (nq0 * nq1 * nq2 * e);
    TData *outptr1 = out1 + (nq0 * nq1 * nq2 * e);
    TData *outptr2 = out2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t k = 0, cnt_kji = 0; k < nq2; ++k)
    {
        TData f2 = 2.0 / (1.0 - Z2[k]);
        for (size_t j = 0; j < nq1; ++j)
        {
            TData f3 = 0.5 * (1.0 + Z1[j]);
            TData f0 = 2.0 * f2 / (1.0 - Z1[j]);
            for (size_t i = 0; i < nq0; ++i, ++cnt_kji)
            {
                size_t dfindex   = DEFORMED ? cnt_kji : 0;
                outptr0[cnt_kji] = dfptr[0][dfindex] * inptr[0][cnt_kji];
                outptr1[cnt_kji] = dfptr[1][dfindex] * inptr[0][cnt_kji];
                outptr2[cnt_kji] = dfptr[2][dfindex] * inptr[0][cnt_kji];
                for (size_t d = 1; d < ncoord; ++d)
                {
                    outptr0[cnt_kji] +=
                        (dfptr[3 * d][dfindex] * inptr[d][cnt_kji]);
                    outptr1[cnt_kji] +=
                        (dfptr[3 * d + 1][dfindex] * inptr[d][cnt_kji]);
                    outptr2[cnt_kji] +=
                        (dfptr[3 * d + 2][dfindex] * inptr[d][cnt_kji]);
                }

                // Multiply by geometric factors
                TData f1 = 0.5 * (1.0 + Z0[i]);
                outptr0[cnt_kji] += (outptr1[cnt_kji] + outptr2[cnt_kji]) * f1;
                outptr0[cnt_kji] *= f0;
                outptr1[cnt_kji] += outptr2[cnt_kji] * f3;
                outptr1[cnt_kji] *= f2;
            }
        }
    }
    delete[] inptr;
    delete[] dfptr;
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBasePrismKernel(
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t ncoord,
    const size_t nelmt, const TData *Z0, const TData *Z2, const size_t dfSize,
    TData *df, TData *in0, TData *in1, TData *in2, TData *out0, TData *out1,
    TData *out2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const auto ndf = 3 * ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * nq1 * nq2 * e);
    inptr[1]      = in1 + (nq0 * nq1 * nq2 * e);
    inptr[2]      = in2 + (nq0 * nq1 * nq2 * e);

    TData *outptr0 = out0 + (nq0 * nq1 * nq2 * e);
    TData *outptr1 = out1 + (nq0 * nq1 * nq2 * e);
    TData *outptr2 = out2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t k = 0, cnt_kji = 0; k < nq2; ++k)
    {
        TData f0 = 2.0 / (1.0 - Z2[k]);
        for (size_t j = 0; j < nq1; ++j)
        {
            for (size_t i = 0; i < nq0; ++i, ++cnt_kji)
            {
                size_t dfindex   = DEFORMED ? cnt_kji : 0;
                outptr0[cnt_kji] = dfptr[0][dfindex] * inptr[0][cnt_kji];
                outptr1[cnt_kji] = dfptr[1][dfindex] * inptr[0][cnt_kji];
                outptr2[cnt_kji] = dfptr[2][dfindex] * inptr[0][cnt_kji];
                for (size_t d = 1; d < ncoord; ++d)
                {
                    outptr0[cnt_kji] +=
                        (dfptr[3 * d][dfindex] * inptr[d][cnt_kji]);
                    outptr1[cnt_kji] +=
                        (dfptr[3 * d + 1][dfindex] * inptr[d][cnt_kji]);
                    outptr2[cnt_kji] +=
                        (dfptr[3 * d + 2][dfindex] * inptr[d][cnt_kji]);
                }

                // Multiply by geometric factors
                TData f1 = 0.5 * (1.0 + Z0[i]);
                outptr0[cnt_kji] += outptr2[cnt_kji] * f1;
                outptr0[cnt_kji] *= f0;
            }
        }
    }
    delete[] inptr;
    delete[] dfptr;
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBasePyrKernel(
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t ncoord,
    const size_t nelmt, const TData *Z0, const TData *Z1, const TData *Z2,
    const size_t dfSize, TData *df, TData *in0, TData *in1, TData *in2,
    TData *out0, TData *out1, TData *out2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    const auto ndf = 3 * ncoord;

    // Assign pointers.
    TData **inptr = new TData *[ncoord];
    inptr[0]      = in0 + (nq0 * nq1 * nq2 * e);
    inptr[1]      = in1 + (nq0 * nq1 * nq2 * e);
    inptr[2]      = in2 + (nq0 * nq1 * nq2 * e);

    TData *outptr0 = out0 + (nq0 * nq1 * nq2 * e);
    TData *outptr1 = out1 + (nq0 * nq1 * nq2 * e);
    TData *outptr2 = out2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    for (size_t k = 0, cnt_kji = 0; k < nq2; ++k)
    {
        TData f0 = 2.0 / (1.0 - Z2[k]);
        for (size_t j = 0; j < nq1; ++j)
        {
            TData f2 = 0.5 * (1.0 + Z1[j]);
            for (size_t i = 0; i < nq0; ++i, ++cnt_kji)
            {
                size_t dfindex   = DEFORMED ? cnt_kji : 0;
                outptr0[cnt_kji] = dfptr[0][dfindex] * inptr[0][cnt_kji];
                outptr1[cnt_kji] = dfptr[1][dfindex] * inptr[0][cnt_kji];
                outptr2[cnt_kji] = dfptr[2][dfindex] * inptr[0][cnt_kji];
                for (size_t d = 1; d < ncoord; ++d)
                {
                    outptr0[cnt_kji] +=
                        (dfptr[3 * d][dfindex] * inptr[d][cnt_kji]);
                    outptr1[cnt_kji] +=
                        (dfptr[3 * d + 1][dfindex] * inptr[d][cnt_kji]);
                    outptr2[cnt_kji] +=
                        (dfptr[3 * d + 2][dfindex] * inptr[d][cnt_kji]);
                }

                // Multiply by geometric factors
                TData f1 = 0.5 * (1.0 + Z0[i]);
                outptr0[cnt_kji] += outptr2[cnt_kji] * f1;
                outptr0[cnt_kji] *= f0;
                outptr1[cnt_kji] += outptr2[cnt_kji] * f2;
                outptr1[cnt_kji] *= f0;
            }
        }
    }
    delete[] inptr;
    delete[] dfptr;
}
} // namespace Nektar::Operators::detail
