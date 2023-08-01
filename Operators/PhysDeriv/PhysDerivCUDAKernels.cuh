namespace Nektar::Operators::detail
{
template <typename TData, bool DEFORMED>
__global__ void PhysDerivSegKernel(const size_t nq0, const size_t ncoord,
                                   const size_t nelmt, const size_t dfSize,
                                   TData *df, TData *inout0, TData *inout1,
                                   TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t ndf = ncoord;

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * e);
    if (ncoord > 1)
    {
        inoutptr[1] = inout1 + (nq0 * e);
    }
    if (ncoord > 2)
    {
        inoutptr[2] = inout2 + (nq0 * e);
    }

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * e : e;
    }

    // Compute derivative.
    for (size_t j = 0; j < nq0; ++j)
    {
        size_t dfindex = DEFORMED ? j : 0;
        for (size_t d = ndf; d > 0; d--)
        {
            inoutptr[d - 1][j] = inoutptr[0][j] * dfptr[d - 1][dfindex];
        }
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDerivQuadKernel(const size_t nq0, const size_t nq1,
                                    const size_t ncoord, const size_t nelmt,
                                    const size_t dfSize, TData *df,
                                    TData *inout0, TData *inout1, TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    auto ndf = 2 * ncoord;

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * nq1 * e);
    inoutptr[1]      = inout1 + (nq0 * nq1 * e);
    if (ncoord > 2)
    {
        inoutptr[2] = inout2 + (nq0 * nq1 * e);
    }

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * e : e;
    }

    // Compute derivative.
    for (size_t j = 0, cnt_ji = 0; j < nq1; ++j)
    {
        for (size_t i = 0; i < nq0; ++i, ++cnt_ji)
        {
            TData d0 = inoutptr[0][cnt_ji];
            TData d1 = inoutptr[1][cnt_ji];

            size_t dfindex = DEFORMED ? cnt_ji : 0;
            for (size_t d = 0; d < ncoord; d++)
            {
                inoutptr[d][cnt_ji] =
                    d0 * dfptr[2 * d][dfindex] + d1 * dfptr[2 * d + 1][dfindex];
            }
        }
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDerivTriKernel(const size_t nq0, const size_t nq1,
                                   const size_t ncoord, const size_t nelmt,
                                   const TData *Z0, const TData *Z1,
                                   const size_t dfSize, TData *df,
                                   TData *inout0, TData *inout1, TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    auto ndf = 2 * ncoord;

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * nq1 * e);
    inoutptr[1]      = inout1 + (nq0 * nq1 * e);
    if (ncoord > 2)
    {
        inoutptr[2] = inout2 + (nq0 * nq1 * e);
    }

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * e : e;
    }

    // Compute derivative.
    for (int j = 0, cnt_ji = 0; j < nq1; ++j)
    {
        TData xfrm0 = 2.0 / (1.0 - Z1[j]);
        for (int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            TData d0 = inoutptr[0][cnt_ji];
            TData d1 = inoutptr[1][cnt_ji];

            // Moving from standard to collapsed coordinates.
            TData xfrm1 = 0.5 * (1.0 + Z0[i]);
            d0 *= xfrm0;
            d1 += d0 * xfrm1;

            // Multiply by derivative factors.
            size_t dfindex = DEFORMED ? cnt_ji : 0;
            for (size_t d = 0; d < ncoord; d++)
            {
                inoutptr[d][cnt_ji] =
                    d0 * dfptr[2 * d][dfindex] + d1 * dfptr[2 * d + 1][dfindex];
            }
        }
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDerivHexKernel(const size_t nq0, const size_t nq1,
                                   const size_t nq2, const size_t ncoord,
                                   const size_t nelmt, const size_t dfSize,
                                   TData *df, TData *inout0, TData *inout1,
                                   TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t ndf = 3 * ncoord;

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * nq1 * nq2 * e);
    inoutptr[1]      = inout1 + (nq0 * nq1 * nq2 * e);
    inoutptr[2]      = inout2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Compute derivative.
    for (size_t k = 0, cnt_ijk = 0; k < nq2; ++k)
    {
        for (size_t j = 0; j < nq1; ++j)
        {
            for (size_t i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                TData tmp[] = {inoutptr[0][cnt_ijk], inoutptr[1][cnt_ijk],
                               inoutptr[2][cnt_ijk]};

                // Multiply by derivative factors.
                size_t dfindex = DEFORMED ? cnt_ijk : 0;
                for (size_t d = 0; d < ncoord; ++d)
                {
                    TData sum = 0.0;
                    for (size_t n = 0; n < ncoord; ++n)
                    {
                        sum += tmp[n] * dfptr[d * ncoord + n][dfindex];
                    }
                    inoutptr[d][cnt_ijk] = sum;
                }
            }
        }
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDerivTetKernel(const size_t nq0, const size_t nq1,
                                   const size_t nq2, const size_t ncoord,
                                   const size_t nelmt, const TData *Z0,
                                   const TData *Z1, const TData *Z2,
                                   const size_t dfSize, TData *df,
                                   TData *inout0, TData *inout1, TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t ndf = 3 * ncoord;

    // Allocate workspace memory.
    TData *wsp0 = new TData[nq0 * nq1 * nq2];
    TData *wsp1 = new TData[nq0 * nq1 * nq2];

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * nq1 * nq2 * e);
    inoutptr[1]      = inout1 + (nq0 * nq1 * nq2 * e);
    inoutptr[2]      = inout2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Moving from standard to collapsed coordinates.
    for (int k = 0, eta0 = 0; k < nq2; ++k)
    {
        TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
        for (int j = 0; j < nq1; ++j)
        {
            TData xfrm_eta1 = 2.0 / (1.0 - Z1[j]);
            TData xfrm      = xfrm_eta1 * xfrm_eta2;
            for (int i = 0; i < nq0; ++i, ++eta0)
            {
                inoutptr[0][eta0] *= xfrm;
                wsp0[eta0] = inoutptr[0][eta0];
            }
        }
    }

    for (int k = 0, eta0 = 0; k < nq2; ++k)
    {
        TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
        for (int j = 0; j < nq1; ++j)
        {
            for (int i = 0; i < nq0; ++i, ++eta0)
            {
                TData xfrm_eta0   = 0.5 * (1.0 + Z0[i]);
                wsp0[eta0]        = xfrm_eta0 * wsp0[eta0];
                wsp1[eta0]        = xfrm_eta2 * inoutptr[1][eta0];
                inoutptr[1][eta0] = wsp0[eta0] + wsp1[eta0];
            }
        }
    }

    for (int k = 0, eta0 = 0; k < nq2; ++k)
    {
        for (int j = 0; j < nq1; ++j)
        {
            TData xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
            for (int i = 0; i < nq0; ++i, ++eta0)
            {
                inoutptr[2][eta0] += wsp0[eta0] + wsp1[eta0] * xfrm_eta1;
            }
        }
    }

    // Compute derivative.
    for (size_t k = 0, cnt_ijk = 0; k < nq2; ++k)
    {
        for (size_t j = 0; j < nq1; ++j)
        {
            for (size_t i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                TData tmp[] = {inoutptr[0][cnt_ijk], inoutptr[1][cnt_ijk],
                               inoutptr[2][cnt_ijk]};

                // Multiply by derivative factors.
                size_t dfindex = DEFORMED ? cnt_ijk : 0;
                for (size_t d = 0; d < ncoord; ++d)
                {
                    TData sum = 0.0;
                    for (size_t n = 0; n < ncoord; ++n)
                    {
                        sum += tmp[n] * dfptr[d * ncoord + n][dfindex];
                    }
                    inoutptr[d][cnt_ijk] = sum;
                }
            }
        }
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDerivPrismKernel(
    const size_t nq0, const size_t nq1, const size_t nq2, const size_t ncoord,
    const size_t nelmt, const TData *Z0, const TData *Z1, const TData *Z2,
    const size_t dfSize, TData *df, TData *inout0, TData *inout1, TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t ndf = 3 * ncoord;

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * nq1 * nq2 * e);
    inoutptr[1]      = inout1 + (nq0 * nq1 * nq2 * e);
    inoutptr[2]      = inout2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Compute derivative.
    for (size_t k = 0, cnt_ijk = 0; k < nq2; ++k)
    {
        TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
        for (size_t j = 0; j < nq1; ++j)
        {
            for (size_t i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                TData d0 = inoutptr[0][cnt_ijk];
                TData d1 = inoutptr[1][cnt_ijk];
                TData d2 = inoutptr[2][cnt_ijk];

                // Chain-rule for eta_0 and eta_2.
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                d0 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0;

                // Multiply by derivative factors.
                size_t dfindex       = DEFORMED ? cnt_ijk : 0;
                inoutptr[0][cnt_ijk] = d0 * dfptr[0][dfindex] +
                                       d1 * dfptr[1][dfindex] +
                                       d2 * dfptr[2][dfindex];
                inoutptr[1][cnt_ijk] = d0 * dfptr[3][dfindex] +
                                       d1 * dfptr[4][dfindex] +
                                       d2 * dfptr[5][dfindex];
                inoutptr[2][cnt_ijk] = d0 * dfptr[6][dfindex] +
                                       d1 * dfptr[7][dfindex] +
                                       d2 * dfptr[8][dfindex];
            }
        }
    }
}

template <typename TData, bool DEFORMED>
__global__ void PhysDerivPyrKernel(const size_t nq0, const size_t nq1,
                                   const size_t nq2, const size_t ncoord,
                                   const size_t nelmt, const TData *Z0,
                                   const TData *Z1, const TData *Z2,
                                   const size_t dfSize, TData *df,
                                   TData *inout0, TData *inout1, TData *inout2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    size_t ndf = 3 * ncoord;

    // Assign pointers.
    TData **inoutptr = new TData *[ncoord];
    inoutptr[0]      = inout0 + (nq0 * nq1 * nq2 * e);
    inoutptr[1]      = inout1 + (nq0 * nq1 * nq2 * e);
    inoutptr[2]      = inout2 + (nq0 * nq1 * nq2 * e);

    TData **dfptr = new TData *[ndf];
    for (size_t d = 0; d < ndf; d++)
    {
        dfptr[d] = df + d * dfSize;
        dfptr[d] += DEFORMED ? nq0 * nq1 * nq2 * e : e;
    }

    // Compute derivative.
    for (size_t k = 0, cnt_ijk = 0; k < nq2; ++k)
    {
        TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
        for (size_t j = 0; j < nq1; ++j)
        {
            TData xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
            for (size_t i = 0; i < nq0; ++i, ++cnt_ijk)
            {
                TData d0 = inoutptr[0][cnt_ijk];
                TData d1 = inoutptr[1][cnt_ijk];
                TData d2 = inoutptr[2][cnt_ijk];

                // Chain-rule for eta_0 and eta_2.
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                d0 *= xfrm_eta2;
                d1 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;

                // Multiply by derivative factors.
                size_t dfindex       = DEFORMED ? cnt_ijk : 0;
                inoutptr[0][cnt_ijk] = d0 * dfptr[0][dfindex] +
                                       d1 * dfptr[1][dfindex] +
                                       d2 * dfptr[2][dfindex];
                inoutptr[1][cnt_ijk] = d0 * dfptr[3][dfindex] +
                                       d1 * dfptr[4][dfindex] +
                                       d2 * dfptr[5][dfindex];
                inoutptr[2][cnt_ijk] = d0 * dfptr[6][dfindex] +
                                       d1 * dfptr[7][dfindex] +
                                       d2 * dfptr[8][dfindex];
            }
        }
    }
}

template <typename TData>
__global__ void PhysDerivTensor1DKernel(const size_t nq0, const size_t nelmt,
                                        const TData *D0, const TData *in,
                                        TData *out0)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Assign pointers.
    const TData *inptr = in + (nq0 * e);
    TData *outptr0     = out0 + (nq0 * e);

    // Direction 1
    for (size_t i = 0; i < nq0; ++i)
    {
        TData sum = 0.0;
        for (size_t k = 0; k < nq0; ++k)
        {
            sum += D0[k * nq0 + i] * inptr[k];
        }
        outptr0[i] = sum;
    }
}

template <typename TData>
__global__ void PhysDerivTensor2DKernel(const size_t nq0, const size_t nq1,
                                        const size_t nelmt, const TData *D0,
                                        const TData *D1, const TData *in,
                                        TData *out0, TData *out1)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Assign pointers.
    const TData *inptr = in + (nq0 * nq1 * e);
    TData *outptr0     = out0 + (nq0 * nq1 * e);
    TData *outptr1     = out1 + (nq0 * nq1 * e);

    // Direction 1
    for (size_t i = 0; i < nq0; ++i)
    {
        for (size_t j = 0; j < nq1; ++j)
        {
            TData sum = 0.0;
            for (size_t k = 0; k < nq0; ++k)
            {
                sum += D0[k * nq0 + i] * inptr[j * nq0 + k];
            }
            outptr0[j * nq0 + i] = sum;
        }
    }

    // Direction 2
    for (size_t i = 0; i < nq0; ++i)
    {
        for (size_t j = 0; j < nq1; ++j)
        {
            TData sum = 0.0;
            for (size_t k = 0; k < nq1; ++k)
            {
                sum += D1[k * nq1 + j] * inptr[k * nq0 + i];
            }
            outptr1[j * nq0 + i] = sum;
        }
    }
}

template <typename TData>
__global__ void PhysDerivTensor3DKernel(const size_t nq0, const size_t nq1,
                                        const size_t nq2, const size_t nelmt,
                                        const TData *D0, const TData *D1,
                                        const TData *D2, const TData *in,
                                        TData *out0, TData *out1, TData *out2)
{
    size_t e = blockDim.x * blockIdx.x + threadIdx.x;

    if (e >= nelmt)
    {
        return;
    }

    // Assign pointers.
    const TData *inptr = in + (nq0 * nq1 * nq2 * e);
    TData *outptr0     = out0 + (nq0 * nq1 * nq2 * e);
    TData *outptr1     = out1 + (nq0 * nq1 * nq2 * e);
    TData *outptr2     = out2 + (nq0 * nq1 * nq2 * e);

    // Direction 1
    for (size_t i = 0; i < nq0; ++i)
    {
        for (size_t j = 0; j < nq1 * nq2; ++j)
        {
            TData sum = 0.0;
            for (size_t k = 0; k < nq0; ++k)
            {
                sum += D0[k * nq0 + i] * inptr[j * nq0 + k];
            }
            outptr0[j * nq0 + i] = sum;
        }
    }

    // Direction 2
    for (size_t block = 0; block < nq2; ++block)
    {
        size_t start = block * nq0 * nq1;
        for (size_t i = 0; i < nq0; ++i)
        {
            for (size_t j = 0; j < nq1; ++j)
            {
                TData sum = 0.0;
                for (size_t k = 0; k < nq1; ++k)
                {
                    sum += D1[k * nq1 + j] * inptr[start + k * nq0 + i];
                }
                outptr1[start + j * nq0 + i] = sum;
            }
        }
    }

    // Direction 3
    for (size_t i = 0; i < nq0 * nq1; ++i)
    {
        for (size_t j = 0; j < nq2; ++j)
        {
            TData sum = 0.0;
            for (size_t k = 0; k < nq2; ++k)
            {
                sum += D2[k * nq2 + j] * inptr[k * nq0 * nq1 + i];
            }
            outptr2[j * nq0 * nq1 + i] = sum;
        }
    }
}
} // namespace Nektar::Operators::detail
