#pragma once

namespace Nektar::Operators::detail
{

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBase1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nSize, const unsigned int dfSize,
    const TData *__restrict df, const TData *__restrict in,
    TData *__restrict out)
{
    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * e;

        for (unsigned int i = 0; i < nq0; ++i)
        {
            unsigned int index   = offset + i;
            unsigned int dfindex = DEFORMED ? index : e;

            TData sum = 0.0;
            for (unsigned int d = 0; d < ncoord; ++d)
            {
                sum += df[dfindex + d * dfSize] * in[index + d * nSize];
            }
            out[index] = sum;
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, bool DEFORMED>
__global__ void IProductWRTDerivBase1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nSize, const unsigned int dfSize,
    const TData *__restrict df, const TData *__restrict in,
    TData *__restrict out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * e;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            unsigned int index   = offset + i;
            unsigned int dfindex = DEFORMED ? index : e;

            TData sum = 0.0;
            for (unsigned int d = 0; d < ncoord; ++d)
            {
                sum += df[dfindex + d * dfSize] * in[index + d * nSize];
            }
            out[index] = sum;
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nSize,
    const unsigned int dfSize, const TData *__restrict Z0,
    const TData *__restrict Z1, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out)
{
    extern __shared__ TData shared[];
    TData *s_f0, *s_f1;

    // Copy to shared memory.
    if constexpr (SHAPETYPE == LibUtilities::Tri)
    {
        s_f0 = shared;
        s_f1 = s_f0 + nq1;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_f0[sIndex] = 2.0 / (1.0 - Z1[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_f1[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * e;

        for (unsigned int j = 0, cnt_ji = 0; j < nq1; ++j)
        {
            for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
            {
                unsigned int index   = offset + cnt_ji;
                unsigned int dfindex = DEFORMED ? index : e;

                TData sum1 = 0.0, sum2 = 0.0;
                for (unsigned int d = 0; d < ncoord; ++d)
                {
                    sum1 += (df[dfindex + (2 * d) * dfSize] *
                             in[index + d * nSize]);
                    sum2 += (df[dfindex + (2 * d + 1) * dfSize] *
                             in[index + d * nSize]);
                }

                if constexpr (SHAPETYPE == LibUtilities::Quad)
                {
                    out[index]         = sum1;
                    out[index + nSize] = sum2;
                }
                else if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    out[index]         = (sum1 + sum2 * s_f1[i]) * s_f0[j];
                    out[index + nSize] = sum2;
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const unsigned int nSize,
    const unsigned int dfSize, const TData *__restrict Z0,
    const TData *__restrict Z1, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out)
{
    TData f0, f1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * e;

        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                f0 = 2.0 / (1.0 - Z1[j]);
            }

            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                unsigned int cnt_ji  = nq0 * j + i;
                unsigned int index   = offset + cnt_ji;
                unsigned int dfindex = DEFORMED ? index : e;

                TData sum1 = 0.0, sum2 = 0.0;
                for (unsigned int d = 0; d < ncoord; ++d)
                {
                    sum1 += (df[dfindex + (2 * d) * dfSize] *
                             in[index + d * nSize]);
                    sum2 += (df[dfindex + (2 * d + 1) * dfSize] *
                             in[index + d * nSize]);
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    f1 = 0.5 * (1.0 + Z0[i]);
                }

                if constexpr (SHAPETYPE == LibUtilities::Quad)
                {
                    out[index]         = sum1;
                    out[index + nSize] = sum2;
                }
                else if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    out[index]         = (sum1 + sum2 * f1) * f0;
                    out[index + nSize] = sum2;
                }
            }
        }

        e += gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nSize, const unsigned int dfSize,
    const TData *__restrict Z0, const TData *__restrict Z1,
    const TData *__restrict Z2, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out)
{
    extern __shared__ TData shared[];
    TData *s_f0, *s_f1, *s_f2, *s_f3;

    // Copy to shared memory.
    if constexpr (SHAPETYPE == LibUtilities::Tet)
    {
        s_f0 = shared;
        s_f1 = s_f0 + nq1;
        s_f2 = s_f1 + nq0;
        s_f3 = s_f2 + nq2;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_f0[sIndex] = 2.0 / (1.0 - Z1[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_f1[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq2)
        {
            s_f2[sIndex] = 2.0 / (1.0 - Z2[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_f3[sIndex] = 0.5 * (1.0 + Z1[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Prism)
    {
        s_f1 = shared;
        s_f2 = s_f1 + nq0;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_f1[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq2)
        {
            s_f2[sIndex] = 2.0 / (1.0 - Z2[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
    {
        s_f1 = shared;
        s_f2 = s_f1 + nq0;
        s_f3 = s_f2 + nq2;

        unsigned int sIndex = threadIdx.x;
        while (sIndex < nq0)
        {
            s_f1[sIndex] = 0.5 * (1.0 + Z0[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq2)
        {
            s_f2[sIndex] = 2.0 / (1.0 - Z2[sIndex]);
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nq1)
        {
            s_f3[sIndex] = 0.5 * (1.0 + Z1[sIndex]);
            sIndex += blockDim.x;
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * nq2 * e;

        for (unsigned int k = 0, cnt_kji = 0; k < nq2; ++k)
        {
            for (unsigned int j = 0; j < nq1; ++j)
            {
                for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
                {
                    unsigned int index   = offset + cnt_kji;
                    unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0; d < ncoord; ++d)
                    {
                        sum1 += (df[dfindex + (3 * d) * dfSize] *
                                 in[index + d * nSize]);
                        sum2 += (df[dfindex + (3 * d + 1) * dfSize] *
                                 in[index + d * nSize]);
                        sum3 += (df[dfindex + (3 * d + 2) * dfSize] *
                                 in[index + d * nSize]);
                    }

                    if constexpr (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]             = sum1;
                        out[index + nSize]     = sum2;
                        out[index + 2 * nSize] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index] = (sum1 + (sum2 + sum3) * s_f1[i]) *
                                     s_f2[k] * s_f0[j];
                        out[index + nSize] = (sum2 + sum3 * s_f3[j]) * s_f2[k];
                        out[index + 2 * nSize] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        out[index]         = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                        out[index + nSize] = sum2;
                        out[index + 2 * nSize] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        out[index]         = (sum1 + sum3 * s_f1[i]) * s_f2[k];
                        out[index + nSize] = (sum2 + sum3 * s_f3[j]) * s_f2[k];
                        out[index + 2 * nSize] = sum3;
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename TData, LibUtilities::ShapeType SHAPETYPE, bool DEFORMED>
__global__ void IProductWRTDerivBase3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int ncoord, const unsigned int nelmt,
    const unsigned int nSize, const unsigned int dfSize,
    const TData *__restrict Z0, const TData *__restrict Z1,
    const TData *__restrict Z2, const TData *__restrict df,
    const TData *__restrict in, TData *__restrict out)
{
    TData f0, f1, f2, f3;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        unsigned int offset = nq0 * nq1 * nq2 * e;

        for (unsigned int k = threadIdx.z; k < nq2; k += blockDim.z)
        {
            if constexpr (SHAPETYPE == LibUtilities::Tet ||
                          SHAPETYPE == LibUtilities::Prism ||
                          SHAPETYPE == LibUtilities::Pyr)
            {
                f2 = 2.0 / (1.0 - Z2[k]);
            }

            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                if constexpr (SHAPETYPE == LibUtilities::Tet ||
                              SHAPETYPE == LibUtilities::Pyr)
                {
                    f3 = 0.5 * (1.0 + Z1[j]);
                }
                else if constexpr (SHAPETYPE == LibUtilities::Tet)
                {
                    f0 = 2.0 * f2 / (1.0 - Z1[j]);
                }

                for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
                {
                    unsigned int index   = offset + nq0 * nq1 * k + nq0 * j + i;
                    unsigned int dfindex = DEFORMED ? index : e;

                    TData sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (unsigned int d = 0; d < ncoord; ++d)
                    {
                        sum1 += (df[dfindex + (3 * d) * dfSize] *
                                 in[index + d * nSize]);
                        sum2 += (df[dfindex + (3 * d + 1) * dfSize] *
                                 in[index + d * nSize]);
                        sum3 += (df[dfindex + (3 * d + 2) * dfSize] *
                                 in[index + d * nSize]);
                    }

                    if constexpr (SHAPETYPE == LibUtilities::Tet ||
                                  SHAPETYPE == LibUtilities::Prism ||
                                  SHAPETYPE == LibUtilities::Pyr)
                    {
                        f1 = 0.5 * (1.0 + Z0[i]);
                    }

                    if constexpr (SHAPETYPE == LibUtilities::Hex)
                    {
                        out[index]             = sum1;
                        out[index + nSize]     = sum2;
                        out[index + 2 * nSize] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        out[index] = (sum1 + (sum2 + sum3) * f1) * f2 * f0;
                        out[index + nSize]     = (sum2 + sum3 * f3) * f2;
                        out[index + 2 * nSize] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        out[index]             = (sum1 + sum3 * f1) * f2;
                        out[index + nSize]     = sum2;
                        out[index + 2 * nSize] = sum3;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        out[index]             = (sum1 + sum3 * f1) * f2;
                        out[index + nSize]     = (sum2 + sum3 * f3) * f2;
                        out[index + 2 * nSize] = sum3;
                    }
                }
            }
        }

        e += gridDim.x;
    }
}

} // namespace Nektar::Operators::detail
