///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivCUDASumFacKernels.cuh
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

#pragma once

#if defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename TData>
__global__ void PhysDeriv1DKernel(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ D0, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int i = 0u; i < nq0; ++i)
        {
            const unsigned int index =
                nq0 * warpsize * iwarp + warpsize * i + ilane;
            const unsigned int dfindex =
                DEFORMED ? nq0 * ncoord * warpsize * iwarp +
                               warpsize * i * ncoord + ilane
                         : ncoord * warpsize * iwarp + ilane;

            // Compute tensorial derivative.
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] *
                      in[nq0 * warpsize * iwarp + warpsize * q + ilane];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nq0 + index] = d0 * df[d * warpsize + dfindex];
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <bool DEFORMED, typename TData>
__global__ void PhysDeriv1DKernel_QP(
    const unsigned int nq0, const unsigned int ncoord, const unsigned int nelmt,
    const TData *__restrict__ D0, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nq0 : 1;
        const unsigned int dfoffset = ncoord * dfsize * e;
        const unsigned int offset   = nq0 * e;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            const unsigned int index   = offset + i;
            const unsigned int dfindex = DEFORMED ? dfoffset + i : dfoffset;

            // Compute tensorial derivative.
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[offset + q];
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nq0 + index] = d0 * df[d * dfsize + dfindex];
            }
        }

        e += gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, typename TData>
__global__ void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    TData *s_xfrm0, *s_xfrm1;

    // Precompute geometric factors.
    if constexpr (SHAPETYPE == LibUtilities::Tri)
    {
        s_xfrm0 = (TData *)shared;
        s_xfrm1 = s_xfrm0 + nq1;

        for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
        {
            s_xfrm0[idx] = 2.0 / (1.0 - Z1[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
        {
            s_xfrm1[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
        {
            for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
            {
                const unsigned int index =
                    nqTot * warpsize * iwarp + warpsize * cnt_ji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                   warpsize * cnt_ji * ndf + ilane
                             : ndf * warpsize * iwarp + ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 +=
                        D0[q * nq0 + i] * in[nqTot * warpsize * iwarp +
                                             warpsize * (nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 +=
                        D1[q * nq1 + j] * in[nqTot * warpsize * iwarp +
                                             warpsize * (nq0 * q + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPETYPE == LibUtilities::Tri)
                {
                    d0 *= s_xfrm0[j];
                    d1 += d0 * s_xfrm1[i];
                }

                // Multiply by derivative factors.
                for (unsigned int d = 0u; d < ncoord; d++)
                {
                    out[d * nelmt * nqTot + index] =
                        d0 * df[(2u * d) * warpsize + dfindex] +
                        d1 * df[(2u * d + 1u) * warpsize + dfindex];
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, typename TData>
__global__ void PhysDeriv2DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ Z0,
    const TData *__restrict__ Z1, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    const unsigned int ndf   = 2 * ncoord;
    const unsigned int nqTot = nq0 * nq1;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = idx / nq0;
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[offset + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[offset + nq0 * q + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tri)
            {
                TData xfrm0 = 2.0 / (1.0 - Z1[j]);
                TData xfrm1 = 0.5 * (1.0 + Z0[i]);
                d0 *= xfrm0;
                d1 += d0 * xfrm1;
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nqTot + index] =
                    d0 * df[(2u * d) * dfsize + dfindex] +
                    d1 * df[(2u * d + 1u) * dfsize + dfindex];
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, typename TData>
__global__ void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ Z2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    extern __shared__ __align__(sizeof(TData)) unsigned char shared[];

    constexpr unsigned int warpsize = NektarSpaces::vector_width<TData>::value;

    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    TData *s_xfrm_eta0, *s_xfrm_eta1, *s_xfrm_eta1m, *s_xfrm_eta2;

    // Precompute geometric factors.
    if constexpr (SHAPETYPE == LibUtilities::Tet)
    {
        s_xfrm_eta0  = (TData *)shared;
        s_xfrm_eta1  = s_xfrm_eta0 + nq0;
        s_xfrm_eta1m = s_xfrm_eta1 + nq1;
        s_xfrm_eta2  = s_xfrm_eta1m + nq1;

        for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
        {
            s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
        {
            s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
        {
            s_xfrm_eta1m[idx] = 2.0 / (1.0 - Z1[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq2; idx += blockDim.x)
        {
            s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Prism)
    {
        s_xfrm_eta0 = (TData *)shared;
        s_xfrm_eta2 = s_xfrm_eta0 + nq0;

        for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
        {
            s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq2; idx += blockDim.x)
        {
            s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        __syncthreads();
    }
    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
    {
        s_xfrm_eta0 = (TData *)shared;
        s_xfrm_eta1 = s_xfrm_eta0 + nq0;
        s_xfrm_eta2 = s_xfrm_eta1 + nq1;

        for (unsigned int idx = threadIdx.x; idx < nq0; idx += blockDim.x)
        {
            s_xfrm_eta0[idx] = 0.5 * (1.0 + Z0[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq1; idx += blockDim.x)
        {
            s_xfrm_eta1[idx] = 0.5 * (1.0 + Z1[idx]);
        }

        for (unsigned int idx = threadIdx.x; idx < nq2; idx += blockDim.x)
        {
            s_xfrm_eta2[idx] = 2.0 / (1.0 - Z2[idx]);
        }

        __syncthreads();
    }

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        const unsigned int iwarp = e / warpsize;
        const unsigned int ilane = e % warpsize;

        for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
        {
            for (unsigned int j = 0u; j < nq1; j++)
            {
                for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
                {
                    const unsigned int index =
                        nqTot * warpsize * iwarp + warpsize * cnt_kji + ilane;
                    const unsigned int dfindex =
                        DEFORMED ? nqTot * ndf * warpsize * iwarp +
                                       warpsize * cnt_kji * ndf + ilane
                                 : ndf * warpsize * iwarp + ilane;

                    // Compute tensorial derivative.
                    // Direction 0
                    TData d0 = 0.0;
                    for (unsigned int q = 0u; q < nq0; ++q)
                    {
                        d0 += D0[q * nq0 + i] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * k + nq0 * j + q) +
                                 ilane];
                    }

                    // Direction 1
                    TData d1 = 0.0;
                    for (unsigned int q = 0u; q < nq1; ++q)
                    {
                        d1 += D1[q * nq1 + j] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * k + nq0 * q + i) +
                                 ilane];
                    }

                    // Direction 2
                    TData d2 = 0.0;
                    for (unsigned int q = 0u; q < nq2; ++q)
                    {
                        d2 += D2[q * nq2 + k] *
                              in[nqTot * warpsize * iwarp +
                                 warpsize * (nq0 * nq1 * q + nq0 * j + i) +
                                 ilane];
                    }

                    // Moving from standard to collapsed coordinates.
                    if constexpr (SHAPETYPE == LibUtilities::Tet)
                    {
                        TData xfrm = s_xfrm_eta1m[j] * s_xfrm_eta2[k];
                        TData tmp0 = xfrm * d0;
                        TData tmp1 = s_xfrm_eta0[i] * tmp0;
                        TData tmp2 = s_xfrm_eta2[k] * d1;
                        d0         = tmp0;
                        d1         = tmp1 + tmp2;
                        d2 += tmp1 + s_xfrm_eta1[j] * tmp2;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Prism)
                    {
                        d0 *= s_xfrm_eta2[k];
                        d2 += s_xfrm_eta0[i] * d0;
                    }
                    else if constexpr (SHAPETYPE == LibUtilities::Pyr)
                    {
                        d0 *= s_xfrm_eta2[k];
                        d1 *= s_xfrm_eta2[k];
                        d2 += s_xfrm_eta0[i] * d0 + s_xfrm_eta1[j] * d1;
                    }

                    // Multiply by derivative factors.
                    for (unsigned int d = 0u; d < ncoord; d++)
                    {
                        out[d * nelmt * nqTot + index] =
                            d0 * df[(3u * d) * warpsize + dfindex] +
                            d1 * df[(3u * d + 1u) * warpsize + dfindex] +
                            d2 * df[(3u * d + 2u) * warpsize + dfindex];
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <LibUtilities::ShapeType SHAPETYPE, bool DEFORMED, typename TData>
__global__ void PhysDeriv3DKernel_QP(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *__restrict__ D0,
    const TData *__restrict__ D1, const TData *__restrict__ D2,
    const TData *__restrict__ Z0, const TData *__restrict__ Z1,
    const TData *__restrict__ Z2, const TData *__restrict__ df,
    const TData *__restrict__ in, TData *__restrict__ out)
{
    constexpr unsigned int ncoord = 3u;
    constexpr unsigned int ndf    = 9u;

    const unsigned int nqTot = nq0 * nq1 * nq2;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const unsigned int dfsize   = DEFORMED ? nqTot : 1;
        const unsigned int dfoffset = ndf * dfsize * e;
        const unsigned int offset   = nqTot * e;

        for (unsigned int idx = threadIdx.x; idx < nqTot; idx += blockDim.x)
        {
            const unsigned int i       = idx % nq0;
            const unsigned int j       = (idx / nq0) % nq1;
            const unsigned int k       = idx / (nq0 * nq1);
            const unsigned int index   = offset + idx;
            const unsigned int dfindex = DEFORMED ? dfoffset + idx : dfoffset;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 +=
                    D0[q * nq0 + i] * in[offset + nq0 * nq1 * k + nq0 * j + q];
            }

            // Direction 1
            TData d1 = 0.0;
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 +=
                    D1[q * nq1 + j] * in[offset + nq0 * nq1 * k + nq0 * q + i];
            }

            // Direction 2
            TData d2 = 0.0;
            for (unsigned int q = 0u; q < nq2; ++q)
            {
                d2 +=
                    D2[q * nq2 + k] * in[offset + nq0 * nq1 * q + nq0 * j + i];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPETYPE == LibUtilities::Tet)
            {
                TData xfrm_eta0  = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta1  = 0.5 * (1.0 + Z1[j]);
                TData xfrm_eta1m = 2.0 / (1.0 - Z1[j]);
                TData xfrm_eta2  = 2.0 / (1.0 - Z2[k]);

                TData xfrm = xfrm_eta1m * xfrm_eta2;
                TData tmp0 = xfrm * d0;
                TData tmp1 = xfrm_eta0 * tmp0;
                TData tmp2 = xfrm_eta2 * d1;
                d0         = tmp0;
                d1         = tmp1 + tmp2;
                d2 += tmp1 + xfrm_eta1 * tmp2;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Prism)
            {
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0;
            }
            else if constexpr (SHAPETYPE == LibUtilities::Pyr)
            {
                TData xfrm_eta0 = 0.5 * (1.0 + Z0[i]);
                TData xfrm_eta1 = 0.5 * (1.0 + Z1[j]);
                TData xfrm_eta2 = 2.0 / (1.0 - Z2[k]);
                d0 *= xfrm_eta2;
                d1 *= xfrm_eta2;
                d2 += xfrm_eta0 * d0 + xfrm_eta1 * d1;
            }

            // Multiply by derivative factors.
            for (unsigned int d = 0u; d < ncoord; d++)
            {
                out[d * nelmt * nqTot + index] =
                    d0 * df[(3u * d) * dfsize + dfindex] +
                    d1 * df[(3u * d + 1u) * dfsize + dfindex] +
                    d2 * df[(3u * d + 2u) * dfsize + dfindex];
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

// Launchers
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int nq0,
                                               const unsigned int ncoord,
                                               const unsigned int nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    if constexpr (MULTILEVEL)
    {
        const unsigned int blocksize =
            std::min(nq0, NektarSpaces::CUDA::defaultBlockSize);
        const unsigned int gridsize = std::min(nelmt, 2147483647u);

        PhysDeriv1DKernel_QP<DEFORMED>
            <<<gridsize, blocksize>>>(nq0, ncoord, nelmt, D0, df, in, out);
    }
    else
    {
        const unsigned int blocksize = NektarSpaces::CUDA::defaultBlockSize;
        const unsigned int gridsize =
            std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);

        PhysDeriv1DKernel<DEFORMED>
            <<<gridsize, blocksize>>>(nq0, ncoord, nelmt, D0, df, in, out);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int ncoord,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *Z0,
    const TData *Z1, const TData *df, const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    if constexpr (MULTILEVEL)
    {
        const unsigned int blocksize =
            std::min(nq0 * nq1, NektarSpaces::CUDA::defaultBlockSize);
        const unsigned int gridsize = std::min(nelmt, 2147483647u);

        PhysDeriv2DKernel_QP<SHAPE_TYPE, DEFORMED><<<gridsize, blocksize>>>(
            nq0, nq1, ncoord, nelmt, D0, D1, Z0, Z1, df, in, out);
    }
    else
    {
        const unsigned int shmemsize =
            sizeof(TData) * PhysDerivSharedMemorySize<SHAPE_TYPE>(nq0, nq1);
        const unsigned int blocksize = NektarSpaces::CUDA::defaultBlockSize;
        const unsigned int gridsize =
            std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);

        PhysDeriv2DKernel<SHAPE_TYPE, DEFORMED>
            <<<gridsize, blocksize, shmemsize>>>(nq0, nq1, ncoord, nelmt, D0,
                                                 D1, Z0, Z1, df, in, out);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *Z0, const TData *Z1, const TData *Z2, const TData *df,
    const TData *in, TData *out)
{
    constexpr bool MULTILEVEL =
        std::is_same_v<Implementation, Operators::SumFacQP>;

    if constexpr (MULTILEVEL)
    {
        const unsigned int blocksize =
            std::min(nq0 * nq1 * nq2, NektarSpaces::CUDA::defaultBlockSize);
        const unsigned int gridsize = std::min(nelmt, 2147483647u);

        PhysDeriv3DKernel_QP<SHAPE_TYPE, DEFORMED><<<gridsize, blocksize>>>(
            nq0, nq1, nq2, nelmt, D0, D1, D2, Z0, Z1, Z2, df, in, out);
    }
    else
    {
        const unsigned int shmemsize =
            sizeof(TData) *
            PhysDerivSharedMemorySize<SHAPE_TYPE>(nq0, nq1, nq2);
        const unsigned int blocksize = NektarSpaces::CUDA::defaultBlockSize;
        const unsigned int gridsize =
            std::min((nelmt + blocksize - 1u) / blocksize, 2147483647u);

        PhysDeriv3DKernel<SHAPE_TYPE, DEFORMED>
            <<<gridsize, blocksize, shmemsize>>>(nq0, nq1, nq2, nelmt, D0, D1,
                                                 D2, Z0, Z1, Z2, df, in, out);
    }
}

} // namespace Nektar::Operators::detail

#endif
