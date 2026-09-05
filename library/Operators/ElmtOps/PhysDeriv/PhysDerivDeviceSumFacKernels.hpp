///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFacKernels.hpp
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

#include <LibUtilities/Backends/Backends_Device_API.hpp>
#include <LibUtilities/Backends/DeviceProperties.hpp>
#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
// Helper function
template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter1D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter1D sizeParam1D)
{
    return 0;
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter2D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
    [[maybe_unused]] const TPhysSizeParameter2D sizeParam2D)
{
    if constexpr (SHAPE_TYPE == LibUtilities::Quad)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                       SHAPE_TYPE == LibUtilities::NodalTri)
    {
        const unsigned int nq0 = sizeParam2D.nq0();
        const unsigned int nq1 = sizeParam2D.nq1();

        return nq0 + nq1;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          typename TPhysSizeParameter3D,
          std::enable_if_t<std::is_same_v<Implementation, SumFac> &&
                               IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                           bool>
              Enable = true>
inline constexpr unsigned int PhysDerivSharedMemorySize(
    const TPhysSizeParameter3D sizeParam3D)
{
    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    if constexpr (SHAPE_TYPE == LibUtilities::Hex)
    {
        return 0;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                       SHAPE_TYPE == LibUtilities::NodalTet)
    {
        return nq0 + 2u * nq1 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        return nq0 + nq2;
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        return nq0 + nq1 + nq2;
    }
}

template <bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[warpsize * q + ilane];
        }

        // Multiply by derivative factors.
        for (unsigned int k = 0u; k < ncoord; k++)
        {
            out[k * outoffset + index] = d0 * df[k * warpsize + dfindex];
        }
    }
}

template <bool APPEND, bool DEFORMED, unsigned int DIR, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir1DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        const unsigned int index = warpsize * i + ilane;
        const unsigned int dfindex =
            DEFORMED ? ncoord * warpsize * i + ilane : ilane;

        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[q * nq0 + i] * in[warpsize * q + ilane];
        }

        // Multiply by derivative factors.
        if constexpr (APPEND)
        {
            out[index] += d0 * df[DIR * warpsize + dfindex];
        }
        else
        {
            out[index] = d0 * df[DIR * warpsize + dfindex];
        }
    }
}

template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor1DKernel(
    const unsigned int ilane, const unsigned int nq0,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT in0,
    TData *NEK_RESTRICT out, const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int i = 0u; i < nq0; ++i)
    {
        // Compute tensorial derivative.
        TData d0 = 0.0;
#pragma unroll
        for (unsigned int q = 0u; q < nq0; ++q)
        {
            d0 += D0[i * nq0 + q] * in0[warpsize * q + ilane];
        }

        if constexpr (SCALE)
        {
            d0 *= scale;
        }

        if constexpr (APPEND)
        {
            out[warpsize * i + ilane] += d0;
        }
        else
        {
            out[warpsize * i + ilane] = d0;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[warpsize * (nq0 * q + i) + ilane];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors.
            out[index] = d0 * df[0u * warpsize + dfindex] +
                         d1 * df[1u * warpsize + dfindex];
            out[outoffset + index] = d0 * df[2u * warpsize + dfindex] +
                                     d1 * df[3u * warpsize + dfindex];
            if (ncoord == 3u)
            {
                out[2u * outoffset + index] = d0 * df[4u * warpsize + dfindex] +
                                              d1 * df[5u * warpsize + dfindex];
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir2DSumFacKernel(
    const unsigned int ilane, const unsigned int ncoord, const unsigned int nq0,
    const unsigned int nq1, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    const unsigned int ndf = 2 * ncoord;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            const unsigned int index = warpsize * cnt_ji + ilane;
            const unsigned int dfindex =
                DEFORMED ? ndf * warpsize * cnt_ji + ilane : ilane;

            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[q * nq0 + i] * in[warpsize * (nq0 * j + q) + ilane];
            }

            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[q * nq1 + j] * in[warpsize * (nq0 * q + i) + ilane];
            }

            // Moving from standard to collapsed coordinates.
            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                d0 *= f1[j];
                d1 += d0 * f0[i];
            }

            // Multiply by derivative factors.
            if constexpr (APPEND)
            {
                out[index] += d0 * df[2u * DIR * warpsize + dfindex] +
                              d1 * df[(2u * DIR + 1u) * warpsize + dfindex];
            }
            else
            {
                out[index] = d0 * df[2u * DIR * warpsize + dfindex] +
                             d1 * df[(2u * DIR + 1u) * warpsize + dfindex];
            }
        }
    }
}

template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor2DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    TData *NEK_RESTRICT out, const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int j = 0u, cnt_ji = 0u; j < nq1; ++j)
    {
        for (unsigned int i = 0u; i < nq0; ++i, ++cnt_ji)
        {
            // Compute tensorial derivative.
            // Direction 0
            TData d0 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq0; ++q)
            {
                d0 += D0[i * nq0 + q] * in0[warpsize * (nq0 * j + q) + ilane];
            }

            // Direction 1
            TData d1 = 0.0;
#pragma unroll
            for (unsigned int q = 0u; q < nq1; ++q)
            {
                d1 += D1[j * nq1 + q] * in1[warpsize * (nq0 * q + i) + ilane];
            }

            if constexpr (APPEND)
            {
                if constexpr (SCALE)
                {
                    out[warpsize * cnt_ji + ilane] += scale * (d0 + d1);
                }
                else
                {
                    out[warpsize * cnt_ji + ilane] += d0 + d1;
                }
            }
            else
            {
                if constexpr (SCALE)
                {
                    out[warpsize * cnt_ji + ilane] = scale * (d0 + d1);
                }
                else
                {
                    out[warpsize * cnt_ji + ilane] = d0 + d1;
                }
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename TData>
NEK_DEVICE_INLINE static void PhysDeriv3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const size_t outoffset,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[q * nq1 + j] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[q * nq2 + k] *
                          in[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                              SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp0 = f1m[j] * f2[k] * d0;
                    TData tmp1 = f0[i] * tmp0;
                    TData tmp2 = f2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + f1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    d0 *= f2[k];
                    d2 += f0[i] * d0;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= f2[k];
                    d1 *= f2[k];
                    d2 += f0[i] * d0 + f1[j] * d1;
                }

                // Multiply by derivative factors.
                out[index] = d0 * df[0u * warpsize + dfindex] +
                             d1 * df[1u * warpsize + dfindex] +
                             d2 * df[2u * warpsize + dfindex];
                out[outoffset + index] = d0 * df[3u * warpsize + dfindex] +
                                         d1 * df[4u * warpsize + dfindex] +
                                         d2 * df[5u * warpsize + dfindex];
                out[2u * outoffset + index] = d0 * df[6u * warpsize + dfindex] +
                                              d1 * df[7u * warpsize + dfindex] +
                                              d2 * df[8u * warpsize + dfindex];
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool APPEND, bool DEFORMED,
          unsigned int DIR, typename TData>
NEK_DEVICE_INLINE static void PhysDerivDir3DSumFacKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    [[maybe_unused]] const TData *NEK_RESTRICT f0,
    [[maybe_unused]] const TData *NEK_RESTRICT f1,
    [[maybe_unused]] const TData *NEK_RESTRICT f1m,
    [[maybe_unused]] const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    constexpr unsigned int ndf = 9u;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                const unsigned int index = warpsize * cnt_kji + ilane;
                const unsigned int dfindex =
                    DEFORMED ? ndf * warpsize * cnt_kji + ilane : ilane;

                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[q * nq0 + i] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[q * nq1 + j] *
                          in[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[q * nq2 + k] *
                          in[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                // Moving from standard to collapsed coordinates.
                if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                              SHAPE_TYPE == LibUtilities::NodalTet)
                {
                    TData tmp0 = f1m[j] * f2[k] * d0;
                    TData tmp1 = f0[i] * tmp0;
                    TData tmp2 = f2[k] * d1;
                    d0         = tmp0;
                    d1         = tmp1 + tmp2;
                    d2 += tmp1 + f1[j] * tmp2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                                   SHAPE_TYPE == LibUtilities::NodalPrism)
                {
                    d0 *= f2[k];
                    d2 += f0[i] * d0;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    d0 *= f2[k];
                    d1 *= f2[k];
                    d2 += f0[i] * d0 + f1[j] * d1;
                }

                // Multiply by derivative factors.
                if constexpr (APPEND)
                {
                    out[index] += d0 * df[(3u * DIR) * warpsize + dfindex] +
                                  d1 * df[(3u * DIR + 1) * warpsize + dfindex] +
                                  d2 * df[(3u * DIR + 2) * warpsize + dfindex];
                }
                else
                {
                    out[index] = d0 * df[(3u * DIR) * warpsize + dfindex] +
                                 d1 * df[(3u * DIR + 1) * warpsize + dfindex] +
                                 d2 * df[(3u * DIR + 2) * warpsize + dfindex];
                }
            }
        }
    }
}

template <bool SCALE, bool APPEND, typename TData>
NEK_DEVICE_INLINE static void SumDerivTensor3DKernel(
    const unsigned int ilane, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT in0, const TData *NEK_RESTRICT in1,
    const TData *NEK_RESTRICT in2, TData *NEK_RESTRICT out,
    const TData scale = 1.0)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    for (unsigned int k = 0u, cnt_kji = 0u; k < nq2; k++)
    {
        for (unsigned int j = 0u; j < nq1; j++)
        {
            for (unsigned int i = 0u; i < nq0; i++, cnt_kji++)
            {
                // Compute tensorial derivative.
                // Direction 0
                TData d0 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq0; ++q)
                {
                    d0 += D0[i * nq0 + q] *
                          in0[warpsize * (nq0 * nq1 * k + nq0 * j + q) + ilane];
                }

                // Direction 1
                TData d1 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq1; ++q)
                {
                    d1 += D1[j * nq1 + q] *
                          in1[warpsize * (nq0 * nq1 * k + nq0 * q + i) + ilane];
                }

                // Direction 2
                TData d2 = 0.0;
#pragma unroll
                for (unsigned int q = 0u; q < nq2; ++q)
                {
                    d2 += D2[k * nq2 + q] *
                          in2[warpsize * (nq0 * nq1 * q + nq0 * j + i) + ilane];
                }

                if constexpr (APPEND)
                {
                    if constexpr (SCALE)
                    {
                        out[warpsize * cnt_kji + ilane] +=
                            scale * (d0 + d1 + d2);
                    }
                    else
                    {
                        out[warpsize * cnt_kji + ilane] += d0 + d1 + d2;
                    }
                }
                else
                {
                    if constexpr (SCALE)
                    {
                        out[warpsize * cnt_kji + ilane] =
                            scale * (d0 + d1 + d2);
                    }
                    else
                    {
                        out[warpsize * cnt_kji + ilane] = d0 + d1 + d2;
                    }
                }
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter1D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, [[maybe_unused]] unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e                  = getGlobalIdx<0>(threadBlock);
    const unsigned int m      = getBlockIdx<1>(threadBlock);
    const unsigned int c      = getBlockIdx<2>(threadBlock);
    const unsigned int nmode  = getBlockRange<1>(threadBlock);
    const unsigned int outDim = (nmode > 1) ? 3u : ncoord;
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr =
            in + nq0 * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nq0 * (nelmt * (outDim * nmode * c + m) + warpsize * iwarp);
        PhysDeriv1DSumFacKernel<DEFORMED>(ilane, ncoord, nq0, outoffset, D0,
                                          dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <typename Implementation, bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter1D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivDir1DKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter1D or TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    const unsigned int ndf    = ncoord;
    const unsigned int dfsize = DEFORMED ? nq0 : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nq0 * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nq0 * (nelmt * c + warpsize * iwarp);
        PhysDerivDir1DSumFacKernel<APPEND, DEFORMED, DIR>(
            ilane, ncoord, nq0, 0, D0, dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter2D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const unsigned int outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    const unsigned int ndf    = 2 * ncoord;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e                  = getGlobalIdx<0>(threadBlock);
    const unsigned int m      = getBlockIdx<1>(threadBlock);
    const unsigned int c      = getBlockIdx<2>(threadBlock);
    const unsigned int nmode  = getBlockRange<1>(threadBlock);
    const unsigned int outDim = (nmode > 1) ? 3u : ncoord;
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr =
            in + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (outDim * nmode * c + m) + warpsize * iwarp);
        PhysDeriv2DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, ncoord, nq0, nq1, outoffset, D0, D1, s_f0, s_f1, dfptr,
            inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter2D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivDir2DKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter2D or TemplatedPhysSizeParameter2D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    const unsigned int ndf    = 4u;
    const unsigned int nqTot  = nq0 * nq1;
    const unsigned int dfsize = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0 = nullptr;
    TData *s_f1 = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                  SHAPE_TYPE == LibUtilities::NodalTri)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e             = getGlobalIdx<0>(threadBlock);
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);

        PhysDerivDir2DSumFacKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>(
            ilane, nq0, nq1, D0, D1, s_f0, s_f1, dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool DEFORMED, typename TPhysSizeParameter3D, typename TthreadBlock,
          typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const size_t outoffset, const TData *NEK_RESTRICT D0,
    const TData *NEK_RESTRICT D1, const TData *NEK_RESTRICT D2,
    const TData *NEK_RESTRICT f0, const TData *NEK_RESTRICT f1,
    const TData *NEK_RESTRICT f1m, const TData *NEK_RESTRICT f2,
    const TData *NEK_RESTRICT df, const TData *NEK_RESTRICT in,
    TData *NEK_RESTRICT out, unsigned char *shmemptr,
    const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int m     = getBlockIdx<1>(threadBlock);
    const unsigned int c     = getBlockIdx<2>(threadBlock);
    const unsigned int nmode = getBlockRange<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr =
            in + nqTot * (nelmt * (nmode * c + m) + warpsize * iwarp);
        TData *outptr =
            out + nqTot * (nelmt * (3 * nmode * c + m) + warpsize * iwarp);
        PhysDeriv3DSumFacKernel<SHAPE_TYPE, DEFORMED>(
            ilane, nq0, nq1, nq2, outoffset, D0, D1, D2, s_f0, s_f1, s_f1m,
            s_f2, dfptr, inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation,
          bool APPEND, bool DEFORMED, unsigned int DIR,
          typename TPhysSizeParameter3D, typename TthreadBlock, typename TData,
          std::enable_if_t<std::is_same_v<Implementation, SumFac>, bool>
              Enable = true>
NEK_DEVICE_KERNEL void PhysDerivDir3DKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D, const size_t nelmt,
    const TData *NEK_RESTRICT D0, const TData *NEK_RESTRICT D1,
    const TData *NEK_RESTRICT D2, const TData *NEK_RESTRICT f0,
    const TData *NEK_RESTRICT f1, const TData *NEK_RESTRICT f1m,
    const TData *NEK_RESTRICT f2, const TData *NEK_RESTRICT df,
    const TData *NEK_RESTRICT in, TData *NEK_RESTRICT out,
    unsigned char *shmemptr, const TthreadBlock &threadBlock)
{
    static_assert(
        IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
        "Template argument must be either of type "
        "NonTemplatedPhysSizeParameter3D or TemplatedPhysSizeParameter3D.");

    FETCH_SHARED_MEMORY(shmemptr);

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    constexpr unsigned int ndf = 9u;
    const unsigned int nqTot   = nq0 * nq1 * nq2;
    const unsigned int dfsize  = DEFORMED ? nqTot : 1u;

    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    TData *s_f0  = nullptr;
    TData *s_f1  = nullptr;
    TData *s_f1m = nullptr;
    TData *s_f2  = nullptr;

    // Precompute geometric factors.
    const unsigned int idx0   = getLocalIdx<0>(threadBlock);
    const unsigned int stride = getLocalRange<0>(threadBlock);

    if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                  SHAPE_TYPE == LibUtilities::NodalTet)
    {
        s_f0  = (TData *)shmemptr;
        s_f1  = s_f0 + nq0;
        s_f1m = s_f1 + nq1;
        s_f2  = s_f1m + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx]  = f1[idx];
            s_f1m[idx] = f1m[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                       SHAPE_TYPE == LibUtilities::NodalPrism)
    {
        s_f0 = (TData *)shmemptr;
        s_f2 = s_f0 + nq0;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }
    else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
    {
        s_f0 = (TData *)shmemptr;
        s_f1 = s_f0 + nq0;
        s_f2 = s_f1 + nq1;

        for (unsigned int idx = idx0; idx < nq0; idx += stride)
        {
            s_f0[idx] = f0[idx];
        }

        for (unsigned int idx = idx0; idx < nq1; idx += stride)
        {
            s_f1[idx] = f1[idx];
        }

        for (unsigned int idx = idx0; idx < nq2; idx += stride)
        {
            s_f2[idx] = f2[idx];
        }

        localBarrier(threadBlock);
    }

    size_t e = getGlobalIdx<0>(threadBlock); // use size_t to prevent overflow
    const unsigned int c = getBlockIdx<1>(threadBlock);
    while (e < nelmt)
    {
        const size_t ilane = e % warpsize;
        const size_t iwarp = e / warpsize;
        const TData *dfptr = df + ndf * dfsize * warpsize * iwarp;
        const TData *inptr = in + nqTot * (nelmt * c + warpsize * iwarp);
        TData *outptr      = out + nqTot * (nelmt * c + warpsize * iwarp);
        PhysDerivDir3DSumFacKernel<SHAPE_TYPE, APPEND, DEFORMED, DIR>(
            ilane, nq0, nq1, nq2, D0, D1, D2, s_f0, s_f1, s_f1m, s_f2, dfptr,
            inptr, outptr);
        e += getGlobalRange<0>(threadBlock);
    }
}

#endif

} // namespace Nektar::Operators::detail
