///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTDerivBaseSerialAVXSumFacKernels.hpp
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

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void StdAlignDerivBase1D(
    const unsigned int nq0, const unsigned int indim, const simd_type *df_ptr,
    simd_type *df_tmp, const size_t inoffset, const simd_type *in,
    simd_type *out, const simd_type *jac_ptr,
    const typename simd_type::scalarType *w)
{
    simd_type jac = 1.0;
    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if constexpr (!DEFORMED)
    {
        // unroll very small loops
        jac       = jac_ptr[0];
        df_tmp[0] = df_ptr[0];
        if (indim >= 2)
        {
            df_tmp[1] = df_ptr[1];
        }
        if (indim == 3)
        {
            df_tmp[2] = df_ptr[2];
        }
    }

    for (unsigned i = 0; i < nq0; ++i)
    {
        if constexpr (DEFORMED)
        {
            jac       = jac_ptr[i];
            df_tmp[0] = df_ptr[i * indim];
            if (indim >= 2)
            {
                df_tmp[1] = df_ptr[i * indim + 1];
            }
            if (indim == 3)
            {
                df_tmp[2] = df_ptr[i * indim + 2];
            }
        }

        simd_type sum = 0.0;
        for (unsigned int d = 0; d < indim; ++d)
        {
            simd_type inval = in[d * inoffset + i];
            sum.fma(inval, df_tmp[d]);
        }
        out[i] = sum * jac * w[i];
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void StdAlignDerivBase2D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int indim,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    const simd_type *inptr, simd_type *out[2],
    [[maybe_unused]] const simd_type *Fac0,
    [[maybe_unused]] const simd_type *Fac1, const simd_type *jac_ptr,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1)
{
    const auto ndf = 2 * indim;
    simd_type jac  = 1.0;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if constexpr (!DEFORMED)
    {
        jac       = jac_ptr[0];
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];

        if (indim == 3)
        {
            df_tmp[4] = df_ptr[4];
            df_tmp[5] = df_ptr[5];
        }
    }

    simd_type f1;
    unsigned int cnt_ji = 0;
    for (unsigned int j = 0; j < nq1; ++j)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                      SHAPE_TYPE == LibUtilities::NodalTri)
        {
            f1 = simd_type(Fac1[j]);
        }

        for (unsigned int i = 0; i < nq0; ++i, ++cnt_ji)
        {
            if constexpr (DEFORMED)
            {
                jac       = jac_ptr[cnt_ji];
                df_tmp[0] = df_ptr[cnt_ji * ndf];
                df_tmp[1] = df_ptr[cnt_ji * ndf + 1];
                df_tmp[2] = df_ptr[cnt_ji * ndf + 2];
                df_tmp[3] = df_ptr[cnt_ji * ndf + 3];

                if (indim == 3)
                {
                    df_tmp[4] = df_ptr[cnt_ji * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_ji * ndf + 5];
                }
            }

            simd_type in0 = inptr[cnt_ji];
            simd_type in1 = inptr[inoffset + cnt_ji];

            simd_type out0 = df_tmp[0] * in0;
            out0.fma(df_tmp[2], in1);

            simd_type out1 = df_tmp[1] * in0;
            out1.fma(df_tmp[3], in1);

            if (indim == 3)
            {
                simd_type in2 = inptr[2 * inoffset + cnt_ji];
                out0.fma(df_tmp[4], in2);
                out1.fma(df_tmp[5], in2);
            }

            if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri)
            {
                // Multiply by geometric factors
                simd_type f0 = Fac0[i];

                // Scale by geometric factor 2/(1-z1)
                out0 *= f1;
                // Scale by geometric factor (1+z0)/(1-z1)
                simd_type c1 = f0 * out1;
                out0.fma(c1, f1);
            }

            // store outputs
            simd_type WJ   = jac * w0[i] * w1[j];
            out[0][cnt_ji] = out0 * WJ;
            out[1][cnt_ji] = out1 * WJ;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void StdAlignDerivBase3D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    [[maybe_unused]] const simd_type *Fac0,
    [[maybe_unused]] const simd_type *Fac1,
    [[maybe_unused]] const simd_type *Fac1m,
    [[maybe_unused]] const simd_type *Fac2, const simd_type *jac_ptr,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2, const simd_type *inptr,
    simd_type *out[3])
{
    const auto ndf = 9;
    simd_type jac  = 1.0;

    // Calculate dxi/dx in[0] + dxi/dy in[1] + dxi/dz in[2]
    if constexpr (!DEFORMED)
    {
        jac       = jac_ptr[0];
        df_tmp[0] = df_ptr[0];
        df_tmp[1] = df_ptr[1];
        df_tmp[2] = df_ptr[2];
        df_tmp[3] = df_ptr[3];
        df_tmp[4] = df_ptr[4];
        df_tmp[5] = df_ptr[5];
        df_tmp[6] = df_ptr[6];
        df_tmp[7] = df_ptr[7];
        df_tmp[8] = df_ptr[8];
    }

    unsigned int cnt_kji = 0;
    simd_type f0, f1, f1m, f2;

    for (unsigned int k = 0; k < nq2; ++k)
    {
        if constexpr (SHAPE_TYPE != LibUtilities::Hex)
        {
            f2 = simd_type(Fac2[k]);
        }

        for (unsigned int j = 0; j < nq1; ++j)
        {
            if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                          (SHAPE_TYPE == LibUtilities::NodalTet))
            {
                f1  = simd_type(Fac1[j]);
                f1m = simd_type(Fac1m[j]);
            }
            else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                f1 = simd_type(Fac1[j]);
            }

            simd_type w12 = w1[j] * w2[k];

            for (unsigned int i = 0; i < nq0; ++i, ++cnt_kji)
            {
                if constexpr (DEFORMED)
                {
                    jac       = jac_ptr[cnt_kji];
                    df_tmp[0] = df_ptr[cnt_kji * ndf];
                    df_tmp[1] = df_ptr[cnt_kji * ndf + 1];
                    df_tmp[2] = df_ptr[cnt_kji * ndf + 2];
                    df_tmp[3] = df_ptr[cnt_kji * ndf + 3];
                    df_tmp[4] = df_ptr[cnt_kji * ndf + 4];
                    df_tmp[5] = df_ptr[cnt_kji * ndf + 5];
                    df_tmp[6] = df_ptr[cnt_kji * ndf + 6];
                    df_tmp[7] = df_ptr[cnt_kji * ndf + 7];
                    df_tmp[8] = df_ptr[cnt_kji * ndf + 8];
                }

                simd_type in0 = inptr[cnt_kji];
                simd_type in1 = inptr[inoffset + cnt_kji];
                simd_type in2 = inptr[2 * inoffset + cnt_kji];

                simd_type out0 = df_tmp[0] * in0;
                out0.fma(df_tmp[3], in1);
                out0.fma(df_tmp[6], in2);

                simd_type out1 = df_tmp[1] * in0;
                out1.fma(df_tmp[4], in1);
                out1.fma(df_tmp[7], in2);

                simd_type out2 = df_tmp[2] * in0;
                out2.fma(df_tmp[5], in1);
                out2.fma(df_tmp[8], in2);

                if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                              (SHAPE_TYPE == LibUtilities::NodalTet))
                {
                    // (out0 + (out1 + out2)*(1+z0)/2) * 2/(1 - z1) * 2/(1 - z2)
                    f0 = Fac0[i];
                    out0.fma(out1 + out2, f0);
                    out0 *= f1m * f2;

                    // (out1 + out2 * (1+z1)/2 ) * 2/(1 - z2)
                    out1.fma(out2, f1);
                    out1 *= f2;
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    // (out0 +  out2 * (1 + z0)/2 ) * 2/(1 - z2)
                    f0 = Fac0[i];
                    out0.fma(out2, f0);
                    out0 *= f2;

                    // (out1 + out2 * (1+z1)/2 ) * 2/(1 - z2)
                    out1.fma(out2, f1);
                    out1 *= f2;
                }
                else if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                                   (SHAPE_TYPE == LibUtilities::NodalPrism))
                {
                    // (out0 +  out2 * (1 + z0)/2 ) * 2/(1 - z2)
                    f0 = Fac0[i];
                    out0.fma(out2, f0);
                    out0 *= f2;
                }

                // store outputs
                simd_type WJ    = jac * w0[i] * w12;
                out[0][cnt_kji] = out0 * WJ;
                out[1][cnt_kji] = out1 * WJ;
                out[2][cnt_kji] = out2 * WJ;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTDerivBaseKernelLauncher(
    const TSizeParameter1D sizeParam1D, const unsigned int indim,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    const simd_type *in, const simd_type *jac,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *D0, simd_type *tmp0, simd_type *out)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D> ||
                      IsPhysSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be a one dimensional size "
                  "parameter.");

    const unsigned int nq0 = sizeParam1D.nq0();

    // Align the derivative with the standard element.
    StdAlignDerivBase1D<DEFORMED>(nq0, indim, df_ptr, df_tmp, inoffset, in,
                                  tmp0, jac, w0);

    // Apply the transposed derivative and sum up.
    SumDerivTensor1DKernel<false>(nq0, tmp0, D0, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTDerivBaseKernelLauncher(
    const TSizeParameter2D sizeParam2D, const unsigned int indim,
    const simd_type *df_ptr, simd_type *df_tmp, const size_t inoffset,
    const simd_type *in, const simd_type *f0, const simd_type *f1,
    const simd_type *jac, const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, simd_type *tmp0, simd_type *tmp1,
    simd_type *out)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D> ||
                      IsPhysSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be a two dimensional size "
                  "parameter.");

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    simd_type *tmpPtr[2] = {tmp0, tmp1};

    // Align the derivative with the standard element.
    StdAlignDerivBase2D<SHAPE_TYPE, DEFORMED>(nq0, nq1, indim, df_ptr, df_tmp,
                                              inoffset, in, tmpPtr, f0, f1, jac,
                                              w0, w1);

    // Apply the transposed derivative and sum up.
    SumDerivTensor2DKernel<false>(nq0, nq1, tmpPtr[0], tmpPtr[1], D0, D1, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void IProductWRTDerivBaseKernelLauncher(
    const TSizeParameter3D sizeParam3D,
    [[maybe_unused]] const unsigned int indim, const simd_type *df_ptr,
    simd_type *df_tmp, const size_t inoffset, const simd_type *in,
    const simd_type *f0, const simd_type *f1, const simd_type *f1m,
    const simd_type *f2, const simd_type *jac,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, simd_type *tmp0, simd_type *tmp1,
    simd_type *tmp2, simd_type *out)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D> ||
                      IsPhysSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be a three dimensional size "
                  "parameter.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    simd_type *tmpPtr[3] = {tmp0, tmp1, tmp2};

    // Align the derivative with the standard element.
    StdAlignDerivBase3D<SHAPE_TYPE, DEFORMED>(nq0, nq1, nq2, df_ptr, df_tmp,
                                              inoffset, f0, f1, f1m, f2, jac,
                                              w0, w1, w2, in, tmpPtr);

    // Apply the transposed derivative and sum up.
    SumDerivTensor3DKernel<false>(nq0, nq1, nq2, tmpPtr[0], tmpPtr[1],
                                  tmpPtr[2], D0, D1, D2, out);
}

} // namespace Nektar::Operators::detail
