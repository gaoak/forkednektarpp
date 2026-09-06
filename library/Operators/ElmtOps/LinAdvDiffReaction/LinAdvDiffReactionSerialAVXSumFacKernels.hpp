///////////////////////////////////////////////////////////////////////////////
//
// File: LinAdvDiffReactionSerialAVXSumFacKernels.hpp
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

#include "ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"
#include "ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

#include "ElmtOps/Helmholtz/HelmholtzSerialAVXSumFacKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

template <bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvectionSegKernel(
    const unsigned int ncoord, const unsigned int nq0,
    const simd_type *advVel0_ptr, const simd_type *advVel1_ptr,
    const simd_type *advVel2_ptr, const simd_type *df_ptr, simd_type *deriv0,
    simd_type *out, const typename simd_type::scalarType scale)
{
    simd_type vx, vy, vz, df0, df1, df2, d0;

    // Precompute Laplacian metrics.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        if (ncoord > 1)
        {
            df1 = df_ptr[1];
        }
        if (ncoord > 2)
        {
            df2 = df_ptr[2];
        }
    }

    // Apply metrics on all quad points.
    for (unsigned int i = 0; i < nq0; ++i)
    {
        // Set deformed derivative factors.
        if constexpr (DEFORMED)
        {
            df0 = df_ptr[i];
            if (ncoord > 1)
            {
                df1 = df_ptr[i * ncoord + 1];
            }
            if (ncoord > 2)
            {
                df2 = df_ptr[i * ncoord + 2];
            }
        }

        // Get advection velocity.
        vx = advVel0_ptr[i];
        if (ncoord > 1)
        {
            vy = advVel1_ptr[i];
        }
        if (ncoord > 2)
        {
            vz = advVel2_ptr[i];
        }

        // Get derivatives.
        d0 = deriv0[i];

        // x-advection * scale.
        simd_type adv = df0 * d0;
        adv *= vx;

        if (ncoord > 1)
        {
            simd_type tmp = df1 * d0;
            adv.fma(vy, tmp);
        }
        if (ncoord > 2)
        {
            simd_type tmp = df2 * d0;
            adv.fma(vz, tmp);
        }

        adv.fma(out[i], simd_type(scale));
        out[i] = adv;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvection2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    [[maybe_unused]] const simd_type *hfac0,
    [[maybe_unused]] const simd_type *hfac1, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    const simd_type *df_ptr, simd_type *deriv0, simd_type *deriv1,
    simd_type *out, const typename simd_type::scalarType scale)
{
    const auto ndf = 2 * ncoord;
    simd_type vx, vy, vz, df0, df1, df2, df3, df4, df5, d0, d1, h0, h1;

    // Precompute Laplacian metrics.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        if (ncoord)
        {
            df4 = df_ptr[4];
            df5 = df_ptr[5];
        }
    }

    // Apply metrics on all quad points.
    for (unsigned int q = 0; q < nq1; ++q)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                      SHAPE_TYPE == LibUtilities::eNodalTri)
        {
            h1 = hfac1[q];
        }
        for (unsigned int p = 0; p < nq0; ++p)
        {
            unsigned int cnt = q * nq0 + p;

            // Set deformed derivative factors.
            if constexpr (DEFORMED)
            {
                df0 = df_ptr[cnt * ndf];
                df1 = df_ptr[cnt * ndf + 1];
                df2 = df_ptr[cnt * ndf + 2];
                df3 = df_ptr[cnt * ndf + 3];
                if (ncoord == 3)
                {
                    df4 = df_ptr[cnt * ndf + 4];
                    df5 = df_ptr[cnt * ndf + 5];
                }
            }

            // Get advection velocity.
            vx = advVel0_ptr[cnt];
            vy = advVel1_ptr[cnt];
            if (ncoord == 3)
            {
                vz = advVel2_ptr[cnt];
            }

            // Get derivatives.
            d0 = deriv0[cnt];
            d1 = deriv1[cnt];

            if constexpr (SHAPE_TYPE == LibUtilities::eTriangle ||
                          SHAPE_TYPE == LibUtilities::eNodalTri)
            {
                h0 = hfac0[p];
                d0 *= h1;
                d1.fma(d0, h0);
            }

            // x-advection * scale.
            simd_type adv = df0 * d0;
            adv.fma(df1, d1);
            adv *= vx;

            simd_type dy = df2 * d0;
            dy.fma(df3, d1);

            adv.fma(vy, dy);

            if (ncoord == 3)
            {
                simd_type dz = df4 * d0;
                dz.fma(df5, d1);

                adv.fma(vz, dz);
            }

            adv.fma(out[cnt], simd_type(scale));
            out[cnt] = adv;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED, typename simd_type>
NEK_FORCE_INLINE static void AddAdvection3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    [[maybe_unused]] const simd_type *hfac0,
    [[maybe_unused]] const simd_type *hfac1,
    [[maybe_unused]] const simd_type *hfac2,
    [[maybe_unused]] const simd_type *hfac3, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr,
    const simd_type *df_ptr, simd_type *deriv0, simd_type *deriv1,
    simd_type *deriv2, simd_type *out,
    const typename simd_type::scalarType scale)
{
    constexpr auto ndf = 9;
    simd_type vx, vy, vz;
    simd_type df0, df1, df2, df3, df4, df5, df6, df7, df8;
    simd_type d0, d1, d2, h0, h1, h2, h3;

    // Precompute Laplacian metrics.
    if constexpr (!DEFORMED)
    {
        df0 = df_ptr[0];
        df1 = df_ptr[1];
        df2 = df_ptr[2];
        df3 = df_ptr[3];
        df4 = df_ptr[4];
        df5 = df_ptr[5];
        df6 = df_ptr[6];
        df7 = df_ptr[7];
        df8 = df_ptr[8];
    }

    // Apply metrics on all quad points.
    for (unsigned int r = 0; r < nq2; ++r)
    {
        if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                      (SHAPE_TYPE == LibUtilities::NodalPrism))
        {
            h3 = hfac3[r];
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            h3 = hfac3[r];
        }
        else if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                           (SHAPE_TYPE == LibUtilities::NodalTet))
        {
            h3 = hfac3[r];
        }

        for (unsigned int q = 0; q < nq1; ++q)
        {
            if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
            {
                h1 = hfac1[q];
            }
            else if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                               (SHAPE_TYPE == LibUtilities::NodalTet))
            {
                h1 = hfac1[q];
                h2 = hfac2[q];
            }

            for (unsigned int p = 0; p < nq0; ++p)
            {
                unsigned int cnt = r * nq0 * nq1 + q * nq0 + p;

                // Set deformed derivative factors.
                if constexpr (DEFORMED)
                {
                    df0 = df_ptr[cnt * ndf];
                    df1 = df_ptr[cnt * ndf + 1];
                    df2 = df_ptr[cnt * ndf + 2];
                    df3 = df_ptr[cnt * ndf + 3];
                    df4 = df_ptr[cnt * ndf + 4];
                    df5 = df_ptr[cnt * ndf + 5];
                    df6 = df_ptr[cnt * ndf + 6];
                    df7 = df_ptr[cnt * ndf + 7];
                    df8 = df_ptr[cnt * ndf + 8];
                }

                // Get advection velocity.
                vx = advVel0_ptr[cnt];
                vy = advVel1_ptr[cnt];
                vz = advVel2_ptr[cnt];

                // Get derivatives.
                d0 = deriv0[cnt];
                d1 = deriv1[cnt];
                d2 = deriv2[cnt];

                // Chain-rule local to cartesian.
                if constexpr ((SHAPE_TYPE == LibUtilities::Prism) ||
                              (SHAPE_TYPE == LibUtilities::NodalPrism))
                {
                    h0 = hfac0[p];
                    d0 *= h3;
                    d2.fma(h0, d0);
                }
                else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
                {
                    h0 = hfac0[p];
                    d0 *= h3;
                    d1 *= h3;
                    d2.fma(h0, d0);
                    d2.fma(h1, d1);
                }
                else if constexpr ((SHAPE_TYPE == LibUtilities::Tet) ||
                                   (SHAPE_TYPE == LibUtilities::NodalTet))
                {
                    h0 = hfac0[p];
                    d0 *= h3;
                    d0 *= h2;
                    d1 *= h3;
                    d2.fma(h0, d0);
                    d2.fma(h1, d1);
                    d1.fma(h0, d0);
                }

                // Advection.
                // u.dx
                simd_type adv = df0 * d0;
                adv.fma(df1, d1);
                adv.fma(df2, d2);
                adv *= vx;

                // v.dy
                simd_type dy = df3 * d0;
                dy.fma(df4, d1);
                dy.fma(df5, d2);
                adv.fma(vy, dy);

                // w.dz
                simd_type dz = df6 * d0;
                dz.fma(df7, d1);
                dz.fma(df8, d2);
                adv.fma(vz, dz);

                adv.fma(out[cnt], simd_type(scale));
                out[cnt] = adv;
            }
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void LinAdvDiffReactionKernelLauncher(
    const TSizeParameter1D sizeParam1D, const bool isModified,
    const unsigned int ncoord, const typename simd_type::scalarType *diffCoeff,
    const std::vector<typename simd_type::scalarType> &nullVec,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *w0, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, const simd_type *df,
    const simd_type *advVel0_ptr, const simd_type *advVel1_ptr,
    const simd_type *advVel2_ptr, simd_type *deriv0, simd_type *bwd,
    const simd_type *in, simd_type *out,
    const typename simd_type::scalarType lambda)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    const unsigned int nq0 = sizeParam1D.nq0();

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam1D, isModified, basis0,
                                              NtoM, in, bwd);

    // Step 2: Get tensor derivatives.
    PhysDerivTensor1DKernel(nq0, bwd, D0, deriv0);

    // Step 3: Evaluate advection term and add to bwd * lambda.
    AddAdvectionSegKernel<DEFORMED>(ncoord, nq0, advVel0_ptr, advVel1_ptr,
                                    advVel2_ptr, df, deriv0, bwd, lambda);

    // Step 4: Apply diffusion coeff and WJ.
    DiffusionCoeffwithWJ1DKernel<SHAPE_TYPE, true, DEFORMED>(
        ncoord, nq0, true, diffCoeff, false, nullVec, nullVec, nullVec, nullVec,
        nullVec, nullVec, jac, w0, df, deriv0, bwd, 1.0);

    // Step 5: Apply derivative and sum up.
    SumDerivTensor1DKernel<true>(nq0, deriv0, D0, bwd);

    // Step 6: Inner product without WJ.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
        sizeParam1D, isModified, bwd, basis0, NtoMTrans, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void LinAdvDiffReactionKernelLauncher(
    const TSizeParameter2D sizeParam2D, const bool isModified,
    const unsigned int ncoord, const typename simd_type::scalarType *diffCoeff,
    const std::vector<typename simd_type::scalarType> &nullVec,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1, const simd_type *f0,
    const simd_type *f1, const simd_type *NtoM, const simd_type *NtoMTrans,
    const simd_type *jac, const simd_type *df, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr, simd_type *wsp0,
    simd_type *deriv0, simd_type *deriv1, simd_type *bwd, const simd_type *in,
    simd_type *out, const typename simd_type::scalarType lambda)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    const unsigned int nq0 = sizeParam2D.nq0();
    const unsigned int nq1 = sizeParam2D.nq1();

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam2D, isModified, basis0,
                                              basis1, NtoM, wsp0, in, bwd);

    // Step 2: Get tensor derivatives.
    PhysDerivTensor2DKernel(nq0, nq1, bwd, D0, D1, deriv0, deriv1);

    // Step 3: Evaluate advection term and add to bwd * lambda.
    AddAdvection2DKernel<SHAPE_TYPE, DEFORMED>(
        ncoord, nq0, nq1, f0, f1, advVel0_ptr, advVel1_ptr, advVel2_ptr, df,
        deriv0, deriv1, bwd, lambda);

    // Step 4: Apply diffusion coeff and WJ.
    DiffusionCoeffwithWJ2DKernel<SHAPE_TYPE, true, DEFORMED>(
        ncoord, nq0, nq1, true, diffCoeff, false, nullVec, nullVec, nullVec,
        nullVec, nullVec, nullVec, jac, w0, w1, df, f0, f1, deriv0, deriv1, bwd,
        1.0);

    // Step 5: Apply derivative and sum up.
    SumDerivTensor2DKernel<true>(nq0, nq1, deriv0, deriv1, D0, D1, bwd);

    // Step 6: Inner product without WJ.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
        sizeParam2D, isModified, bwd, basis0, basis1, NtoMTrans, wsp0, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void LinAdvDiffReactionKernelLauncher(
    const TSizeParameter3D sizeParam3D, const bool isModified,
    [[maybe_unused]] const unsigned int ncoord,
    const typename simd_type::scalarType *diffCoeff,
    const std::vector<typename simd_type::scalarType> &nullVec,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *basis2,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *NtoM, const simd_type *NtoMTrans, const simd_type *jac,
    const simd_type *df, const simd_type *advVel0_ptr,
    const simd_type *advVel1_ptr, const simd_type *advVel2_ptr, simd_type *wsp0,
    simd_type *wsp1, simd_type *wsp2, simd_type *deriv0, simd_type *deriv1,
    simd_type *deriv2, simd_type *bwd, const simd_type *in, simd_type *out,
    const typename simd_type::scalarType lambda)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam3D, isModified, basis0,
                                              basis1, basis2, NtoM, wsp0, wsp1,
                                              in, bwd);

    // Step 2: Get tensor derivatives.
    PhysDerivTensor3DKernel(nq0, nq1, nq2, bwd, D0, D1, D2, deriv0, deriv1,
                            deriv2);

    // Step 3: Evaluate advection term and add to bwd * lambda.
    AddAdvection3DKernel<SHAPE_TYPE, DEFORMED>(
        nq0, nq1, nq2, f0, f1, f1m, f2, advVel0_ptr, advVel1_ptr, advVel2_ptr,
        df, deriv0, deriv1, deriv2, bwd, lambda);

    // Step 4: Apply diffusion coeff and WJ.
    DiffusionCoeffwithWJ3DKernel<SHAPE_TYPE, true, DEFORMED>(
        nq0, nq1, nq2, true, diffCoeff, false, nullVec, nullVec, nullVec,
        nullVec, nullVec, nullVec, jac, w0, w1, w2, df, f0, f1, f1m, f2, deriv0,
        deriv1, deriv2, bwd, 1.0);

    // Step 5: Apply derivative and sum up.
    SumDerivTensor3DKernel<true>(nq0, nq1, nq2, deriv0, deriv1, deriv2, D0, D1,
                                 D2, bwd);

    // Step 6: Inner product without WJ.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false>(
        sizeParam3D, isModified, bwd, basis0, basis1, basis2, NtoMTrans, wsp0,
        wsp1, wsp2, out);
}

} // namespace Nektar::Operators::detail
