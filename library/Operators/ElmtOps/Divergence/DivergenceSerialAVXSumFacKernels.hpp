///////////////////////////////////////////////////////////////////////////////
//
// File: DivergenceSerialAVXSumFacKernels.hpp
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

#include <LibUtilities/BasicUtils/NekInline.hpp>

#include "Operators/ElmtOps/PhysDeriv/PhysDerivSerialAVXSumFacKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void DivergenceKernelLauncher(
    const TPhysSizeParameter1D sizeParam1D,
    const typename simd_type::scalarType *D0, const simd_type *df_ptr,
    simd_type *wsp0, const simd_type *in0, simd_type *out)
{
    static_assert(IsPhysSizeParameter1D_v<TPhysSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter1D or "
                  "TemplatedPhysSizeParameter1D.");

    const unsigned int ncoord = sizeParam1D.ncoord();
    const unsigned int nq0    = sizeParam1D.nq0();

    // du/dx
    PhysDerivTensor1DKernel(nq0, in0, D0, wsp0);
    PhysDerivDir1DKernel<SHAPE_TYPE, false, DEFORMED, 0>(nq0, ncoord, df_ptr,
                                                         wsp0, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void DivergenceKernelLauncher(
    const TPhysSizeParameter2D sizeParam2D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1, const simd_type *f0,
    const simd_type *f1, const simd_type *df_ptr, simd_type *wsp0,
    simd_type *wsp1, const simd_type *in0, const simd_type *in1, simd_type *out)
{
    static_assert(IsPhysSizeParameter2D_v<TPhysSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter2D or "
                  "TemplatedPhysSizeParameter2D.");

    const unsigned int ncoord = sizeParam2D.ncoord();
    const unsigned int nq0    = sizeParam2D.nq0();
    const unsigned int nq1    = sizeParam2D.nq1();

    // du/dx
    PhysDerivTensor2DKernel(nq0, nq1, in0, D0, D1, wsp0, wsp1);
    PhysDerivDir2DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
        nq0, nq1, ncoord, f0, f1, df_ptr, wsp0, wsp1, out);

    // dv/dy
    PhysDerivTensor2DKernel(nq0, nq1, in1, D0, D1, wsp0, wsp1);
    PhysDerivDir2DKernel<SHAPE_TYPE, true, DEFORMED, 1>(
        nq0, nq1, ncoord, f0, f1, df_ptr, wsp0, wsp1, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TPhysSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void DivergenceKernelLauncher(
    const TPhysSizeParameter3D sizeParam3D,
    const typename simd_type::scalarType *D0,
    const typename simd_type::scalarType *D1,
    const typename simd_type::scalarType *D2, const simd_type *f0,
    const simd_type *f1, const simd_type *f1m, const simd_type *f2,
    const simd_type *df_ptr, simd_type *wsp0, simd_type *wsp1, simd_type *wsp2,
    const simd_type *in0, const simd_type *in1, const simd_type *in2,
    simd_type *out)
{
    static_assert(IsPhysSizeParameter3D_v<TPhysSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedPhysSizeParameter3D or "
                  "TemplatedPhysSizeParameter3D.");

    const unsigned int nq0 = sizeParam3D.nq0();
    const unsigned int nq1 = sizeParam3D.nq1();
    const unsigned int nq2 = sizeParam3D.nq2();

    // du/dx
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in0, D0, D1, D2, wsp0, wsp1, wsp2);
    PhysDerivDir3DKernel<SHAPE_TYPE, false, DEFORMED, 0>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, wsp0, wsp1, wsp2, out);

    // dv/dy
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in1, D0, D1, D2, wsp0, wsp1, wsp2);
    PhysDerivDir3DKernel<SHAPE_TYPE, true, DEFORMED, 1>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, wsp0, wsp1, wsp2, out);

    // dw/dz
    PhysDerivTensor3DKernel(nq0, nq1, nq2, in2, D0, D1, D2, wsp0, wsp1, wsp2);
    PhysDerivDir3DKernel<SHAPE_TYPE, true, DEFORMED, 2>(
        nq0, nq1, nq2, f0, f1, f1m, f2, df_ptr, wsp0, wsp1, wsp2, out);
}

} // namespace Nektar::Operators::detail
