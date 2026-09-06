///////////////////////////////////////////////////////////////////////////////
//
// File: MassSerialAVXSumFacKernels.hpp
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
//      Mass operator kernel file consisting of the header files of
//      BwdTransKernels and IProductWRTBaseKernels.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/BwdTrans/BwdTransSerialAVXSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSerialAVXSumFacKernels.hpp"

#include <Operators/ElmtOps/ElmtHelper.hpp>

namespace Nektar::Operators::detail
{

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter1D, typename simd_type>
NEK_FORCE_INLINE static void MassKernelLauncher(
    const TSizeParameter1D sizeParam1D, const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *w0, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, simd_type *bwd,
    const simd_type *in, simd_type *out)
{
    static_assert(IsSizeParameter1D_v<TSizeParameter1D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam1D, isModified, basis0,
                                              NtoM, in, bwd);

    // Step 2: Inner product for mass matrix operation.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false, DEFORMED>(
        sizeParam1D, isModified, bwd, basis0, w0, NtoMTrans, jac, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter2D, typename simd_type>
NEK_FORCE_INLINE static void MassKernelLauncher(
    const TSizeParameter2D sizeParam2D, const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, simd_type *wsp0,
    simd_type *bwd, const simd_type *in, simd_type *out)
{
    static_assert(IsSizeParameter2D_v<TSizeParameter2D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam2D, isModified, basis0,
                                              basis1, NtoM, wsp0, in, bwd);

    // Step 2: Inner product for mass matrix operation.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false, DEFORMED>(
        sizeParam2D, isModified, bwd, basis0, basis1, w0, w1, NtoMTrans, jac,
        wsp0, out);
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool DEFORMED,
          typename TSizeParameter3D, typename simd_type>
NEK_FORCE_INLINE static void MassKernelLauncher(
    const TSizeParameter3D sizeParam3D, const bool isModified,
    const typename simd_type::scalarType *basis0,
    const typename simd_type::scalarType *basis1,
    const typename simd_type::scalarType *basis2,
    const typename simd_type::scalarType *w0,
    const typename simd_type::scalarType *w1,
    const typename simd_type::scalarType *w2, const simd_type *NtoM,
    const simd_type *NtoMTrans, const simd_type *jac, simd_type *wsp0,
    simd_type *wsp1, simd_type *wsp2, simd_type *bwd, const simd_type *in,
    simd_type *out)
{
    static_assert(IsSizeParameter3D_v<TSizeParameter3D>,
                  "Template argument must be either of type "
                  "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");

    // Step 1: BwdTrans.
    BwdTransKernelLauncher<SHAPE_TYPE, false>(sizeParam3D, isModified, basis0,
                                              basis1, basis2, NtoM, wsp0, wsp1,
                                              in, bwd);

    // Step 2: Inner product for mass matrix operation.
    IProductWRTBaseKernelLauncher<SHAPE_TYPE, false, false, DEFORMED>(
        sizeParam3D, isModified, bwd, basis0, basis1, basis2, w0, w1, w2,
        NtoMTrans, jac, wsp0, wsp1, wsp2, out);
}

} // namespace Nektar::Operators::detail
