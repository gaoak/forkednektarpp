///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceOnHostSumFacKernelLaunchers.hpp
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

#if defined(NEKTAR_ENABLE_DEVICEONHOST)

namespace Nektar::Operators::detail
{

// Kernel Launchers.
// Non-size based version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const unsigned int ncoord,
                                               const unsigned int nq0,
                                               const size_t nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    PhysDeriv1DKernel<Implementation, DEFORMED>(ncoord, nq0, nelmt, D0, df, in,
                                                out, deviceOnHostBlock1D());
}

// Size based template version.
template <typename ExecSpace, typename Implementation, bool DEFORMED,
          unsigned int ncoord, unsigned int nq0, typename TData>
NEK_FORCE_INLINE static void PhysDeriv1DKernel(const size_t nelmt,
                                               const TData *D0, const TData *df,
                                               const TData *in, TData *out)
{
    PhysDeriv1DKernel<Implementation, DEFORMED>(ncoord, nq0, nelmt, D0, df, in,
                                                out, deviceOnHostBlock1D());
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(
    const unsigned int ncoord, const unsigned int nq0, const unsigned int nq1,
    const size_t nelmt, const TData *D0, const TData *D1, const TData *f0,
    const TData *f1, const TData *df, const TData *in, TData *out)
{
    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
    std::vector<TData> shmem(shmemsize);

    PhysDeriv2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out, shmem.data(),
        deviceOnHostBlock1D());
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int ncoord,
          unsigned int nq0, unsigned int nq1, typename TData>
NEK_FORCE_INLINE static void PhysDeriv2DKernel(const size_t nelmt,
                                               const TData *D0, const TData *D1,
                                               const TData *f0, const TData *f1,
                                               const TData *df, const TData *in,
                                               TData *out)
{
    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1);
    std::vector<TData> shmem(shmemsize);

    PhysDeriv2DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        ncoord, nq0, nq1, nelmt, D0, D1, f0, f1, df, in, out, shmem.data(),
        deviceOnHostBlock1D());
}

// Non-size based version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const size_t nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *in, TData *out)
{
    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2);
    std::vector<TData> shmem(shmemsize);

    PhysDeriv3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out,
        shmem.data(), deviceOnHostBlock1D());
}

// Size based template version.
template <LibUtilities::ShapeType SHAPE_TYPE, typename ExecSpace,
          typename Implementation, bool DEFORMED, unsigned int nq0,
          unsigned int nq1, unsigned int nq2, typename TData>
NEK_FORCE_INLINE static void PhysDeriv3DKernel(
    const size_t nelmt, const TData *D0, const TData *D1, const TData *D2,
    const TData *f0, const TData *f1, const TData *f1m, const TData *f2,
    const TData *df, const TData *in, TData *out)
{
    const unsigned int shmemsize =
        PhysDerivSharedMemorySize<SHAPE_TYPE, Implementation>(nq0, nq1, nq2);
    std::vector<TData> shmem(shmemsize);

    PhysDeriv3DKernel<SHAPE_TYPE, Implementation, DEFORMED>(
        nq0, nq1, nq2, nelmt, D0, D1, D2, f0, f1, f1m, f2, df, in, out,
        shmem.data(), deviceOnHostBlock1D());
}

} // namespace Nektar::Operators::detail

#endif
