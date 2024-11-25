///////////////////////////////////////////////////////////////////////////////
//
// File: HelmholtzSYCLSumFacKernels.hpp
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

#if defined(NEKTAR_ENABLE_SYCL)

#include "Operators/Common/Spaces.hpp"
#include "Operators/Utils/SYCLQueue.hpp"

namespace Nektar::Operators::detail
{

// SYCL Kernels
template <typename TData>
void DiffusionCoeff1DKernel(const unsigned int nsize, const TData *diffCoeff,
                            TData *deriv0, const sycl::nd_item<3> &item_ct1)
{
    unsigned int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (i < nsize)
    {
        deriv0[i] *= diffCoeff[0];

        i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
void DiffusionCoeff2DKernel(const unsigned int nsize, const TData *diffCoeff,
                            TData *deriv0, TData *deriv1,
                            TData *__restrict shared,
                            const sycl::nd_item<3> &item_ct1)
{
    TData *s_diffCoeff = shared;

    // Copy to shared memory.
    unsigned int idx = item_ct1.get_local_id(2);
    if (idx < 4)
    {
        s_diffCoeff[idx] = diffCoeff[idx];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    unsigned int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (i < nsize)
    {
        TData deriv[2] = {deriv0[i], deriv1[i]};

        deriv0[i] = s_diffCoeff[0] * deriv[0] + s_diffCoeff[1] * deriv[1];
        deriv1[i] = s_diffCoeff[2] * deriv[0] + s_diffCoeff[3] * deriv[1];

        i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename TData>
void DiffusionCoeff3DKernel(const unsigned int nsize, const TData *diffCoeff,
                            TData *deriv0, TData *deriv1, TData *deriv2,
                            TData *__restrict shared,
                            const sycl::nd_item<3> &item_ct1)
{
    TData *s_diffCoeff = shared;

    // Copy to shared memory.
    unsigned int idx = item_ct1.get_local_id(2);
    if (idx < 9)
    {
        s_diffCoeff[idx] = diffCoeff[idx];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    unsigned int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

    while (i < nsize)
    {
        TData deriv[3] = {deriv0[i], deriv1[i], deriv2[i]};

        deriv0[i] = s_diffCoeff[0] * deriv[0] + s_diffCoeff[1] * deriv[1] +
                    s_diffCoeff[2] * deriv[2];
        deriv1[i] = s_diffCoeff[3] * deriv[0] + s_diffCoeff[4] * deriv[1] +
                    s_diffCoeff[5] * deriv[2];
        deriv2[i] = s_diffCoeff[6] * deriv[0] + s_diffCoeff[7] * deriv[1] +
                    s_diffCoeff[8] * deriv[2];

        i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    }
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
DiffusionCoeff1DKernel(const unsigned int nsize, const TData *diffCoeff,
                       TData *deriv0)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize * blockSize),
                               sycl::range<3>(1, 1, blockSize)),
             [=](sycl::nd_item<3> item) {
                 DiffusionCoeff1DKernel(nsize, diffCoeff, deriv0, item);
             });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
DiffusionCoeff2DKernel(const unsigned int nsize, const TData *diffCoeff,
                       TData *deriv0, TData *deriv1)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         // Create local shared memory
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(4), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize * blockSize),
                               sycl::range<3>(1, 1, blockSize)),
             [=](sycl::nd_item<3> item) {
                 TData *shmptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
                 DiffusionCoeff2DKernel(nsize, diffCoeff, deriv0, deriv1,
                                        shmptr, item);
             });
     }).wait();
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::SYCL>,
                               void>::type
DiffusionCoeff3DKernel(const unsigned int nsize, const TData *diffCoeff,
                       TData *deriv0, TData *deriv1, TData *deriv2)
{
    const unsigned int blockSize = NektarSpaces::SYCL::defaultBlockSize;
    const unsigned int gridSize  = (nsize + blockSize - 1u) / blockSize;

    sycl::queue &Q = SYCLQueue::GetInstance();
    Q.submit([=](sycl::handler &cgh) {
         // Create local shared memory
         sycl::local_accessor<TData, 1> shared(sycl::range<1>(9), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize * blockSize),
                               sycl::range<3>(1, 1, blockSize)),
             [=](sycl::nd_item<3> item) {
                 TData *shmptr =
                     shared
                         .template get_multi_ptr<sycl::access::decorated::no>()
                         .get();
                 DiffusionCoeff3DKernel(nsize, diffCoeff, deriv0, deriv1,
                                        deriv2, shmptr, item);
             });
     }).wait();
}

} // namespace Nektar::Operators::detail

#endif
