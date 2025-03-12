///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrDeviceOnHostKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same_v<ExecSpace, NektarSpaces::DeviceOnHost>, void>::type
AssembleKernel(const unsigned int nsize, const int *assmbPtr,
               const TData *signPtr, const TData *inptr, TData *outptr)
{
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const unsigned int i) {
            atomic_add<NektarSpaces::GlobalScope>(outptr + assmbPtr[i],
                                                  signPtr[i] * inptr[i]);
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same_v<ExecSpace, NektarSpaces::DeviceOnHost>, void>::type
AssembleKernel(const unsigned int nsize, const int *assmbPtr, const TData sign,
               const TData *inptr, TData *outptr)
{
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const unsigned int i) {
            atomic_add<NektarSpaces::GlobalScope>(outptr + assmbPtr[i],
                                                  sign * inptr[i]);
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same_v<ExecSpace, NektarSpaces::DeviceOnHost>, void>::type
AssembleKernel(const unsigned int nsize, const int *assmbPtr,
               const TData *inptr, TData *outptr)
{
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const unsigned int i) {
            atomic_add<NektarSpaces::GlobalScope>(outptr + assmbPtr[i],
                                                  inptr[i]);
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same_v<ExecSpace, NektarSpaces::DeviceOnHost>, void>::type
GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                    const TData *signPtr, const TData *inptr, TData *outptr)
{
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const unsigned int i) {
            outptr[i] = signPtr[i] * inptr[assmbPtr[i]];
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same_v<ExecSpace, NektarSpaces::DeviceOnHost>, void>::type
GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                    const TData sign, const TData *inptr, TData *outptr)
{
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const unsigned int i) {
            outptr[i] = sign * inptr[assmbPtr[i]];
        });
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<
    std::is_same_v<ExecSpace, NektarSpaces::DeviceOnHost>, void>::type
GlobalToLocalKernel(const unsigned int nsize, const int *assmbPtr,
                    const TData *inptr, TData *outptr)
{
    Nektar::parallel_for<ExecSpace>(
        0u, nsize, NEKTAR_LAMBDA(const unsigned int i) {
            outptr[i] = inptr[assmbPtr[i]];
        });
}

} // namespace Nektar::Operators::detail

#endif
