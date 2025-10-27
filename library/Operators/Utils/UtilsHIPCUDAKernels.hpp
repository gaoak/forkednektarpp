///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsHIPCUDAKernels.hpp
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

#if (defined(NEKTAR_ENABLE_CUDA) && defined(__CUDACC__)) ||                    \
    (defined(NEKTAR_ENABLE_HIP) && defined(__HIPCC__))

namespace Nektar
{

template <typename TData>
__global__ void interleaveKernelLauncher(const unsigned int interleaveWidth,
                                         size_t numElmtGroups,
                                         const unsigned int npts, TData *buffer,
                                         TData *inout,
                                         const hipcudaBlock1D &threadBlock)
{
    interleaveKernel(interleaveWidth, numElmtGroups, npts, buffer, inout,
                     threadBlock);
}

template <typename TData>
__global__ void deInterleaveKernelLauncher(const unsigned int interleaveWidth,
                                           size_t numElmtGroups,
                                           const unsigned int npts,
                                           TData *buffer, TData *inout,
                                           const hipcudaBlock1D &threadBlock)
{
    deInterleaveKernel(interleaveWidth, numElmtGroups, npts, buffer, inout,
                       threadBlock);
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
interleave(const unsigned int interleaveWidth, const size_t numElmtGroups,
           const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numElmtGroups;
    const size_t bufferSize =
        sizeof(TData) * interleaveWidth * numElmtGroups * npts;

    TData *buffer;
    GetDeviceProperties::CheckGlobalMemoryUsage(bufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMalloc(&buffer, bufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMalloc(&buffer, bufferSize));
#endif
    interleaveKernelLauncher<<<gridSize, blockSize>>>(
        interleaveWidth, numElmtGroups, npts, buffer, inout, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(buffer));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(buffer));
#endif
}

template <typename ExecSpace, typename TData>
inline typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Device>,
                               void>::type
deInterleave(const unsigned int interleaveWidth, size_t numElmtGroups,
             const unsigned int npts, TData *inout)
{
    const unsigned int blockSize = NektarSpaces::Device::defaultBlockSize;
    const unsigned int gridSize  = numElmtGroups;
    const size_t bufferSize =
        sizeof(TData) * interleaveWidth * numElmtGroups * npts;

    TData *buffer;
    GetDeviceProperties::CheckGlobalMemoryUsage(bufferSize);
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaMalloc(&buffer, bufferSize));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipMalloc(&buffer, bufferSize));
#endif
    deInterleaveKernelLauncher<<<gridSize, blockSize>>>(
        interleaveWidth, numElmtGroups, npts, buffer, inout, hipcudaBlock1D());
    CHECK_LAST_HIPCUDA_ERROR();
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaFree(buffer));
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipFree(buffer));
#endif
}

} // namespace Nektar

#endif
