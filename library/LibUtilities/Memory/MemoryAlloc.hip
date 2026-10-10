///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryAlloc.hip
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

#include <LibUtilities/Memory/MemoryAlloc.hpp>

namespace Nektar
{

bool isSetDeviceMemoryPool = false;

template <typename TData>
__global__ void deviceFillKernel(TData *dst, const TData val, const size_t size)
{
    const unsigned int idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = idx0; idx < size; idx += stride)
    {
        dst[idx] = val;
    }
}

template <typename TData>
void deviceFillKernelLauncher(TData *dst, const TData val, const size_t size,
                              const unsigned int streamID)
{
    auto stream                  = HIPStream::GetInstance(streamID);
    const unsigned int blocksize = NektarSpaces::Device::maximumBlockSize;
    const unsigned int gridsize  = (size + blocksize - 1) / blocksize;
    deviceFillKernel<<<gridsize, blocksize, 0, stream>>>(dst, val, size);
    CHECK_LAST_HIPCUDA_ERROR()
}

template void deviceFillKernelLauncher<double>(double *dst, const double val,
                                               const size_t size,
                                               const unsigned int streamID);
template void deviceFillKernelLauncher<float>(float *dst, const float val,
                                              const size_t size,
                                              const unsigned int streamID);

template <typename TData>
__global__ void deviceZeroPaddingKernel(TData *dst, const size_t size,
                                        const size_t compSize,
                                        const size_t prefix, const size_t nelmt,
                                        const unsigned int ndata,
                                        const unsigned int width)
{
    const size_t tail   = compSize - prefix;
    const size_t idx0   = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = idx0; idx < size; idx += stride)
    {
        const size_t i = prefix + idx % tail;
        if ((i / (ndata * width)) * width + i % width >= nelmt)
        {
            dst[(idx / tail) * compSize + i] = TData(0);
        }
    }
}

template <typename TData>
void deviceZeroPaddingKernelLauncher(TData *dst, const unsigned int ncomp,
                                     const size_t compSize, const size_t nelmt,
                                     const unsigned int ndata,
                                     const unsigned int width,
                                     const unsigned int streamID)
{
    const size_t prefix = (nelmt / width) * width * ndata;
    const size_t size   = ncomp * (compSize - prefix);

    auto stream                  = HIPStream::GetInstance(streamID);
    const unsigned int blocksize = NektarSpaces::Device::maximumBlockSize;
    const unsigned int gridsize  = (size + blocksize - 1) / blocksize;
    deviceZeroPaddingKernel<<<gridsize, blocksize, 0, stream>>>(
        dst, size, compSize, prefix, nelmt, ndata, width);
    CHECK_LAST_HIPCUDA_ERROR()
}

template void deviceZeroPaddingKernelLauncher<double>(
    double *dst, const unsigned int ncomp, const size_t compSize,
    const size_t nelmt, const unsigned int ndata, const unsigned int width,
    const unsigned int streamID);
template void deviceZeroPaddingKernelLauncher<float>(
    float *dst, const unsigned int ncomp, const size_t compSize,
    const size_t nelmt, const unsigned int ndata, const unsigned int width,
    const unsigned int streamID);
template void deviceZeroPaddingKernelLauncher<uint8_t>(
    uint8_t *dst, const unsigned int ncomp, const size_t compSize,
    const size_t nelmt, const unsigned int ndata, const unsigned int width,
    const unsigned int streamID);

} // namespace Nektar
