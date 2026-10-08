///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryAlloc.cu
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
    auto stream                  = CUDAStream::GetInstance(streamID);
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

} // namespace Nektar
