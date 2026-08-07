///////////////////////////////////////////////////////////////////////////////
//
// File: magmaHandle.hpp
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

#include <iostream>
#include <stdio.h>
#include <unordered_map>

#include <magma_v2.h>

#if defined(NEKTAR_ENABLE_CUDA)
#include <Operators/Common/Backends/CUDAStream.hpp>
#define CUBLAS_CHECK(condition)                                                \
    {                                                                          \
        const cublasStatus_t status = condition;                               \
        if (status != CUBLAS_STATUS_SUCCESS)                                   \
        {                                                                      \
            std::cerr << "cuBLAS error encountered: \""                        \
                      << cublasGetStatusString(status) << "\" at " << __FILE__ \
                      << ':' << __LINE__ << std::endl;                         \
            exit(0);                                                           \
        }                                                                      \
    }
#elif defined(NEKTAR_ENABLE_HIP)
#include <Operators/Common/Backends/HIPStream.hpp>
#define HIPBLAS_CHECK(condition)                                               \
    {                                                                          \
        const hipblasStatus_t status = condition;                              \
        if (status != HIPBLAS_STATUS_SUCCESS)                                  \
        {                                                                      \
            std::cerr << "hipBLAS error encountered: \""                       \
                      << hipblasStatusToString(status) << "\" at " << __FILE__ \
                      << ':' << __LINE__ << std::endl;                         \
            exit(0);                                                           \
        }                                                                      \
    }
#endif

class magmaHandle
{
public:
    static magma_queue_t &GetInstance(const unsigned int streamID)
    {
        if (handle.find(streamID) == handle.end())
        {
            magma_queue_t magma_queue;
            int device_rank = 0;
#if defined(NEKTAR_ENABLE_CUDA)
            (void)cudaGetDevice(&device_rank);
            cudaStream_t stream = CUDAStream::GetInstance(streamID);
            cublasHandle_t cublas_handle;
            cusparseHandle_t cusparse_handle;
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
            (void)cusparseCreate(&cusparse_handle);
            CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
            (void)cusparseSetStream(cusparse_handle, stream);
            magma_queue_create_from_cuda(device_rank, stream, cublas_handle,
                                         cusparse_handle, &magma_queue);
#elif defined(NEKTAR_ENABLE_HIP)
            (void)hipGetDevice(&device_rank);
            hipStream_t stream = HIPStream::GetInstance(streamID);
            hipblasHandle_t hipblas_handle;
            hipsparseHandle_t hipsparse_handle;
            HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
            (void)hipsparseCreate(&hipsparse_handle);
            HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
            (void)hipsparseSetStream(hipsparse_handle, stream);
            magma_queue_create_from_hip(device_rank, stream, hipblas_handle,
                                        hipsparse_handle, &magma_queue);
#endif
            handle[streamID] = magma_queue;
        }

        return handle[streamID];
    }

private:
    static std::unordered_map<unsigned int, magma_queue_t> handle;
};
