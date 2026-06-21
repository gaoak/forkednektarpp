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

#include <magma_v2.h>

#if defined(NEKTAR_ENABLE_CUDA)
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
    static magma_queue_t &GetInstance()
    {
        if (!handle)
        {
            magma_int_t dev = 0;
            magma_queue_create(dev, &handle);
        }

        return handle;
    }

private:
    static magma_queue_t handle;
};
