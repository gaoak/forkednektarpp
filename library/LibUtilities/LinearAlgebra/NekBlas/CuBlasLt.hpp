///////////////////////////////////////////////////////////////////////////////
//
// File: CuBlasLt.hpp
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
// Description: cuBLASLt helpers shared by the CUDA NekBlas back-ends.
// Private to the NekBlas sources, never installed.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LibUtilities/LinearAlgebra/NekBlas/NekBlas.hpp"

#include <LibUtilities/Backends/CUDAStream.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

#include <cublasLt.h>

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

namespace Nektar::NekBlas::detail
{
inline cublasLtHandle_t GetCuBlasLtHandle()
{
    static cublasLtHandle_t handle = nullptr;

    if (!handle)
    {
        CUBLAS_CHECK(cublasLtCreate(&handle));
    }

    return handle;
}

// Layout of the stored (pre-op) matrix whose op() is M x N.
inline cublasLtMatrixLayout_t CreateCuBlasLtLayout(
    const cudaDataType_t dataType, const cublasOperation_t trans, const int M,
    const int N, const int ld, const int stride, const int batchSize)
{
    const bool transposed = (trans == CUBLAS_OP_T);

    cublasLtMatrixLayout_t layout;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &layout, dataType, transposed ? N : M, transposed ? M : N, ld));

    const int32_t batchCount  = batchSize;
    const int64_t batchOffset = stride;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
        sizeof(batchCount)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batchOffset,
        sizeof(batchOffset)));

    return layout;
}

// cuBLAS has no strided-batched geam; cublasLtMatrixTransform computes
// C = alpha*op(A) + beta*op(B) with the batching carried by the layouts.
template <typename TData>
void CuBlasLtGeamStridedBatched(Handle<NektarSpaces::Device> handle,
                                std::string transposeA, std::string transposeB,
                                const int M, const int N, const TData alpha,
                                const TData *a, const int lda,
                                const int strideA, const TData beta,
                                const TData *b, const int ldb,
                                const int strideB, TData *c, const int ldc,
                                const int strideC, const int batchSize)
{
    constexpr cudaDataType_t dataType =
        std::is_same_v<TData, float> ? CUDA_R_32F : CUDA_R_64F;

    auto transA = (transposeA == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto transB = (transposeB == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasLtMatrixLayout_t layoutA =
        CreateCuBlasLtLayout(dataType, transA, M, N, lda, strideA, batchSize);
    cublasLtMatrixLayout_t layoutC = CreateCuBlasLtLayout(
        dataType, CUBLAS_OP_N, M, N, ldc, strideC, batchSize);

    // A null B (beta == 0) is passed with a null layout.
    cublasLtMatrixLayout_t layoutB = nullptr;
    if (b != nullptr)
    {
        layoutB = CreateCuBlasLtLayout(dataType, transB, M, N, ldb, strideB,
                                       batchSize);
    }

    cublasLtMatrixTransformDesc_t transformDesc;
    CUBLAS_CHECK(cublasLtMatrixTransformDescCreate(&transformDesc, dataType));
    CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
        transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transA,
        sizeof(transA)));
    if (b != nullptr)
    {
        CUBLAS_CHECK(cublasLtMatrixTransformDescSetAttribute(
            transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &transB,
            sizeof(transB)));
    }

    CUBLAS_CHECK(cublasLtMatrixTransform(
        GetCuBlasLtHandle(), transformDesc, &alpha, a, layoutA, &beta, b,
        layoutB, c, layoutC, CUDAStream::GetInstance(handle.GetStreamID())));

    CUBLAS_CHECK(cublasLtMatrixTransformDescDestroy(transformDesc));
    if (layoutB != nullptr)
    {
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
}
} // namespace Nektar::NekBlas::detail
