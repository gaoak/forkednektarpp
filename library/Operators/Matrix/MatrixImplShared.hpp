///////////////////////////////////////////////////////////////////////////////
//
// File: MatrixImplShared.hpp
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

#include "Operators/Matrix/MatrixImplBase.hpp"
#include "Operators/Matrix/MatrixKernels.hpp"

namespace Nektar::Operators::detail
{

// Shared implementation
template <typename ExecSpace, typename Implementation, FieldState TFieldState,
          typename TData,
          typename = typename std::enable_if<
#if defined(NEKTAR_ENABLE_CUDA)
              (std::is_same<ExecSpace, NektarSpaces::CUDA>::value &&
               std::is_same<Implementation, Operators::SumFac>::value) ||
#endif
              (std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value &&
               std::is_same<Implementation, Operators::StdMat>::value)>::type>
class OperatorMatrixImpl
    : public OperatorMatrixImplBase<ExecSpace, Implementation, TFieldState,
                                    TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    OperatorMatrixImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMatrixImplBase<ExecSpace, Implementation, TFieldState, TData>(
              expansionList)
    {
    }

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        // Copy memory to the device, if necessary and get raw pointers.
        auto *inPtr     = in.template GetConstPtr<MemSpace>();
        auto *outPtr    = out.template GetPtr<MemSpace>();
        auto *matrixPtr = this->m_matrix.template GetConstPtr<MemSpace>();

        // Initialise index
        size_t exp_idx = 0;

        // Loop over the blocks.
        for (auto const &block : in.GetBlocks())
        {
            // Determine shape and type of the element.
            auto nElmts = block.num_elements;

            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(exp_idx);
            auto numPts       = (TFieldState == FieldState::Coeff)
                                    ? expPtr->GetNcoeffs()
                                    : expPtr->GetTotPoints();

            // Deterime CUDA grid size.
#if defined(NEKTAR_ENABLE_CUDA)
            if constexpr (std::is_same<ExecSpace, NektarSpaces::CUDA>::value)
            {
                m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);
            }
#endif
            MatrixKernel<ExecSpace, TData>(m_gridSize, m_blockSize, nElmts,
                                           numPts, this->m_size, matrixPtr,
                                           inPtr, outPtr);

            // Increment pointer and index for next element type.
            matrixPtr += numPts * nElmts;
            inPtr += numPts * nElmts;
            outPtr += numPts * nElmts;
            exp_idx += nElmts;
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorMatrixImpl<ExecSpace, Implementation, TFieldState, TData>>(
            expansionList);
    }

private:
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
