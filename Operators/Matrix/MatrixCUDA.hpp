#pragma once

#include "MemoryRegionCUDA.hpp"
#include "Operators/Matrix/MatrixCUDAKernels.cuh"
#include "Operators/OperatorHelper.cuh"
#include "Operators/OperatorMatrix.hpp"

namespace Nektar::Operators::detail
{

// Matrix implementation
template <typename TData, FieldState TFieldState>
class OperatorMatrixImpl<TData, TFieldState, ImplCUDA>
    : public OperatorMatrix<TData, TFieldState>
{
public:
    OperatorMatrixImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMatrix<TData, TFieldState>(expansionList)
    {
        // get size of the matrix from the blocks given by state and
        // expansionlist
        auto blocks = GetBlockAttributes(TFieldState, expansionList);
        m_size      = std::accumulate(blocks.begin(), blocks.end(), 0,
                                      [](size_t acc, const BlockAttributes &block)
                                      { return acc + block.block_size; });
        // create memory for square matrix of given size
        // allocate memory device
        cudaMalloc((void **)&m_matrix, sizeof(TData) * m_size * m_size);
    }

    ~OperatorMatrixImpl()
    {
        cudaFree(m_matrix);
    }

    size_t size()
    {
        return m_size;
    }

    void fill(std::vector<TData> src)
    {
        // copy host memory to device memory
        cudaMemcpy(m_matrix, src.data(), sizeof(TData) * m_size * m_size,
                   cudaMemcpyHostToDevice);
    }

    std::string toString()
    {
        std::vector<TData> matrix_print(m_size * m_size);
        // Copy device memory to host memory for printing
        cudaMemcpy(matrix_print.data(), m_matrix,
                   m_size * m_size * sizeof(TData), cudaMemcpyDeviceToHost);

        auto pMat = matrix_print.cbegin();
        std::string str;
        for (size_t i = 0; i < m_size; ++i)
        {
            for (size_t j = 0; j < m_size; ++j)
            {
                if (j > 0)
                {
                    str += "\t";
                }
                str += std::to_string(*(pMat++));
            }
            str += "\n";
        }
        return str;
    }

    void apply(Field<TData, TFieldState> &in,
               Field<TData, TFieldState> &out) override
    {
        // Copy memory to GPU, if necessary and get raw pointers.
        auto const *inptr =
            in.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();
        auto *outptr = out.template GetStorage<MemoryRegionCUDA>().GetGPUPtr();

        // Initialise index
        size_t expIdx = 0;

        // Loop over the blocks.
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // Determine shape and type of the element.
            auto const expPtr = this->m_expansionList->GetExp(expIdx);
            auto nElmts       = in.GetBlocks()[block_idx].num_elements;
            auto numPts       = (TFieldState == FieldState::Coeff)
                                    ? expPtr->GetNcoeffs()
                                    : expPtr->GetTotPoints();

            // Deterime CUDA grid size.
            m_gridSize = GetCUDAGridSize(nElmts, m_blockSize);

            MatrixKernel<<<m_gridSize, m_blockSize>>>(numPts, nElmts, m_size,
                                                      m_matrix, inptr, outptr);

            // Increment pointer and index for next element type.
            inptr += numPts * nElmts;
            outptr += numPts * nElmts;
            m_matrix += numPts * nElmts;
            expIdx += nElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<
            OperatorMatrixImpl<TData, TFieldState, ImplCUDA>>(expansionList);
    }

    static std::string className;

private:
    size_t m_size;
    TData *m_matrix;
    size_t m_gridSize  = 1024;
    size_t m_blockSize = 32;
};

} // namespace Nektar::Operators::detail
