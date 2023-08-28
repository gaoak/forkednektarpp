#pragma once

#include <numeric>

#include "Field.hpp"
#include "Operators/OperatorMatrix.hpp"

namespace Nektar::Operators::detail
{

template <typename TData, FieldState TFieldState>
class OperatorMatrixImpl : public OperatorMatrix<TData, TFieldState>
{
public:
    OperatorMatrixImpl(const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorMatrix<TData, TFieldState>(std::move(expansionList))
    {
        // get size of the matrix from the blocks given by state and expansionlist
        auto blocks = GetBlockAttributes(TFieldState, expansionList);
        m_size = std::accumulate(
            blocks.begin(), blocks.end(), 0,
            [](size_t acc, const BlockAttributes &block)
            { return acc + block.block_size; });
        
        // create memory for square matrix of given size
        m_matrix = std::vector<TData>(m_size * m_size);
    }

    size_t size()
    {
        return m_size;
    }

    void fill(const TData *src)
    {
        std::copy(src, src + (m_size * m_size), m_matrix.begin()); 
    }

    std::string toString()
    {
        auto pMat = m_matrix.cbegin();
        std::string str;
        for (size_t i = 0; i < m_size; ++i)
        {
            for (size_t j = 0; j < m_size; ++j)
            {
                if (j > 0)
                    str += "\t";
                str += std::to_string(*(pMat++));
            }
            str += "\n";
        }
        return str;
    }

    void apply(Field<TData, TFieldState> &in, Field<TData, TFieldState> &out)
    {
        auto *pIn = in.GetStorage().GetCPUPtr();
        auto *pOut = out.GetStorage().GetCPUPtr();
        auto pMatrix = m_matrix.cbegin();

        for (size_t i = 0; i < m_size; ++i)
        {
            *(pOut) = 0.;
            for (size_t j = 0; j < m_size; ++j)
            {
                *(pOut) += *(pIn) * *(pMatrix);
                pIn++;
                pMatrix++;
            }
            pOut++;
            pIn = in.GetStorage().GetCPUPtr();
        }
    }

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorMatrixImpl<TData, TFieldState>>(
            expansionList);
    }

    // className - for OperatorFactory
    static std::string className;

protected:
    size_t m_size;
    std::vector<TData> m_matrix;

};

}