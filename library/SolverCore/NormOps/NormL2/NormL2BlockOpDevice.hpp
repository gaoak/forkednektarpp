///////////////////////////////////////////////////////////////////////////////
//
// File: NormL2BlockOpDevice.hpp
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

#include <array>

#include <SolverCore/NormOps/NormL2/NormL2BlockOp.hpp>

#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class NormL2BlockOpImpl : public NormL2BlockOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    NormL2BlockOpImpl(const unsigned int block_idx,
                      const LocalRegions::ExpansionSharedPtr &exp,
                      LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
        : NormL2BlockOp<TData>(block_idx, exp, dataWarehouse)
    {
        m_streamID = block_idx + 1;
        m_nqTot    = exp->GetTotPoints();
    }

    // className - for BlockOperatorFactory
    static std::string className;

    // Instantiation function for CreatorFunction in BlockOperatorFactory.
    static std::unique_ptr<NormL2BlockOp<TData>> Instantiate(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        LibUtilities::NekDataWarehouseSharedPtr dataWarehouse)
    {
        return std::make_unique<NormL2BlockOpImpl<ExecSpace, TData>>(
            block_idx, exp, dataWarehouse);
    }

protected:
    unsigned int m_streamID;
    unsigned int m_nqTot = 0;
    /// Quadrature metric of every point of the block for non-interleaved
    /// data (first) and for interleaved data (second); null until first read
    /// at that layout.
    std::array<const TData *, 2> m_metric{};

    /// The quadrature metric at @p interleaveWidth, fetched from the data
    /// warehouse with the layout of the first input read in it, and kept.
    const TData *GetMetric(const unsigned int interleaveWidth)
    {
        const TData *&metric = m_metric[interleaveWidth == 1 ? 0 : 1];
        if (metric == nullptr)
        {
            metric = this->m_dataWarehouse->template GetData<MemSpace>(
                LocalRegions::QuadratureMetricKey<TData>(this->m_block_idx,
                                                         interleaveWidth));
        }
        return metric;
    }

    void v_Apply(LibUtilities::BlockAccessor<TData, FieldState::Phys> &inblock,
                 LibUtilities::MemoryRegion<TData> &data) override
    {
        const unsigned int nComp = inblock.GetNumComponents();
        const unsigned int nHomo = inblock.GetNumHomoModes();
        const size_t compSize    = inblock.CompSize();

        // Initialize pointers.
        auto inptr   = inblock.template GetPtr<MemSpace, ReadOnly>(m_streamID);
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        // The metric is laid out like the input, so one storage index
        // addresses the same point in both.
        const TData *metric = GetMetric(inblock.GetInterleaveWidth());

        // With a homogeneous direction the planes of a component, stored one
        // after the other, are summed with their weights in one reduction.
        const TData *planeWeights =
            (this->m_planeWeights.size() == 0)
                ? nullptr
                : this->m_planeWeights.template GetPtr<MemSpace, ReadOnly>(
                      m_streamID);

        // Compute norm, as many components to a reduction as there are left.
        unsigned int cnt = 0;
        cnt = Integrals<8>(cnt, nComp, nHomo, compSize, metric, planeWeights,
                           inptr, dataptr);
        cnt = Integrals<4>(cnt, nComp, nHomo, compSize, metric, planeWeights,
                           inptr, dataptr);
        cnt = Integrals<2>(cnt, nComp, nHomo, compSize, metric, planeWeights,
                           inptr, dataptr);
        Integrals<1>(cnt, nComp, nHomo, compSize, metric, planeWeights, inptr,
                     dataptr);
    }

    /// Add the integrals of the squares of components [@p cnt, nComp), each
    /// over its
    /// @p nHomo planes, to data[@p cnt, nComp), @p NRED components to a
    /// reduction while that many are left. Returns the first component not
    /// done.
    template <unsigned int NRED>
    unsigned int Integrals(unsigned int cnt, const unsigned int nComp,
                           const unsigned int nHomo, const size_t compSize,
                           const TData *metric, const TData *planeWeights,
                           const TData *inptr, TData *dataptr)
    {
        for (; cnt + NRED <= nComp; cnt += NRED)
        {
            if (planeWeights == nullptr)
            {
                IntegralKernelLauncher<ExecSpace, IntegralOp::Square, NRED>(
                    compSize, metric, inptr + cnt * compSize, dataptr + cnt,
                    m_streamID);
            }
            else
            {
                IntegralKernelLauncher<ExecSpace, IntegralOp::Square, NRED>(
                    compSize, nHomo, metric, planeWeights,
                    inptr + cnt * nHomo * compSize, dataptr + cnt, m_streamID);
            }
        }
        return cnt;
    }

    void v_Volume(LibUtilities::MemoryRegion<TData> &data) override
    {
        // Number of elements in the block with padding.
        const size_t nelmt      = this->m_nelmt;
        constexpr auto vecWidth = NektarSpaces::max_vector_width<TData>::value;
        const size_t nelmtPad = ((nelmt + vecWidth - 1) / vecWidth) * vecWidth;

        // Initialize pointers.
        auto dataptr = data.template GetPtr<MemSpace, ReadWrite>(m_streamID);

        // The metric at the device's default interleave width.
        const TData *metric = GetMetric(NektarSpaces::Device::warpSize);

        // Volume of the block's elements
        VolumeKernelLauncher<ExecSpace>(nelmtPad * m_nqTot, metric, dataptr,
                                        m_streamID);
    }
};

} // namespace Nektar::SolverCore::detail
