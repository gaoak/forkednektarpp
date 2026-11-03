///////////////////////////////////////////////////////////////////////////////
//
// File: EnergyOpImpl.hpp
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
// Description: Device-portable kinetic-energy and enstrophy operator.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>

#include <LocalRegions/DataWarehouse/GeometricDataWarehouse.hpp>
#include <SolverCore/Energy/EnergyKernels.hpp>
#include <SolverCore/Energy/EnergyOp.hpp>

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TData>
class EnergyOpImpl : public EnergyOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    EnergyOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                 const std::vector<std::string> &components)
        : EnergyOp<TData>(expansionList, components)
    {
        m_dimension = expansionList->GetCoordim(0);
        ASSERTL0(m_dimension == 2 || m_dimension == 3,
                 "Energy supports only 2D and 3D expansions.");
        ASSERTL0(expansionList->GetShapeDimension() == m_dimension,
                 "Energy does not support embedded-manifold expansions.");
        ASSERTL0(components.size() >= m_dimension,
                 "Energy requires one velocity component per dimension.");

        // The quadrature metric of each block is fetched from the data
        // warehouse at the interleave width of the first velocity read at it.
        const size_t nBlocks =
            MultiRegions::GetCollections(expansionList).size();
        for (auto &metric : m_metric)
        {
            metric.assign(nBlocks, nullptr);
        }

        this->m_result = LibUtilities::MemoryRegion<TData>(
            "Energy result", EnergyOp<TData>::resultSize, eHostPinned);
    }

    static std::string className;

    static std::unique_ptr<MultiRegions::Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<EnergyOpImpl<ExecSpace, TData>>(expansionList,
                                                                components);
    }

protected:
    /// Interleave width of the execution space: an input block is read at
    /// this width or at width one.
    static constexpr unsigned int m_implInterleaveWidth =
        NektarSpaces::vector_width<ExecSpace, TData>::value;

    /// Quadrature metric of every point of each block, indexed by block, at
    /// interleave width one (first) and at m_implInterleaveWidth (second);
    /// null until a velocity is first read at that width.
    std::array<std::vector<const TData *>, 2> m_metric;
    std::array<unsigned int, 3> m_velocityComponents{};
    unsigned int m_dimension = 0;

    void v_Configure(
        const std::vector<unsigned int> &velocityComponents) override
    {
        ASSERTL0(velocityComponents.size() == m_dimension,
                 "Energy velocity selection must match the dimension.");
        std::copy(velocityComponents.begin(), velocityComponents.end(),
                  m_velocityComponents.begin());
    }

    template <unsigned int DIMENSION>
    void ApplyEnergy(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient)
    {
        static_assert(DIMENSION == 2 || DIMENSION == 3,
                      "Energy supports only dimensions 2 and 3.");

        for (unsigned int block = 0; block < velocity.GetBlocks().size();
             ++block)
        {
            const unsigned int streamID = block + 1;
            auto &velocityBlock         = velocity.GetBlocks()[block];
            auto &gradientBlock         = velocityGradient.GetBlocks()[block];
            const unsigned int interleaveWidth =
                velocityBlock.GetInterleaveWidth();
            ASSERTL0(interleaveWidth == 1 ||
                         interleaveWidth == m_implInterleaveWidth,
                     "Energy reads a velocity at interleave width 1 or " +
                         std::to_string(m_implInterleaveWidth) + ", not " +
                         std::to_string(interleaveWidth) + ".");
            ASSERTL1(velocityBlock.CompSize() == gradientBlock.CompSize() &&
                         gradientBlock.GetInterleaveWidth() == interleaveWidth,
                     "Energy velocity-gradient block layout mismatch.");

            // The metric is laid out like the velocity, so one storage index
            // addresses the same point in both.
            const TData *&metric =
                m_metric[interleaveWidth == 1 ? 0 : 1][block];
            if (metric == nullptr)
            {
                metric = this->m_dataWarehouse->template GetData<MemSpace>(
                    LocalRegions::QuadratureMetricKey<TData>(block,
                                                             interleaveWidth));
            }

            const auto velocityPtr =
                velocityBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            const auto gradientPtr =
                gradientBlock.template GetPtr<MemSpace, ReadOnly>(streamID);
            auto result =
                this->m_result.template GetPtr<MemSpace, ReadWrite>(streamID);

            EnergyKernel<DIMENSION, ExecSpace>(
                velocityBlock.CompSize(), m_velocityComponents[0],
                m_velocityComponents[1], m_velocityComponents[2], metric,
                velocityPtr, gradientPtr, result, streamID);
        }
    }

    void v_Apply(
        LibUtilities::Field<TData, FieldState::Phys> &velocity,
        LibUtilities::Field<TData, FieldState::Phys> &velocityGradient) override
    {
        ASSERTL1(velocity.GetNumHomoModes() == 1,
                 "Energy does not support homogeneous expansions.");
        ASSERTL1(velocity.GetNumComponents() >= m_dimension,
                 "Energy velocity component count mismatch.");
        ASSERTL1(velocity.GetBlocks().size() == m_metric[0].size() &&
                     velocityGradient.GetBlocks().size() == m_metric[0].size(),
                 "Energy velocity block count changed after configuration.");
        // Each block is integrated on its own stream, between the zeroing of
        // the result and its reduction on the null stream.
        this->m_result.template Initialize<MemSpace>(TData{0});

        switch (m_dimension)
        {
            case 2:
                ApplyEnergy<2>(velocity, velocityGradient);
                break;
            case 3:
                ApplyEnergy<3>(velocity, velocityGradient);
                break;
            default:
                ASSERTL0(false, "Energy supports only dimensions 2 and 3.");
        }

        auto comm = this->m_expansionList->GetComm();
        comm->template Reduce<MemSpace>(this->m_result, LibUtilities::ReduceSum,
                                        0);
    }
};

} // namespace Nektar::SolverCore::detail
